"""Lightweight critics, ported from the SingingVocoders `fast_D` / `bigvgan_D` recipe.

Ported from `H:/Chouwa/SingingVocoders/modules/fast_D/discriminator.py` and the
Fast* branch wrappers in `modules/bigvgan_D/discriminator.py`, adapted to
Apollo's `(outputs, feature_maps)` + `branches` interface.

Two ideas make these much cheaper than the textbook HiFi-GAN / UnivNet critics,
without giving up what a critic is for:

* **Space-to-depth instead of strided convolutions.** A standard MPD downsamples
  with stride-3 convolutions and doubles the channel count each time, ending at
  1024 channels -- roughly 8M parameters *per period*. FastPD gets the same
  effect with a `view`: time is folded into the feature axis, which is free and
  throws nothing away. The convolutions then run on short sequences.
* **RMSNorm instead of weight/spectral norm.** Spectral norm runs a power
  iteration on every forward, and the critic is called four times per training
  step. RMSNorm costs a reduction.

The final projection carries a **SAN** head by default: L2-normalising the
projection weight turns the logit into a direction-only (cosine) similarity,
which bounds each branch's output without spectral norm or a gradient penalty.
That is the cheapest available answer to a critic that overpowers the generator.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


class L2Normalize(nn.Module):
    """Parametrisation that L2-normalises a weight tensor to unit norm (SAN)."""

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        flat = weight.reshape(weight.shape[0], -1)
        return F.normalize(flat, dim=1).reshape_as(weight)


def apply_san(layer: nn.Module) -> nn.Module:
    parametrize.register_parametrization(layer, "weight", L2Normalize())
    return layer


class SoftSignGLUFunction(torch.autograd.Function):
    """Gated softsign with a hand-written backward.

    Saves the two partial derivatives instead of the inputs, so the block holds
    two tensors through the backward pass rather than four.
    """

    @staticmethod
    def forward(ctx, out, gate):
        denom_out = out.abs().add(1.0)
        denom_gate = gate.abs().add(1.0)
        out = out / denom_out
        gate = gate / denom_gate
        ctx.save_for_backward(out / denom_gate / denom_gate,
                              gate / denom_out / denom_out)
        return out * gate

    @staticmethod
    def backward(ctx, grad_output):
        out_d_gate, gate_d_out = ctx.saved_tensors
        return grad_output * gate_d_out, grad_output * out_d_gate


class SoftSignGLU(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return SoftSignGLUFunction.apply(out, gate)


class _Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class _Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class LYNXNet2Block(nn.Module):
    """Depthwise conv followed by two gated linear layers."""

    def __init__(self, dim, kernel_size=11, use_dwconv=True):
        super().__init__()
        self.net = nn.Sequential(
            _Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2,
                      groups=dim if use_dwconv else 1),
            _Transpose((1, 2)),
            nn.Linear(dim, dim * 2),
            SoftSignGLU(),
            nn.Linear(dim, dim * 2),
            SoftSignGLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, norm_x=None):
        if norm_x is None:
            norm_x = F.rms_norm(x, (x.size(-1),))
        return x + self.net(norm_x)


class ResBlock2d(nn.Module):
    def __init__(self, dim, use_dwconv=False):
        super().__init__()
        self.net = nn.Sequential(
            _Permute((0, 3, 1, 2)),
            nn.Conv2d(dim, dim * 2, 3, padding=1, groups=dim if use_dwconv else 1),
            _Permute((0, 2, 3, 1)),
            SoftSignGLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, norm_x=None):
        if norm_x is None:
            norm_x = F.rms_norm(x, (x.size(-1),))
        return x + self.net(norm_x)


# ---------------------------------------------------------------- branches


class FastPD(nn.Module):
    """Single-period branch. Folds the waveform by `period`, then by `strides`."""

    def __init__(self, period, init_channel=8, strides=(4, 4, 4), kernel_size=11,
                 use_san=True):
        super().__init__()
        self.period = period
        self.strides = list(strides)
        self.hop_length = period * int(np.prod(self.strides))

        self.pre = nn.Linear(self.strides[0], init_channel * self.strides[0])
        self.residual_layers = nn.ModuleList([
            LYNXNet2Block(dim=init_channel * int(np.prod(self.strides[: i + 1])),
                          kernel_size=kernel_size, use_dwconv=(i > 0))
            for i in range(len(self.strides))
        ])
        self.post = nn.Linear(init_channel * int(np.prod(self.strides)), 1)
        if use_san:
            apply_san(self.post)

    def forward(self, x):
        # x: (B, 1, T)
        b, _, t = x.shape
        usable = (t // self.hop_length) * self.hop_length
        if usable == 0:
            raise ValueError(
                f"input of {t} samples is too short for period branch {self.period} "
                f"(needs at least {self.hop_length})"
            )
        x = x[:, :, :usable].view(b, -1, self.period)
        x = x.transpose(1, 2).reshape(b * self.period, -1, self.strides[0])

        x = F.gelu(self.pre(x))
        fmap = []
        for i, layer in enumerate(self.residual_layers):
            if i > 0 and self.strides[i] > 1:
                x = x.view(b * self.period, -1, x.size(2) * self.strides[i])
            norm_x = F.rms_norm(x, (x.size(-1),))
            if i > 0:
                fmap.append(norm_x.reshape(b, -1))
            x = layer(x, norm_x)

        x = F.rms_norm(x, (x.size(-1),))
        return self.post(x).reshape(b, -1), fmap


class FastSpecD(nn.Module):
    """Single-resolution branch on the log-magnitude STFT."""

    def __init__(self, init_channel=8, strides=(4, 2, 2), fft_size=1024,
                 shift_size=128, win_length=1024, use_san=True):
        super().__init__()
        self.strides = list(strides)
        self.expansion = int(np.prod(self.strides))
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer("freq_coords",
                             torch.linspace(-1, 1, fft_size // 2 + 1).view(-1, 1, 1),
                             persistent=False)

        self.pre = nn.Linear(self.strides[0], init_channel * self.strides[0])
        self.freq = nn.Linear(1, init_channel * self.strides[0])
        self.layers = nn.ModuleList([
            ResBlock2d(init_channel * int(np.prod(self.strides[: i + 1])))
            for i in range(len(self.strides))
        ])
        self.post = nn.Linear(init_channel * self.expansion, 1)
        if use_san:
            apply_san(self.post)

    def _front_end(self, x):
        spec = torch.stft(x, self.fft_size, self.shift_size, self.win_length,
                          self.window.to(x.device), return_complex=True).abs()
        return torch.log10(spec.clamp(min=1e-5))

    def forward(self, x):
        x = self._front_end(x.squeeze(1).float())
        return self._backbone(x)

    def _backbone(self, x):
        b, f, t = x.shape
        usable = (t // self.expansion) * self.expansion
        if usable == 0:
            raise ValueError(f"spectrogram of {t} frames is too short for this branch")
        x = x[:, :, :usable, None].view(b, f, -1, self.strides[0])

        x = self.pre(x) + self.freq(self.freq_coords.to(x.device))
        fmap = []
        for i, layer in enumerate(self.layers):
            if i > 0:
                x = x.view(b, f, -1, x.size(3) * self.strides[i])
            norm_x = F.rms_norm(x, (x.size(-1),))
            if i > 0:
                fmap.append(norm_x)
            x = layer(x, norm_x)

        x = F.rms_norm(x, (x.size(-1),))
        return self.post(x), fmap


class FastMelD(FastSpecD):
    """Log-mel branch: a cheap stand-in for a constant-Q critic.

    Mel is approximately logarithmic above ~1 kHz, so this keeps the harmonic
    emphasis a CQT branch is chosen for, at the cost of one STFT and one matmul
    rather than a filterbank run at several scales.
    """

    def __init__(self, sample_rate=44100, init_channel=8, strides=(4, 2, 2),
                 fft_size=2048, shift_size=512, win_length=2048, n_mels=128,
                 fmin=0.0, fmax=None, use_san=True):
        super().__init__(init_channel=init_channel, strides=strides,
                         fft_size=fft_size, shift_size=shift_size,
                         win_length=win_length, use_san=use_san)

        from ..losses.perceptual import _mel_basis

        self.register_buffer("mel_basis",
                             _mel_basis(sample_rate, fft_size, n_mels, fmin, fmax),
                             persistent=False)
        # the frequency-position embedding must span mel bins, not FFT bins
        self.register_buffer("freq_coords",
                             torch.linspace(-1, 1, n_mels).view(-1, 1, 1),
                             persistent=False)

    def _front_end(self, x):
        spec = torch.stft(x, self.fft_size, self.shift_size, self.win_length,
                          self.window.to(x.device), return_complex=True).abs()
        mel = torch.matmul(self.mel_basis.to(x.device), spec)
        return torch.log10(mel.clamp(min=1e-5))


# ---------------------------------------------------------------- banks


class _FastBank(nn.Module):
    """Shared plumbing: power normalisation, channel folding, branch selection.

    Subclasses set ``discriminators`` and should set ``branch_names`` to a
    same-length list. The names are what the training loop's per-branch schedule
    matches its glob patterns against; a bank that does not set them falls back to
    positional names.
    """

    branch_names = None

    def __init__(self, nch=1):
        super().__init__()
        self.nch = nch
        self.eps = torch.finfo(torch.float32).eps

    def names(self):
        if self.branch_names is not None:
            return list(self.branch_names)
        return [f"branch_{i}" for i in range(len(self.discriminators))]

    def _prepare(self, est):
        b, nch, t = est.shape
        assert nch == self.nch, f"discriminator built for {self.nch}ch, got {nch}ch"
        est = est / (est.pow(2).sum((1, 2)) + self.eps).sqrt().reshape(b, 1, 1)
        return est.reshape(b * nch, 1, t)

    def forward(self, est, sample_rate=44100, branches=None):
        folded = self._prepare(est)
        indices = range(len(self.discriminators)) if branches is None else list(branches)

        outputs, feature_maps = [], []
        for i in indices:
            score, features = self.discriminators[i](folded)
            outputs.append(score)
            feature_maps.append(features)
        return outputs, feature_maps


class FastMPD(_FastBank):
    """Multi-period critic. Coprime periods so the branches disagree about
    what counts as periodic."""

    def __init__(self, nch=1, periods=(2, 3, 5, 7, 11), init_channel=8,
                 strides=(4, 4, 4), kernel_size=11, use_san=True):
        super().__init__(nch=nch)
        self.periods = list(periods)
        self.discriminators = nn.ModuleList([
            FastPD(p, init_channel=init_channel, strides=strides,
                   kernel_size=kernel_size, use_san=use_san)
            for p in self.periods
        ])
        self.branch_names = [f"mpd_{p}" for p in self.periods]


class FastMRD(_FastBank):
    """Multi-resolution magnitude critic."""

    def __init__(self, nch=1, fft_sizes=(2048, 1024, 512), hop_sizes=(256, 128, 64),
                 win_lengths=(2048, 1024, 512), init_channel=8, strides=(4, 2, 2),
                 use_san=True):
        super().__init__(nch=nch)
        self.discriminators = nn.ModuleList([
            FastSpecD(init_channel=init_channel, strides=strides, fft_size=n,
                      shift_size=h, win_length=w, use_san=use_san)
            for n, h, w in zip(fft_sizes, hop_sizes, win_lengths)
        ])
        self.branch_names = [f"mrd_{n}" for n in fft_sizes]


class FastMelBank(_FastBank):
    """Log-frequency critic, the cheap substitute for a CQT branch."""

    def __init__(self, nch=1, sample_rate=44100, fft_sizes=(2048,), hop_sizes=(512,),
                 win_lengths=(2048,), n_mels=128, init_channel=8, strides=(4, 2, 2),
                 fmin=0.0, fmax=None, use_san=True):
        super().__init__(nch=nch)
        self.discriminators = nn.ModuleList([
            FastMelD(sample_rate=sample_rate, init_channel=init_channel,
                     strides=strides, fft_size=n, shift_size=h, win_length=w,
                     n_mels=n_mels, fmin=fmin, fmax=fmax, use_san=use_san)
            for n, h, w in zip(fft_sizes, hop_sizes, win_lengths)
        ])
        self.branch_names = [f"mel_{n}" for n in fft_sizes]
