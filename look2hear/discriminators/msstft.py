"""Multi-scale complex-STFT critic (EnCodec family).

Ported from `H:/Chouwa/SingingVocoders/modules/chouwa_D/discriminator.py`
(`DiscriminatorSTFT`), whose two non-obvious parameters both turned out to matter
a lot here:

* **`freq_stride`.** Striding only the time axis leaves every layer running over
  all `n_fft/2+1` frequency bins, and the deep wide layers then dominate the cost
  of the entire critic. A first attempt at this file did exactly that and measured
  52 seconds per training step. Striding frequency as well brings it in line with
  everything else. Layer-1 feature maps still supervise at full resolution through
  feature matching, so the acuity is not lost.

* **`mag_compression`.** With linear real/imaginary input, bins 30-50 dB below the
  fundamental have almost no influence on the logit -- so a "full-band" branch
  behaves as a low-band critic in practice. For Apollo that is precisely backwards:
  the high band is the part being reconstructed and the part where artefacts live.
  Compressing to `mag^alpha * e^(i*phase)` keeps phase intact while giving quiet
  bins real gradient weight.

This is the only critic family in the bank that sees phase. Apollo's reconstruction
loss compares magnitudes only, so without one of these, nothing in the objective
constrains phase at all.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize, weight_norm, spectral_norm

from .fast import L2Normalize


class STFTScaleDiscriminator(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, channels=32,
                 max_channels=128, n_layers=3, freq_stride=2, mag_compression=0.3,
                 use_spectral_norm=False, use_san=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mag_compression = float(mag_compression)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList()
        self.convs.append(norm_f(nn.Conv2d(2, channels, (3, 7), padding=(1, 3))))

        in_ch = channels
        for i in range(n_layers - 1):
            out_ch = min(channels * (2 ** (i + 1)), max_channels)
            self.convs.append(norm_f(nn.Conv2d(
                in_ch, out_ch, (3, 7),
                stride=(max(1, int(freq_stride)), 2),
                padding=(1, 3),
            )))
            in_ch = out_ch

        self.convs.append(norm_f(nn.Conv2d(in_ch, in_ch, (3, 3), padding=(1, 1))))

        self.conv_post = nn.Conv2d(in_ch, 1, (3, 3), padding=(1, 1))
        if use_san:
            parametrize.register_parametrization(self.conv_post, "weight", L2Normalize())
        else:
            self.conv_post = norm_f(self.conv_post)

        # cuDNN picks much faster NHWC kernels for these shapes; identical math
        self.to(memory_format=torch.channels_last)

    def _stack_input(self, real, imag):
        if self.mag_compression < 1.0:
            # x * mag^(alpha-1) == mag^alpha * e^(i*phase), fused as one pow on
            # the squared magnitude. The additive floor bounds the backward gain
            # and keeps exactly-zero bins (digital silence) NaN-free.
            scale = (real * real + imag * imag + 1e-4).pow(
                (self.mag_compression - 1.0) * 0.5)
            real = real * scale
            imag = imag * scale
        return torch.stack([real, imag], dim=1).contiguous(
            memory_format=torch.channels_last)

    def forward(self, x):
        x_sq = x.squeeze(1).float()
        pad = (self.n_fft - self.hop_length) // 2
        x_sq = F.pad(x_sq, (pad, pad), mode="constant")
        spec = torch.stft(
            x_sq, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=x_sq.dtype, device=x_sq.device),
            center=False, return_complex=True,
        )

        h = self._stack_input(spec.real, spec.imag)
        fmap = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            fmap.append(h)

        out = self.conv_post(h)
        return torch.flatten(out, 1, -1), fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """Complex-STFT critic at several resolutions.

    Args:
        freq_stride: 2 halves the bin count per strided layer, roughly halving
            the cost. 1 keeps full per-bin acuity for narrow artefacts.
        mag_compression: exponent applied to the magnitude, phase untouched.
            1.0 disables it and reproduces plain real/imaginary input.
    """

    def __init__(self, nch=1, n_ffts=(2048, 1024, 512), hop_ratio=4,
                 channels=32, max_channels=128, n_layers=3, freq_stride=2,
                 mag_compression=0.3, use_spectral_norm=False, use_san=True):
        super().__init__()
        self.nch = nch
        self.n_ffts = list(n_ffts)
        self.eps = torch.finfo(torch.float32).eps
        self.discriminators = nn.ModuleList([
            STFTScaleDiscriminator(
                n_fft=n, hop_length=n // hop_ratio, win_length=n,
                channels=channels, max_channels=max_channels, n_layers=n_layers,
                freq_stride=freq_stride, mag_compression=mag_compression,
                use_spectral_norm=use_spectral_norm, use_san=use_san)
            for n in self.n_ffts
        ])
        self.branch_names = [f"msstft_{n}" for n in self.n_ffts]

    def forward(self, est, sample_rate=44100, branches=None):
        b, nch, t = est.shape
        assert nch == self.nch, f"discriminator built for {self.nch}ch, got {nch}ch"

        est = est / (est.pow(2).sum((1, 2)) + self.eps).sqrt().reshape(b, 1, 1)
        folded = est.reshape(b * nch, 1, t)

        indices = range(len(self.discriminators)) if branches is None else list(branches)

        outputs, feature_maps = [], []
        for i in indices:
            score, features = self.discriminators[i](folded)
            outputs.append(score)
            feature_maps.append(features)

        return outputs, feature_maps
