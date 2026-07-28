"""Apollo v2: a lighter, wider-context generator.

Not checkpoint-compatible with v1 -- this is a from-scratch architecture. v1 is
untouched in `apollo.py` and still loads the released weights.

The starting point was measuring where v1 actually spends its 16.5M parameters:

    seq_net (ICB)          9.50M   57.4%     <- temporal path
    band_net/MLP           4.72M   28.5%
    band_net/attn          1.57M    9.5%     <- the paper's actual contribution
    projections            0.74M    4.5%

86% of the model is feed-forward, and the largest block of it -- the ICB -- is
three stacked 4x-expansion FFNs whose whole job is a kernel-7 convolution over
time: 19 frames (190 ms) per layer, 109 frames (1.09 s) across the six-layer
stack. That is an expensive way to buy temporal context.

v2 changes four things:

1. **Temporal path.** 3 blocks of 4x expansion at kernel 7 become 2 blocks of 2.5x
   expansion at kernel 11 with dilations (1, 2): 58% fewer parameters, and the
   stack's context grows from 1.09 s to 1.81 s. The depthwise convolution is
   nearly free, so most of that comes for nothing.
2. **SwiGLU at standard width.** v1 expands to 8x the feature dim and chunks it in
   two, which is roughly 3x wider than a normal SwiGLU. v2 uses the conventional
   2.67x, matching a 4x vanilla FFN's parameter count. It also drops v1's
   accidental double-SiLU on the gate branch.
3. **Residual output.** v1 synthesises the entire complex spectrum from scratch,
   including the low band a codec left intact, and its loss never constrains
   phase -- so everything it gets slightly wrong comes back as broadband noise.
   v2 predicts a correction to the input spectrum instead, which makes "leave it
   alone" the cheap default rather than something the model must learn.
4. **Optional mid/side.** Lossy codecs joint-stereo-code, so their artefacts are
   structured in M/S, not L/R. Processing in the same domain matches the damage.

Roughly 9.5M parameters against v1's 16.5M.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .apollo import RMSNorm, grouped_rms_norm
from .base_model import BaseModel


class SwiGLU(nn.Module):
    """Gated feed-forward at the conventional expansion ratio.

    v1 builds this as ``Conv1d(N, N*8) -> SiLU -> chunk -> SiLU(gate) * z``, which
    both over-widens the block and applies SiLU to the gate twice. This is the
    standard formulation.
    """

    def __init__(self, dim, mult=2.67):
        super().__init__()
        hidden = int(dim * mult)
        self.norm = RMSNorm(dim)
        self.proj_in = nn.Conv1d(dim, hidden * 2, 1, bias=False)
        self.proj_out = nn.Conv1d(hidden, dim, 1, bias=False)

    def forward(self, x):
        gate, value = self.proj_in(self.norm(x)).chunk(2, dim=1)
        return self.proj_out(F.silu(gate) * value)


class BandAttention(nn.Module):
    """Self-attention across the 80 bands, with rotary position encoding.

    Structurally the same as v1's Roformer -- this is the part of the architecture
    that earns its keep, at under 10% of the parameters -- with the feed-forward
    swapped for a properly sized SwiGLU.
    """

    def __init__(self, dim, num_head=8, theta=10000, window=128, ffn_mult=2.67,
                 attention_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_head = num_head
        self.head_dim = dim // num_head
        self.attention_drop = attention_drop

        cos_freq, sin_freq = self._rotary_tables(theta, window)
        self.register_buffer("cos_freq", cos_freq, persistent=False)
        self.register_buffer("sin_freq", sin_freq, persistent=False)

        self.norm = RMSNorm(dim)
        self.qkv = nn.Conv1d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv1d(dim, dim, 1, bias=False)
        self.ffn = SwiGLU(dim, mult=ffn_mult)

    def _rotary_tables(self, theta, window):
        freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2)[: self.head_dim // 2] / self.head_dim))
        pos = torch.arange(0, window).reshape(-1, 1)
        angles = pos * freq.reshape(1, -1)
        cos = torch.stack([angles.cos()] * 2, -1).reshape(window, self.head_dim)
        sin = torch.stack([angles.sin()] * 2, -1).reshape(window, self.head_dim)
        return cos, sin

    def _rotate(self, feature):
        # interleaved (a, b) -> (-b, a)
        n = feature.shape[-1]
        pairs = feature.reshape(*feature.shape[:-1], n // 2, 2)
        return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).reshape(feature.shape)

    def _apply_rope(self, feature):
        seq = feature.shape[-2]
        cos = self.cos_freq[:seq].to(feature.dtype)
        sin = self.sin_freq[:seq].to(feature.dtype)
        return feature * cos + self._rotate(feature) * sin

    def forward(self, x):
        # x: (B, dim, nband)
        B, _, L = x.shape

        qkv = self.qkv(self.norm(x)).reshape(B, self.num_head, self.head_dim * 3, L).mT
        q, k, v = torch.split(qkv, self.head_dim, dim=-1)

        q = self._apply_rope(q).contiguous()
        k = self._apply_rope(k).contiguous()

        attended = F.scaled_dot_product_attention(
            q, k, v.contiguous(),
            dropout_p=self.attention_drop if self.training else 0.0,
            is_causal=False,
        )
        x = x + self.proj(attended.mT.reshape(B, -1, L))

        return x + self.ffn(x)


class TemporalBlock(nn.Module):
    """Dilated depthwise conv plus a gated feed-forward, over time."""

    def __init__(self, dim, kernel=11, dilation=1, mult=2.5):
        super().__init__()
        hidden = int(dim * mult)
        padding = (kernel - 1) // 2 * dilation

        self.depthwise = nn.Conv1d(dim, dim, kernel, padding=padding,
                                   dilation=dilation, groups=dim)
        self.norm = RMSNorm(dim)
        self.proj_in = nn.Conv1d(dim, hidden, 1)
        self.proj_out = nn.Conv1d(hidden, dim, 1)

    def forward(self, x):
        h = self.norm(self.depthwise(x))
        return x + self.proj_out(F.silu(self.proj_in(h)))


class TemporalStack(nn.Module):
    """The v2 replacement for v1's ICB.

    Per layer: v1 is 3 blocks, kernel 7, no dilation -> 19 frames for 1.58M
    params; v2 is 2 blocks, kernel 11, dilations (1, 2) -> 31 frames for 0.66M.
    Across a six-layer stack that is 1.09 s of context for v1 against 1.81 s here.

    The depthwise convolution is nearly free in parameters -- the FFN either side
    is what actually holds them -- so kernel and dilation are the cheapest knobs in
    the whole architecture.

    Do not push this past the training segment length: context the model can never
    have seen during training is context it cannot use, and it forces a larger
    overlap at chunked inference to avoid truncating it.
    """

    def __init__(self, dim, blocks=2, kernel=11, mult=2.5, dilations=(1, 2)):
        super().__init__()
        dilations = list(dilations)[:blocks] or [1]
        while len(dilations) < blocks:
            dilations.append(dilations[-1] * 3)
        self.blocks = nn.Sequential(*[
            TemporalBlock(dim, kernel=kernel, dilation=d, mult=mult) for d in dilations
        ])
        self.receptive_field = 1 + sum((kernel - 1) * d for d in dilations)

    def forward(self, x):
        return self.blocks(x)


class BSNetV2(nn.Module):
    def __init__(self, feature_dim, ffn_mult=2.67, icb_blocks=2, icb_kernel=11,
                 icb_mult=2.5, icb_dilations=(1, 2)):
        super().__init__()
        self.feature_dim = feature_dim
        self.band_net = BandAttention(feature_dim, ffn_mult=ffn_mult)
        self.seq_net = TemporalStack(feature_dim, blocks=icb_blocks, kernel=icb_kernel,
                                     mult=icb_mult, dilations=icb_dilations)
        self.slice_elems = 0

        # widest intermediate per row, taken from the real layer widths rather than
        # a guessed multiplier -- getting this wrong silently defeats the budget
        self.band_peak_channels = self.band_net.ffn.proj_in.out_channels
        self.seq_peak_channels = self.seq_net.blocks[0].proj_in.out_channels

    def _run_sliced(self, fn, x, elems_per_row):
        if self.slice_elems <= 0 or torch.is_grad_enabled():
            return fn(x)
        chunk = max(1, int(self.slice_elems // max(1, elems_per_row)))
        if x.shape[0] <= chunk:
            return fn(x)
        return torch.cat([fn(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)], 0)

    def forward(self, x):
        # x: (B, nband, N, T)
        B, nband, N, T = x.shape

        band_input = x.permute(0, 3, 2, 1).reshape(B * T, N, nband)
        band_output = self._run_sliced(self.band_net, band_input,
                                       self.band_peak_channels * nband)
        band_output = band_output.reshape(B, T, N, nband).permute(0, 3, 2, 1)

        seq_input = band_output.reshape(B * nband, N, T)
        return self._run_sliced(self.seq_net, seq_input,
                                self.seq_peak_channels * T).reshape(B, nband, N, T)


class ApolloV2(BaseModel):
    """Band-split restoration generator, second generation.

    Args:
        output_mode: ``"residual"`` predicts a correction to the input spectrum --
            the default, and the reason v2 is quieter by construction. ``"direct"``
            reproduces v1's behaviour of synthesising the whole spectrum.
        stereo_mode: ``"ms"`` converts to mid/side before analysis. Lossy codecs
            joint-stereo-code, so their artefacts are structured that way. Ignored
            for mono input.
    """

    def __init__(
        self,
        sr: int = 44100,
        win: int = 20,
        feature_dim: int = 256,
        layer: int = 6,
        nband: int = 80,
        ffn_mult: float = 2.67,
        icb_blocks: int = 2,
        icb_kernel: int = 11,
        icb_mult: float = 2.5,
        icb_dilations=(1, 2),
        output_mode: str = "residual",
        stereo_mode: str = "lr",
        output_init_scale: float = 0.1,
    ):
        super().__init__(sample_rate=sr)

        if output_mode not in ("residual", "direct"):
            raise ValueError(f"output_mode must be residual|direct, got {output_mode!r}")
        if stereo_mode not in ("lr", "ms"):
            raise ValueError(f"stereo_mode must be lr|ms, got {stereo_mode!r}")

        self.sr = sr
        self.win = int(sr * win // 1000)
        self.stride = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.output_mode = output_mode
        self.stereo_mode = stereo_mode
        self.eps = torch.finfo(torch.float32).eps

        bandwidth = int(self.win / (nband * 2))
        self.band_width = [bandwidth] * (nband - 1)
        self.band_width.append(self.enc_dim - int(np.sum(self.band_width)))
        self.nband = len(self.band_width)
        self.band_groups = self._build_band_groups(self.band_width)

        # Band projections stored as one stacked parameter per group rather than a
        # ModuleList of 80 tiny convs -- v1 needed the ModuleList for checkpoint
        # compatibility, v2 does not.
        self.bn_norm = nn.ParameterList()
        self.bn_weight = nn.ParameterList()
        self.bn_bias = nn.ParameterList()
        self.out_norm = nn.ParameterList()
        self.out_weight = nn.ParameterList()
        self.out_bias = nn.ParameterList()

        for _, count, width in self.band_groups:
            in_dim = width * 2 + 1
            self.bn_norm.append(nn.Parameter(torch.ones(count, in_dim)))
            self.bn_weight.append(nn.Parameter(torch.randn(count, feature_dim, in_dim) / in_dim ** 0.5))
            self.bn_bias.append(nn.Parameter(torch.zeros(count, feature_dim)))

            self.out_norm.append(nn.Parameter(torch.ones(count, feature_dim)))
            # scaled-down init: with a residual output this starts the model close
            # to a pass-through, which converges far faster than learning identity
            self.out_weight.append(nn.Parameter(
                torch.randn(count, width * 4, feature_dim) * output_init_scale / feature_dim ** 0.5
            ))
            self.out_bias.append(nn.Parameter(torch.zeros(count, width * 4)))

        self.net = nn.Sequential(*[
            BSNetV2(feature_dim, ffn_mult=ffn_mult, icb_blocks=icb_blocks,
                    icb_kernel=icb_kernel, icb_mult=icb_mult, icb_dilations=icb_dilations)
            for _ in range(layer)
        ])

        self.register_buffer("stft_window", torch.hann_window(self.win), persistent=False)
        self.grad_checkpointing = False

    # ------------------------------------------------------------------

    @staticmethod
    def _build_band_groups(band_width):
        groups, start = [], 0
        for i in range(1, len(band_width) + 1):
            if i == len(band_width) or band_width[i] != band_width[start]:
                groups.append((start, i - start, band_width[start]))
                start = i
        return groups

    @property
    def receptive_field_frames(self):
        """Temporal context in STFT frames, summed over the stack."""
        per_layer = self.net[0].seq_net.receptive_field - 1
        return 1 + per_layer * len(self.net)

    @property
    def receptive_field_ms(self):
        return self.receptive_field_frames * self.stride / self.sr * 1000

    def set_gradient_checkpointing(self, enabled=True):
        self.grad_checkpointing = bool(enabled)
        return self

    def set_vram_budget(self, megabytes):
        elems = int(float(megabytes) * (2 ** 20) / 4) if megabytes else 0
        for module in self.modules():
            if isinstance(module, BSNetV2):
                module.slice_elems = elems
        return self

    # ------------------------------------------------------------------

    def _to_working_channels(self, wav):
        if self.stereo_mode == "ms" and wav.shape[1] == 2:
            left, right = wav[:, 0], wav[:, 1]
            return torch.stack([(left + right) * 0.5, (left - right) * 0.5], 1)
        return wav

    def _from_working_channels(self, wav):
        if self.stereo_mode == "ms" and wav.shape[1] == 2:
            mid, side = wav[:, 0], wav[:, 1]
            return torch.stack([mid + side, mid - side], 1)
        return wav

    def _encode(self, spec):
        """Complex STFT -> per-band features, batched over equally sized bands."""
        features = []
        bin_idx = 0
        for (_, count, width), norm_w, weight, bias in zip(
            self.band_groups, self.bn_norm, self.bn_weight, self.bn_bias
        ):
            nbins = count * width
            band = spec[:, bin_idx:bin_idx + nbins]
            bin_idx += nbins

            Bc, _, T = band.shape
            real = band.real.reshape(Bc, count, width, T)
            imag = band.imag.reshape(Bc, count, width, T)

            power = (real.pow(2).sum(2) + imag.pow(2).sum(2) + self.eps).sqrt().unsqueeze(2)
            concat = torch.cat([real / power, imag / power, torch.log(power)], 2)

            normed = grouped_rms_norm(concat, norm_w)
            features.append(torch.matmul(weight, normed) + bias.unsqueeze(0).unsqueeze(-1))

        return torch.cat(features, 1)

    def _decode(self, feature):
        """Per-band features -> complex spectrum."""
        out = []
        band_idx = 0
        for (_, count, width), norm_w, weight, bias in zip(
            self.band_groups, self.out_norm, self.out_weight, self.out_bias
        ):
            feat = feature[:, band_idx:band_idx + count]
            band_idx += count

            normed = grouped_rms_norm(feat, norm_w)
            proj = torch.matmul(weight, normed) + bias.unsqueeze(0).unsqueeze(-1)
            RI = F.glu(proj, dim=2)

            Bc, _, _, T = RI.shape
            RI = RI.float().reshape(Bc, count, 2, width, T)
            out.append(torch.complex(RI[:, :, 0], RI[:, :, 1]).reshape(Bc, count * width, T))

        return torch.cat(out, 1)

    def forward(self, input):
        B, nch, nsample = input.shape

        working = self._to_working_channels(input)
        window = self.stft_window.to(dtype=working.dtype, device=working.device)

        spec = torch.stft(working.reshape(B * nch, nsample), n_fft=self.win,
                          hop_length=self.stride, window=window, return_complex=True)

        feature = self._encode(spec)

        if self.grad_checkpointing and torch.is_grad_enabled():
            for block in self.net:
                feature = torch.utils.checkpoint.checkpoint(block, feature, use_reentrant=False)
        else:
            feature = self.net(feature)

        est_spec = self._decode(feature)
        if self.output_mode == "residual":
            # the model corrects the input rather than replacing it, so anything
            # the codec left intact survives untouched by default
            est_spec = spec + est_spec

        output = torch.istft(est_spec, n_fft=self.win, hop_length=self.stride,
                             window=window.to(est_spec.real.dtype),
                             length=nsample).reshape(B, nch, -1)

        return self._from_working_channels(output)

    def get_model_args(self):
        """Everything needed to rebuild this model.

        Unlike v1, several v2 settings (dilations, output mode, stereo mode) leave
        no trace in the weight shapes, so they cannot be inferred from a state
        dict. Checkpoints carry this dict and the loader reads it back.
        """
        first = self.net[0]
        block = first.seq_net.blocks[0]
        return {
            "sr": self.sr,
            "win": round(self.win * 1000 / self.sr),
            "feature_dim": self.feature_dim,
            "layer": len(self.net),
            "nband": self.nband,
            "ffn_mult": first.band_net.ffn.proj_in.out_channels / 2 / self.feature_dim,
            "icb_blocks": len(first.seq_net.blocks),
            "icb_kernel": block.depthwise.kernel_size[0],
            "icb_mult": block.proj_in.out_channels / self.feature_dim,
            "icb_dilations": [b.depthwise.dilation[0] for b in first.seq_net.blocks],
            "output_mode": self.output_mode,
            "stereo_mode": self.stereo_mode,
        }
