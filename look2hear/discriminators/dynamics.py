"""Critics that judge how energy *moves*, rather than what the spectrum contains.

Ported from `H:/Chouwa/SingingVocoders/modules/med_D/discriminator.py` and
`modules/chouwa_D/discriminator.py`, adapted to Apollo's
`(outputs, feature_maps)` + `branches` interface.

Every branch Apollo had before this file judges a static picture: the magnitude or
real/imaginary content of a frame (`FastMRD`, `FastMelBank`, `MultiScaleSTFTDiscriminator`),
or the waveform folded by a period (`FastMPD`). None of them can express *dynamics* --
how the energy in a band rises and falls over tens or hundreds of milliseconds.

That is a strange gap for a restoration model, because dynamics is most of what a
low-bitrate codec destroys. Pre-echo spreads a transient backwards across the
window it lives in. The attack of a consonant is smeared. Quantisation noise fills
the pauses with a static hiss that has the right *level* and the wrong *movement*.
A critic that only sees per-frame magnitude will happily accept all three, and the
reconstruction terms cannot object either -- `freq_MAE` is a magnitude comparison
and `fullness`/`bleedless` are mel-domain level penalties.

The four families here close that gap, and they are cheap because the analysis
front-ends have no trainable parameters and everything after them runs at the
envelope rate rather than the sample rate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize, spectral_norm, weight_norm

from .fast import L2Normalize, _FastBank


def _norm(use_spectral_norm):
    return spectral_norm if use_spectral_norm else weight_norm


def _head(conv, use_san, use_spectral_norm):
    """Final projection: SAN-normalised, or plain weight/spectral norm."""
    if use_san:
        parametrize.register_parametrization(conv, "weight", L2Normalize())
        return conv
    return _norm(use_spectral_norm)(conv)


class FixedSincSubBand(nn.Module):
    """Fixed sinc filterbank, a lightweight stand-in for a PQMF analysis.

    No trainable parameters: the bands are windowed-sinc bandpasses built once at
    construction. That is the whole reason the branches below are cheap.
    """

    def __init__(self, sample_rate=44100, n_bands=8, kernel_size=127):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_bands = max(2, int(n_bands))
        ks = int(kernel_size)
        self.kernel_size = ks if ks % 2 == 1 else ks + 1
        self.register_buffer("filters", self._build_filters(), persistent=False)

    def _build_filters(self):
        nyquist = self.sample_rate / 2.0
        edges = torch.linspace(0.0, nyquist, self.n_bands + 1)
        n = torch.arange(self.kernel_size, dtype=torch.float32) - self.kernel_size // 2
        window = torch.hamming_window(self.kernel_size, periodic=False)

        filters = []
        for low, high in zip(edges[:-1], edges[1:]):
            h = (self._lowpass(float(high), n) - self._lowpass(float(low), n)) * window
            filters.append(h / h.abs().sum().clamp_min(1e-6))
        return torch.stack(filters, dim=0).unsqueeze(1)

    def _lowpass(self, cutoff_hz, n):
        cutoff = min(max(cutoff_hz / self.sample_rate, 0.0), 0.5)
        if cutoff <= 0.0:
            return torch.zeros_like(n)
        return 2.0 * cutoff * torch.sinc(2.0 * cutoff * n)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x):
        # Autocast off: under bf16 cuDNN picks a much slower kernel for a 127-tap
        # conv1d at full sample rate. The branch body after this still autocasts.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.float()
        pad = self.kernel_size // 2
        filters = self.filters.to(device=x.device, dtype=x.dtype)
        return F.conv1d(F.pad(x, (pad, pad), mode="reflect"), filters)


# ------------------------------------------------------------------ envelope


class EnvelopeDiscriminator(nn.Module):
    """One MED branch: multi-band amplitude envelope at a single time constant.

    Pipeline: split into `n_bands` with the fixed sinc bank, rectify, smooth with
    an average pool of width `pool` and stride `stride` -- that pair *is* the
    envelope detector and sets this branch's time constant -- then compress with
    `log1p` and derive three planes:

    * **envelope** -- the level itself;
    * **delta** -- its first difference, which is what separates an attack from
      a fade, and the plane pre-echo shows up in;
    * **local contrast** -- envelope minus its own moving average, so the branch
      judges the *shape* of the dynamics independently of absolute level.

    A small 2D conv stack then reads the (band x time) map. "Multi" in MED means
    several of these at different time constants; at 44.1 kHz the defaults are
    roughly 1.5 ms (transients), 6 ms (phoneme body) and 23 ms (syllabic
    dynamics, tremolo, breathing).
    """

    def __init__(self, sample_rate=44100, n_bands=8, filter_kernel_size=127,
                 pool=256, stride=64, channels=32, max_channels=128, n_layers=4,
                 use_spectral_norm=False, use_san=True):
        super().__init__()
        self.analysis = FixedSincSubBand(sample_rate, n_bands, filter_kernel_size)
        self.pool = max(4, int(pool))
        self.stride = max(1, int(stride))
        norm_f = _norm(use_spectral_norm)

        self.convs = nn.ModuleList()
        in_ch = 3   # envelope, delta, contrast
        for i in range(n_layers):
            out_ch = min(channels * (2 ** i), max_channels)
            # Stride on the time axis only: the band axis is 8-16 wide already,
            # and decimating it would erase the structure this branch exists for.
            self.convs.append(norm_f(nn.Conv2d(
                in_ch, out_ch, (3, 9), stride=(1, 2), padding=(1, 4))))
            in_ch = out_ch
        self.conv_final = norm_f(nn.Conv2d(in_ch, in_ch, (3, 3), padding=(1, 1)))
        self.conv_post = _head(nn.Conv2d(in_ch, 1, (3, 3), padding=(1, 1)),
                               use_san, use_spectral_norm)

    def _features(self, x):
        bands = self.analysis(x)
        env = F.avg_pool1d(
            F.pad(bands.abs(), (self.pool // 2, self.pool // 2), mode="reflect"),
            kernel_size=self.pool, stride=self.stride)
        log_env = torch.log1p(env)
        delta = F.pad(log_env[..., 1:] - log_env[..., :-1], (1, 0))
        local = F.avg_pool1d(F.pad(log_env, (2, 2), mode="replicate"),
                             kernel_size=5, stride=1)
        return torch.stack([log_env, delta, log_env - local], dim=1)

    def forward(self, x, compute_fmaps=True):
        h = self._features(x).to(next(self.parameters()).dtype)
        fmap = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            if compute_fmaps:
                fmap.append(h)
        h = F.leaky_relu(self.conv_final(h), 0.1)
        if compute_fmaps:
            fmap.append(h)
        h = self.conv_post(h)
        if compute_fmaps:
            fmap.append(h)
        return torch.flatten(h, 1, -1), fmap


# Default time constants at 44.1 kHz. pool/stride in samples:
#   64/16    ~ 1.5 ms  -> transients, consonant attacks, pre-echo
#   256/64   ~ 5.8 ms  -> phoneme body, fricative texture
#   1024/256 ~ 23 ms   -> syllabic dynamics, tremolo, breathing
DEFAULT_MED_SCALES = (
    {"n_bands": 8, "pool": 64, "stride": 16},
    {"n_bands": 8, "pool": 256, "stride": 64},
    {"n_bands": 16, "pool": 1024, "stride": 256},
)


# ----------------------------------------------------------------- transient


class TransientDiscriminator(nn.Module):
    """Waveform-derivative critic: very small, aimed squarely at attacks.

    The first difference of the waveform emphasises exactly what a codec's window
    smears, and pairing it with its absolute value gives the stack both the signed
    slope and the unsigned activity.
    """

    def __init__(self, channels=16, max_channels=64, n_layers=3,
                 use_spectral_norm=False, use_san=True):
        super().__init__()
        norm_f = _norm(use_spectral_norm)
        self.convs = nn.ModuleList([norm_f(nn.Conv1d(2, channels, 15, padding=7))])
        in_ch = channels
        for i in range(n_layers - 1):
            out_ch = min(channels * (2 ** (i + 1)), max_channels)
            self.convs.append(norm_f(nn.Conv1d(in_ch, out_ch, 9, stride=4, padding=4)))
            in_ch = out_ch
        self.conv_final = norm_f(nn.Conv1d(in_ch, in_ch, 5, padding=2))
        self.conv_post = _head(nn.Conv1d(in_ch, 1, 3, padding=1),
                               use_san, use_spectral_norm)

    def _features(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        dx = F.pad(x.float()[..., 1:] - x.float()[..., :-1], (1, 0))
        return torch.cat([dx, dx.abs()], dim=1)

    def forward(self, x, compute_fmaps=True):
        h = self._features(x).to(next(self.parameters()).dtype)
        fmap = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            if compute_fmaps:
                fmap.append(h)
        h = F.leaky_relu(self.conv_final(h), 0.1)
        if compute_fmaps:
            fmap.append(h)
        h = self.conv_post(h)
        if compute_fmaps:
            fmap.append(h)
        return torch.flatten(h, 1, -1), fmap


# ------------------------------------------------------------ HF modulation


class HFModulationDiscriminator(nn.Module):
    """Modulation-spectrum critic for high-band texture.

    Targets the failure Apollo's `bleedless_weight` exists to fight, but attacks
    it where it is actually visible. Generated 2-10 kHz energy can match the real
    band's level frame by frame and still be static noise, where real speech and
    real music deliver that band in bursts -- syllabic modulation around 2-8 Hz,
    phonemic around 20-50 Hz. The spectrogram branches judge per-frame magnitude
    and cannot see across hundreds of milliseconds; this one only sees that.

    Each band is envelope-detected, framed, and transformed along *time*. The
    modulation magnitude spectrum is then divided by its own DC bin, which makes
    the feature level-invariant by construction: the generator cannot satisfy this
    branch by adding energy, only by moving it like the real thing.
    """

    def __init__(self, sample_rate=44100, low_hz=2000.0, high_hz=10000.0,
                 n_bands=4, filter_kernel_size=127, env_pool=64, env_stride=32,
                 frame_size=256, frame_hop=64, max_mod_hz=64.0, channels=16,
                 max_channels=64, n_layers=3, use_spectral_norm=False,
                 use_san=True, dc_floor=1e-2):
        super().__init__()
        self.env_pool = max(4, int(env_pool))
        self.env_stride = max(1, int(env_stride))
        self.frame_size = max(16, int(frame_size))
        self.frame_hop = max(1, int(frame_hop))
        self.dc_floor = float(dc_floor)

        high_hz = min(float(high_hz), sample_rate / 2.0 * 0.95)
        self.register_buffer("filters",
                             self._build_bandpass(sample_rate, float(low_hz), high_hz,
                                                  max(2, int(n_bands)),
                                                  int(filter_kernel_size)),
                             persistent=False)
        self.kernel_size = self.filters.shape[-1]
        self.register_buffer("frame_window",
                             torch.hann_window(self.frame_size, periodic=False),
                             persistent=False)

        # Modulation bins kept: 1..n_mod. DC is dropped -- it is the normaliser.
        env_rate = float(sample_rate) / self.env_stride
        mod_res = env_rate / self.frame_size
        self.n_mod = max(8, min(int(round(float(max_mod_hz) / mod_res)),
                                self.frame_size // 2))

        norm_f = _norm(use_spectral_norm)
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(self.filters.shape[0], channels, (5, 3), padding=(2, 1)))])
        in_ch = channels
        for i in range(max(1, int(n_layers)) - 1):
            out_ch = min(channels * (2 ** (i + 1)), max_channels)
            self.convs.append(norm_f(nn.Conv2d(
                in_ch, out_ch, (5, 3), stride=(2, 1), padding=(2, 1))))
            in_ch = out_ch
        self.conv_final = norm_f(nn.Conv2d(in_ch, in_ch, (3, 3), padding=(1, 1)))
        self.conv_post = _head(nn.Conv2d(in_ch, 1, (3, 3), padding=(1, 1)),
                               use_san, use_spectral_norm)

    @staticmethod
    def _build_bandpass(sample_rate, low_hz, high_hz, n_bands, kernel_size):
        ks = int(kernel_size) if int(kernel_size) % 2 == 1 else int(kernel_size) + 1
        n = torch.arange(ks, dtype=torch.float32) - ks // 2
        window = torch.hamming_window(ks, periodic=False)
        edges = torch.linspace(low_hz, high_hz, n_bands + 1)

        def lowpass(cut_hz):
            cut = min(max(cut_hz / sample_rate, 1e-4), 0.5)
            return 2.0 * cut * torch.sinc(2.0 * cut * n)

        filters = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            h = (lowpass(float(hi)) - lowpass(float(lo))) * window
            filters.append(h / h.abs().sum().clamp_min(1e-6))
        return torch.stack(filters, dim=0).unsqueeze(1)

    @torch.amp.autocast("cuda", enabled=False)
    def _features(self, x):
        """(B, 1, T) waveform -> (B, n_bands, n_mod, frames).

        fp32 by design: rfft has no bf16 kernel and the envelope math is
        precision-sensitive. Under autocast the 127-tap bandpass also lands on a
        much slower cuDNN kernel.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.float()
        pad = self.kernel_size // 2
        bands = F.conv1d(F.pad(x, (pad, pad), mode="reflect"), self.filters.float())

        env = F.avg_pool1d(
            F.pad(bands.abs(), (self.env_pool // 2, self.env_pool // 2), mode="reflect"),
            kernel_size=self.env_pool, stride=self.env_stride)
        env = torch.log1p(env)
        if env.shape[-1] < self.frame_size:
            env = F.pad(env, (0, self.frame_size - env.shape[-1]), mode="replicate")

        frames = env.unfold(-1, self.frame_size, self.frame_hop) * self.frame_window.float()
        mod = torch.fft.rfft(frames, dim=-1).abs()

        # An additive floor rather than `dc.clamp_min`: on a silent frame the
        # generated HF envelope is ~0, and a clamped division carries a backward
        # factor up to 1/eps, which overflows under a fp16 GradScaler. The floor
        # soft-gates those frames toward zero -- their modulation pattern is
        # meaningless anyway -- and bounds the gradient gain at 1/dc_floor.
        dc = mod[..., 0:1]
        mod = mod[..., 1:1 + self.n_mod] / (dc + self.dc_floor)
        return mod.permute(0, 1, 3, 2)

    def forward(self, x, compute_fmaps=True):
        h = self._features(x).to(next(self.parameters()).dtype)
        fmap = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            if compute_fmaps:
                fmap.append(h)
        h = F.leaky_relu(self.conv_final(h), 0.1)
        if compute_fmaps:
            fmap.append(h)
        h = self.conv_post(h)
        if compute_fmaps:
            fmap.append(h)
        return torch.flatten(h, 1, -1), fmap


# ------------------------------------------------------------- periodicity


class PeriodicityACFDiscriminator(nn.Module):
    """Lag-domain critic for over-regular periodicity.

    A GAN vocoder or restorer that has learned the *average* harmonic structure
    produces a comb that is too clean and too regular -- the "plastic" voice. On
    the lag axis that signature is explicit and level-invariant: a synthetic comb
    gives a normalised autocorrelation peak near 1.0 at the period with slow decay
    across its multiples, while a natural voice decays faster through jitter and
    shimmer.

    The waveform is lowpassed and decimated into the band where this lives, then
    each frame yields a normalised ACF. The second channel is the frame-to-frame
    ACF difference: real prosody keeps the lag pattern moving, a frozen comb does
    not.
    """

    def __init__(self, sample_rate=44100, decimation=8, frame_size=512,
                 frame_hop=128, f0_min=70.0, f0_max=400.0, channels=16,
                 max_channels=64, n_layers=3, use_spectral_norm=False,
                 use_san=True, eps=1e-8):
        super().__init__()
        self.decimation = max(1, int(decimation))
        self.frame_size = int(frame_size)
        self.frame_hop = max(1, int(frame_hop))
        self.eps = float(eps)
        sr_d = float(sample_rate) / self.decimation

        # Fixed anti-alias sinc lowpass for the decimation.
        k = 127
        n = torch.arange(k, dtype=torch.float32) - k // 2
        cutoff = 0.45 / self.decimation      # just under the new Nyquist
        h = 2.0 * cutoff * torch.sinc(2.0 * cutoff * n)
        h = h * torch.hamming_window(k, periodic=False)
        self.register_buffer("lp_filter", (h / h.sum().clamp_min(1e-6)).view(1, 1, k),
                             persistent=False)
        self.register_buffer("frame_window",
                             torch.hann_window(self.frame_size, periodic=False),
                             persistent=False)

        # Lag band: half a period below f0_max up to ~2.5 periods of f0_min, so
        # the decay across period multiples -- the regularity cue -- is in view.
        self.lag_lo = max(2, int(round(sr_d / float(f0_max) * 0.5)))
        self.lag_hi = min(int(round(sr_d / float(f0_min) * 2.5)), self.frame_size // 2)
        if self.lag_hi <= self.lag_lo:
            self.lag_hi = self.lag_lo + 8

        norm_f = _norm(use_spectral_norm)
        self.convs = nn.ModuleList([norm_f(nn.Conv2d(2, channels, (5, 3), padding=(2, 1)))])
        in_ch = channels
        for i in range(max(1, int(n_layers)) - 1):
            out_ch = min(channels * (2 ** (i + 1)), max_channels)
            self.convs.append(norm_f(nn.Conv2d(
                in_ch, out_ch, (5, 3), stride=(2, 1), padding=(2, 1))))
            in_ch = out_ch
        self.conv_final = norm_f(nn.Conv2d(in_ch, in_ch, (3, 3), padding=(1, 1)))
        self.conv_post = _head(nn.Conv2d(in_ch, 1, (3, 3), padding=(1, 1)),
                               use_san, use_spectral_norm)

    @torch.amp.autocast("cuda", enabled=False)
    def _features(self, x):
        """(B, 1, T) waveform -> (B, 2, lags, frames). fp32 by design (fft path)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.float()
        pad = self.lp_filter.shape[-1] // 2
        x = F.conv1d(F.pad(x, (pad, pad), mode="reflect"), self.lp_filter.float(),
                     stride=self.decimation)

        frames = x.squeeze(1).unfold(-1, self.frame_size, self.frame_hop)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        frames = frames * self.frame_window.float()

        # ACF through the power spectrum; length 2N gives linear, not circular, lags
        spec = torch.fft.rfft(frames, n=2 * self.frame_size, dim=-1)
        acf = torch.fft.irfft(spec.real ** 2 + spec.imag ** 2, dim=-1)
        acf = acf[..., : self.frame_size // 2]
        acf = acf / acf[..., 0:1].clamp_min(self.eps)
        acf = acf[..., self.lag_lo: self.lag_hi].transpose(1, 2)

        delta = F.pad(acf[..., 1:] - acf[..., :-1], (1, 0))
        return torch.stack([acf, delta], dim=1)

    def forward(self, x, compute_fmaps=True):
        h = self._features(x).to(next(self.parameters()).dtype)
        fmap = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            if compute_fmaps:
                fmap.append(h)
        h = F.leaky_relu(self.conv_final(h), 0.1)
        if compute_fmaps:
            fmap.append(h)
        h = self.conv_post(h)
        if compute_fmaps:
            fmap.append(h)
        return torch.flatten(h, 1, -1), fmap


# ----------------------------------------------------------------- banks


class MultiEnvelopeDiscriminator(_FastBank):
    """MED: several envelope branches, one per time constant."""

    def __init__(self, nch=1, sample_rate=44100, scales=None, channels=32,
                 max_channels=128, n_layers=4, filter_kernel_size=127,
                 use_spectral_norm=False, use_san=True):
        super().__init__(nch=nch)
        self.branch_names = []
        self.discriminators = nn.ModuleList()
        for scale in (scales or DEFAULT_MED_SCALES):
            scale = dict(scale)
            self.discriminators.append(EnvelopeDiscriminator(
                sample_rate=sample_rate,
                n_bands=int(scale.get("n_bands", 8)),
                filter_kernel_size=int(scale.get("filter_kernel_size", filter_kernel_size)),
                pool=int(scale.get("pool", 256)),
                stride=int(scale.get("stride", 64)),
                channels=int(scale.get("channels", channels)),
                max_channels=int(scale.get("max_channels", max_channels)),
                n_layers=int(scale.get("n_layers", n_layers)),
                use_spectral_norm=use_spectral_norm, use_san=use_san))
            self.branch_names.append(
                f"med_{int(scale.get('n_bands', 8))}b_{int(scale.get('pool', 256))}")


class TransientBank(_FastBank):
    def __init__(self, nch=1, channels=16, max_channels=64, n_layers=3,
                 use_spectral_norm=False, use_san=True):
        super().__init__(nch=nch)
        self.discriminators = nn.ModuleList([TransientDiscriminator(
            channels=channels, max_channels=max_channels, n_layers=n_layers,
            use_spectral_norm=use_spectral_norm, use_san=use_san)])
        self.branch_names = ["transient"]


class HFModulationBank(_FastBank):
    def __init__(self, nch=1, sample_rate=44100, low_hz=2000.0, high_hz=10000.0,
                 n_bands=4, channels=16, max_channels=64, n_layers=3,
                 env_pool=64, env_stride=32, frame_size=256, frame_hop=64,
                 max_mod_hz=64.0, use_spectral_norm=False, use_san=True):
        super().__init__(nch=nch)
        self.discriminators = nn.ModuleList([HFModulationDiscriminator(
            sample_rate=sample_rate, low_hz=low_hz, high_hz=high_hz, n_bands=n_bands,
            env_pool=env_pool, env_stride=env_stride, frame_size=frame_size,
            frame_hop=frame_hop, max_mod_hz=max_mod_hz, channels=channels,
            max_channels=max_channels, n_layers=n_layers,
            use_spectral_norm=use_spectral_norm, use_san=use_san)])
        self.branch_names = ["hf_modulation"]


class PeriodicityBank(_FastBank):
    def __init__(self, nch=1, sample_rate=44100, decimation=8, frame_size=512,
                 frame_hop=128, f0_min=70.0, f0_max=400.0, channels=16,
                 max_channels=64, n_layers=3, use_spectral_norm=False, use_san=True):
        super().__init__(nch=nch)
        self.discriminators = nn.ModuleList([PeriodicityACFDiscriminator(
            sample_rate=sample_rate, decimation=decimation, frame_size=frame_size,
            frame_hop=frame_hop, f0_min=f0_min, f0_max=f0_max, channels=channels,
            max_channels=max_channels, n_layers=n_layers,
            use_spectral_norm=use_spectral_norm, use_san=use_san)])
        self.branch_names = ["periodicity"]
