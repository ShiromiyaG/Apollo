"""Post-processing that trades away the artefacts Apollo introduces.

Apollo regenerates the *entire* spectrum, low frequencies included, even though a
lossy codec left those largely intact. Its training objective makes that costly:
`freq_MAE` compares magnitudes only, so nothing in the generator loss ties the
reconstructed phase to the input. Anything the GAN does not pin down comes back
as a diffuse, noise-like residual spread across the band.

The tools here let you keep the part of the output that is actually worth having
-- the regenerated mid/high band -- while reusing the input's own low band, where
it was already correct to begin with.
"""

import torch


def _stft(x, n_fft, hop):
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    return torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)


def _istft(spec, n_fft, hop, length, device, dtype):
    window = torch.hann_window(n_fft, device=device, dtype=dtype)
    return torch.istft(spec, n_fft=n_fft, hop_length=hop, window=window, length=length)


def spectral_crossover(
    dry: torch.Tensor,
    wet: torch.Tensor,
    sr: int,
    crossover_hz: float,
    width_octaves: float = 0.5,
    n_fft: int = 4096,
) -> torch.Tensor:
    """Keep ``dry`` below ``crossover_hz`` and ``wet`` above it.

    The blend runs over ``width_octaves`` around the crossover with a raised-cosine
    ramp, so there is no audible seam. Because the low band is copied with its
    original phase, this removes the model's low-frequency phase noise outright
    rather than masking it.

    Args:
        dry: (nch, n) the original input.
        wet: (nch, n) the model output.
        crossover_hz: frequency at which the blend is 50/50.
        width_octaves: transition width; 0 gives a brick wall.
    """
    if crossover_hz <= 0:
        return wet

    length = min(dry.shape[-1], wet.shape[-1])
    dry, wet = dry[..., :length], wet[..., :length]
    hop = n_fft // 4

    spec_dry = _stft(dry, n_fft, hop)
    spec_wet = _stft(wet, n_fft, hop)

    freqs = torch.linspace(0, sr / 2, spec_dry.shape[-2], device=dry.device, dtype=dry.dtype)
    if width_octaves <= 0:
        mask = (freqs >= crossover_hz).to(dry.dtype)
    else:
        lo = crossover_hz * (2.0 ** (-width_octaves / 2))
        hi = crossover_hz * (2.0 ** (width_octaves / 2))
        t = ((freqs.clamp(min=1e-6).log2() - torch.log2(torch.tensor(lo, device=dry.device, dtype=dry.dtype)))
             / (torch.log2(torch.tensor(hi, device=dry.device, dtype=dry.dtype))
                - torch.log2(torch.tensor(lo, device=dry.device, dtype=dry.dtype))))
        t = t.clamp(0.0, 1.0)
        mask = 0.5 - 0.5 * torch.cos(torch.pi * t)

    mask = mask.reshape(1, -1, 1)
    blended = spec_dry * (1 - mask) + spec_wet * mask

    return _istft(blended, n_fft, hop, length, dry.device, dry.dtype)


def residual_gate(
    dry: torch.Tensor,
    wet: torch.Tensor,
    sr: int,
    threshold_db: float = -55.0,
    knee_db: float = 12.0,
    attack_ms: float = 20.0,
) -> torch.Tensor:
    """Fade the model's contribution out wherever the input is near-silent.

    Apollo never saw quiet material during training -- the data pipeline ran a
    source-activity detector and then peak-normalised every segment -- so silence
    and fade-outs are out of distribution and come back with an invented noise
    floor. This suppresses the added residual there while leaving loud passages
    completely untouched.
    """
    length = min(dry.shape[-1], wet.shape[-1])
    dry, wet = dry[..., :length], wet[..., :length]

    win = max(1, int(sr * attack_ms / 1000.0))
    # smoothed local level of the input, in dBFS relative to the file peak
    energy = dry.pow(2).mean(0, keepdim=True)
    kernel = torch.ones(1, 1, win, device=dry.device, dtype=dry.dtype) / win
    padded = torch.nn.functional.pad(energy.unsqueeze(0), (win // 2, win - win // 2 - 1), mode="reflect")
    local_rms = torch.nn.functional.conv1d(padded, kernel).squeeze(0).sqrt()

    peak = dry.abs().max().clamp(min=1e-12)
    level_db = 20.0 * torch.log10((local_rms / peak).clamp(min=1e-12))

    # 0 below the threshold, 1 above threshold+knee, raised cosine in between
    t = ((level_db - threshold_db) / max(knee_db, 1e-6)).clamp(0.0, 1.0)
    gate = 0.5 - 0.5 * torch.cos(torch.pi * t)

    return dry + gate * (wet - dry)


def dry_wet_mix(dry: torch.Tensor, wet: torch.Tensor, amount: float) -> torch.Tensor:
    """Linear blend; ``amount`` 1.0 is the pure model output, 0.0 the input."""
    if amount >= 1.0:
        return wet
    length = min(dry.shape[-1], wet.shape[-1])
    return dry[..., :length] * (1.0 - amount) + wet[..., :length] * amount


def match_loudness(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rescale ``target`` so its RMS matches ``reference``.

    Guards against the slow level drift a GAN generator can pick up, which would
    otherwise make A/B comparisons misleading.
    """
    ref_rms = reference.pow(2).mean().sqrt()
    tgt_rms = target.pow(2).mean().sqrt()
    if tgt_rms <= 0:
        return target
    return target * (ref_rms / tgt_rms)
