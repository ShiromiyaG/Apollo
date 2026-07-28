"""Chunked overlap-add inference.

The reference `inference.py` pushes the whole file through the model in one go.
Peak VRAM therefore scales with the length of the track (~110 MB per second of
44.1 kHz stereo), which puts anything past ~30 s out of reach on an 8 GB card.

Processing fixed-length chunks with a crossfaded overlap-add keeps peak memory
constant regardless of track length, and has a second benefit: the model was
trained exclusively on 3 s segments that were peak-normalised to 1.0, so feeding
it similarly scaled chunks keeps it inside its training distribution.
"""

import math

import torch
import torch.nn.functional as F


def _pad_edges(wav: torch.Tensor, front: int, back: int) -> torch.Tensor:
    """Extend a waveform with mirrored audio rather than silence.

    Zero-padding the head of the file hands the model a silence-to-signal step it
    never saw in training; its receptive field then smears that transient across
    the opening ~200 ms. Reflecting the signal keeps the context plausible.
    Anything beyond what reflection can supply falls back to zeros.
    """
    n = wav.shape[-1]
    if n < 2:
        return F.pad(wav, (front, back))

    f, b = min(front, n - 1), min(back, n - 1)
    out = F.pad(wav, (f, b), mode="reflect")
    if f < front or b < back:
        out = F.pad(out, (front - f, back - b))
    return out


def taper_window(length: int, fade: int, device, dtype=torch.float32) -> torch.Tensor:
    """Flat-top window with complementary sin^2 / cos^2 edges.

    Two adjacent windows offset by ``length - fade`` sum to exactly 1 across the
    overlap, so the crossfade is amplitude-preserving for correlated content --
    which is what neighbouring chunks of the same restoration produce.
    """
    w = torch.ones(length, device=device, dtype=dtype)
    if fade > 0:
        t = torch.arange(fade, device=device, dtype=dtype)
        ramp = torch.sin(math.pi * (t + 0.5) / (2 * fade)) ** 2
        w[:fade] = ramp
        w[length - fade:] = ramp.flip(0)
    return w


@torch.inference_mode()
def chunked_restore(
    model,
    wav: torch.Tensor,
    chunk_samples: int,
    overlap_samples: int,
    batch_size: int = 1,
    device=None,
    normalize: str = "chunk",
    silence_db: float = -60.0,
    max_gain_db: float = 40.0,
    accum_on_cpu: bool = True,
    progress=None,
) -> torch.Tensor:
    """Run ``model`` over ``wav`` in overlapping chunks.

    Args:
        wav: (nch, n) float tensor, any device.
        chunk_samples: samples per chunk fed to the model.
        overlap_samples: crossfade length between neighbouring chunks.
        batch_size: chunks pushed through the model at once.
        normalize: ``"chunk"`` peak-normalises every chunk to 1.0 before the model
            and undoes the gain afterwards (closest match to how Apollo was
            trained); ``"global"`` uses one gain for the whole file; ``"none"``
            feeds raw amplitudes, reproducing the reference behaviour.
        silence_db: chunks whose peak sits below this (dBFS, relative to the file
            peak) bypass the model entirely and are copied through. Stops the
            model from hallucinating a noise floor into silent passages.
        max_gain_db: ceiling on the per-chunk normalisation boost.
        accum_on_cpu: keep the output accumulator in host memory so GPU usage
            stays flat for long files.

    Returns:
        (nch, n) tensor on the same device as ``wav``.
    """
    if normalize not in ("chunk", "global", "none"):
        raise ValueError(f"normalize must be chunk|global|none, got {normalize!r}")
    if wav.dim() != 2:
        raise ValueError(f"expected (nch, n) waveform, got shape {tuple(wav.shape)}")
    if overlap_samples >= chunk_samples:
        raise ValueError("overlap_samples must be smaller than chunk_samples")

    src_device = wav.device
    device = device or src_device
    nch, n = wav.shape
    hop = chunk_samples - overlap_samples

    file_peak = float(wav.abs().max())
    if file_peak <= 0:
        return wav.clone()

    global_gain = 1.0 / file_peak if normalize == "global" else 1.0

    # Pad by one overlap at the front so the very first sample is already inside a
    # window plateau, and at the back so the final chunk is complete.
    n_chunks = max(1, math.ceil((n + overlap_samples) / hop))
    padded_len = overlap_samples + (n_chunks - 1) * hop + chunk_samples
    padded = _pad_edges(wav, overlap_samples, padded_len - overlap_samples - n)

    accum_device = torch.device("cpu") if accum_on_cpu else device
    out = torch.zeros(nch, padded_len, device=accum_device, dtype=torch.float32)
    wsum = torch.zeros(padded_len, device=accum_device, dtype=torch.float32)
    window = taper_window(chunk_samples, overlap_samples, accum_device)

    silence_threshold = file_peak * (10.0 ** (silence_db / 20.0))
    max_gain = 10.0 ** (max_gain_db / 20.0)

    for start in range(0, n_chunks, batch_size):
        stop = min(start + batch_size, n_chunks)
        offsets = [i * hop for i in range(start, stop)]
        batch = torch.stack([padded[:, o:o + chunk_samples] for o in offsets], 0)
        batch = batch.to(device, non_blocking=True)

        # (B, 1, 1) peaks -- one gain per chunk across all channels, matching the
        # single `max_scale` the training pipeline applied to each segment.
        peaks = batch.abs().amax(dim=(1, 2), keepdim=True)
        loud = (peaks > silence_threshold).squeeze(-1).squeeze(-1)

        if normalize == "chunk":
            gain = torch.where(peaks > 0, 1.0 / peaks.clamp(min=1e-12), torch.ones_like(peaks))
            gain = gain.clamp(max=max_gain)
        elif normalize == "global":
            gain = torch.full_like(peaks, global_gain)
        else:
            gain = torch.ones_like(peaks)

        if loud.any():
            est = torch.empty_like(batch)
            est[~loud] = batch[~loud]
            fed = batch[loud] * gain[loud]
            est[loud] = model(fed) / gain[loud]
        else:
            est = batch

        est = est.to(accum_device, dtype=torch.float32)
        for k, o in enumerate(offsets):
            out[:, o:o + chunk_samples] += est[k] * window
            wsum[o:o + chunk_samples] += window

        if progress is not None:
            progress(stop, n_chunks)

    out = out / wsum.clamp(min=1e-8).unsqueeze(0)
    out = out[:, overlap_samples:overlap_samples + n]

    return out.to(src_device, dtype=wav.dtype)
