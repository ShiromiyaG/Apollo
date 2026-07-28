"""End-to-end restoration of an audio file."""

import os
from dataclasses import dataclass

import torch

from .audio_io import AUDIO_EXTS, load_audio, resample, save_audio
from .chunked import chunked_restore
from .postprocess import dry_wet_mix, match_loudness, residual_gate, spectral_crossover


@dataclass
class RestoreOptions:
    """Everything that shapes the output, with defaults tuned for an 8 GB card."""

    chunk_seconds: float = 10.0
    overlap_seconds: float = 1.0
    batch_size: int = 1
    normalize: str = "chunk"
    silence_db: float = -60.0

    # post-processing -- all default to "off" so the plain output matches upstream
    crossover_hz: float = 0.0
    crossover_width_octaves: float = 0.5
    dry_wet: float = 1.0
    gate_threshold_db: float = -55.0
    gate: bool = False
    match_input_loudness: bool = False


def restore_waveform(model, wav, sr, options: RestoreOptions, device=None, progress=None):
    """Restore a (nch, n) waveform already at the model's sample rate."""
    device = device or next(model.parameters()).device
    chunk = int(sr * options.chunk_seconds)
    overlap = int(sr * options.overlap_seconds)

    if wav.shape[-1] <= chunk:
        # short clip: one pass, but keep the same normalisation semantics
        chunk = max(wav.shape[-1], int(sr * 0.5))
        overlap = min(overlap, max(0, chunk // 4))

    wet = chunked_restore(
        model,
        wav,
        chunk_samples=chunk,
        overlap_samples=overlap,
        batch_size=options.batch_size,
        device=device,
        normalize=options.normalize,
        silence_db=options.silence_db,
        progress=progress,
    )

    if options.crossover_hz > 0:
        wet = spectral_crossover(wav, wet, sr, options.crossover_hz,
                                 width_octaves=options.crossover_width_octaves)
    if options.gate:
        wet = residual_gate(wav, wet, sr, threshold_db=options.gate_threshold_db)
    if options.dry_wet < 1.0:
        wet = dry_wet_mix(wav, wet, options.dry_wet)
    if options.match_input_loudness:
        wet = match_loudness(wav, wet)

    return wet


def restore_file(model, in_path, out_path, options: RestoreOptions, device=None,
                 model_sr=44100, keep_input_sr=True, bit_depth=None, progress=None):
    """Load, restore and write one file. Returns a short status string."""
    wav, file_sr = load_audio(in_path)

    work = resample(wav, file_sr, model_sr)

    out = restore_waveform(model, work, model_sr, options, device=device, progress=progress)

    if file_sr != model_sr and keep_input_sr:
        out = resample(out, model_sr, file_sr)
        out = out[..., :wav.shape[-1]]
        out_sr = file_sr
    else:
        out_sr = model_sr

    peak = out.abs().max()
    clipped = bool(peak > 1.0)
    if clipped:
        out = out / peak

    save_audio(out_path, out, out_sr, bit_depth=bit_depth)

    note = f"{out.shape[0]}ch {out.shape[-1] / out_sr:.1f}s @ {out_sr} Hz"
    if file_sr != model_sr:
        note += f" (resampled {file_sr}->{model_sr}->{out_sr})"
    if clipped:
        note += f" [normalised, peak was {float(peak):.2f}]"
    return note


def iter_audio_files(root):
    """Yield audio files under ``root``, recursively."""
    if os.path.isfile(root):
        yield root
        return
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(AUDIO_EXTS):
                yield os.path.join(dirpath, name)
