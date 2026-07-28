"""Audio file I/O without the heavyweight backends.

torchaudio 2.9+ routes ``load``/``save`` through TorchCodec, which drags in an
ffmpeg build and is a common install failure on Windows. libsndfile (via the
``soundfile`` wheel) already covers WAV/FLAC/OGG/MP3/AIFF and installs with a
plain ``pip install``, so it is tried first. torchaudio and an ffmpeg subprocess
remain as fallbacks for the container formats libsndfile does not read.
"""

import os
import shutil
import subprocess

import numpy as np
import torch

# formats libsndfile handles directly
NATIVE_EXTS = (".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".aiff", ".aif", ".au", ".w64")
# extra containers we can reach through a fallback
EXTRA_EXTS = (".m4a", ".aac", ".wma", ".mp4", ".webm")
AUDIO_EXTS = NATIVE_EXTS + EXTRA_EXTS

_SUBTYPE_FOR_DEPTH = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}


def load_audio(path):
    """Return ``(wav, sr)`` with ``wav`` shaped (nch, n) as float32."""
    try:
        import soundfile as sf

        data, sr = sf.read(path, always_2d=True, dtype="float32")
        return torch.from_numpy(data.T.copy()), int(sr)
    except Exception as sf_error:
        pass

    try:
        import torchaudio

        wav, sr = torchaudio.load(path)
        return wav.float(), int(sr)
    except Exception:
        pass

    wav, sr = _load_via_ffmpeg(path)
    if wav is None:
        raise RuntimeError(
            f"could not read {path!r}. soundfile failed ({sf_error}); install ffmpeg "
            f"or convert the file to WAV/FLAC first."
        )
    return wav, sr


def save_audio(path, wav, sr, bit_depth=None):
    """Write a (nch, n) float tensor. ``bit_depth`` is 16, 24, 32 or None."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    data = wav.detach().cpu().float().numpy().T  # soundfile wants (n, nch)

    ext = os.path.splitext(path)[1].lower()
    subtype = _SUBTYPE_FOR_DEPTH.get(bit_depth)
    if subtype is None:
        subtype = "PCM_24" if ext in (".wav", ".flac", ".aiff", ".aif") else None
    if ext == ".flac" and subtype == "FLOAT":
        subtype = "PCM_24"  # FLAC has no float subtype

    try:
        import soundfile as sf

        sf.write(path, data, sr, subtype=subtype)
        return
    except Exception:
        pass

    import torchaudio

    kwargs = {}
    if bit_depth in (16, 24):
        kwargs = dict(encoding="PCM_S", bits_per_sample=bit_depth)
    elif bit_depth == 32:
        kwargs = dict(encoding="PCM_F", bits_per_sample=32)
    torchaudio.save(path, wav.detach().cpu().float(), sr, **kwargs)


def _load_via_ffmpeg(path):
    """Decode to raw float32 through an ffmpeg subprocess, if one is on PATH."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        return None, None

    try:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        )
        sr_text, ch_text = probe.stdout.strip().split("\n")[0].split(",")[:2]
        sr, nch = int(sr_text), int(ch_text)

        raw = subprocess.run(
            [ffmpeg, "-v", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le", "-"],
            capture_output=True, check=True,
        ).stdout
    except Exception:
        return None, None

    data = np.frombuffer(raw, dtype=np.float32).reshape(-1, nch)
    return torch.from_numpy(data.T.copy()), sr


def resample(wav, orig_sr, new_sr):
    """Resample a (nch, n) tensor. Uses torchaudio's kernel, which is backend-free."""
    if orig_sr == new_sr:
        return wav
    import torchaudio.functional as AF

    return AF.resample(wav, orig_sr, new_sr)
