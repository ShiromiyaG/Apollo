"""Data pipeline for the vocal/speech restoration mode.

Differs from the original music pipeline in three ways that matter:

* It reads plain audio folders instead of preprocessed HDF5, so getting started
  means pointing it at a directory rather than running a conversion step.
* Segments are seeked, not loaded whole, so a corpus of long files costs the same
  per sample as a corpus of short ones.
* The degradation chain covers what actually happens to voice recordings -- low
  bitrate codecs that collapse to mono and halve the sample rate, band-limiting,
  clipping, requantisation -- rather than MP3 alone.
"""

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .codec import SPEECH_BITRATES, encode_decode, lowpass_resample, resolve_formats

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".opus")


def find_audio_files(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(AUDIO_EXTS):
                files.append(os.path.join(dirpath, name))
    return files


def _read_segment(path, start_frame, frames, channels):
    """Read ``frames`` samples starting at ``start_frame`` without loading the file."""
    import soundfile as sf

    with sf.SoundFile(path) as handle:
        handle.seek(start_frame)
        data = handle.read(frames, dtype="float32", always_2d=True)

    wav = torch.from_numpy(data.T.copy())
    if channels == 1 and wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    elif channels == 2 and wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif channels == 2 and wav.shape[0] > 2:
        wav = wav[:2]
    return wav


def _file_info(path):
    import soundfile as sf

    info = sf.info(path)
    return info.frames, info.samplerate


class VocalDegradation:
    """Randomised degradation chain applied to a clean vocal segment."""

    def __init__(
        self,
        sr=44100,
        codec_prob=0.9,
        codec_formats=("mp3", "ogg"),
        codec_weights=None,
        bitrates=None,
        bandlimit_prob=0.3,
        bandlimit_hz=(3000, 11000),
        clip_prob=0.1,
        clip_range=(0.5, 0.95),
        quantize_prob=0.1,
        quantize_bits=(8, 12),
    ):
        self.sr = sr
        self.codec_prob = codec_prob
        # Expands bundle names ("voice", "music", ...) and drops anything this
        # install cannot encode. Doing it here, once, is what stops a missing
        # ffmpeg from quietly pairing every clean segment with an identical copy:
        # `encode_decode` returns its input unchanged when no backend can handle
        # the format, and nothing downstream can tell that apart from a codec that
        # simply did very little.
        self.codec_formats = resolve_formats(codec_formats, sr=sr)
        # Per-codec sampling weights, defaulting to uniform. Uniform is the wrong
        # default once the narrowband codecs are in the set: the "voice" bundle is
        # twelve entries of which six resample to 8 or 16 kHz, so half of every
        # epoch would be telephone audio. That is a real degradation worth
        # training on and a terrible thing to train mostly on -- the model would
        # spend its capacity on bandwidth extension from 4 kHz and treat ordinary
        # 128 kbps MP3 as an edge case.
        weights = {str(k): float(v) for k, v in dict(codec_weights or {}).items()}
        self.codec_weights = [weights.get(name, 1.0) for name in self.codec_formats]
        self.bitrates = list(bitrates or SPEECH_BITRATES)
        self.bandlimit_prob = bandlimit_prob
        self.bandlimit_hz = bandlimit_hz
        self.clip_prob = clip_prob
        self.clip_range = clip_range
        self.quantize_prob = quantize_prob
        self.quantize_bits = quantize_bits

    def __call__(self, wav, rng=random):
        out = wav

        if rng.random() < self.codec_prob:
            fmt = rng.choices(self.codec_formats, weights=self.codec_weights)[0]
            bitrate = rng.choice(self.bitrates)
            out = encode_decode(out, self.sr, fmt=fmt, bitrate_kbps=bitrate,
                                quality=rng.randint(0, 5))

        if rng.random() < self.bandlimit_prob:
            cutoff = rng.uniform(*self.bandlimit_hz)
            out = lowpass_resample(out, self.sr, cutoff)

        if rng.random() < self.clip_prob:
            ceiling = rng.uniform(*self.clip_range) * out.abs().max().clamp(min=1e-9)
            out = out.clamp(-ceiling, ceiling)

        if rng.random() < self.quantize_prob:
            bits = rng.randint(*self.quantize_bits)
            steps = 2 ** (bits - 1)
            out = torch.round(out * steps) / steps

        return out


class VocalRestorationDataset(Dataset):
    """Random clean segments paired with an on-the-fly degraded copy."""

    def __init__(
        self,
        data_dir: str,
        sr: int = 44100,
        segments: float = 3.0,
        channels: int = 1,
        num_samples: int = 20000,
        gain_range_db: Tuple[float, float] = (-6.0, 6.0),
        min_active_db: float = -45.0,
        degradation: Optional[VocalDegradation] = None,
        seed: Optional[int] = None,
    ):
        self.files = find_audio_files(data_dir)
        if not self.files:
            raise FileNotFoundError(f"no audio files found under {data_dir!r}")

        self.sr = sr
        self.segment_frames = int(segments * sr)
        self.channels = channels
        self.num_samples = num_samples
        self.gain_range_db = gain_range_db
        self.min_active_db = min_active_db
        self.degradation = degradation or VocalDegradation(sr=sr)
        self.seed = seed
        self._info_cache = {}
        self._check_usable()

    def _check_usable(self, sample_size=200):
        """Fail loudly if the corpus cannot supply segments.

        Files at the wrong sample rate or shorter than one segment are skipped
        during sampling. Without this check a fully mismatched corpus trains
        happily on silence, which is a very expensive thing to discover late.
        """
        probe = self.files[:sample_size]
        usable, wrong_sr, too_short = 0, 0, 0
        for path in probe:
            try:
                frames, file_sr = self._info(path)
            except Exception:
                continue
            if file_sr != self.sr:
                wrong_sr += 1
            elif frames < self.segment_frames:
                too_short += 1
            else:
                usable += 1

        if usable:
            if wrong_sr or too_short:
                print(f"[VocalRestorationDataset] {usable}/{len(probe)} sampled files usable "
                      f"({wrong_sr} at the wrong sample rate, {too_short} shorter than "
                      f"{self.segment_frames / self.sr:.1f}s)")
            return

        detail = []
        if wrong_sr:
            detail.append(f"{wrong_sr} are not {self.sr} Hz")
        if too_short:
            detail.append(f"{too_short} are shorter than {self.segment_frames / self.sr:.1f}s")
        raise ValueError(
            f"none of the {len(probe)} sampled files under the training directory are usable"
            + (": " + ", ".join(detail) if detail else "")
            + f". Resample the corpus to {self.sr} Hz, or set `sr`/`segments` to match it."
        )

    def __len__(self):
        return self.num_samples

    def _info(self, path):
        if path not in self._info_cache:
            self._info_cache[path] = _file_info(path)
        return self._info_cache[path]

    def _sample_clean(self, rng):
        """Draw a segment that actually contains voice, not a silent gap."""
        for _ in range(12):
            path = rng.choice(self.files)
            try:
                frames, file_sr = self._info(path)
            except Exception:
                continue
            if file_sr != self.sr or frames < self.segment_frames:
                continue

            start = rng.randint(0, frames - self.segment_frames)
            try:
                wav = _read_segment(path, start, self.segment_frames, self.channels)
            except Exception:
                continue
            if wav.shape[-1] < self.segment_frames:
                continue

            rms = wav.pow(2).mean().sqrt()
            if rms > 0 and 20 * torch.log10(rms) > self.min_active_db:
                return wav

        return torch.zeros(self.channels, self.segment_frames)

    def __getitem__(self, idx):
        rng = random.Random(self.seed + idx) if self.seed is not None else random

        clean = self._sample_clean(rng)
        clean = clean * (10.0 ** (rng.uniform(*self.gain_range_db) / 20.0))

        degraded = self.degradation(clean, rng=rng)

        # Match the reference pipeline: one scale for the pair so their relative
        # level is preserved, and the model always sees a peak of 1.0.
        scale = torch.maximum(clean.abs().max(), degraded.abs().max())
        if scale > 0:
            clean = clean / scale
            degraded = degraded / scale

        return clean, degraded


class PairedVocalEval(Dataset):
    """Fixed evaluation pairs.

    Either two parallel folders (``clean_dir``/``degraded_dir``, matched by relative
    path) or one clean folder degraded deterministically with a fixed seed.
    """

    def __init__(self, clean_dir, degraded_dir=None, sr=44100, channels=1,
                 segments=None, degradation=None, seed=1234):
        self.clean_files = find_audio_files(clean_dir)
        if not self.clean_files:
            raise FileNotFoundError(f"no audio files found under {clean_dir!r}")

        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        self.sr = sr
        self.channels = channels
        self.segment_frames = int(segments * sr) if segments else None
        self.degradation = degradation or VocalDegradation(sr=sr)
        self.seed = seed

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        path = self.clean_files[idx]
        frames, _ = _file_info(path)
        want = min(self.segment_frames or frames, frames)
        clean = _read_segment(path, 0, want, self.channels)

        if self.degraded_dir:
            rel = os.path.relpath(path, self.clean_dir)
            candidate = os.path.join(self.degraded_dir, rel)
            if not os.path.isfile(candidate):
                stem = os.path.splitext(rel)[0]
                matches = [os.path.join(self.degraded_dir, stem + ext) for ext in AUDIO_EXTS]
                candidate = next((m for m in matches if os.path.isfile(m)), None)
            if candidate is None:
                raise FileNotFoundError(f"no degraded counterpart for {path!r}")
            degraded = _read_segment(candidate, 0, want, self.channels)
        else:
            degraded = self.degradation(clean, rng=random.Random(self.seed + idx))

        scale = torch.maximum(clean.abs().max(), degraded.abs().max())
        if scale > 0:
            clean, degraded = clean / scale, degraded / scale

        return clean, degraded


class VocalDataModule(LightningDataModule):
    """Lightning wrapper. Defaults are sized for a single 8 GB GPU."""

    def __init__(
        self,
        train_dir: str,
        eval_dir: str,
        eval_degraded_dir: Optional[str] = None,
        sr: int = 44100,
        segments: float = 3.0,
        channels: int = 1,
        num_samples: int = 20000,
        gain_range_db: Tuple[float, float] = (-6.0, 6.0),
        min_active_db: float = -45.0,
        eval_segments: Optional[float] = 6.0,
        codec_prob: float = 0.9,
        codec_formats: Tuple[str, ...] = ("mp3", "ogg"),
        codec_weights: Optional[dict] = None,
        bitrates: Optional[Tuple[int, ...]] = None,
        bandlimit_prob: float = 0.3,
        clip_prob: float = 0.1,
        quantize_prob: float = 0.1,
        batch_size: int = 1,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is not None:
            return

        hp = self.hparams
        degradation = VocalDegradation(
            sr=hp.sr,
            codec_prob=hp.codec_prob,
            codec_formats=hp.codec_formats,
            codec_weights=hp.codec_weights,
            bitrates=hp.bitrates,
            bandlimit_prob=hp.bandlimit_prob,
            clip_prob=hp.clip_prob,
            quantize_prob=hp.quantize_prob,
        )

        self.data_train = VocalRestorationDataset(
            data_dir=hp.train_dir,
            sr=hp.sr,
            segments=hp.segments,
            channels=hp.channels,
            num_samples=hp.num_samples,
            gain_range_db=tuple(hp.gain_range_db),
            min_active_db=hp.min_active_db,
            degradation=degradation,
        )
        self.data_val = PairedVocalEval(
            clean_dir=hp.eval_dir,
            degraded_dir=hp.eval_degraded_dir,
            sr=hp.sr,
            channels=hp.channels,
            segments=hp.eval_segments,
            degradation=degradation,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=1,
            num_workers=min(self.hparams.num_workers, 2),
            shuffle=False,
            pin_memory=True,
        )
