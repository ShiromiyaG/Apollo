"""Dataset preparation for both training modes.

Turns a folder of arbitrary audio into something the training pipeline can use
without silently skipping half of it. The training dataset seeks segments
directly out of the files, so there is no packing step -- this only has to make
the corpus uniform:

* resample everything to the target rate (files at the wrong rate are skipped at
  training time, which is a very expensive thing to discover late);
* fix the channel count -- stereo for restoration, mono for vocal;
* drop files that are too short, silent, or clipped past usefulness;
* peak-normalise so the on-the-fly gain augmentation starts from a known place;
* split into train/valid and write a manifest of what was kept and why.

    # stereo, for lossy -> lossless music restoration
    python scripts/preprocess.py --mode restoration --in_dir ./raw_music --out_dir ./data/music

    # mono, for vocal / speech
    python scripts/preprocess.py --mode vocal --in_dir ./raw_vocals --out_dir ./data/vocals

Nothing here degrades the audio: the degraded pairs are generated on the fly
during training so that every epoch sees different codec settings. This produces
the *clean* side only.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look2hear.inference.audio_io import AUDIO_EXTS, load_audio, resample, save_audio

MODE_DEFAULTS = {
    "restoration": {"channels": 2, "min_seconds": 4.0, "min_active_db": -50.0},
    "vocal": {"channels": 1, "min_seconds": 4.0, "min_active_db": -45.0},
}


def find_audio_files(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(AUDIO_EXTS):
                files.append(os.path.join(dirpath, name))
    return files


def fix_channels(wav, channels):
    """Force the channel count, keeping stereo width where it exists."""
    have = wav.shape[0]
    if have == channels:
        return wav
    if channels == 1:
        return wav.mean(0, keepdim=True)
    if have == 1:
        return wav.repeat(channels, 1)
    return wav[:channels]


def spectral_cutoff_hz(wav, sr, n_fft=4096, drop_db=25.0, search_from_hz=11000.0,
                       span_hz=2000.0):
    """Frequency of a lossy encoder's brick wall, or None if there isn't one.

    A "clean" training target that is itself an MP3 teaches the model to
    *reproduce* codec artefacts, which is the exact opposite of the task, and
    nothing else in the pipeline can notice: the degradation stage will cheerfully
    encode it a second time and hand the trainer a pair whose clean side is already
    damaged. This is the single most damaging thing that can be wrong with a corpus
    here, and it is invisible in every other check.

    What it looks for is the **cliff**, not the absence of treble. Every lossy
    encoder brick-walls -- LAME at 128 kbps near 16 kHz, AAC at 128 near 17 -- and
    the cut is tens of dB across a narrow band, where natural roll-off is gradual.
    Measuring bandwidth instead would reject a solo piano, an analogue transfer or
    any legitimately dark recording, so the test is the size of the step: the mean
    level in the ``span_hz`` below a candidate frequency against the mean above it.

    The search starts at ``search_from_hz`` because below that a cliff means
    something else entirely -- a narrowband source, not a transcode -- and sparse
    material (a handful of tones) would otherwise register as one.

    Returns the cutoff in Hz, or None.
    """
    mono = wav.mean(0) if wav.dim() > 1 else wav
    if mono.shape[-1] < n_fft * 4:
        return None

    window = torch.hann_window(n_fft)
    spec = torch.stft(mono, n_fft, n_fft // 2, window=window, return_complex=True)
    power = spec.abs().pow(2).mean(-1)
    power = power / power.max().clamp(min=1e-20)
    db = 10 * torch.log10(power.clamp(min=1e-20))

    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr)
    nyquist = sr / 2.0
    span = max(1, int(span_hz / (sr / n_fft)))

    best_drop, best_freq = 0.0, None
    for index in range(len(freqs)):
        centre = float(freqs[index])
        if centre < search_from_hz or centre > 0.98 * nyquist:
            continue
        below = db[max(0, index - span):index]
        above = db[index:index + span]
        if below.numel() < span // 2 or above.numel() < span // 2:
            continue
        drop = float(below.mean() - above.mean())
        if drop > best_drop:
            best_drop, best_freq = drop, centre

    return best_freq if best_drop >= drop_db else None


def inspect(wav, sr, min_seconds, min_active_db, max_clip_ratio, min_cutoff_hz=0.0):
    """Return a rejection reason, or None if the file is usable."""
    if wav.numel() == 0:
        return "empty"
    if wav.shape[-1] < min_seconds * sr:
        return "too_short"
    if not torch.isfinite(wav).all():
        return "non_finite"

    peak = float(wav.abs().max())
    if peak <= 0:
        return "silent"

    rms = float(wav.pow(2).mean().sqrt())
    if 20 * torch.log10(torch.tensor(max(rms, 1e-12))) < min_active_db:
        return "too_quiet"

    # a file that is already hard-clipped teaches the model to reproduce clipping
    clipped = float((wav.abs() >= 0.999).float().mean())
    if clipped > max_clip_ratio:
        return "clipped"

    # ...and a file that is already a transcode teaches it to reproduce codec
    # artefacts, which is worse, because it is invisible in every other check
    if min_cutoff_hz > 0:
        cutoff = spectral_cutoff_hz(wav, sr)
        if cutoff is not None and cutoff < min_cutoff_hz:
            return "lossy_source"

    return None


def process_one(path, args):
    wav, file_sr = load_audio(path)
    wav = resample(wav, file_sr, args.sr)
    wav = fix_channels(wav, args.channels)

    reason = inspect(wav, args.sr, args.min_seconds, args.min_active_db,
                     args.max_clip_ratio, args.min_cutoff_hz)
    if reason is not None:
        return None, reason

    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * args.peak

    return wav, None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["restoration", "vocal"], required=True,
                        help="restoration is stereo music, vocal is mono voice")
    parser.add_argument("--in_dir", required=True, help="folder of source audio, searched recursively")
    parser.add_argument("--out_dir", required=True, help="destination; train/ and valid/ are created inside")

    parser.add_argument("--sr", type=int, default=44100, help="target sample rate")
    parser.add_argument("--channels", type=int, default=None,
                        help="override the mode default (2 for restoration, 1 for vocal)")
    parser.add_argument("--min_seconds", type=float, default=None,
                        help="reject files shorter than this")
    parser.add_argument("--min_active_db", type=float, default=None,
                        help="reject files quieter than this dBFS RMS")
    parser.add_argument("--max_clip_ratio", type=float, default=0.01,
                        help="reject files with more than this fraction of samples at full scale")
    parser.add_argument("--peak", type=float, default=0.95,
                        help="peak-normalise survivors to this level")
    parser.add_argument("--min_cutoff_hz", type=float, default=0.0,
                        help="reject sources that are themselves transcodes: files whose "
                             "spectrum brick-walls below this. 0 (default) only REPORTS "
                             "them in the manifest; 19000 rejects anything at or below "
                             "192 kbps, 17500 keeps 192 kbps and drops 128 kbps")

    parser.add_argument("--valid_ratio", type=float, default=0.02, help="fraction held out for validation")
    parser.add_argument("--max_valid", type=int, default=50, help="cap on validation files")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ext", default=".flac", choices=[".wav", ".flac"],
                        help="output format; flac is lossless and about half the size")
    parser.add_argument("--bit_depth", type=int, default=24, choices=[16, 24, 32])
    parser.add_argument("--dry_run", action="store_true", help="report what would happen, write nothing")

    args = parser.parse_args(argv)

    defaults = MODE_DEFAULTS[args.mode]
    if args.channels is None:
        args.channels = defaults["channels"]
    if args.min_seconds is None:
        args.min_seconds = defaults["min_seconds"]
    if args.min_active_db is None:
        args.min_active_db = defaults["min_active_db"]

    files = find_audio_files(args.in_dir)
    if not files:
        raise SystemExit(f"no audio files found under {args.in_dir}")

    print(f"mode={args.mode}  channels={args.channels}  sr={args.sr}")
    print(f"found {len(files)} candidate file(s) under {args.in_dir}\n")

    random.Random(args.seed).shuffle(files)
    n_valid = min(args.max_valid, max(1, int(len(files) * args.valid_ratio)))

    rejected = Counter()
    manifest = {"train": [], "valid": []}
    kept_seconds = 0.0
    kept = 0

    for index, path in enumerate(files):
        try:
            wav, reason = process_one(path, args)
        except Exception as exc:
            rejected["unreadable"] += 1
            print(f"  skip {os.path.basename(path)}: {exc}")
            continue

        if reason is not None:
            rejected[reason] += 1
            continue

        split = "valid" if kept < n_valid else "train"
        stem = os.path.splitext(os.path.relpath(path, args.in_dir))[0].replace(os.sep, "__")
        out_path = os.path.join(args.out_dir, split, stem + args.ext)

        seconds = wav.shape[-1] / args.sr
        kept_seconds += seconds
        kept += 1
        manifest[split].append({"source": path, "output": out_path,
                                "seconds": round(seconds, 2)})

        if not args.dry_run:
            save_audio(out_path, wav, args.sr, bit_depth=args.bit_depth)

        if kept % 25 == 0 or index == len(files) - 1:
            print(f"\r  kept {kept}/{index + 1}  ({kept_seconds / 3600:.2f} h)", end="", flush=True)

    print()
    print(f"\nkept    {kept} file(s), {kept_seconds / 3600:.2f} h total")
    print(f"  train {len(manifest['train'])}")
    print(f"  valid {len(manifest['valid'])}")

    if rejected:
        print("\nrejected:")
        for reason, count in rejected.most_common():
            print(f"  {reason:12s} {count}")

    if kept == 0:
        raise SystemExit("nothing survived preprocessing; loosen --min_seconds / --min_active_db")
    if not manifest["valid"]:
        print("\nWARNING: no validation files -- raise --valid_ratio")

    if not args.dry_run:
        manifest_path = os.path.join(args.out_dir, "manifest.json")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({"mode": args.mode, "sr": args.sr, "channels": args.channels,
                       "hours": round(kept_seconds / 3600, 3),
                       "rejected": dict(rejected), "files": manifest}, handle, indent=2)
        print(f"\nmanifest -> {manifest_path}")
        print(f"now set datas.train_dir to {os.path.join(args.out_dir, 'train')}")
    else:
        print("\n(dry run, nothing written)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
