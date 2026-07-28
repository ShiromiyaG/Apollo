"""Apollo audio restoration.

Single file:
    python inference.py --in_wav input.wav --out_wav output.wav --ckpt apollo_model.ckpt

Whole folder:
    python inference.py --in_dir ./lossy --out_dir ./restored --ckpt apollo_model.ckpt

Peak VRAM is set by --chunk and --vram_budget, not by how long the track is.
The defaults stay under ~1.5 GB, so a 4 minute song fits comfortably on 8 GB.

If the output sounds noisy or "grainy", the knobs that help most are
--crossover (keep the input's own low band) and --gate (stop the model
inventing a noise floor in silence). See README for the reasoning.
"""

import argparse
import os
import sys
import time

import torch

from look2hear.inference import RestoreOptions, iter_audio_files, load_apollo, restore_file


def build_parser():
    p = argparse.ArgumentParser(
        description="Restore lossy audio with Apollo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("input / output")
    io.add_argument("--in_wav", type=str, help="input audio file")
    io.add_argument("--out_wav", type=str, help="output audio file")
    io.add_argument("--in_dir", type=str, help="input folder (processed recursively)")
    io.add_argument("--out_dir", type=str, help="output folder")
    io.add_argument("--bit_depth", type=int, choices=[16, 24, 32], default=None,
                    help="output PCM bit depth; default follows the container")
    io.add_argument("--out_ext", type=str, default=".wav", help="extension for folder mode")

    mdl = p.add_argument_group("model")
    mdl.add_argument("--ckpt", type=str, default=None,
                     help="local checkpoint; omit to download from --repo_id")
    mdl.add_argument("--repo_id", type=str, default="JusperLee/Apollo")
    mdl.add_argument("--sr", type=int, default=44100, help="model sample rate")
    mdl.add_argument("--win", type=int, default=None, help="window in ms (inferred if unset)")
    mdl.add_argument("--feature_dim", type=int, default=None, help="inferred if unset")
    mdl.add_argument("--layer", type=int, default=None, help="inferred if unset")
    mdl.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    perf = p.add_argument_group("performance")
    perf.add_argument("--chunk", type=float, default=10.0,
                      help="seconds of audio per model pass; lower it if you run out of VRAM")
    perf.add_argument("--overlap", type=float, default=1.0, help="crossfade between chunks, seconds")
    perf.add_argument("--batch_size", type=int, default=1, help="chunks per forward pass")
    perf.add_argument("--vram_budget", type=float, default=128.0,
                      help="MB cap on the widest intermediate tensor; 0 disables slicing")

    qual = p.add_argument_group("quality")
    qual.add_argument("--normalize", choices=["chunk", "global", "none"], default="chunk",
                      help="'chunk' matches how Apollo was trained (peak-normalised segments)")
    qual.add_argument("--crossover", type=float, default=0.0, metavar="HZ",
                      help="keep the input below this frequency and the model above it; "
                           "try 1000-4000 to kill low-band noise. 0 disables")
    qual.add_argument("--crossover_width", type=float, default=0.5, metavar="OCTAVES",
                      help="width of the crossover transition")
    qual.add_argument("--dry_wet", type=float, default=1.0,
                      help="1.0 = pure model output, 0.0 = untouched input")
    qual.add_argument("--gate", action="store_true",
                      help="suppress the model's contribution where the input is near-silent")
    qual.add_argument("--gate_threshold", type=float, default=-55.0, metavar="DB")
    qual.add_argument("--silence_db", type=float, default=-60.0,
                      help="chunks quieter than this bypass the model entirely")
    qual.add_argument("--match_loudness", action="store_true",
                      help="rescale the output to the input's RMS")

    return p


def resolve_jobs(args):
    """Return a list of (in_path, out_path) pairs."""
    if args.in_dir:
        if not args.out_dir:
            raise SystemExit("--in_dir requires --out_dir")
        jobs = []
        for path in iter_audio_files(args.in_dir):
            rel = os.path.relpath(path, args.in_dir)
            stem = os.path.splitext(rel)[0]
            jobs.append((path, os.path.join(args.out_dir, stem + args.out_ext)))
        if not jobs:
            raise SystemExit(f"no audio files found under {args.in_dir}")
        return jobs

    if not args.in_wav or not args.out_wav:
        raise SystemExit("provide either --in_wav/--out_wav or --in_dir/--out_dir")
    return [(args.in_wav, args.out_wav)]


def main(argv=None):
    args = build_parser().parse_args(argv)
    jobs = resolve_jobs(args)

    print(f"Loading model on {args.device} ...", flush=True)
    model = load_apollo(
        checkpoint=args.ckpt,
        repo_id=args.repo_id,
        sr=args.sr,
        win=args.win,
        feature_dim=args.feature_dim,
        layer=args.layer,
        device=args.device,
        vram_budget_mb=args.vram_budget,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"  win={model.win} samples  feature_dim={model.feature_dim}  "
          f"bands={model.nband}  params={params/1e6:.1f}M")

    options = RestoreOptions(
        chunk_seconds=args.chunk,
        overlap_seconds=args.overlap,
        batch_size=args.batch_size,
        normalize=args.normalize,
        silence_db=args.silence_db,
        crossover_hz=args.crossover,
        crossover_width_octaves=args.crossover_width,
        dry_wet=args.dry_wet,
        gate=args.gate,
        gate_threshold_db=args.gate_threshold,
        match_input_loudness=args.match_loudness,
    )

    failures = 0
    for idx, (src, dst) in enumerate(jobs, 1):
        label = os.path.basename(src)
        started = time.perf_counter()

        def progress(done, total, _label=label, _idx=idx):
            print(f"\r[{_idx}/{len(jobs)}] {_label}  chunk {done}/{total}", end="", flush=True)

        try:
            note = restore_file(model, src, dst, options, device=args.device,
                                model_sr=args.sr, bit_depth=args.bit_depth, progress=progress)
        except Exception as exc:  # keep going through a batch
            failures += 1
            print(f"\r[{idx}/{len(jobs)}] {label}  FAILED: {exc}")
            continue

        elapsed = time.perf_counter() - started
        print(f"\r[{idx}/{len(jobs)}] {label}  {note}  in {elapsed:.1f}s -> {dst}")

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        print(f"peak VRAM: {torch.cuda.max_memory_allocated()/2**20:.0f} MB")

    if failures:
        print(f"{failures} file(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
