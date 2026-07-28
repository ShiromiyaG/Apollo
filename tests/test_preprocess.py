"""Tests for the dataset preprocessing script."""

import json
import math
import os
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts.preprocess import find_audio_files, fix_channels, inspect, main

SR = 44100


def _tone(sr=SR, secs=8.0, nch=2, amp=0.6, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * secs)) / sr
    wave = sum(np.sin(2 * math.pi * f * t) / (i + 1) for i, f in enumerate([110, 220, 440, 880]))
    wave = wave / np.abs(wave).max()
    wave = (wave + rng.normal(0, noise, len(t))) * amp
    data = np.stack([wave] * nch, -1) if nch > 1 else wave[:, None]
    return data.astype(np.float32)


# ------------------------------------------------------------- channels


@pytest.mark.parametrize("have,want", [(1, 1), (2, 2), (1, 2), (2, 1), (6, 2)])
def test_fix_channels(have, want):
    out = fix_channels(torch.randn(have, 1000), want)
    assert out.shape[0] == want


def test_mono_downmix_averages_rather_than_dropping():
    left = torch.ones(1, 100)
    right = -torch.ones(1, 100)
    out = fix_channels(torch.cat([left, right], 0), 1)
    assert torch.allclose(out, torch.zeros(1, 100))


def test_mono_to_stereo_duplicates():
    src = torch.randn(1, 100)
    out = fix_channels(src, 2)
    assert torch.allclose(out[0], out[1]) and torch.allclose(out[0], src[0])


# ------------------------------------------------------------- rejection


def test_accepts_a_good_file():
    wav = torch.from_numpy(_tone().T.copy())
    assert inspect(wav, SR, min_seconds=4.0, min_active_db=-50.0, max_clip_ratio=0.01) is None


@pytest.mark.parametrize("wav,secs,reason", [
    (torch.zeros(2, SR * 8), 4.0, "silent"),
    (torch.randn(2, SR), 4.0, "too_short"),
    (torch.zeros(2, 0), 4.0, "empty"),
])
def test_rejects_unusable(wav, secs, reason):
    assert inspect(wav, SR, secs, -50.0, 0.01) == reason


def test_rejects_too_quiet():
    wav = torch.from_numpy(_tone(amp=0.0002).T.copy())
    assert inspect(wav, SR, 4.0, -50.0, 0.01) == "too_quiet"


def test_rejects_clipped():
    """Training on already-clipped audio teaches the model to reproduce clipping."""
    wav = torch.from_numpy(_tone(amp=1.0).T.copy()) * 8
    wav = wav.clamp(-1.0, 1.0)
    assert inspect(wav, SR, 4.0, -50.0, 0.01) == "clipped"


def test_rejects_non_finite():
    wav = torch.from_numpy(_tone().T.copy())
    wav[0, 100] = float("nan")
    assert inspect(wav, SR, 4.0, -50.0, 0.01) == "non_finite"


# ------------------------------------------------------------ end to end


def _build_corpus(root):
    os.makedirs(os.path.join(root, "album"), exist_ok=True)
    sf.write(os.path.join(root, "good_a.wav"), _tone(seed=1), SR)
    sf.write(os.path.join(root, "album", "good_b.wav"), _tone(seed=2), SR)
    sf.write(os.path.join(root, "album", "good_48k.flac"), _tone(sr=48000, nch=1, seed=3), 48000)
    sf.write(os.path.join(root, "short.wav"), _tone(secs=1.0, seed=4), SR)
    sf.write(os.path.join(root, "silent.wav"), np.zeros((SR * 8, 2), dtype=np.float32), SR)


def test_find_audio_files_is_recursive(tmp_path):
    _build_corpus(str(tmp_path))
    assert len(find_audio_files(str(tmp_path))) == 5


def test_restoration_mode_outputs_stereo_at_target_rate(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    _build_corpus(str(src))

    main(["--mode", "restoration", "--in_dir", str(src), "--out_dir", str(dst),
          "--valid_ratio", "0.3"])

    produced = find_audio_files(str(dst))
    assert len(produced) == 3           # the short and silent files are dropped
    for path in produced:
        info = sf.info(path)
        assert info.channels == 2
        assert info.samplerate == SR


def test_vocal_mode_outputs_mono(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    _build_corpus(str(src))

    main(["--mode", "vocal", "--in_dir", str(src), "--out_dir", str(dst), "--valid_ratio", "0.3"])

    for path in find_audio_files(str(dst)):
        assert sf.info(path).channels == 1


def test_output_is_peak_normalised(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    os.makedirs(src)
    sf.write(src / "quiet_but_valid.wav", _tone(amp=0.05), SR)

    main(["--mode", "restoration", "--in_dir", str(src), "--out_dir", str(dst),
          "--peak", "0.95", "--valid_ratio", "1.0"])

    path = find_audio_files(str(dst))[0]
    data, _ = sf.read(path)
    assert abs(float(np.abs(data).max()) - 0.95) < 0.01


def test_manifest_records_what_happened(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    _build_corpus(str(src))

    main(["--mode", "restoration", "--in_dir", str(src), "--out_dir", str(dst),
          "--valid_ratio", "0.3"])

    with open(dst / "manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert manifest["mode"] == "restoration"
    assert manifest["channels"] == 2
    assert manifest["rejected"]["too_short"] == 1
    assert manifest["rejected"]["silent"] == 1
    assert len(manifest["files"]["train"]) + len(manifest["files"]["valid"]) == 3


def test_train_and_valid_do_not_overlap(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    _build_corpus(str(src))

    main(["--mode", "restoration", "--in_dir", str(src), "--out_dir", str(dst),
          "--valid_ratio", "0.34"])

    with open(dst / "manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)

    train = {entry["source"] for entry in manifest["files"]["train"]}
    valid = {entry["source"] for entry in manifest["files"]["valid"]}
    assert train and valid and not (train & valid)


def test_dry_run_writes_nothing(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    _build_corpus(str(src))

    main(["--mode", "restoration", "--in_dir", str(src), "--out_dir", str(dst), "--dry_run"])
    assert not dst.exists() or not find_audio_files(str(dst))


def test_errors_when_nothing_survives(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    os.makedirs(src)
    sf.write(src / "silent.wav", np.zeros((SR * 8, 2), dtype=np.float32), SR)

    with pytest.raises(SystemExit, match="nothing survived"):
        main(["--mode", "restoration", "--in_dir", str(src), "--out_dir", str(dst)])


def test_errors_on_empty_input(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    os.makedirs(src)
    with pytest.raises(SystemExit, match="no audio files"):
        main(["--mode", "vocal", "--in_dir", str(src), "--out_dir", str(dst)])


def test_preprocessed_output_feeds_the_dataset(tmp_path):
    """The whole point: what comes out must be directly loadable by the trainer."""
    from look2hear.datas.vocal_datamodule import VocalRestorationDataset

    src, dst = tmp_path / "in", tmp_path / "out"
    _build_corpus(str(src))
    main(["--mode", "vocal", "--in_dir", str(src), "--out_dir", str(dst), "--valid_ratio", "0.3"])

    dataset = VocalRestorationDataset(str(dst / "train"), sr=SR, segments=3.0,
                                      channels=1, num_samples=2, seed=0)
    clean, degraded = dataset[0]
    assert clean.shape == (1, 3 * SR) and degraded.shape == (1, 3 * SR)
    assert torch.isfinite(clean).all() and torch.isfinite(degraded).all()


# ------------------------------------------------------- lossy-source detection


from preprocess import inspect, spectral_cutoff_hz  # noqa: E402
from look2hear.datas.codec import encode_decode  # noqa: E402


def _broadband(secs=3.0, sr=44100, nch=2):
    torch.manual_seed(0)
    t = torch.arange(int(sr * secs), dtype=torch.float32) / sr
    wave = sum(torch.sin(2 * math.pi * f * t) / (i + 1)
               for i, f in enumerate([110, 220, 440, 880, 1760, 3520, 7040, 14080, 19000]))
    wave = wave + torch.randn_like(wave) * 0.05
    return (wave / wave.abs().max() * 0.7).unsqueeze(0).repeat(nch, 1)


def test_lossless_source_reports_no_cutoff():
    assert spectral_cutoff_hz(_broadband(), 44100) is None


@pytest.mark.parametrize("fmt,bitrate,ceiling", [
    ("mp3", 128, 17500),
    ("mp3", 192, 19500),
    ("aac", 128, 18500),
])
def test_transcodes_report_their_brick_wall(fmt, bitrate, ceiling):
    clean = _broadband()
    degraded = encode_decode(clean, 44100, fmt=fmt, bitrate_kbps=bitrate, strict=True)
    if degraded is None:
        pytest.skip(f"{fmt} not encodable on this install")

    cutoff = spectral_cutoff_hz(degraded, 44100)
    assert cutoff is not None and cutoff < ceiling


def test_inspect_rejects_a_transcoded_source():
    """A 'clean' target that is itself an MP3 teaches the model to reproduce codec
    artefacts -- the exact opposite of the task, and invisible to every other check."""
    clean = _broadband()
    degraded = encode_decode(clean, 44100, fmt="mp3", bitrate_kbps=128, strict=True)
    if degraded is None:
        pytest.skip("mp3 not encodable on this install")

    assert inspect(clean, 44100, 1.0, -50.0, 0.01, 19000.0) is None
    assert inspect(degraded, 44100, 1.0, -50.0, 0.01, 19000.0) == "lossy_source"


def test_the_check_can_be_disabled():
    degraded = encode_decode(_broadband(), 44100, fmt="mp3", bitrate_kbps=64, strict=True)
    if degraded is None:
        pytest.skip("mp3 not encodable on this install")
    assert inspect(degraded, 44100, 1.0, -50.0, 0.01, 0.0) is None


def test_short_files_are_not_judged():
    """Too few frames to estimate a long-term spectrum from."""
    assert spectral_cutoff_hz(torch.randn(2, 512), 44100) is None


def test_dark_material_is_not_mistaken_for_a_transcode():
    """The reason the test is a cliff and not a bandwidth measure: a solo piano, an
    analogue transfer or anything with a gentle roll-off must survive."""
    torch.manual_seed(1)
    noise = torch.randn(2, 44100 * 3)
    spectrum = torch.fft.rfft(noise)
    freqs = torch.fft.rfftfreq(noise.shape[-1], 1 / 44100)
    spectrum = spectrum * (1.0 / (1.0 + (freqs / 2000.0) ** 2))   # -12 dB/oct
    dark = torch.fft.irfft(spectrum, n=noise.shape[-1])
    dark = dark / dark.abs().max() * 0.7

    assert spectral_cutoff_hz(dark, 44100) is None


def test_sparse_material_is_not_mistaken_for_a_transcode():
    """A handful of tones drops to the floor above the top partial, but that is not
    a codec cutoff -- which is why the search starts at 11 kHz."""
    assert spectral_cutoff_hz(torch.from_numpy(_tone(secs=3.0)).T.float(), 44100) is None


def test_the_check_is_report_only_by_default():
    """It is a heuristic; silently shrinking someone's corpus is not its job."""
    parser_default = 0.0
    degraded = encode_decode(_broadband(), 44100, fmt="mp3", bitrate_kbps=96, strict=True)
    if degraded is None:
        pytest.skip("mp3 not encodable on this install")
    assert inspect(degraded, 44100, 1.0, -50.0, 0.01, parser_default) is None
