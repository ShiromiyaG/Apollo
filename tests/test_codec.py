"""Tests for the codec simulation that replaces torchaudio's removed apply_codec."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look2hear.datas.codec import (
    align_to_reference,
    codec_simu,
    encode_decode,
    lowpass_resample,
)

SR = 44100


def _tone(freq, secs=1.0, nch=2, sr=SR):
    t = torch.arange(int(sr * secs), dtype=torch.float32) / sr
    wave = torch.sin(2 * math.pi * freq * t) * 0.5
    return wave.unsqueeze(0).repeat(nch, 1)


def _music_like(secs=1.0, nch=2, sr=SR):
    """Harmonic stack plus a little noise -- enough structure for a codec to chew on."""
    torch.manual_seed(0)
    t = torch.arange(int(sr * secs), dtype=torch.float32) / sr
    wave = sum(torch.sin(2 * math.pi * f * t) / (i + 1) for i, f in enumerate([220, 440, 880, 1760, 3520]))
    wave = wave + torch.randn_like(wave) * 0.05
    wave = wave / wave.abs().max() * 0.7
    return wave.unsqueeze(0).repeat(nch, 1)


# ------------------------------------------------------------------ alignment


def test_align_recovers_known_shift():
    wav = _music_like(secs=0.5)
    shifted = torch.roll(wav, 137, dims=-1)
    recovered = align_to_reference(wav, shifted)
    # compare away from the wrap-around region
    assert (recovered[:, 500:-500] - wav[:, 500:-500]).abs().max() < 1e-4


def test_align_leaves_aligned_signal_alone():
    wav = _music_like(secs=0.5)
    assert torch.allclose(align_to_reference(wav, wav.clone()), wav)


# --------------------------------------------------------------------- mp3


@pytest.mark.parametrize("bitrate", [32, 128])
@pytest.mark.parametrize("nch", [1, 2])
def test_mp3_roundtrip_shape_and_degradation(bitrate, nch):
    wav = _music_like(nch=nch)
    out = encode_decode(wav, SR, fmt="mp3", bitrate_kbps=bitrate)

    assert out.shape == wav.shape
    assert torch.isfinite(out).all()
    # it must actually degrade the signal, but still resemble it
    diff = (out - wav).pow(2).mean().sqrt()
    assert diff > 1e-4, "codec did not change the signal at all"
    assert diff < wav.pow(2).mean().sqrt(), "codec output is unrelated to the input"


def test_lower_bitrate_degrades_more():
    wav = _music_like()
    err = {}
    for bitrate in (32, 128):
        out = encode_decode(wav, SR, fmt="mp3", bitrate_kbps=bitrate)
        err[bitrate] = float((out - wav).pow(2).mean().sqrt())
    assert err[32] > err[128]


def test_mp3_is_roughly_time_aligned():
    """Misaligned pairs would teach the model to smear transients."""
    wav = _music_like()
    out = encode_decode(wav, SR, fmt="mp3", bitrate_kbps=128)

    mid = slice(SR // 4, -SR // 4)
    ref, est = wav[0, mid], out[0, mid]
    corr = torch.dot(ref, est) / (ref.norm() * est.norm() + 1e-9)
    assert float(corr) > 0.9


def test_low_bitrate_removes_high_frequencies():
    """A 24 kbps encode should band-limit; that is the content Apollo must invent."""
    wav = _music_like()
    out = encode_decode(wav, SR, fmt="mp3", bitrate_kbps=24)

    def hf_energy(x):
        spec = torch.fft.rfft(x[0]).abs()
        freqs = torch.fft.rfftfreq(x.shape[-1], 1 / SR)
        return float(spec[freqs > 12000].pow(2).sum())

    assert hf_energy(out) < 0.5 * hf_energy(wav)


# -------------------------------------------------------------------- misc


def test_codec_simu_accepts_legacy_options():
    wav = _music_like(secs=0.5)
    out = codec_simu(wav, sr=SR, options={"bitrate": "random", "compression": "random",
                                          "complexity": "random", "vbr": "random"})
    assert out.shape == wav.shape
    assert torch.isfinite(out).all()


def test_codec_simu_does_not_mutate_caller_options():
    options = {"bitrate": "random"}
    codec_simu(_music_like(secs=0.3), sr=SR, options=options)
    assert options == {"bitrate": "random"}


def test_unsupported_format_raises():
    with pytest.raises(ValueError):
        encode_decode(_music_like(secs=0.2), SR, fmt="wavpack")


def test_rejects_wrong_rank():
    with pytest.raises(ValueError):
        encode_decode(torch.zeros(1, 2, 100), SR)


def test_lowpass_resample_band_limits():
    wav = _tone(15000, secs=0.5)
    out = lowpass_resample(wav, SR, cutoff_hz=4000)
    assert out.shape[-1] == wav.shape[-1]
    assert out.abs().max() < 0.1 * wav.abs().max()


def test_lowpass_resample_passes_low_content():
    wav = _tone(300, secs=0.5)
    out = lowpass_resample(wav, SR, cutoff_hz=8000)
    mid = slice(SR // 10, -SR // 10)
    assert out[:, mid].abs().max() > 0.8 * wav[:, mid].abs().max()


def test_lowpass_noop_above_nyquist():
    wav = _music_like(secs=0.3)
    assert torch.equal(lowpass_resample(wav, SR, cutoff_hz=SR), wav)


# ---------------------------------------------------------- codec registry


from look2hear.datas.codec import (  # noqa: E402
    CODEC_BUNDLES,
    CODECS,
    available_codecs,
    resolve_formats,
)


def test_every_bundle_names_only_known_codecs():
    for bundle, names in CODEC_BUNDLES.items():
        unknown = [n for n in names if n not in CODECS]
        assert not unknown, f"bundle {bundle} references {unknown}"


def test_resolve_expands_bundles_and_dedupes():
    resolved = resolve_formats(["mdct", "mp3"], check=False)
    assert resolved[0] == "mp3" and resolved.count("mp3") == 1
    assert "aac" in resolved


def test_resolve_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown codec"):
        resolve_formats(["definitely_not_a_codec"], check=False)


def test_resolve_raises_when_nothing_is_encodable(monkeypatch):
    """A codec that cannot be encoded makes encode_decode return its input, which
    would pair every clean segment with an identical copy and teach identity."""
    monkeypatch.setattr("look2hear.datas.codec.available_codecs", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="identical"):
        resolve_formats(["mp3"])


def test_strict_mode_reports_failure_instead_of_returning_the_input(monkeypatch):
    monkeypatch.setattr("look2hear.datas.codec._mp3_lameenc",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr("look2hear.datas.codec._via_ffmpeg", lambda *a, **k: None)
    monkeypatch.setattr("look2hear.datas.codec._via_soundfile",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    wav = _tone(440, secs=0.2)
    assert encode_decode(wav, SR, fmt="mp3", strict=True) is None
    assert torch.equal(encode_decode(wav, SR, fmt="mp3"), wav)


@pytest.mark.parametrize("name,expected", [
    ("opus", 48000),      # libopus takes 8/12/16/24/48; 44.1k content goes UP to 48
    ("amr_nb", 8000),
    ("amr_wb", 16000),
    ("ac3", 44100),       # 32/44.1/48 only, and 44.1 is exact
    ("mp3", 44100),       # unconstrained
])
def test_target_rate_picks_the_nearest_supported_rate(name, expected):
    assert CODECS[name].target_rate(SR) == expected


def test_bitrate_is_clamped_into_the_codec_range():
    # AMR-NB tops out at 12.2 kbps; asking for 96 must not be passed through
    assert CODECS["amr_nb"].clamp_bitrate(96) == 12
    assert CODECS["amr_nb"].clamp_bitrate(1) == 5
    assert CODECS["mp3"].clamp_bitrate(128) == 128


@pytest.mark.parametrize("name", sorted(set(CODECS) - {"m4a", "vorbis"}))
def test_codec_round_trips_and_degrades(name):
    if not available_codecs([name], sr=SR):
        pytest.skip(f"{name} not encodable on this install")
    wav = _music_like(secs=0.5, nch=2)
    out = encode_decode(wav, SR, fmt=name, bitrate_kbps=48, strict=True)

    assert out is not None
    assert out.shape == wav.shape, "the caller's shape must come back unchanged"
    assert torch.isfinite(out).all()
    # it must actually do something -- a no-op would be a silent identity pair
    assert (out - wav).abs().mean() > 1e-4


@pytest.mark.parametrize("name", ["amr_nb", "gsm", "g722"])
def test_narrowband_codecs_remove_the_high_band(name):
    if not available_codecs([name], sr=SR):
        pytest.skip(f"{name} not encodable on this install")
    wav = _music_like(secs=0.5, nch=2)
    out = encode_decode(wav, SR, fmt=name, bitrate_kbps=16, strict=True)

    def high_energy(x):
        spec = torch.fft.rfft(x[0]).abs() ** 2
        freqs = torch.fft.rfftfreq(x.shape[-1], 1 / SR)
        return float(spec[freqs >= 9000].sum() / spec.sum().clamp(min=1e-12))

    # these resample to 8 or 16 kHz, so everything above ~8 kHz is gone by
    # construction -- that is the degradation, and it is why they are in the set
    assert high_energy(out) < 0.02 * max(high_energy(wav), 1e-9) + 1e-4
