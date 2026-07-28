"""Tests for the chunked overlap-add pipeline and the post-processing helpers.

Run with:  python -m pytest tests/ -v
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look2hear.inference.chunked import chunked_restore, taper_window
from look2hear.inference.loader import _unwrap_state_dict, infer_hparams
from look2hear.inference.postprocess import (
    dry_wet_mix,
    match_loudness,
    residual_gate,
    spectral_crossover,
)
from look2hear.models.apollo import Apollo

SR = 44100


class Identity(torch.nn.Module):
    """Stand-in for Apollo that returns its input untouched."""

    def forward(self, x):
        return x


class Gain(torch.nn.Module):
    """Scale-sensitive stand-in, to check the normalisation round-trip."""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


# ---------------------------------------------------------------- windowing


def test_taper_window_pairs_sum_to_one():
    length, fade = 512, 128
    w = taper_window(length, fade, device="cpu")
    hop = length - fade
    # the fade-out of one chunk plus the fade-in of the next must reconstruct unity
    overlap_sum = w[hop:] + w[:fade]
    assert torch.allclose(overlap_sum, torch.ones(fade), atol=1e-6)
    assert torch.allclose(w[fade:hop], torch.ones(hop - fade), atol=1e-6)


def test_taper_window_no_fade():
    w = taper_window(256, 0, device="cpu")
    assert torch.allclose(w, torch.ones(256))


# ------------------------------------------------------------ overlap-add


@pytest.mark.parametrize("n", [SR, SR * 3 + 137, SR // 4])
@pytest.mark.parametrize("nch", [1, 2])
def test_ola_reconstructs_identity(n, nch):
    """An identity model through the chunker must return the signal unchanged."""
    torch.manual_seed(0)
    wav = torch.randn(nch, n) * 0.2
    out = chunked_restore(
        Identity(), wav,
        chunk_samples=SR // 2, overlap_samples=SR // 8,
        normalize="chunk", silence_db=-200.0,
    )
    assert out.shape == wav.shape
    assert (out - wav).abs().max() < 1e-5


def test_normalisation_round_trip_is_gain_neutral():
    """Per-chunk peak normalisation must be undone exactly on the way out."""
    torch.manual_seed(0)
    wav = torch.randn(2, SR * 2) * 0.03  # deliberately quiet
    out = chunked_restore(
        Gain(1.0), wav,
        chunk_samples=SR // 2, overlap_samples=SR // 8,
        normalize="chunk", silence_db=-200.0,
    )
    assert (out - wav).abs().max() < 1e-5


def test_normalize_chunk_actually_rescales_model_input():
    """The model should see peak ~1.0 even when the file is quiet."""
    seen = []

    class Recorder(torch.nn.Module):
        def forward(self, x):
            seen.append(float(x.abs().max()))
            return x

    wav = torch.randn(1, SR) * 0.01
    chunked_restore(Recorder(), wav, chunk_samples=SR // 2, overlap_samples=SR // 8,
                    normalize="chunk", silence_db=-200.0)
    assert seen and all(abs(p - 1.0) < 1e-3 for p in seen)

    seen.clear()
    chunked_restore(Recorder(), wav, chunk_samples=SR // 2, overlap_samples=SR // 8,
                    normalize="none", silence_db=-200.0)
    assert seen and all(p < 0.1 for p in seen)


def test_silent_chunks_bypass_the_model():
    """Silence must be copied through, not hallucinated over."""

    class Noise(torch.nn.Module):
        def forward(self, x):
            return torch.full_like(x, 0.5)

    wav = torch.zeros(1, SR)
    wav[:, : SR // 2] = torch.randn(1, SR // 2) * 0.5  # loud first half, silent second

    out = chunked_restore(Noise(), wav, chunk_samples=SR // 4, overlap_samples=SR // 16,
                          normalize="chunk", silence_db=-60.0)
    # the tail was silent, so it must stay silent
    assert out[:, -SR // 8:].abs().max() < 1e-4


def test_all_silent_input_returns_silence():
    out = chunked_restore(Identity(), torch.zeros(2, SR),
                          chunk_samples=SR // 2, overlap_samples=SR // 8)
    assert out.abs().max() == 0


def test_rejects_overlap_larger_than_chunk():
    with pytest.raises(ValueError):
        chunked_restore(Identity(), torch.zeros(1, SR), chunk_samples=100, overlap_samples=100)


def test_rejects_bad_normalize_mode():
    with pytest.raises(ValueError):
        chunked_restore(Identity(), torch.zeros(1, SR), chunk_samples=100,
                        overlap_samples=10, normalize="loudness")


def test_batch_size_does_not_change_output():
    torch.manual_seed(0)
    wav = torch.randn(2, SR * 2) * 0.2
    kwargs = dict(chunk_samples=SR // 2, overlap_samples=SR // 8, silence_db=-200.0)
    a = chunked_restore(Gain(0.9), wav, batch_size=1, **kwargs)
    b = chunked_restore(Gain(0.9), wav, batch_size=4, **kwargs)
    assert (a - b).abs().max() < 1e-6


# ------------------------------------------------------------ postprocess


def test_crossover_is_transparent_when_inputs_match():
    torch.manual_seed(0)
    wav = torch.randn(2, SR) * 0.2
    out = spectral_crossover(wav, wav.clone(), SR, crossover_hz=2000.0)
    n = out.shape[-1]
    # ignore STFT edge ramps
    assert (out[:, SR // 10 : n - SR // 10] - wav[:, SR // 10 : n - SR // 10]).abs().max() < 1e-4


def test_crossover_keeps_low_band_from_dry():
    """A tone below the crossover must survive from the dry signal only."""
    t = torch.arange(SR * 2, dtype=torch.float32) / SR
    low = torch.sin(2 * math.pi * 200 * t).unsqueeze(0) * 0.5
    dry = low
    wet = torch.zeros_like(low)  # model "deleted" everything

    out = spectral_crossover(dry, wet, SR, crossover_hz=4000.0, width_octaves=0.5)
    mid = out[:, SR // 2 : -SR // 2]
    ref = dry[:, SR // 2 : -SR // 2]
    # the 200 Hz tone is far below the crossover, so it should come back nearly intact
    assert mid.abs().max() > 0.4 * ref.abs().max()


def test_crossover_disabled_returns_wet():
    dry = torch.randn(1, 1000)
    wet = torch.randn(1, 1000)
    assert torch.equal(spectral_crossover(dry, wet, SR, crossover_hz=0.0), wet)


def test_gate_suppresses_residual_in_silence():
    wav = torch.zeros(1, SR)
    wav[:, : SR // 2] = torch.randn(1, SR // 2) * 0.5
    wet = wav + 0.05  # model added a constant noise floor everywhere

    out = residual_gate(wav, wet, SR, threshold_db=-55.0)
    # silent tail: the added floor should be gated away
    assert (out[:, -SR // 4 :] - wav[:, -SR // 4 :]).abs().max() < 0.01
    # loud head: the model's contribution must be preserved
    assert (out[:, : SR // 4] - wav[:, : SR // 4]).abs().mean() > 0.04


def test_dry_wet_mix_endpoints():
    dry, wet = torch.zeros(1, 100), torch.ones(1, 100)
    assert torch.allclose(dry_wet_mix(dry, wet, 1.0), wet)
    assert torch.allclose(dry_wet_mix(dry, wet, 0.0), dry)
    assert torch.allclose(dry_wet_mix(dry, wet, 0.25), torch.full((1, 100), 0.25))


def test_match_loudness():
    torch.manual_seed(0)
    ref = torch.randn(1, 1000)
    out = match_loudness(ref, ref * 4.0)
    assert abs(float(out.pow(2).mean().sqrt() - ref.pow(2).mean().sqrt())) < 1e-5


# ---------------------------------------------------------------- loader


def test_infer_hparams_matches_construction():
    model = Apollo(sr=SR, win=20, feature_dim=64, layer=2)
    win, dim, layer = infer_hparams(model.state_dict(), sr=SR)
    assert (win, dim, layer) == (20, 64, 2)


def test_unwrap_lightning_checkpoint():
    model = Apollo(sr=SR, win=20, feature_dim=32, layer=1)
    ckpt = {
        "state_dict": {f"audio_model.{k}": v for k, v in model.state_dict().items()}
        | {"discriminator.foo.weight": torch.zeros(3)},
        "epoch": 7,
    }
    state = _unwrap_state_dict(ckpt)
    assert set(state) == set(model.state_dict())


def test_unwrap_rejects_junk():
    with pytest.raises(ValueError):
        _unwrap_state_dict({"epoch": 1, "notes": "hello"})


# --------------------------------------------------- real model, seam check


def test_chunk_seams_are_inaudible_on_real_model():
    """Interior chunk seams must sit far below audibility on the real architecture.

    Only the interior is compared. At the very start and end of a file the chunked
    path deliberately feeds the model mirrored context, which a single full-length
    pass does not have, so the two legitimately differ there -- that is an edge
    handling choice, not a seam.
    """
    torch.manual_seed(0)
    model = Apollo(sr=SR, win=20, feature_dim=32, layer=1).eval()
    wav = torch.randn(1, SR * 3) * 0.2
    wav = wav / wav.abs().max()

    chunk, overlap = SR, SR // 4
    with torch.inference_mode():
        full = model(wav.unsqueeze(0)).squeeze(0)

    chunked = chunked_restore(model, wav, chunk_samples=chunk, overlap_samples=overlap,
                              normalize="none", silence_db=-200.0)

    interior = slice(chunk, wav.shape[-1] - chunk)
    err = (chunked[:, interior] - full[:, interior]).pow(2).mean().sqrt()
    ref = full[:, interior].pow(2).mean().sqrt()
    assert 20 * math.log10(float(err / ref)) < -40  # >40 dB below the signal

    # and the seams must not be worse than the chunk interiors around them
    seam = chunked[:, chunk - overlap : chunk + overlap] - full[:, chunk - overlap : chunk + overlap]
    quiet = chunked[:, chunk + overlap : chunk + 3 * overlap] - full[:, chunk + overlap : chunk + 3 * overlap]
    assert float(seam.pow(2).mean().sqrt()) < 10 * float(quiet.pow(2).mean().sqrt()) + 1e-6


def test_reflect_padding_beats_zero_padding_at_file_start():
    """Mirrored edge context should leave the opening of the file cleaner."""
    torch.manual_seed(0)
    model = Apollo(sr=SR, win=20, feature_dim=32, layer=1).eval()
    wav = torch.randn(1, SR * 2) * 0.2
    wav = wav / wav.abs().max()

    with torch.inference_mode():
        full = model(wav.unsqueeze(0)).squeeze(0)
    chunked = chunked_restore(model, wav, chunk_samples=SR, overlap_samples=SR // 4,
                              normalize="none", silence_db=-200.0)

    # the head should not blow up relative to the body of the track
    head = (chunked[:, : SR // 10] - full[:, : SR // 10]).pow(2).mean().sqrt()
    body = full[:, SR // 2 :].pow(2).mean().sqrt()
    assert float(head / body) < 0.5


def test_vram_budget_does_not_change_output():
    torch.manual_seed(0)
    model = Apollo(sr=SR, win=20, feature_dim=32, layer=1).eval()
    wav = torch.randn(1, 1, SR) * 0.2

    with torch.inference_mode():
        model.set_vram_budget(0)
        a = model(wav)
        model.set_vram_budget(1)  # forces many tiny slices
        b = model(wav)
    assert (a - b).abs().max() < 1e-5
