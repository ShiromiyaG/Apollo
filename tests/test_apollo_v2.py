"""Tests for the v2 generator and its round-trip through the checkpoint loader."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look2hear.inference.loader import _unwrap_state_dict, is_v2_state_dict, load_apollo
from look2hear.models.apollo import Apollo
from look2hear.models.apollo_v2 import ApolloV2, SwiGLU, TemporalStack

SR = 44100


def _tiny(**kwargs):
    kwargs.setdefault("feature_dim", 32)
    kwargs.setdefault("layer", 1)
    return ApolloV2(sr=SR, win=20, **kwargs)


# ------------------------------------------------------------- components


def test_swiglu_halves_the_projection():
    block = SwiGLU(64, mult=2.0)
    assert block.proj_in.out_channels == 256   # 2 * (64 * 2.0)
    assert block.proj_out.in_channels == 128
    assert block(torch.randn(2, 64, 16)).shape == (2, 64, 16)


def test_temporal_stack_receptive_field():
    stack = TemporalStack(32, blocks=2, kernel=11, dilations=(1, 2))
    # 1 + (11-1)*1 + (11-1)*2
    assert stack.receptive_field == 31


def test_temporal_stack_pads_missing_dilations():
    stack = TemporalStack(32, blocks=3, kernel=5, dilations=(1,))
    assert len(stack.blocks) == 3


def test_receptive_field_exceeds_v1_but_fits_the_training_segment():
    v2 = ApolloV2(sr=SR, win=20, feature_dim=32, layer=6)
    # v1's ICB is 3 blocks of kernel 7, no dilation: 18 frames per layer over 6
    # layers -> 109 frames (1.09 s). Note that is the *stack*, not one layer.
    v1_frames = 1 + 3 * (7 - 1) * 6
    assert v1_frames == 109

    assert v2.receptive_field_frames > 1.5 * v1_frames
    # context past the training segment cannot be learned, and forces a larger
    # overlap at chunked inference
    assert 1000 < v2.receptive_field_ms < 3000


# ---------------------------------------------------------------- forward


@pytest.mark.parametrize("nch", [1, 2])
@pytest.mark.parametrize("secs", [0.5, 1.3])
def test_forward_preserves_shape(nch, secs):
    model = _tiny().eval()
    wav = torch.randn(1, nch, int(SR * secs)) * 0.1
    with torch.inference_mode():
        out = model(wav)
    assert out.shape == wav.shape
    assert torch.isfinite(out).all()


def test_residual_mode_starts_close_to_identity():
    """The small output init plus a residual path means training starts near pass-through."""
    torch.manual_seed(0)
    model = _tiny(output_mode="residual").eval()
    wav = torch.randn(1, 1, SR // 2) * 0.1
    with torch.inference_mode():
        out = model(wav)

    err = (out - wav).pow(2).mean().sqrt() / wav.pow(2).mean().sqrt()
    assert float(err) < 0.5, "residual init should not be far from the input"


def test_direct_mode_is_not_identity():
    torch.manual_seed(0)
    model = _tiny(output_mode="direct").eval()
    wav = torch.randn(1, 1, SR // 2) * 0.1
    with torch.inference_mode():
        out = model(wav)
    assert (out - wav).abs().max() > 1e-3


def test_mid_side_round_trips():
    """The M/S transform must be exactly invertible, or stereo would drift."""
    model = _tiny(stereo_mode="ms")
    wav = torch.randn(1, 2, 1000)
    restored = model._from_working_channels(model._to_working_channels(wav))
    assert torch.allclose(restored, wav, atol=1e-6)


def test_mid_side_is_ignored_for_mono():
    model = _tiny(stereo_mode="ms").eval()
    wav = torch.randn(1, 1, SR // 2) * 0.1
    with torch.inference_mode():
        assert torch.isfinite(model(wav)).all()


def test_rejects_bad_modes():
    with pytest.raises(ValueError, match="output_mode"):
        _tiny(output_mode="mask")
    with pytest.raises(ValueError, match="stereo_mode"):
        _tiny(stereo_mode="joint")


# ------------------------------------------------------- memory features


def test_vram_budget_does_not_change_output():
    torch.manual_seed(0)
    model = _tiny().eval()
    wav = torch.randn(1, 1, SR // 2) * 0.1
    with torch.inference_mode():
        model.set_vram_budget(0)
        full = model(wav)
        model.set_vram_budget(1)      # forces many tiny slices
        sliced = model(wav)
    assert (full - sliced).abs().max() < 1e-5


def test_gradient_checkpointing_is_exact():
    torch.manual_seed(0)
    model = _tiny(layer=2).train()
    wav = torch.randn(1, 1, SR // 4) * 0.1

    model.set_gradient_checkpointing(False)
    plain = model(wav)
    plain.pow(2).mean().backward()
    plain_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    model.zero_grad(set_to_none=True)
    model.set_gradient_checkpointing(True)
    ckpt = model(wav)
    ckpt.pow(2).mean().backward()
    ckpt_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    assert (plain - ckpt).abs().max() < 1e-6
    for a, b in zip(plain_grads, ckpt_grads):
        assert (a - b).abs().max() < 1e-5


def test_v2_is_smaller_than_v1_at_equal_dims():
    v1 = Apollo(sr=SR, win=20, feature_dim=256, layer=6)
    v2 = ApolloV2(sr=SR, win=20, feature_dim=256, layer=6)
    p1 = sum(p.numel() for p in v1.parameters())
    p2 = sum(p.numel() for p in v2.parameters())
    assert p2 < 0.65 * p1, f"v2 {p2/1e6:.2f}M vs v1 {p1/1e6:.2f}M"


# --------------------------------------------------------- checkpointing


def test_state_dict_is_recognised_as_v2():
    assert is_v2_state_dict(_tiny().state_dict())
    assert not is_v2_state_dict(Apollo(sr=SR, win=20, feature_dim=32, layer=1).state_dict())


def test_model_args_rebuild_an_identical_model():
    model = _tiny(layer=2, output_mode="direct", stereo_mode="ms",
                  icb_kernel=9, icb_dilations=(1, 3))
    rebuilt = ApolloV2(**model.get_model_args())
    assert set(rebuilt.state_dict()) == set(model.state_dict())
    for key, tensor in model.state_dict().items():
        assert rebuilt.state_dict()[key].shape == tensor.shape


def test_export_round_trip(tmp_path):
    """Train-side export must be directly loadable by the inference loader."""
    model = _tiny(layer=2, stereo_mode="ms")
    path = tmp_path / "v2.pth"
    torch.save({"state_dict": model.state_dict(),
                "model_name": "ApolloV2",
                "model_args": model.get_model_args()}, path)

    loaded = load_apollo(checkpoint=str(path), device="cpu")
    assert isinstance(loaded, ApolloV2)
    assert loaded.stereo_mode == "ms"
    assert len(loaded.net) == 2

    wav = torch.randn(1, 2, SR // 4) * 0.1
    with torch.inference_mode():
        assert torch.allclose(loaded(wav), model.eval()(wav), atol=1e-6)


def test_v2_without_model_args_fails_clearly(tmp_path):
    path = tmp_path / "bare.pth"
    torch.save({"state_dict": _tiny().state_dict()}, path)
    with pytest.raises(ValueError, match="model_args"):
        load_apollo(checkpoint=str(path), device="cpu")


def test_v1_checkpoints_still_load(tmp_path):
    """v2 support must not disturb the v1 path."""
    model = Apollo(sr=SR, win=20, feature_dim=32, layer=1)
    path = tmp_path / "v1.pth"
    torch.save({"state_dict": model.state_dict()}, path)

    loaded = load_apollo(checkpoint=str(path), device="cpu")
    assert isinstance(loaded, Apollo) and not isinstance(loaded, ApolloV2)
    assert loaded.feature_dim == 32
