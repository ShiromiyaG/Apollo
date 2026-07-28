"""Wiring tests for the Lightning modules and the model's training-mode features."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look2hear.discriminators.frequencydis import MultiFrequencyDiscriminator
from look2hear.losses.gan_losses import MultiFrequencyDisLoss, VocalGenLoss
from look2hear.models.apollo import Apollo
from look2hear.system.vocal_litmodule import VocalLightningModule, _as_module_dict

SR = 44100


def _make_module(**kwargs):
    model = Apollo(sr=SR, win=20, feature_dim=16, layer=1)
    disc = MultiFrequencyDiscriminator(nch=1, window=[64])
    losses = {"g": VocalGenLoss(mel_n_fft=1024, mel_hop=256, mel_bins=32),
              "d": MultiFrequencyDisLoss()}
    opt = [torch.optim.AdamW(model.parameters()), torch.optim.AdamW(disc.parameters())]
    sch = [torch.optim.lr_scheduler.StepLR(o, 1) for o in opt]
    return VocalLightningModule(model=model, discriminator=disc, optimizer=opt,
                                loss_func=losses, metrics=None, scheduler=sch, **kwargs)


# --------------------------------------------------- loss registration


def test_losses_are_registered_as_children():
    """Regression: a bare mapping leaves the mel buffers stranded on the CPU."""
    module = _make_module()
    assert isinstance(module.loss_func, torch.nn.ModuleDict)
    assert "loss_func" in dict(module.named_children())

    names = [n for n, _ in module.named_buffers()]
    assert any("loss_func.g.fullness.mel.mel_basis" in n for n in names)


def test_as_module_dict_handles_omegaconf_mapping():
    """Hydra hands the loss mapping over as a DictConfig, which is not a dict."""
    omegaconf = pytest.importorskip("omegaconf")

    losses = omegaconf.DictConfig({}, flags={"allow_objects": True})
    losses.g = VocalGenLoss(mel_n_fft=512, mel_hop=128, mel_bins=16)
    losses.d = MultiFrequencyDisLoss()

    assert not isinstance(losses, dict)  # the trap this guards against
    assert isinstance(_as_module_dict(losses), torch.nn.ModuleDict)


def test_as_module_dict_passes_through_non_module_values():
    plain = {"g": "not a module"}
    assert _as_module_dict(plain) is plain


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_loss_buffers_follow_the_module_to_cuda():
    module = _make_module().to("cuda")
    for name, buf in module.named_buffers():
        assert buf.device.type == "cuda", f"{name} was left behind on {buf.device}"


# ------------------------------------------------------------- options


def test_autocast_dtype_parsing():
    assert _make_module(autocast_dtype="none").autocast_dtype is None
    assert _make_module(autocast_dtype="bf16").autocast_dtype is torch.bfloat16
    assert _make_module(autocast_dtype="fp16").autocast_dtype is torch.float16


def test_unknown_autocast_dtype_rejected():
    with pytest.raises(KeyError):
        _make_module(autocast_dtype="int8")


def test_gradient_checkpointing_flag_reaches_the_model():
    module = _make_module(gradient_checkpointing=True)
    assert module.audio_model.grad_checkpointing is True


def test_accumulate_is_at_least_one():
    assert _make_module(accumulate_grad_batches=0).accumulate == 1


# ---------------------------------------------- model training features


def test_gradient_checkpointing_matches_plain_forward():
    """Checkpointing recomputes rather than approximates: values and grads must match."""
    torch.manual_seed(0)
    model = Apollo(sr=SR, win=20, feature_dim=16, layer=2).train()
    wav = torch.randn(1, 1, SR // 2) * 0.2

    model.set_gradient_checkpointing(False)
    out_plain = model(wav)
    out_plain.pow(2).mean().backward()
    grads_plain = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    model.zero_grad(set_to_none=True)
    model.set_gradient_checkpointing(True)
    out_ckpt = model(wav)
    out_ckpt.pow(2).mean().backward()
    grads_ckpt = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    assert (out_plain - out_ckpt).abs().max() < 1e-6
    assert len(grads_plain) == len(grads_ckpt)
    for a, b in zip(grads_plain, grads_ckpt):
        assert (a - b).abs().max() < 1e-5


def test_model_runs_under_bf16_autocast():
    """torch.complex rejects reduced precision; the output head must pin back to fp32."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU for bf16 autocast")

    model = Apollo(sr=SR, win=20, feature_dim=16, layer=1).eval().cuda()
    wav = torch.randn(1, 1, SR // 2, device="cuda") * 0.2
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(wav)
    assert out.shape == wav.shape
    assert torch.isfinite(out).all()


def test_state_dict_stays_compatible_with_upstream_naming():
    """The fused forward must not change any parameter name or shape."""
    model = Apollo(sr=SR, win=20, feature_dim=32, layer=1)
    keys = model.state_dict()

    assert "BN.0.1.weight" in keys and "output.0.1.bias" in keys
    assert "net.0.band_net.weight.weight" in keys
    # the STFT window is a non-persistent buffer, so it must not leak into the file
    assert not any("stft_window" in k for k in keys)
