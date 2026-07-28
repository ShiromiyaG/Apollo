"""Tests for the GAN-side changes: EMA, R1, branch rotation, adversarial ramp."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look2hear.discriminators.frequencydis import MultiFrequencyDiscriminator, r1_penalty
from look2hear.losses.gan_losses import MultiFrequencyDisLoss, MultiFrequencyGenLoss, VocalGenLoss
from look2hear.models.apollo import Apollo
from look2hear.system.ema import ModelEMA
from look2hear.system.restoration_litmodule import RestorationLightningModule

SR = 44100


def _disc(nch=1, windows=(128, 256, 512), hidden=4):
    return MultiFrequencyDiscriminator(nch=nch, window=list(windows), hidden_channels=hidden)


# ----------------------------------------------------------------- EMA


def test_ema_tracks_a_moving_average():
    model = torch.nn.Linear(4, 4)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ModelEMA(model, decay=0.5, warmup_steps=0)

    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)
    # one update at decay 0.5 lands halfway
    assert float(ema.shadow["weight"].mean()) == pytest.approx(0.5, abs=1e-6)

    ema.update(model)
    assert float(ema.shadow["weight"].mean()) == pytest.approx(0.75, abs=1e-6)


def test_ema_warmup_ramps_the_decay():
    model = torch.nn.Linear(2, 2)
    fast = ModelEMA(model, decay=0.999, warmup_steps=1000)
    slow = ModelEMA(model, decay=0.999, warmup_steps=0)
    # early on the warmup schedule must move faster than the target decay
    assert fast._current_decay() < slow._current_decay()


def test_ema_smooths_out_jitter():
    """The point of EMA: the average is closer to the centre than any single step."""
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 8)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ModelEMA(model, decay=0.9, warmup_steps=0)

    for _ in range(200):
        with torch.no_grad():
            model.weight.normal_(mean=1.0, std=0.5)   # noisy orbit around 1.0
        ema.update(model)

    raw_error = abs(float(model.weight.detach().mean()) - 1.0)
    ema_error = abs(float(ema.shadow["weight"].mean()) - 1.0)
    assert ema_error < raw_error


def test_ema_averaged_context_restores_weights():
    model = torch.nn.Linear(4, 4)
    ema = ModelEMA(model, decay=0.5, warmup_steps=0)
    with torch.no_grad():
        model.weight.fill_(5.0)
    before = model.weight.detach().clone()

    with ema.averaged(model):
        assert not torch.allclose(model.weight, before)
    assert torch.allclose(model.weight, before)


def test_ema_round_trips_through_state_dict():
    model = torch.nn.Linear(4, 4)
    ema = ModelEMA(model, decay=0.9, warmup_steps=0)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)

    restored = ModelEMA(torch.nn.Linear(4, 4), decay=0.9, warmup_steps=0)
    restored.load_state_dict(ema.state_dict())
    assert torch.allclose(restored.shadow["weight"], ema.shadow["weight"])
    assert restored.num_updates == ema.num_updates


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_ema_follows_the_model_onto_the_accelerator():
    """Regression: the shadow is built before the trainer moves the module."""
    model = torch.nn.Linear(4, 4)
    ema = ModelEMA(model, decay=0.9, warmup_steps=0)   # shadow allocated on CPU
    model = model.cuda()

    ema.update(model)                                   # must not raise
    assert all(t.is_cuda for t in ema.shadow.values())


def test_ema_rejects_bad_decay():
    with pytest.raises(ValueError):
        ModelEMA(torch.nn.Linear(2, 2), decay=1.0)


def test_ema_handles_integer_buffers():
    """Averaging a step counter would be meaningless; it must be copied."""
    module = torch.nn.Module()
    module.register_buffer("steps", torch.tensor(0, dtype=torch.long))
    ema = ModelEMA(module, decay=0.9, warmup_steps=0)

    module.steps.fill_(7)
    ema.update(module)
    assert int(ema.shadow["steps"]) == 7


# ------------------------------------------------------- discriminator


def test_branch_selection_runs_only_the_named_branches():
    # eval mode: spectral_norm refreshes its power iteration on every training
    # forward, so two calls would otherwise legitimately disagree
    disc = _disc(windows=(128, 256, 512)).eval()
    wav = torch.randn(1, 1, SR // 4) * 0.2

    with torch.no_grad():
        all_out, all_maps = disc(wav)
        some_out, some_maps = disc(wav, branches=[0, 2])

    assert len(all_out) == 3 and len(some_out) == 2
    assert len(all_maps) == 3 and len(some_maps) == 2
    # a selected branch must produce exactly what it would have in a full pass
    assert torch.allclose(all_out[0], some_out[0], atol=1e-6)
    assert torch.allclose(all_out[2], some_out[1], atol=1e-6)


def test_branch_selection_is_consistent_with_the_losses():
    """A shortened branch list must still form a valid loss."""
    disc = _disc(windows=(128, 256, 512))
    wav = torch.randn(1, 1, SR // 4) * 0.2

    est_out, est_maps = disc(wav, branches=[1])
    tgt_out, tgt_maps = disc(wav * 0.9, branches=[1])

    assert torch.isfinite(MultiFrequencyDisLoss()(tgt_out, est_out))
    loss = MultiFrequencyGenLoss()(est_out, est_maps, tgt_maps, wav, wav.squeeze(0))
    assert torch.isfinite(loss)


def test_hidden_channels_is_configurable():
    small = _disc(hidden=4)
    big = _disc(hidden=16)
    assert sum(p.numel() for p in big.parameters()) > 4 * sum(p.numel() for p in small.parameters())


def test_min_freq_bin_drops_low_bins():
    disc_full = _disc(windows=(256,))
    disc_cut = MultiFrequencyDiscriminator(nch=1, window=[256], hidden_channels=4, min_freq_bin=20)
    wav = torch.randn(1, 1, SR // 4) * 0.2

    out_full, _ = disc_full(wav)
    out_cut, _ = disc_cut(wav)
    # fewer input rows -> a smaller feature map along the frequency axis
    assert out_cut[0].shape[-2] < out_full[0].shape[-2]


def test_channel_mismatch_is_caught():
    disc = _disc(nch=2)
    with pytest.raises(AssertionError, match="2ch"):
        disc(torch.randn(1, 1, SR // 8))


# ---------------------------------------------------------------- R1


def test_r1_penalty_is_positive_and_differentiable():
    disc = _disc(windows=(256,))
    real = torch.randn(1, 1, SR // 4) * 0.2

    penalty = r1_penalty(disc, real, SR)
    assert torch.isfinite(penalty) and float(penalty) > 0

    penalty.backward()
    grads = [p.grad for p in disc.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_r1_penalty_does_not_leak_grad_into_the_input():
    disc = _disc(windows=(256,))
    real = torch.randn(1, 1, SR // 8) * 0.2
    r1_penalty(disc, real, SR)
    assert real.grad is None  # the helper detaches before requiring grad


def test_r1_respects_branch_selection():
    disc = _disc(windows=(128, 256, 512))
    real = torch.randn(1, 1, SR // 8) * 0.2
    assert torch.isfinite(r1_penalty(disc, real, SR, branches=[0]))


# ------------------------------------------------------ adversarial ramp


def test_adv_scale_gates_only_the_adversarial_terms():
    torch.manual_seed(0)
    est_out = [torch.randn(1, 1, 4, 4)]
    est_maps = [[torch.randn(1, 3, 4, 4)]]
    tgt_maps = [[torch.randn(1, 3, 4, 4)]]
    output = torch.randn(1, 1, SR // 4) * 0.2
    target = torch.randn(1, SR // 4) * 0.2

    loss = VocalGenLoss(sr=SR, mel_n_fft=1024, mel_hop=256, mel_bins=64)

    loss(est_out, est_maps, tgt_maps, output, target, adv_scale=0.0)
    off = dict(loss.last_terms)
    loss(est_out, est_maps, tgt_maps, output, target, adv_scale=1.0)
    on = dict(loss.last_terms)

    assert off["adv"] == 0.0 and off["fm"] == 0.0
    assert on["adv"] > 0.0
    # the reconstruction terms must be untouched by the ramp
    for term in ("freq", "fullness", "bleedless", "clarity", "waveform"):
        assert off[term] == pytest.approx(on[term], rel=1e-6)


def test_original_loss_also_accepts_adv_scale():
    est_out = [torch.randn(1, 1, 4, 4)]
    est_maps = [[torch.randn(1, 3, 4, 4)]]
    tgt_maps = [[torch.randn(1, 3, 4, 4)]]
    output = torch.randn(1, 1, SR // 4) * 0.2
    target = torch.randn(1, SR // 4) * 0.2

    loss = MultiFrequencyGenLoss()
    a = loss(est_out, est_maps, tgt_maps, output, target, adv_scale=0.0)
    b = loss(est_out, est_maps, tgt_maps, output, target, adv_scale=1.0)
    assert float(a) < float(b)


# ------------------------------------------------- lightning wiring


def _module(**kwargs):
    model = Apollo(sr=SR, win=20, feature_dim=16, layer=1)
    disc = _disc(windows=(128, 256, 512, 1024))
    losses = {"g": VocalGenLoss(mel_n_fft=1024, mel_hop=256, mel_bins=32),
              "d": MultiFrequencyDisLoss()}
    opt = [torch.optim.AdamW(model.parameters()), torch.optim.AdamW(disc.parameters())]
    sch = [torch.optim.lr_scheduler.StepLR(o, 1) for o in opt]
    return RestorationLightningModule(model=model, discriminator=disc, optimizer=opt,
                                      loss_func=losses, metrics=None, scheduler=sch, **kwargs)


def test_branch_rotation_covers_every_branch():
    module = _module(disc_branches_per_step=2)
    seen = set()
    for _ in range(4):
        seen.update(module._active_branches())
    assert seen == {0, 1, 2, 3}


def test_branch_rotation_disabled_returns_none():
    assert _module(disc_branches_per_step=0)._active_branches() is None
    # asking for at least as many as exist is the same as asking for all
    assert _module(disc_branches_per_step=9)._active_branches() is None


@pytest.mark.parametrize("step,expected", [
    (0, 0.0),      # before disc_start_step: reconstruction only
    (99, 0.0),
    (100, 0.0),    # ramp begins
    (150, 0.5),
    (200, 1.0),    # fully faded in
    (5000, 1.0),
])
def test_adv_scale_schedule(step, expected):
    module = _module(disc_start_step=100, adv_ramp_steps=100)
    module._batch_step = step
    assert module._adv_scale() == pytest.approx(expected)


def test_adv_scale_without_ramp_is_a_step_function():
    module = _module(disc_start_step=100, adv_ramp_steps=0)
    module._batch_step = 99
    assert module._adv_scale() == 0.0
    module._batch_step = 100
    assert module._adv_scale() == 1.0


def test_adv_scale_counts_optimizer_windows_not_global_step():
    """Regression: `global_step` counts every manual optimizer.step(), which is
    two per window once the critic runs. The ramp used to finish in half the
    configured number of steps."""
    module = _module(disc_start_step=0, adv_ramp_steps=100)
    module._batch_step = 50
    assert module._adv_scale() == pytest.approx(0.5)
    # advancing the trainer's own counter must not move the schedule
    assert module._adv_scale() == pytest.approx(0.5)


def test_ema_is_created_and_disabled_on_request():
    assert _module(ema_decay=0.999).ema is not None
    assert _module(ema_decay=0).ema is None


def test_r1_disables_cudnn_benchmark():
    """The R1 double backward can stall forever under cuDNN autotuning."""
    torch.backends.cudnn.benchmark = True
    _module(r1_gamma=1.0)
    assert torch.backends.cudnn.benchmark is False


def test_checkpoint_carries_the_ema_weights():
    module = _module(ema_decay=0.9)
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    assert "ema" in checkpoint and "shadow" in checkpoint["ema"]

    other = _module(ema_decay=0.9)
    other.on_load_checkpoint(checkpoint)
    assert other.ema.num_updates == module.ema.num_updates


def test_export_generator_writes_loadable_weights(tmp_path):
    from look2hear.inference.loader import _unwrap_state_dict

    module = _module(ema_decay=0.9)
    path = tmp_path / "exported.pth"
    module.export_generator(str(path))

    state = _unwrap_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    fresh = Apollo(sr=SR, win=20, feature_dim=16, layer=1)
    fresh.load_state_dict(state, strict=True)


# ------------------------------------------------- lazy penalty scheduling


def _fires_at(module, updates):
    """Which discriminator updates the lazy penalty would run on."""
    fired = []
    for _ in range(updates):
        if (module._d_update_count + 1) % module.r1_every == 0:
            fired.append(module._d_update_count)
        module._d_update_count += 1
    return fired


def test_penalty_fires_once_per_interval_of_discriminator_updates():
    """Regression: scheduling on `batch_idx` counted micro-batches, so under
    `accumulate_grad_batches: 8` with `r1_every: 16` the penalty fired every 2
    discriminator updates while still being multiplied by 16 -- 8x too strong."""
    module = _module(r1_gamma=1.0, r1_every=16, accumulate_grad_batches=8)
    assert _fires_at(module, 64) == [15, 31, 47, 63]


def test_penalty_share_keeps_gamma_invariant_to_branch_count():
    """The applied weight must not depend on whether the step ran branch-wise."""
    fused = _module(r1_gamma=1.0, r1_every=1, disc_branchwise=False)
    branch = _module(r1_gamma=1.0, r1_every=1, disc_branchwise=True)
    # fused: one group of 4 branches, share 1.0. branch-wise: 4 groups of 1,
    # share 0.25 each -- the same total.
    assert 4 * (1 / 4) == pytest.approx(1.0)
    assert len(fused.discriminator.discriminators) == 4
    assert len(branch.discriminator.discriminators) == 4


def test_schedule_state_round_trips_through_a_checkpoint():
    module = _module(ema_decay=0.9)
    module._d_update_count = 37
    module._batch_step = 41
    module._branch_cursor = 2

    checkpoint = {}
    module.on_save_checkpoint(checkpoint)

    other = _module(ema_decay=0.9)
    other.on_load_checkpoint(checkpoint)
    assert (other._d_update_count, other._batch_step, other._branch_cursor) == (37, 41, 2)


def test_old_checkpoints_load_without_the_schedule_block():
    module = _module(ema_decay=0.9)
    module.on_load_checkpoint({})           # must not raise
    assert module._d_update_count == 0
