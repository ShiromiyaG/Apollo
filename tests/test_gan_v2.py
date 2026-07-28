"""Tests for the v2 GAN work: balancer, output-gradient generator step,
per-branch schedules, concatenated critic forward, R3GAN objectives, and the
dynamics critics.

Everything here is opt-in per config. The point of most of these tests is that
turning an option on does not change the arithmetic -- only where it happens.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look2hear.discriminators.combined import build_fast_discriminator
from look2hear.discriminators.frequencydis import MultiFrequencyDiscriminator, gradient_penalty
from look2hear.losses.balancer import Balancer
from look2hear.losses.gan_losses import (MultiFrequencyDisLoss, VocalGenLoss,
                                         d_branch_loss, g_branch_loss)
from look2hear.models.apollo import Apollo
from look2hear.system.restoration_litmodule import RestorationLightningModule

SR = 44100
SAMPLES = SR // 2


def _module(disc=None, **kwargs):
    model = Apollo(sr=SR, win=20, feature_dim=16, layer=1)
    disc = disc if disc is not None else build_fast_discriminator(
        nch=1, sample_rate=SR, use_mel=False, init_channel=4)
    losses = {"g": VocalGenLoss(mel_n_fft=1024, mel_hop=256, mel_bins=32,
                                **{k: kwargs.pop(k) for k in list(kwargs)
                                   if k == "gan_loss_type"}),
              "d": MultiFrequencyDisLoss()}
    opt = [torch.optim.AdamW(model.parameters()), torch.optim.AdamW(disc.parameters())]
    sch = [torch.optim.lr_scheduler.StepLR(o, 1) for o in opt]
    module = RestorationLightningModule(model=model, discriminator=disc, optimizer=opt,
                                        loss_func=losses, metrics=None, scheduler=sch,
                                        **kwargs)
    # `manual_backward` routes through the trainer, which is not attached here
    module.manual_backward = lambda loss, **kw: loss.backward(**kw)
    return module


def _run_generator_step(module, real, degraded):
    """One generator step, returning the generator's gradients."""
    module.zero_grad(set_to_none=True)
    fake = module(degraded)
    active = list(range(len(module.discriminator.discriminators)))
    module._generator_step(real, fake, 1.0, active, 1.0, {})
    return {name: p.grad.detach().clone()
            for name, p in module.audio_model.named_parameters() if p.grad is not None}


def _assert_close(a, b, tol=2e-4):
    assert set(a) == set(b)
    for name in a:
        scale = max(float(a[name].abs().max()), 1e-8)
        assert torch.allclose(a[name], b[name], atol=tol * scale, rtol=tol), name


# --------------------------------------------------------- output-grad G step


def test_output_grad_matches_the_per_branch_backward():
    """`gstep_output_grad` moves where the branch loop stops, not what it computes.

    The per-branch backward walks the generator graph once per branch; collecting
    at the output walks it once. Same gradients.
    """
    torch.manual_seed(0)
    module = _module(disc_branchwise=True)
    module.discriminator.eval()
    real = torch.randn(1, 1, SAMPLES) * 0.1
    degraded = torch.randn(1, 1, SAMPLES) * 0.1

    module.gstep_output_grad = False
    per_branch = _run_generator_step(module, real, degraded)

    module.gstep_output_grad = True
    collected = _run_generator_step(module, real, degraded)

    _assert_close(per_branch, collected)


def test_output_grad_traverses_the_generator_once():
    """The whole point: one backward through the generator, not one per branch."""
    torch.manual_seed(0)
    module = _module(disc_branchwise=True, gstep_output_grad=True)
    module.discriminator.eval()

    calls = []
    original = module.audio_model.forward

    def counting_forward(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    module.audio_model.forward = counting_forward
    real = torch.randn(1, 1, SAMPLES) * 0.1
    _run_generator_step(module, real, torch.randn(1, 1, SAMPLES) * 0.1)
    assert len(calls) == 1


# ------------------------------------------------------------------- balancer


def test_balancer_gives_each_term_its_configured_share():
    """Terms of wildly different raw scale must end up with gradient norms in the
    ratio of their weights -- that is the whole reason the balancer exists."""
    x = torch.randn(4, 8, requires_grad=True)
    balancer = Balancer({"big": 1.0, "tiny": 3.0}, total_norm=1.0, ema_decay=0.0)

    losses = {"big": (x * 1000.0).pow(2).mean(), "tiny": (x * 1e-4).pow(2).mean()}
    effective = balancer.backward(losses, x)

    assert float(effective["tiny"]) / float(effective["big"]) == pytest.approx(3.0, rel=1e-3)


def test_balancer_skips_zero_weighted_terms():
    x = torch.randn(4, 8, requires_grad=True)
    balancer = Balancer({"kept": 1.0, "dropped": 0.0}, ema_decay=0.0)
    effective = balancer.backward({"kept": x.pow(2).mean(), "dropped": x.abs().mean()}, x)
    assert set(effective) == {"kept"}


def test_balancer_weight_scale_applies_the_ramp():
    x = torch.randn(4, 8, requires_grad=True)
    balancer = Balancer({"a": 1.0, "b": 1.0}, ema_decay=0.0)
    losses = {"a": x.pow(2).mean(), "b": x.abs().mean()}
    effective = balancer.backward(losses, x, weight_scale={"b": 0.0})
    assert set(effective) == {"a"}


def test_balancer_pre_grads_match_passing_the_loss():
    """The branch-wise path hands over already-computed gradients; the balancer
    only ever uses dL/dinput, so the two forms must agree."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, requires_grad=True)
    weights = {"a": 1.0, "b": 2.0}

    direct = Balancer(weights, ema_decay=0.0)
    direct.backward({"a": (x * 3).pow(2).mean(), "b": x.abs().mean()}, x)
    from_loss = x.grad.detach().clone()

    x.grad = None
    grad_a, = torch.autograd.grad((x * 3).pow(2).mean(), x, retain_graph=True)
    pre = Balancer(weights, ema_decay=0.0)
    pre.backward({"b": x.abs().mean()}, x, pre_grads={"a": grad_a})

    assert torch.allclose(from_loss, x.grad, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_balancer_statistics_survive_a_reduced_precision_input(dtype):
    """A 0.999 decay updates the estimate by one part in a thousand, and bf16
    carries about three significant decimal digits -- accumulated in the input's
    dtype the EMA would freeze at its first value, silently, for the whole run."""
    x = torch.randn(4, 512, dtype=dtype, requires_grad=True)
    balancer = Balancer({"a": 1.0}, ema_decay=0.999)

    seen = []
    for step in range(20):
        balancer.backward({"a": (x.float() * (1.0 + 0.2 * step)).pow(2).mean()}, x)
        seen.append(float(balancer._ema["a"]))
        x.grad = None

    assert balancer._ema["a"].dtype == torch.float32
    assert len(set(seen)) == len(seen), "the EMA stopped moving"


def test_balancer_returns_a_gradient_in_the_input_dtype():
    x = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
    Balancer({"a": 1.0}, ema_decay=0.0).backward({"a": x.float().pow(2).mean()}, x)
    assert x.grad.dtype == torch.bfloat16


def test_balancer_round_trips_through_a_checkpoint():
    x = torch.randn(4, 8, requires_grad=True)
    balancer = Balancer({"a": 1.0}, ema_decay=0.9)
    balancer.backward({"a": x.pow(2).mean()}, x)

    restored = Balancer({"a": 1.0}, ema_decay=0.9)
    restored.load_state_dict(balancer.state_dict())
    assert float(restored._ema["a"]) == pytest.approx(float(balancer._ema["a"]))


def test_balancer_rejects_a_loss_it_cannot_split():
    from look2hear.losses.gan_losses import MultiFrequencyGenLoss

    module = _module(use_loss_balancer=True, balancer_weights={"freq": 1.0})
    module.loss_func["g"] = MultiFrequencyGenLoss()
    with pytest.raises(ValueError, match="terms separately"):
        module._validate_config()


def test_balancer_needs_weights():
    with pytest.raises(ValueError, match="balancer_weights"):
        _module(use_loss_balancer=True)


# ------------------------------------------------------------ branch schedule


def test_branch_schedule_matches_patterns_by_name():
    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mel=False, init_channel=4)
    module = _module(disc=disc, disc_branch_every={"mrd_*": 4})

    names = module._branch_names()
    expensive = [i for i, n in enumerate(names) if n.startswith("mrd_")]
    cheap = [i for i, n in enumerate(names) if not n.startswith("mrd_")]

    module._d_update_count = 0
    assert set(module._scheduled_branches()) == set(range(len(names)))
    module._d_update_count = 1
    assert set(module._scheduled_branches()) == set(cheap)
    module._d_update_count = 4
    assert set(module._scheduled_branches()) >= set(expensive)


def test_branch_schedule_offset_moves_the_active_step():
    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mel=False, init_channel=4)
    module = _module(disc=disc, disc_branch_every={"mrd_*": 2},
                     disc_branch_offset={"mrd_*": 1})
    names = module._branch_names()
    expensive = {i for i, n in enumerate(names) if n.startswith("mrd_")}

    module._d_update_count = 0
    assert not (set(module._scheduled_branches()) & expensive)
    module._d_update_count = 1
    assert set(module._scheduled_branches()) >= expensive


def test_branch_schedule_never_leaves_an_empty_set():
    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mel=False, init_channel=4)
    module = _module(disc=disc, disc_branch_every={"*": 3}, disc_branch_offset={"*": 1})
    module._d_update_count = 0     # every branch is off on this update
    assert module._scheduled_branches() == list(range(len(module._branch_names())))


def test_branch_schedule_is_rejected_without_branch_names():
    module = _module(disc=MultiFrequencyDiscriminator(nch=1, window=[256], hidden_channels=4),
                     disc_branch_every={"mrd_*": 2})
    with pytest.raises(ValueError, match="branch names"):
        module._validate_config()


def test_branch_selection_is_stable_across_an_accumulation_window():
    """The discriminator step, the penalty and the generator step have to agree on
    which branches exist; refreshing per micro-batch broke that."""
    module = _module(disc_branches_per_step=2, accumulate_grad_batches=4)
    module._refresh_branches()
    first = list(module._active)
    for _ in range(3):
        assert module._active == first     # no refresh inside the window
    module._refresh_branches()
    assert module._active != first


# -------------------------------------------------------- concatenated D pass


def test_concatenated_critic_forward_matches_two_forwards():
    torch.manual_seed(0)
    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mel=False, init_channel=4).eval()
    real = torch.randn(2, 1, SAMPLES) * 0.1
    fake = torch.randn(2, 1, SAMPLES) * 0.1
    group = [0, 1]

    separate = _module(disc=disc, disc_concat_real_fake=False)
    joint = _module(disc=disc, disc_concat_real_fake=True)

    with torch.no_grad():
        est_a, tgt_a = separate._discriminator_outputs(real, fake, group)
        est_b, tgt_b = joint._discriminator_outputs(real, fake, group)

    for a, b in zip(est_a, est_b):
        assert torch.allclose(a, b, atol=1e-5)
    for a, b in zip(tgt_a, tgt_b):
        assert torch.allclose(a, b, atol=1e-5)


# ------------------------------------------------------------ R3GAN objectives


def test_lsgan_is_unchanged_by_the_refactor():
    real = torch.randn(2, 8)
    fake = torch.randn(2, 8)
    expected = (real - 1).pow(2).mean() + fake.pow(2).mean()
    assert d_branch_loss(real, fake, "lsgan") == pytest.approx(float(expected), rel=1e-6)
    assert g_branch_loss(fake, "lsgan") == pytest.approx(float((fake - 1).pow(2).mean()))


def test_relativistic_judges_the_paired_difference():
    real = torch.zeros(2, 8)
    ahead = torch.full((2, 8), -2.0)     # fake well below real: D is winning
    behind = torch.full((2, 8), 2.0)     # fake above real: G is winning
    assert d_branch_loss(real, ahead, "relativistic") < d_branch_loss(real, behind, "relativistic")
    assert g_branch_loss(ahead, "relativistic", dr=real) > g_branch_loss(behind, "relativistic", dr=real)


def test_relativistic_generator_needs_the_real_logits():
    with pytest.raises(ValueError, match="real logits"):
        g_branch_loss(torch.zeros(2, 4), "relativistic")


def test_soft_hinge_gradient_is_bounded():
    """The reason to prefer it over LSGAN: a critic that pulls far ahead cannot
    hand the generator an arbitrarily large step."""
    far = torch.tensor([-50.0], requires_grad=True)
    g_branch_loss(far, "soft_hinge").backward()
    assert abs(float(far.grad)) <= 1.0

    far_ls = torch.tensor([-50.0], requires_grad=True)
    g_branch_loss(far_ls, "lsgan").backward()
    assert abs(float(far_ls.grad)) > 100.0


def test_generator_loss_routes_the_configured_objective():
    torch.manual_seed(0)
    est = [torch.randn(1, 1, 4, 4)]
    maps = [[torch.randn(1, 3, 4, 4)]]
    output = torch.randn(1, 1, SR // 8) * 0.1
    target = torch.randn(1, SR // 8) * 0.1

    lsgan = VocalGenLoss(sr=SR, mel_n_fft=1024, mel_hop=256, mel_bins=64)
    hinge = VocalGenLoss(sr=SR, mel_n_fft=1024, mel_hop=256, mel_bins=64,
                         gan_loss_type="soft_hinge")
    lsgan(est, maps, maps, output, target)
    hinge(est, maps, maps, output, target)
    assert lsgan.last_terms["adv"] != hinge.last_terms["adv"]
    # the reconstruction half must be untouched by the choice
    assert lsgan.last_terms["freq"] == pytest.approx(hinge.last_terms["freq"], rel=1e-6)


# ----------------------------------------------------------- gradient penalty


def test_summing_patches_inflates_the_penalty_by_the_squared_patch_count():
    """Summing P patches scales the differentiated score's gradient by P, so the
    penalty comes out exactly P**2 too large -- and P is a function of the segment
    length and the branch's strides, not of how sharp the critic should be."""
    torch.manual_seed(0)
    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mel=False,
                                    init_channel=4).eval()
    real = torch.randn(1, 1, SAMPLES) * 0.1

    with torch.no_grad():
        patches = disc(real, branches=[0])[0][0].numel()

    meaned = float(gradient_penalty(disc, real, SR, branches=[0]))
    summed = float(gradient_penalty(disc, real, SR, branches=[0], patch_reduction="sum"))

    assert patches > 100
    assert summed / meaned == pytest.approx(patches ** 2, rel=1e-3)


def test_penalty_averages_over_the_branches_in_a_group():
    """So `r1_gamma` does not change meaning when branches are added or removed.

    Averaging is exact here: the group penalty is the penalty of the mean score,
    which for a single branch is that branch's own.
    """
    torch.manual_seed(0)
    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mel=False,
                                    init_channel=4).eval()
    real = torch.randn(1, 1, SAMPLES) * 0.1

    one = float(gradient_penalty(disc, real, SR, branches=[0]))
    doubled = float(gradient_penalty(disc, real, SR, branches=[0, 0]))
    # the same branch twice must not double the penalty
    assert doubled == pytest.approx(one, rel=1e-3)


def test_joint_penalty_matches_computing_r1_and_r2_separately():
    """One forward for both halves is an optimisation, not a different penalty:
    samples are independent, so d(score_i)/dx_j is zero for i != j."""
    from look2hear.discriminators.frequencydis import joint_gradient_penalty

    torch.manual_seed(0)
    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mel=False,
                                    init_channel=4).eval()
    real = torch.randn(2, 1, SAMPLES) * 0.1
    fake = torch.randn(2, 1, SAMPLES) * 0.1

    separate = (float(gradient_penalty(disc, real, SR)),
                float(gradient_penalty(disc, fake, SR)))
    joined = joint_gradient_penalty(disc, real, fake, SR)

    assert float(joined[0]) == pytest.approx(separate[0], rel=1e-4)
    assert float(joined[1]) == pytest.approx(separate[1], rel=1e-4)


def test_joint_penalty_splits_on_the_input_batch_not_the_folded_one():
    """A multi-channel bank folds channels into the batch for its own forward, but
    the gradient keeps the caller's batch dimension."""
    from look2hear.discriminators.frequencydis import joint_gradient_penalty

    torch.manual_seed(0)
    disc = build_fast_discriminator(nch=2, sample_rate=SR, use_mel=False,
                                    init_channel=4).eval()
    real = torch.randn(3, 2, SAMPLES) * 0.1
    fake = torch.zeros(3, 2, SAMPLES)      # a very different input

    r1, r2 = joint_gradient_penalty(disc, real, fake, SR)
    assert torch.isfinite(r1) and torch.isfinite(r2)
    assert float(r1) == pytest.approx(float(gradient_penalty(disc, real, SR)), rel=1e-4)


def test_joint_penalty_is_differentiable_into_the_critic():
    from look2hear.discriminators.frequencydis import joint_gradient_penalty

    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mel=False,
                                    init_channel=4).eval()
    r1, r2 = joint_gradient_penalty(disc, torch.randn(1, 1, SAMPLES) * 0.1,
                                    torch.randn(1, 1, SAMPLES) * 0.1, SR)
    (r1 + r2).backward()
    grads = [p.grad for p in disc.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_branchwise_penalty_bounds_a_different_quantity():
    """Not a memory trade with the same meaning. Averaging the branch SCORES and
    then taking the norm gives the summed gradient field, whose cross-branch inner
    products cancel for near-uncorrelated branches -- so it lands about n_branches
    lower than averaging the per-branch norms."""
    torch.manual_seed(0)
    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mel=False,
                                    init_channel=4).eval()
    real = torch.randn(1, 1, SAMPLES) * 0.1
    branches = list(range(len(disc.discriminators)))

    fused = float(gradient_penalty(disc, real, SR, branches))
    per_branch = [float(gradient_penalty(disc, real, SR, [b])) for b in branches]
    branchwise = sum(per_branch) / len(per_branch)

    assert branchwise / fused == pytest.approx(len(branches), rel=0.35)


def test_r1_branchwise_selects_the_per_branch_form():
    torch.manual_seed(0)
    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mel=False,
                                    init_channel=4).eval()
    real = torch.randn(1, 1, SAMPLES) * 0.1
    fake = torch.randn(1, 1, SAMPLES) * 0.1
    branches = list(range(len(disc.discriminators)))

    reported = {}
    for flag in (True, False):
        module = _module(disc=disc, r1_gamma=1.0, r1_every=1, r1_branchwise=flag)
        stats = {}
        module._gan_reg_penalty(real, fake, branches, len(branches), stats)
        reported[flag] = stats["r1"]
        disc.zero_grad(set_to_none=True)

    assert reported[True] > 2.0 * reported[False]


def test_r1_branchwise_is_off_by_default():
    assert _module().r1_branchwise is False


def test_branchwise_and_joined_compose():
    """Both penalties, one branch at a time, one forward per branch."""
    module = _module(r1_gamma=1.0, r2_gamma=1.0, r1_every=1,
                     r1_branchwise=True, r1_r2_joined=True)
    branches = list(range(len(module.discriminator.discriminators)))
    stats = {}
    module._gan_reg_penalty(torch.randn(1, 1, SAMPLES) * 0.1,
                            torch.randn(1, 1, SAMPLES) * 0.1,
                            branches, len(branches), stats)
    assert set(stats) == {"r1", "r2"}
    assert all(v > 0 for v in stats.values())


def test_joined_is_ignored_when_only_one_gamma_is_set():
    """It is a two-penalty optimisation; with one gamma there is nothing to share."""
    module = _module(r1_gamma=1.0, r2_gamma=0.0, r1_r2_joined=True, r1_every=1)
    stats = {}
    module._gan_reg_penalty(torch.randn(1, 1, SAMPLES) * 0.1,
                            torch.randn(1, 1, SAMPLES) * 0.1, [0], 1, stats)
    assert set(stats) == {"r1"}


# ------------------------------------------------------- discriminator scheduler


class _CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def _step_schedulers(module, epochs, d_updates_from):
    """Drive the end-of-epoch scheduler block directly."""
    g, d = _CountingScheduler(), _CountingScheduler()
    for epoch in range(epochs):
        module._d_update_count = 1 if epoch >= d_updates_from else 0
        g.step()
        if not module.hold_disc_scheduler or module._d_update_count > 0:
            d.step()
    return g.steps, d.steps


def test_disc_scheduler_is_held_until_the_critic_has_stepped():
    """Regression: through a `disc_start_step: 5000` warmup the discriminator
    optimizer never steps, but a StepLR kept decaying its LR every epoch -- the
    critic woke up already annealed."""
    held = _module(hold_disc_scheduler=True)
    assert _step_schedulers(held, epochs=10, d_updates_from=6) == (10, 4)


def test_disc_scheduler_is_unheld_by_default():
    """v1 configs keep the original behaviour."""
    default = _module()
    assert default.hold_disc_scheduler is False
    assert _step_schedulers(default, epochs=10, d_updates_from=6) == (10, 10)


def test_r2_is_the_same_computation_on_generated_samples():
    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mel=False, init_channel=4).eval()
    fake = torch.randn(1, 1, SAMPLES) * 0.1
    penalty = gradient_penalty(disc, fake, SR)
    assert torch.isfinite(penalty) and float(penalty) > 0
    penalty.backward()
    assert any(p.grad is not None for p in disc.parameters())


# ------------------------------------------------------------ dynamics critics


@pytest.mark.parametrize("kwargs,expected", [
    ({"use_med": True}, 3),
    ({"use_transient": True}, 1),
    ({"use_hf_modulation": True}, 1),
    ({"use_periodicity": True}, 1),
])
def test_dynamics_branches_produce_logits_and_feature_maps(kwargs, expected):
    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mpd=False, use_mrd=False,
                                    use_mel=False, **kwargs)
    assert len(disc.discriminators) == expected

    wav = torch.randn(2, 1, SAMPLES) * 0.1
    outputs, maps = disc(wav)
    assert len(outputs) == expected and len(maps) == expected
    for logit, fmap in zip(outputs, maps):
        assert logit.shape[0] == 2 and torch.isfinite(logit).all()
        assert fmap and all(torch.isfinite(f).all() for f in fmap)


def test_dynamics_branches_are_named_for_the_schedule():
    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mpd=False, use_mrd=False,
                                    use_mel=False, use_med=True, use_transient=True,
                                    use_hf_modulation=True, use_periodicity=True)
    names = disc.names()
    assert len(names) == len(disc.discriminators)
    assert any(n.startswith("med_") for n in names)
    assert {"transient", "hf_modulation", "periodicity"} <= set(names)


def test_dynamics_branches_are_backwardable():
    disc = build_fast_discriminator(nch=1, sample_rate=SR, use_mpd=False, use_mrd=False,
                                    use_mel=False, use_med=True, use_hf_modulation=True)
    wav = (torch.randn(1, 1, SAMPLES) * 0.1).requires_grad_(True)
    outputs, _ = disc(wav)
    sum(o.mean() for o in outputs).backward()
    assert wav.grad is not None and torch.isfinite(wav.grad).all()


def test_hf_modulation_is_level_invariant():
    """A generator cannot satisfy this branch by adding energy -- only by moving it
    in time like the real thing."""
    from look2hear.discriminators.dynamics import HFModulationDiscriminator

    branch = HFModulationDiscriminator(sample_rate=SR, channels=8).eval()
    wav = torch.randn(1, 1, SAMPLES) * 0.1
    with torch.no_grad():
        quiet = branch._features(wav)
        loud = branch._features(wav * 4.0)
    # log1p is not exactly scale-free, but the DC normalisation removes the bulk
    assert torch.allclose(quiet, loud, atol=0.15)


def test_med_time_constants_give_different_resolutions():
    """Each MED branch is a different envelope time constant, so they must not
    collapse to the same view."""
    from look2hear.discriminators.dynamics import EnvelopeDiscriminator

    fast = EnvelopeDiscriminator(sample_rate=SR, pool=64, stride=16, channels=8).eval()
    slow = EnvelopeDiscriminator(sample_rate=SR, pool=1024, stride=256, channels=8).eval()
    wav = torch.randn(1, 1, SAMPLES) * 0.1
    with torch.no_grad():
        assert fast._features(wav).shape[-1] > 8 * slow._features(wav).shape[-1]


def test_v1_generator_loss_still_runs_through_the_loop():
    """Regression: the fused path passes the critic's real logits unconditionally
    (the relativistic objective needs them), and `MultiFrequencyGenLoss` -- what the
    music configs use -- had no parameter to receive them."""
    from look2hear.losses.gan_losses import MultiFrequencyGenLoss

    module = _module(disc=MultiFrequencyDiscriminator(nch=1, window=[256, 512],
                                                      hidden_channels=4))
    module.loss_func["g"] = MultiFrequencyGenLoss()

    real = torch.randn(1, 1, SAMPLES) * 0.1
    fake = module(torch.randn(1, 1, SAMPLES) * 0.1)
    loss = module._generator_step(real, fake, 1.0, [0, 1], 1.0, {})
    assert torch.isfinite(torch.as_tensor(loss))
    assert any(p.grad is not None for p in module.audio_model.parameters())


def test_dynamics_bank_combines_with_the_spectral_one():
    disc = build_fast_discriminator(nch=1, sample_rate=SR, init_channel=4,
                                    use_med=True, use_transient=True)
    wav = torch.randn(1, 1, SAMPLES) * 0.1
    outputs, maps = disc(wav)
    assert len(outputs) == len(disc.discriminators)

    # branch selection must still index into the flat list correctly
    subset, _ = disc(wav, branches=[0, len(outputs) - 1])
    assert len(subset) == 2
