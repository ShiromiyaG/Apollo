"""Tests for the critic bank: Fast* branches, SAN heads, composition, branch-wise loss."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look2hear.discriminators import (
    CombinedDiscriminator,
    FastMelBank,
    FastMPD,
    FastMRD,
    MultiFrequencyDiscriminator,
    MultiScaleSTFTDiscriminator,
    build_fast_discriminator,
)
from look2hear.discriminators.fast import L2Normalize, SoftSignGLU
from look2hear.losses.gan_losses import MultiFrequencyDisLoss, VocalGenLoss

SR = 44100


def _wav(nch=1, secs=1.0, batch=1):
    return torch.randn(batch, nch, int(SR * secs)) * 0.2


# ------------------------------------------------------------- primitives


def test_softsign_glu_matches_reference_formula():
    """The custom autograd Function must compute the plain formula."""
    x = torch.randn(4, 16, requires_grad=True)
    out, gate = torch.split(x, 8, dim=-1)
    expected = (out / (out.abs() + 1)) * (gate / (gate.abs() + 1))
    assert torch.allclose(SoftSignGLU()(x), expected, atol=1e-6)


def test_softsign_glu_gradients_match_autograd():
    """Hand-written backward must agree with what autograd would have produced."""
    x = torch.randn(4, 16, dtype=torch.double, requires_grad=True)

    custom = SoftSignGLU()(x).sum()
    custom.backward()
    custom_grad = x.grad.clone()

    x.grad = None
    out, gate = torch.split(x, 8, dim=-1)
    reference = ((out / (out.abs() + 1)) * (gate / (gate.abs() + 1))).sum()
    reference.backward()

    assert torch.allclose(custom_grad, x.grad, atol=1e-9)


def test_l2_normalize_produces_unit_rows():
    weight = torch.randn(5, 7, 3)
    normalized = L2Normalize()(weight)
    norms = normalized.reshape(5, -1).norm(dim=1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-6)


def test_san_head_bounds_the_logit_scale():
    """SAN makes the final projection direction-only, so scaling it cannot inflate output."""
    torch.manual_seed(0)
    with_san = FastMPD(nch=1, periods=(2,), use_san=True).eval()
    without = FastMPD(nch=1, periods=(2,), use_san=False).eval()

    with torch.no_grad():
        # blow up the final projection in both
        with_san.discriminators[0].post.parametrizations.weight.original.mul_(50)
        without.discriminators[0].post.weight.mul_(50)

    wav = _wav()
    with torch.no_grad():
        san_out = with_san(wav)[0][0].abs().max()
        plain_out = without(wav)[0][0].abs().max()

    assert float(san_out) < float(plain_out)


# ---------------------------------------------------------------- banks


@pytest.mark.parametrize("nch", [1, 2])
@pytest.mark.parametrize("bank", ["mpd", "mrd", "mel", "msstft"])
def test_banks_produce_the_expected_protocol(nch, bank):
    builders = {
        "mpd": lambda: FastMPD(nch=nch, periods=(2, 3)),
        "mrd": lambda: FastMRD(nch=nch, fft_sizes=(512,), hop_sizes=(128,), win_lengths=(512,)),
        "mel": lambda: FastMelBank(nch=nch, sample_rate=SR, fft_sizes=(1024,),
                                   hop_sizes=(256,), win_lengths=(1024,), n_mels=32),
        "msstft": lambda: MultiScaleSTFTDiscriminator(nch=nch, n_ffts=(512,)),
    }
    disc = builders[bank]()
    outputs, feature_maps = disc(_wav(nch=nch))

    assert len(outputs) == len(disc.discriminators)
    assert len(feature_maps) == len(outputs)
    for score, features in zip(outputs, feature_maps):
        assert torch.isfinite(score).all()
        assert isinstance(features, list) and features
        assert all(torch.isfinite(f).all() for f in features)


def test_channel_mismatch_is_caught():
    with pytest.raises(AssertionError, match="2ch"):
        FastMPD(nch=2, periods=(2,))(_wav(nch=1))


def test_branch_selection_returns_a_subset():
    disc = FastMPD(nch=1, periods=(2, 3, 5))
    assert len(disc(_wav())[0]) == 3
    assert len(disc(_wav(), branches=[0, 2])[0]) == 2


def test_period_branch_rejects_too_short_input():
    disc = FastMPD(nch=1, periods=(11,))
    with pytest.raises(ValueError, match="too short"):
        disc(torch.randn(1, 1, 64))


def test_mag_compression_lifts_quiet_high_bins():
    """Without it, bins far below the fundamental barely move the logit."""
    torch.manual_seed(0)
    t = torch.arange(SR, dtype=torch.float32) / SR
    loud_low = torch.sin(2 * torch.pi * 200 * t) * 0.5
    quiet_high = torch.sin(2 * torch.pi * 15000 * t) * 0.002

    base = (loud_low).reshape(1, 1, -1)
    perturbed = (loud_low + quiet_high).reshape(1, 1, -1)

    def sensitivity(compression):
        torch.manual_seed(0)
        disc = MultiScaleSTFTDiscriminator(nch=1, n_ffts=(1024,),
                                           mag_compression=compression).eval()
        with torch.no_grad():
            a = disc(base)[0][0]
            b = disc(perturbed)[0][0]
        return float((a - b).abs().mean())

    assert sensitivity(0.3) > sensitivity(1.0)


# ---------------------------------------------------------- composition


def test_combined_flattens_branches():
    bank = CombinedDiscriminator([
        FastMPD(nch=1, periods=(2, 3)),
        FastMelBank(nch=1, sample_rate=SR, fft_sizes=(1024,), hop_sizes=(256,),
                    win_lengths=(1024,), n_mels=32),
    ])
    assert len(bank.discriminators) == 3
    outputs, maps = bank(_wav())
    assert len(outputs) == 3 and len(maps) == 3


def test_combined_branch_indices_span_all_parts():
    bank = CombinedDiscriminator([
        FastMPD(nch=1, periods=(2, 3)),
        FastMelBank(nch=1, sample_rate=SR, fft_sizes=(1024,), hop_sizes=(256,),
                    win_lengths=(1024,), n_mels=32),
    ])
    # index 2 belongs to the second part; selecting it alone must reach it
    outputs, _ = bank(_wav(), branches=[2])
    assert len(outputs) == 1


def test_combined_subset_matches_the_full_pass():
    torch.manual_seed(0)
    bank = CombinedDiscriminator([FastMPD(nch=1, periods=(2, 3, 5))]).eval()
    wav = _wav()
    with torch.no_grad():
        full, _ = bank(wav)
        subset, _ = bank(wav, branches=[1])
    assert torch.allclose(full[1], subset[0], atol=1e-6)


def test_builder_rejects_an_empty_bank():
    with pytest.raises(ValueError, match="at least one"):
        build_fast_discriminator(use_mpd=False, use_mrd=False, use_mel=False)


def test_builder_composes_requested_families():
    bank = build_fast_discriminator(nch=1, sample_rate=SR, use_mrd=False,
                                    periods=(2, 3), mel_fft_sizes=(1024,),
                                    mel_hop_sizes=(256,), mel_win_lengths=(1024,))
    assert len(bank.discriminators) == 3   # 2 periods + 1 mel


# --------------------------------------------------- losses on any bank


def test_losses_accept_every_bank():
    for ctor in [
        lambda: FastMPD(nch=1, periods=(2, 3)),
        lambda: MultiScaleSTFTDiscriminator(nch=1, n_ffts=(512,)),
        lambda: MultiFrequencyDiscriminator(nch=1, window=[256, 512]),
        lambda: build_fast_discriminator(nch=1, sample_rate=SR, periods=(2,),
                                         mrd_fft_sizes=(512,), mrd_hop_sizes=(128,),
                                         mrd_win_lengths=(512,), mel_fft_sizes=(1024,),
                                         mel_hop_sizes=(256,), mel_win_lengths=(1024,)),
    ]:
        disc = ctor()
        est, target = _wav(), _wav()
        est_o, est_fm = disc(est)
        tgt_o, tgt_fm = disc(target)

        assert torch.isfinite(MultiFrequencyDisLoss()(tgt_o, est_o))

        loss = VocalGenLoss(sr=SR, mel_n_fft=1024, mel_hop=256, mel_bins=32)
        total = loss(est_o, est_fm, tgt_fm, est, target.squeeze(0))
        assert torch.isfinite(total)


# ------------------------------------------------- branch-wise splitting


def test_split_loss_sums_to_the_combined_loss():
    """Branch-wise stepping must optimise the same objective as one pass."""
    torch.manual_seed(0)
    disc = FastMPD(nch=1, periods=(2, 3, 5)).eval()
    est, target = _wav(), _wav()

    loss = VocalGenLoss(sr=SR, mel_n_fft=1024, mel_hop=256, mel_bins=32)

    with torch.no_grad():
        est_o, est_fm = disc(est)
        tgt_o, tgt_fm = disc(target)
        combined = float(loss(est_o, est_fm, tgt_fm, est, target.squeeze(0), adv_scale=1.0))

        total = sum(float(v) for v in loss.reconstruction_terms(est, target.squeeze(0)).values())
        n = len(disc.discriminators)
        for branch in range(n):
            e_o, e_fm = disc(est, branches=[branch])
            t_o, t_fm = disc(target, branches=[branch])
            adv = loss.adversarial_terms(e_o, e_fm, t_fm, adv_scale=1.0, n_branches=n)
            total += sum(float(v) for v in adv.values())

    assert total == pytest.approx(combined, rel=1e-4)


def test_adversarial_terms_respect_adv_scale():
    torch.manual_seed(0)
    disc = FastMPD(nch=1, periods=(2,)).eval()
    est, target = _wav(), _wav()
    with torch.no_grad():
        e_o, e_fm = disc(est)
        _, t_fm = disc(target)

    loss = VocalGenLoss(sr=SR, mel_n_fft=1024, mel_hop=256, mel_bins=32)
    off = loss.adversarial_terms(e_o, e_fm, t_fm, adv_scale=0.0)
    on = loss.adversarial_terms(e_o, e_fm, t_fm, adv_scale=1.0)

    assert float(off["adv"]) == 0.0 and float(off["fm"]) == 0.0
    assert float(on["adv"]) > 0.0


def test_reconstruction_terms_exclude_adversarial():
    loss = VocalGenLoss(sr=SR, mel_n_fft=1024, mel_hop=256, mel_bins=32)
    terms = loss.reconstruction_terms(_wav(), _wav().squeeze(0))
    assert "adv" not in terms and "fm" not in terms
    assert {"freq", "fullness", "bleedless", "clarity", "waveform"} <= set(terms)


def test_branchwise_gradients_match_a_single_pass():
    """Splitting the backward must not change what the generator learns."""
    torch.manual_seed(0)
    disc = FastMPD(nch=1, periods=(2, 3)).eval()
    loss = VocalGenLoss(sr=SR, mel_n_fft=1024, mel_hop=256, mel_bins=32)
    target = _wav()

    def grads(branchwise):
        torch.manual_seed(1)
        est = _wav().clone().requires_grad_(True)
        if branchwise:
            recon = loss.reconstruction_terms(est, target.squeeze(0))
            sum(recon.values()).backward(retain_graph=True)
            n = len(disc.discriminators)
            for i in range(n):
                e_o, e_fm = disc(est, branches=[i])
                with torch.no_grad():
                    _, t_fm = disc(target, branches=[i])
                adv = loss.adversarial_terms(e_o, e_fm, t_fm, 1.0, n_branches=n)
                sum(adv.values()).backward(retain_graph=(i != n - 1))
        else:
            e_o, e_fm = disc(est)
            with torch.no_grad():
                _, t_fm = disc(target)
            loss(e_o, e_fm, t_fm, est, target.squeeze(0), adv_scale=1.0).backward()
        return est.grad.clone()

    split, single = grads(True), grads(False)
    assert torch.allclose(split, single, atol=1e-6), \
        f"max diff {float((split - single).abs().max()):.3e}"
