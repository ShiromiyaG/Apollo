"""Tests for the vocal-mode losses and data pipeline."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look2hear.losses.perceptual import (
    BleedFullPenaltyLoss,
    MelClarityLoss,
    MelDbTransform,
    WaveformL1Loss,
    _mel_bin_frequencies,
)

SR = 44100


def _voice_like(secs=1.0, nch=1, sr=SR, seed=0):
    """Harmonic stack in the vocal range with a little noise."""
    torch.manual_seed(seed)
    t = torch.arange(int(sr * secs), dtype=torch.float32) / sr
    wave = sum(torch.sin(2 * math.pi * f * t) / (i + 1)
               for i, f in enumerate([180, 360, 720, 1440, 2880, 5760]))
    wave = wave + torch.randn_like(wave) * 0.02
    wave = wave / wave.abs().max() * 0.7
    return wave.unsqueeze(0).repeat(nch, 1)


# ------------------------------------------------------------ mel transform


def test_mel_transform_shapes():
    mel = MelDbTransform(sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    out = mel(_voice_like(secs=0.5, nch=2))
    assert out.shape[0] == 2 and out.shape[1] == 64


def test_mel_bin_frequencies_are_monotonic_and_in_range():
    freqs = _mel_bin_frequencies(SR, 128)
    assert freqs.shape == (128,)
    assert torch.all(freqs[1:] > freqs[:-1])
    assert 0 < float(freqs[0]) and float(freqs[-1]) < SR / 2


# --------------------------------------------------------- fullness penalty


def test_fullness_is_zero_for_identical_signals():
    loss = BleedFullPenaltyLoss(mode="fullness", sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    wav = _voice_like()
    assert float(loss(wav, wav.clone())) == pytest.approx(0.0, abs=1e-5)


def test_fullness_penalises_missing_content_only():
    """Quieter-than-target must cost; louder-than-target must not."""
    loss = BleedFullPenaltyLoss(mode="fullness", sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    ref = _voice_like()

    too_quiet = ref * 0.25
    too_loud = ref * 4.0
    assert float(loss(too_quiet, ref)) > float(loss(too_loud, ref))


def test_bleedless_penalises_added_content_only():
    """The mirror image: bleedless must react to added energy, not missing energy."""
    loss = BleedFullPenaltyLoss(mode="bleedless", sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    ref = _voice_like()
    assert float(loss(ref * 4.0, ref)) > float(loss(ref * 0.25, ref))


def test_fullness_grows_as_high_band_is_stripped():
    """Removing the top octave -- what a low-bitrate codec does -- must cost more."""
    from look2hear.datas.codec import lowpass_resample

    loss = BleedFullPenaltyLoss(mode="fullness", sr=SR, n_fft=2048, hop_length=512, n_mels=128)
    ref = _voice_like()

    mild = lowpass_resample(ref, SR, 12000)
    harsh = lowpass_resample(ref, SR, 3000)
    assert float(loss(harsh, ref)) > float(loss(mild, ref))


def test_fullness_and_bleedless_disagree_on_direction():
    """The two must pull opposite ways -- that is what makes the pair a balance."""
    full = BleedFullPenaltyLoss(mode="fullness", sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    bleed = BleedFullPenaltyLoss(mode="bleedless", sr=SR, n_fft=1024, hop_length=256, n_mels=64)

    ref = _voice_like()
    noisy = ref + torch.randn_like(ref) * 0.05   # added hiss
    assert float(bleed(noisy, ref)) > float(bleed(ref, ref))
    assert float(full(noisy, ref)) < float(bleed(noisy, ref))


def test_fullness_is_differentiable():
    loss = BleedFullPenaltyLoss(mode="fullness", sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    ref = _voice_like(secs=0.3)
    est = (ref * 0.5).clone().requires_grad_(True)
    loss(est, ref).backward()
    assert est.grad is not None and torch.isfinite(est.grad).all()
    assert float(est.grad.abs().sum()) > 0


def test_band_weights_change_the_score():
    plain = BleedFullPenaltyLoss(mode="fullness", sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    weighted = BleedFullPenaltyLoss(mode="fullness", sr=SR, n_fft=1024, hop_length=256,
                                    n_mels=64, band_weights=[(300.0, 8000.0, 4.0)])
    ref = _voice_like()
    est = ref * 0.3
    assert float(weighted(est, ref)) > float(plain(est, ref))


def test_rejects_bad_mode():
    with pytest.raises(ValueError):
        BleedFullPenaltyLoss(mode="loudness")


# ------------------------------------------------------------------ clarity


def test_clarity_zero_for_identical():
    loss = MelClarityLoss(sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    wav = _voice_like()
    assert float(loss(wav, wav.clone())) == pytest.approx(0.0, abs=1e-4)


def test_clarity_reacts_to_lost_detail():
    from look2hear.datas.codec import lowpass_resample

    loss = MelClarityLoss(sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    ref = _voice_like()
    assert float(loss(lowpass_resample(ref, SR, 3000), ref)) > float(loss(ref, ref))


def test_waveform_l1_handles_length_mismatch():
    loss = WaveformL1Loss()
    a, b = torch.zeros(1, 1000), torch.ones(1, 900)
    assert float(loss(a, b)) == pytest.approx(1.0)


# ------------------------------------------------------- combined objective


def test_vocal_gen_loss_reduces_to_original_when_extras_disabled():
    from look2hear.losses.gan_losses import MultiFrequencyGenLoss, VocalGenLoss

    torch.manual_seed(0)
    est_outputs = [torch.randn(2, 1, 4, 4) for _ in range(2)]
    est_maps = [[torch.randn(2, 3, 4, 4) for _ in range(2)] for _ in range(2)]
    tgt_maps = [[torch.randn(2, 3, 4, 4) for _ in range(2)] for _ in range(2)]
    output = torch.randn(2, 1, SR // 4) * 0.2
    target = torch.randn(2, SR // 4) * 0.2

    original = MultiFrequencyGenLoss()
    stripped = VocalGenLoss(fullness_weight=0, bleedless_weight=0,
                            clarity_weight=0, waveform_weight=0)

    a = original(est_outputs, est_maps, tgt_maps, output, target)
    b = stripped(est_outputs, est_maps, tgt_maps, output, target)
    assert float(a) == pytest.approx(float(b), rel=1e-5)


def test_vocal_gen_loss_reports_every_term():
    from look2hear.losses.gan_losses import VocalGenLoss

    torch.manual_seed(0)
    est_outputs = [torch.randn(1, 1, 4, 4)]
    est_maps = [[torch.randn(1, 3, 4, 4)]]
    tgt_maps = [[torch.randn(1, 3, 4, 4)]]
    output = _voice_like(secs=0.5).unsqueeze(0)
    target = _voice_like(secs=0.5)

    loss = VocalGenLoss(sr=SR, mel_n_fft=1024, mel_hop=256, mel_bins=64)
    total = loss(est_outputs, est_maps, tgt_maps, output, target)

    assert torch.isfinite(total)
    assert set(loss.last_terms) == {"freq", "adv", "fm", "fullness", "bleedless",
                                    "clarity", "waveform"}
    assert float(total) == pytest.approx(sum(loss.last_terms.values()), rel=1e-4)


def test_vocal_gen_loss_backward():
    from look2hear.losses.gan_losses import VocalGenLoss

    torch.manual_seed(0)
    est_outputs = [torch.randn(1, 1, 4, 4, requires_grad=True)]
    est_maps = [[torch.randn(1, 3, 4, 4, requires_grad=True)]]
    tgt_maps = [[torch.randn(1, 3, 4, 4)]]
    output = _voice_like(secs=0.4).unsqueeze(0).clone().requires_grad_(True)
    target = _voice_like(secs=0.4)

    loss = VocalGenLoss(sr=SR, mel_n_fft=1024, mel_hop=256, mel_bins=64)
    loss(est_outputs, est_maps, tgt_maps, output, target).backward()
    assert output.grad is not None and torch.isfinite(output.grad).all()


# ------------------------------------------------------------------- data


def test_vocal_degradation_changes_signal_and_keeps_shape():
    from look2hear.datas.vocal_datamodule import VocalDegradation

    degrade = VocalDegradation(sr=SR, codec_prob=1.0, bandlimit_prob=0.0,
                               clip_prob=0.0, quantize_prob=0.0)
    clean = _voice_like()
    out = degrade(clean)
    assert out.shape == clean.shape
    assert torch.isfinite(out).all()
    assert (out - clean).abs().max() > 1e-4


def test_vocal_degradation_is_reproducible_with_a_seeded_rng():
    import random
    from look2hear.datas.vocal_datamodule import VocalDegradation

    degrade = VocalDegradation(sr=SR)
    clean = _voice_like(secs=0.5)
    a = degrade(clean, rng=random.Random(7))
    b = degrade(clean, rng=random.Random(7))
    assert torch.equal(a, b)


def test_vocal_dataset_yields_normalised_pairs(tmp_path):
    import soundfile as sf
    from look2hear.datas.vocal_datamodule import VocalRestorationDataset

    clean = _voice_like(secs=4.0)
    sf.write(tmp_path / "a.wav", clean.T.numpy(), SR)

    dataset = VocalRestorationDataset(str(tmp_path), sr=SR, segments=2.0,
                                      channels=1, num_samples=3, seed=0)
    assert len(dataset) == 3

    for i in range(3):
        target, degraded = dataset[i]
        assert target.shape == (1, 2 * SR)
        assert degraded.shape == (1, 2 * SR)
        assert torch.isfinite(target).all() and torch.isfinite(degraded).all()
        # the pair shares one scale factor, so the louder of the two peaks at 1.0
        assert max(float(target.abs().max()), float(degraded.abs().max())) == pytest.approx(1.0, abs=1e-5)


def test_vocal_dataset_errors_on_empty_folder(tmp_path):
    from look2hear.datas.vocal_datamodule import VocalRestorationDataset

    with pytest.raises(FileNotFoundError):
        VocalRestorationDataset(str(tmp_path))


def test_vocal_dataset_errors_on_wrong_sample_rate(tmp_path):
    """A corpus at the wrong rate would otherwise silently train on zeros."""
    import soundfile as sf
    from look2hear.datas.vocal_datamodule import VocalRestorationDataset

    sf.write(tmp_path / "a.wav", _voice_like(secs=4.0, sr=22050).T.numpy(), 22050)

    with pytest.raises(ValueError, match="not 44100 Hz"):
        VocalRestorationDataset(str(tmp_path), sr=SR, segments=2.0)


def test_vocal_dataset_errors_when_files_are_too_short(tmp_path):
    import soundfile as sf
    from look2hear.datas.vocal_datamodule import VocalRestorationDataset

    sf.write(tmp_path / "a.wav", _voice_like(secs=0.5).T.numpy(), SR)

    with pytest.raises(ValueError, match="shorter than"):
        VocalRestorationDataset(str(tmp_path), sr=SR, segments=3.0)


def test_paired_eval_uses_parallel_folders(tmp_path):
    import soundfile as sf
    from look2hear.datas.vocal_datamodule import PairedVocalEval

    clean_dir = tmp_path / "clean"
    degraded_dir = tmp_path / "degraded"
    clean_dir.mkdir(); degraded_dir.mkdir()

    clean = _voice_like(secs=2.0)
    sf.write(clean_dir / "a.wav", clean.T.numpy(), SR)
    sf.write(degraded_dir / "a.wav", (clean * 0.5).T.numpy(), SR)

    dataset = PairedVocalEval(str(clean_dir), str(degraded_dir), sr=SR, channels=1)
    target, degraded = dataset[0]
    assert target.shape == degraded.shape
    # the degraded copy really is the quieter file, not a regenerated one
    assert float(degraded.abs().max()) < float(target.abs().max())
