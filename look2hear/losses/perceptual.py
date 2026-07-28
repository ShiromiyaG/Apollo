"""Perceptual losses for the vocal/speech restoration mode.

Apollo's stock generator loss is magnitude-only `freq_MAE` plus adversarial and
feature-matching terms. That combination has two blind spots for voice:

1. An L1 on magnitudes is dominated by whichever bins are loudest. A vocal's
   character lives in low-energy detail -- consonants, breath, the upper
   harmonics that make a voice sound present -- and those barely move the number.
2. Nothing constrains phase, so the reconstruction is free to be right on average
   and wrong sample by sample.

The losses here address the first point directly and asymmetrically, in the same
mel-dB domain the `fullness`/`bleedless` metrics from Music-Source-Separation-
Training use, so what you optimise is what you measure.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mel_basis(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Slaney-style mel filterbank, matching librosa.filters.mel defaults."""
    try:
        import librosa

        return torch.from_numpy(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                                    fmin=fmin, fmax=fmax)).float()
    except ImportError:
        pass

    # self-contained fallback so the loss works without librosa installed
    fmax = fmax or sr / 2.0

    def hz_to_mel(f):
        f_min, f_sp = 0.0, 200.0 / 3
        mel = (f - f_min) / f_sp
        min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        if isinstance(f, torch.Tensor):
            return torch.where(f >= min_log_hz,
                               min_log_mel + torch.log(f / min_log_hz) / logstep, mel)
        return min_log_mel + math.log(f / min_log_hz) / logstep if f >= min_log_hz else mel

    def mel_to_hz(m):
        f_min, f_sp = 0.0, 200.0 / 3
        freqs = f_min + f_sp * m
        min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        return torch.where(m >= min_log_mel, min_log_hz * torch.exp(logstep * (m - min_log_mel)), freqs)

    mel_points = torch.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_freqs = torch.linspace(0, sr / 2.0, n_fft // 2 + 1)

    weights = torch.zeros(n_mels, n_fft // 2 + 1)
    diff = hz_points[1:] - hz_points[:-1]
    ramps = hz_points.unsqueeze(1) - fft_freqs.unsqueeze(0)
    for i in range(n_mels):
        lower = -ramps[i] / diff[i].clamp(min=1e-9)
        upper = ramps[i + 2] / diff[i + 1].clamp(min=1e-9)
        weights[i] = torch.maximum(torch.zeros_like(lower), torch.minimum(lower, upper))

    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels]).clamp(min=1e-9)
    return weights * enorm.unsqueeze(1)


class MelDbTransform(nn.Module):
    """STFT -> mel -> dB, with the filterbank and window held as buffers."""

    def __init__(self, sr=44100, n_fft=4096, hop_length=1024, n_mels=512,
                 fmin=0.0, fmax=None, top_db=80.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.top_db = top_db
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("mel_basis", _mel_basis(sr, n_fft, n_mels, fmin, fmax), persistent=False)

    def forward(self, wav):
        """``wav`` of any shape ending in samples -> (batch*, n_mels, frames) in dB."""
        flat = wav.reshape(-1, wav.shape[-1])
        # follow the input rather than assume the module was moved with it, so the
        # loss also works standalone as a metric
        window = self.window.to(device=flat.device, dtype=flat.dtype)
        spec = torch.stft(flat, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=window, return_complex=True,
                          pad_mode="constant").abs()
        mel = torch.matmul(self.mel_basis.to(device=spec.device, dtype=spec.dtype), spec)

        db = 20.0 * torch.log10(mel.clamp(min=1e-10))
        if self.top_db is not None:
            db = torch.maximum(db, db.amax() - self.top_db)
        return db


class BleedFullPenaltyLoss(nn.Module):
    """Asymmetric mel-dB penalty, the differentiable twin of the MSST metrics.

    ``mode="fullness"`` penalises only bins where the target is *louder* than the
    estimate, i.e. content the model failed to reproduce. Minimising it pushes the
    model to fill in the harmonics and detail a codec stripped out, which is what
    reads as a voice sounding "full" rather than thin.

    ``mode="bleedless"`` is the mirror image: it penalises energy the model *added*
    that the target does not have, which is the definition of an artefact.

    Using both, weighted separately, lets you ask for a fuller voice without also
    inviting the model to invent hiss -- the two pull in opposite directions and
    the balance between them is the knob that matters.

    Args:
        band_weights: optional (fmin_hz, fmax_hz, weight) triples applied on top of
            the per-bin penalty, to emphasise the vocal range.
    """

    def __init__(self, mode="fullness", sr=44100, n_fft=4096, hop_length=1024,
                 n_mels=512, top_db=80.0, band_weights=None):
        super().__init__()
        if mode not in ("fullness", "bleedless"):
            raise ValueError(f"mode must be 'fullness' or 'bleedless', got {mode!r}")
        self.mode = mode
        self.mel = MelDbTransform(sr=sr, n_fft=n_fft, hop_length=hop_length,
                                  n_mels=n_mels, top_db=top_db)

        weights = torch.ones(n_mels)
        if band_weights:
            mel_hz = _mel_bin_frequencies(sr, n_mels)
            for fmin, fmax, weight in band_weights:
                weights[(mel_hz >= fmin) & (mel_hz < fmax)] = float(weight)
        self.register_buffer("band_weights", weights.reshape(1, -1, 1), persistent=False)

    def forward(self, estimate, reference):
        est_db = self.mel(estimate)
        ref_db = self.mel(reference)

        if self.mode == "fullness":
            penalty = F.relu(ref_db - est_db)   # target louder -> content is missing
        else:
            penalty = F.relu(est_db - ref_db)   # estimate louder -> artefact added

        penalty = penalty * self.band_weights.to(penalty.dtype)

        # Average over violating bins only. A plain mean would shrink as the model
        # improves simply because more bins hit exactly zero, flattening the
        # gradient right when it still matters.
        active = torch.count_nonzero(penalty)
        return penalty.sum() / active.clamp(min=1)


def _mel_bin_frequencies(sr, n_mels, fmin=0.0, fmax=None):
    """Centre frequency of each mel bin, for band weighting."""
    fmax = fmax or sr / 2.0
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0

    def hz_to_mel(f):
        return min_log_mel + math.log(f / min_log_hz) / logstep if f >= min_log_hz else f / f_sp

    mels = torch.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)[1:-1]
    return torch.where(mels >= min_log_mel,
                       min_log_hz * torch.exp(logstep * (mels - min_log_mel)),
                       f_sp * mels)


class MelClarityLoss(nn.Module):
    """L1 on log-mel magnitudes, optionally weighted towards the vocal range.

    Where `freq_MAE` normalises by the mean magnitude and so tracks whatever is
    loudest, the log domain gives quiet detail the same say as loud detail. That
    is where consonant definition and breath live, and it is what makes a restored
    voice read as clear rather than merely loud.
    """

    def __init__(self, sr=44100, n_fft=2048, hop_length=512, n_mels=160,
                 fmin=40.0, fmax=16000.0, band_weights=((300.0, 8000.0, 2.0),)):
        super().__init__()
        self.mel = MelDbTransform(sr=sr, n_fft=n_fft, hop_length=hop_length,
                                  n_mels=n_mels, fmin=fmin, fmax=fmax, top_db=None)

        weights = torch.ones(n_mels)
        if band_weights:
            mel_hz = _mel_bin_frequencies(sr, n_mels, fmin, fmax)
            for lo, hi, weight in band_weights:
                weights[(mel_hz >= lo) & (mel_hz < hi)] = float(weight)
        self.register_buffer("band_weights", (weights / weights.mean()).reshape(1, -1, 1),
                             persistent=False)

    def forward(self, estimate, reference):
        est_db = self.mel(estimate)
        ref_db = self.mel(reference)
        return ((est_db - ref_db).abs() * self.band_weights.to(est_db.dtype)).mean()


class WaveformL1Loss(nn.Module):
    """Plain L1 in the time domain.

    The stock generator loss compares magnitude spectra only, which leaves phase
    entirely to the discriminator. An explicit waveform term ties the output to the
    target sample by sample; a small weight is enough to stop the diffuse,
    noise-like residual that unconstrained phase produces, without the
    over-smoothing a large weight would cause.
    """

    def forward(self, estimate, reference):
        length = min(estimate.shape[-1], reference.shape[-1])
        return (estimate[..., :length] - reference[..., :length]).abs().mean()
