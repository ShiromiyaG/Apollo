"""Compose several critics into one.

The losses only ever iterate ``outputs`` and ``feature_maps`` as flat lists, so a
critic made of other critics is just concatenation. That is what lets an MPD and
an MS-STFT-D be used together -- the pairing every modern neural vocoder converged
on, because the two see genuinely different things: one works in the period
domain with no windowing, the other in the time-frequency domain.

Branch rotation from the training loop indexes into the flattened list, so it
spreads across all sub-critics rather than starving one of them.
"""

import torch
import torch.nn as nn


class CombinedDiscriminator(nn.Module):
    """Run several discriminators and concatenate their outputs.

    Args:
        discriminators: already-constructed critics, each following the
            ``forward(est, sample_rate, branches) -> (outputs, feature_maps)``
            protocol.
    """

    def __init__(self, discriminators, nch=1):
        super().__init__()
        self.nch = nch
        self.parts = nn.ModuleList(list(discriminators))

        # flat index -> (part index, branch index within that part), so the
        # training loop's branch rotation works across the whole bank
        self._routing = []
        self.branch_names = []
        for part_idx, part in enumerate(self.parts):
            names = (part.names() if hasattr(part, "names")
                     else getattr(part, "branch_names", None))
            for branch_idx in range(len(part.discriminators)):
                self._routing.append((part_idx, branch_idx))
                self.branch_names.append(
                    names[branch_idx] if names else f"part{part_idx}_{branch_idx}")

    @property
    def discriminators(self):
        """Flat view of every branch, for code that counts or rotates them."""
        return self._routing

    def names(self):
        """Flat branch names, in the same order as ``discriminators``.

        These are what the training loop's ``disc_branch_every`` schedule matches
        its glob patterns against.
        """
        return list(self.branch_names)

    def forward(self, est, sample_rate=44100, branches=None):
        if branches is None:
            wanted = list(range(len(self._routing)))
        else:
            wanted = list(branches)

        per_part = {}
        for flat_index in wanted:
            part_idx, branch_idx = self._routing[flat_index]
            per_part.setdefault(part_idx, []).append(branch_idx)

        outputs, feature_maps = [], []
        for part_idx in sorted(per_part):
            part_out, part_maps = self.parts[part_idx](
                est, sample_rate=sample_rate, branches=sorted(per_part[part_idx])
            )
            outputs.extend(part_out)
            feature_maps.extend(part_maps)

        return outputs, feature_maps


def build_hybrid_discriminator(nch=1, mpd_periods=(2, 3, 5, 7, 11),
                               stft_n_ffts=(2048, 1024, 512, 256, 128),
                               mpd_channels=(32, 128, 512, 1024),
                               stft_channels=32, stft_max_channels=256):
    """Textbook MPD + MS-STFT-D pairing (HiFi-GAN + EnCodec).

    Kept for reference and comparison. `build_fast_discriminator` covers the same
    ground for a fraction of the parameters -- prefer it unless you specifically
    want the standard implementations.
    """
    from .mpd import MultiPeriodDiscriminator
    from .msstft import MultiScaleSTFTDiscriminator

    return CombinedDiscriminator([
        MultiPeriodDiscriminator(nch=nch, periods=mpd_periods, channels=mpd_channels),
        MultiScaleSTFTDiscriminator(nch=nch, n_ffts=stft_n_ffts,
                                    channels=stft_channels,
                                    max_channels=stft_max_channels),
    ], nch=nch)


def build_fast_discriminator(nch=1, sample_rate=44100, use_san=True,
                             periods=(2, 3, 5, 7, 11),
                             mrd_fft_sizes=(2048, 1024, 512),
                             mrd_hop_sizes=(256, 128, 64),
                             mrd_win_lengths=(2048, 1024, 512),
                             mel_fft_sizes=(2048,), mel_hop_sizes=(512,),
                             mel_win_lengths=(2048,), n_mels=128,
                             init_channel=8, use_mpd=True, use_mrd=True, use_mel=True,
                             phase_aware=False, msstft_n_ffts=(2048, 1024, 512),
                             msstft_channels=32, msstft_max_channels=128,
                             msstft_freq_stride=2, msstft_mag_compression=0.3,
                             use_med=False, med_scales=None, med_channels=32,
                             med_max_channels=128, med_layers=4,
                             use_transient=False, transient_channels=16,
                             use_hf_modulation=False, hf_low_hz=2000.0,
                             hf_high_hz=10000.0, hf_bands=4, hf_channels=16,
                             use_periodicity=False, periodicity_decimation=8,
                             periodicity_f0_min=70.0, periodicity_f0_max=400.0,
                             periodicity_channels=16,
                             use_spectral_norm=False):
    """FastMPD + FastMRD + FastMel -- the recommended bank.

    Three views that fail in different ways, which is the point of a bank:

    * **period** (no windowing at all) catches buzz and pitch-domain artefacts;
    * **multi-resolution magnitude** catches spectral balance across time scales;
    * **log-mel** keeps a constant-Q-style harmonic emphasis for a fraction of a
      real CQT branch's cost.

    Set ``phase_aware=True`` to add complex-STFT branches. Worth it for Apollo
    specifically: its reconstruction loss compares magnitudes only, so without a
    branch that sees real and imaginary parts, *nothing* in the objective
    constrains phase. The three fast banks above are all magnitude- or
    time-domain, so they cannot supply that on their own.

    The four ``use_med`` / ``use_transient`` / ``use_hf_modulation`` /
    ``use_periodicity`` families add a fourth *kind* of view -- dynamics rather
    than content. See `dynamics.py` for why a restoration model in particular
    wants them; they are off by default so existing configs are unchanged.
    """
    from .fast import FastMelBank, FastMPD, FastMRD

    parts = []
    if use_mpd:
        parts.append(FastMPD(nch=nch, periods=periods, init_channel=init_channel,
                             use_san=use_san))
    if use_mrd:
        parts.append(FastMRD(nch=nch, fft_sizes=mrd_fft_sizes, hop_sizes=mrd_hop_sizes,
                             win_lengths=mrd_win_lengths, init_channel=init_channel,
                             use_san=use_san))
    if use_mel:
        parts.append(FastMelBank(nch=nch, sample_rate=sample_rate,
                                 fft_sizes=mel_fft_sizes, hop_sizes=mel_hop_sizes,
                                 win_lengths=mel_win_lengths, n_mels=n_mels,
                                 init_channel=init_channel, use_san=use_san))
    if phase_aware:
        from .msstft import MultiScaleSTFTDiscriminator
        parts.append(MultiScaleSTFTDiscriminator(
            nch=nch, n_ffts=msstft_n_ffts, channels=msstft_channels,
            max_channels=msstft_max_channels, freq_stride=msstft_freq_stride,
            mag_compression=msstft_mag_compression, use_san=use_san))

    if use_med or use_transient or use_hf_modulation or use_periodicity:
        from .dynamics import (HFModulationBank, MultiEnvelopeDiscriminator,
                               PeriodicityBank, TransientBank)
        common = dict(nch=nch, use_san=use_san, use_spectral_norm=use_spectral_norm)
        if use_med:
            parts.append(MultiEnvelopeDiscriminator(
                sample_rate=sample_rate, scales=med_scales, channels=med_channels,
                max_channels=med_max_channels, n_layers=med_layers, **common))
        if use_transient:
            parts.append(TransientBank(channels=transient_channels, **common))
        if use_hf_modulation:
            parts.append(HFModulationBank(
                sample_rate=sample_rate, low_hz=hf_low_hz, high_hz=hf_high_hz,
                n_bands=hf_bands, channels=hf_channels, **common))
        if use_periodicity:
            parts.append(PeriodicityBank(
                sample_rate=sample_rate, decimation=periodicity_decimation,
                f0_min=periodicity_f0_min, f0_max=periodicity_f0_max,
                channels=periodicity_channels, **common))

    if not parts:
        raise ValueError("at least one discriminator family must be enabled")

    return CombinedDiscriminator(parts, nch=nch)
