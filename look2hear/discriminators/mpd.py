"""Multi-period discriminator (HiFi-GAN).

Apollo's stock critic is entirely STFT-based. Every branch first buys into one
time/frequency resolution trade, and periodic structure that does not line up with
the analysis window gets smeared across bins -- which is exactly where buzzy and
metallic artefacts hide.

An MPD sidesteps the trade completely. It folds the raw waveform into a 2D grid of
period p, so samples that are p apart become neighbours, and then convolves along
that axis. Anything periodic at (or near) p becomes a *spatial* pattern the critic
can see directly, with no windowing involved. Using coprime periods means the
branches disagree about what counts as periodic, which is what makes the set cover
more than any single one could.

For a vocal this matters more than for a mix: pitch, jitter and shimmer are all
period-domain phenomena.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm, weight_norm


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, channels=(32, 128, 512, 1024), kernel=5, stride=3,
                 use_spectral_norm=False, max_channels=1024):
        super().__init__()
        self.period = period
        norm = spectral_norm if use_spectral_norm else weight_norm

        layers = []
        in_ch = 1
        for out_ch in channels:
            out_ch = min(out_ch, max_channels)
            layers.append(norm(nn.Conv2d(
                in_ch, out_ch, (kernel, 1), (stride, 1), padding=((kernel - 1) // 2, 0)
            )))
            in_ch = out_ch
        layers.append(norm(nn.Conv2d(in_ch, in_ch, (kernel, 1), 1,
                                     padding=((kernel - 1) // 2, 0))))
        self.convs = nn.ModuleList(layers)
        self.post = norm(nn.Conv2d(in_ch, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        # x: (B, 1, T) -> (B, 1, T // period, period)
        b, c, t = x.shape
        if t % self.period:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)

        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)

        return self.post(x), features


class MultiPeriodDiscriminator(nn.Module):
    """A bank of period discriminators over coprime periods.

    Multi-channel input is folded into the batch, matching how the STFT critic
    already treats channels, so this works unchanged for mono or stereo.
    """

    def __init__(self, nch=1, periods=(2, 3, 5, 7, 11), channels=(32, 128, 512, 1024),
                 kernel=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.nch = nch
        self.periods = list(periods)
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p, channels=channels, kernel=kernel, stride=stride,
                                use_spectral_norm=use_spectral_norm)
            for p in self.periods
        ])
        self.branch_names = [f"mpd_{p}" for p in self.periods]
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, est, sample_rate=44100, branches=None):
        b, nch, t = est.shape
        assert nch == self.nch, f"discriminator built for {self.nch}ch, got {nch}ch"

        # same power normalisation the STFT critic uses, so the two see comparable
        # input scales when they are combined
        est = est / (est.pow(2).sum((1, 2)) + self.eps).sqrt().reshape(b, 1, 1)
        folded = est.reshape(b * nch, 1, t)

        indices = range(len(self.discriminators)) if branches is None else list(branches)

        outputs, feature_maps = [], []
        for i in indices:
            score, features = self.discriminators[i](folded)
            outputs.append(score)
            feature_maps.append(features)

        return outputs, feature_maps
