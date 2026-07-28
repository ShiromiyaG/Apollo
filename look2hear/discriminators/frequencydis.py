import torch
import torch.nn as nn
import numpy as np


class MultiFrequencyDiscriminator(nn.Module):
    """A bank of 2D convolutional discriminators, one per STFT resolution.

    Each branch sees the real and imaginary parts of the STFT stacked as channels,
    at its own window length. Working on R/I rather than magnitude is what lets the
    discriminator constrain phase at all -- nothing in the reconstruction loss does.

    Three things here are worth knowing before tuning (see docs/gan.md):

    * ``hidden_channels`` was hard-coded to 8 upstream, giving a branch of roughly
      8 -> 256 channels. That is a very small critic for a 44.1 kHz signal; it is
      now a parameter.
    * The default ``window`` list starts at 32 samples. A 32-point FFT is 17 bins
      over 0.7 ms -- that branch is effectively a time-domain critic, and it is the
      one most likely to reward high-frequency noise. Dropping the two shortest
      windows is usually the first thing to try if the output is grainy.
    * Branches are independent, so they do not all have to run on every step.
      ``branches=`` lets the training loop rotate through them, which is close to a
      linear saving in discriminator cost.
    """

    def __init__(self, nch, window, hidden_channels=8, min_freq_bin=0):
        super(MultiFrequencyDiscriminator, self).__init__()

        self.nch = nch
        self.window = window
        self.hidden_channels = hidden_channels
        self.min_freq_bin = min_freq_bin
        self.eps = torch.finfo(torch.float32).eps
        self.discriminators = nn.ModuleList(
            [FrequencyDiscriminator(2 * nch, self.hidden_channels) for _ in range(len(self.window))]
        )

    def forward(self, est, sample_rate=44100, branches=None):
        """
        Args:
            branches: indices of the branches to evaluate. ``None`` runs all of
                them. Skipped branches contribute nothing to the returned lists,
                so the loss functions see a shorter -- but still consistent -- set.
        """
        B, nch, _ = est.shape
        assert nch == self.nch, f"discriminator built for {self.nch}ch, got {nch}ch"

        # normalize power
        est = est / (est.pow(2).sum((1, 2)) + self.eps).sqrt().reshape(B, 1, 1)
        est = est.view(-1, est.shape[-1])

        indices = range(len(self.discriminators)) if branches is None else list(branches)

        est_outputs = []
        est_feature_maps = []

        for i in indices:
            est_spec = torch.stft(est.float(), self.window[i], self.window[i] // 2,
                                  window=torch.hann_window(self.window[i]).to(est.device).float(),
                                  return_complex=True)
            est_RI = torch.stack([est_spec.real, est_spec.imag], dim=1)
            est_RI = est_RI.view(B, nch * 2, est_RI.shape[-2], est_RI.shape[-1]).type(est.type())

            valid_enc = int(est_RI.shape[2] * sample_rate / 44100)
            # optionally ignore the lowest bins: a codec leaves them intact, so
            # asking the critic to judge them spends capacity on a solved problem
            lo = min(self.min_freq_bin, max(0, valid_enc - 1))
            est_out, est_feat_map = self.discriminators[i](est_RI[:, :, lo:valid_enc].contiguous())
            est_outputs.append(est_out)
            est_feature_maps.append(est_feat_map)

        return est_outputs, est_feature_maps


class FrequencyDiscriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels=512):
        super(FrequencyDiscriminator, self).__init__()

        self.eps = torch.finfo(torch.float32).eps
        self.discriminator = nn.ModuleList()
        self.discriminator += [
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels*4, hidden_channels*8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels*8, hidden_channels*16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels*16, hidden_channels*32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Conv2d(hidden_channels*32, 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        ]

    def forward(self, x):
        hiddens = []
        for layer in self.discriminator:
            x = layer(x)
            hiddens.append(x)
        return x, hiddens[:-1]


def gradient_penalty(discriminator, x, sample_rate=44100, branches=None,
                     patch_reduction="mean"):
    """Zero-centred gradient penalty: the squared gradient of D w.r.t. its input.

    Called with real samples this is R1; called with generated ones it is R2. The
    two are the same computation, and the R3GAN recipe uses both.

    An unregularised critic is free to become arbitrarily sharp around the data.
    The generator then chases a gradient field that changes faster than it can
    follow, and what comes out the other side is high-frequency noise -- the exact
    failure this model exhibits. The penalty bounds that sharpness.

    ``patch_reduction`` controls how each branch's output map is collapsed to the
    per-sample score that gets differentiated:

    * ``mean`` (default) averages over the patch axis, then averages over
      branches. This is the score the adversarial objective itself is built from,
      and it makes the penalty a property of the critic's *decision*.

    Note what averaging over branches does **not** do. It gives
    ``||d(mean_b score_b)/dx||^2``, the norm of the summed gradient field, which is
    not ``mean_b ||d(score_b)/dx||^2`` -- the difference is every cross-branch
    inner product, and since branch gradients are close to uncorrelated those
    cancel and the first form comes out about ``n_branches`` times smaller
    (measured: 7.16e-4 against 5.72e-3 on an 8-branch bank). Pass one branch at a
    time and average outside if you want the per-branch form; the training module's
    ``r1_branchwise`` does exactly that.
    * ``sum`` reproduces the original behaviour. Summing ``P`` patches multiplies
      the differentiated score's gradient by ``P``, so the penalty comes out
      exactly ``P**2`` times larger. ``P`` depends on the segment length and on
      each branch's stride configuration, neither of which has anything to do with
      how sharp the critic should be allowed to get. Measured on the fast bank at
      3 s: ~1e-5 under ``mean`` against ~5 under ``sum``, a factor of about 1e5 --
      so ``r1_gamma: 1.0`` was applying a penalty five orders of magnitude above
      the adversarial loss it was meant to temper, and the number could not be
      reasoned about at all. Kept only for reproducing old runs.

    Note: this needs a double backward through the branch convolutions. With
    ``torch.backends.cudnn.benchmark = True`` the autotuner can stall indefinitely
    on the first such step, so the training module turns benchmark mode off while
    the penalty is enabled.
    """
    x = x.detach().requires_grad_(True)
    outputs, _ = discriminator(x, sample_rate=sample_rate, branches=branches)

    # `_collapse` sums over the batch only because samples are independent, so one
    # backward yields every sample its own gradient.
    total = _collapse(outputs, patch_reduction)
    grad = torch.autograd.grad(outputs=total, inputs=x, create_graph=True)[0]

    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


def _collapse(outputs, patch_reduction):
    """Per-sample score to differentiate. See :func:`gradient_penalty`."""
    if patch_reduction == "sum":
        return sum(output.sum() for output in outputs)
    per_branch = [output.float().reshape(output.shape[0], -1).mean(1)
                  for output in outputs]
    return torch.stack(per_branch).mean(0).sum()


def joint_gradient_penalty(discriminator, real, fake, sample_rate=44100,
                           branches=None, patch_reduction="mean"):
    """R1 and R2 from a single critic forward and a single double backward.

    Running them separately pays for the whole graph twice, and the double
    backward is the expensive part -- measured on the 18-branch bank, a step that
    fires the penalty costs about 7.5 s against 0.7 s for an ordinary one, so the
    penalty *is* the step whenever it lands.

    It does not have to be paid twice. Samples are independent, so one forward
    over ``cat([real, fake])`` produces both halves' per-sample gradients at once:
    ``d(score_i)/dx_j`` is zero for ``i != j``, which is exactly why the summed
    score can be differentiated in one pass without the samples contaminating each
    other. The returned penalties are the same numbers the separate calls give.

    The trade is memory: that one forward carries twice the batch, so the
    regularisation event's activation peak is larger. Branch-wise stepping keeps
    that bounded to one branch at a time.

    Returns:
        ``(r1, r2)`` -- the real-side and fake-side penalties.
    """
    split = real.shape[0]
    x = torch.cat([real.detach(), fake.detach()], dim=0).requires_grad_(True)

    outputs, _ = discriminator(x, sample_rate=sample_rate, branches=branches)
    total = _collapse(outputs, patch_reduction)
    grad = torch.autograd.grad(outputs=total, inputs=x, create_graph=True)[0]

    # The gradient keeps the *input's* batch dimension even when the bank folds
    # channels into the batch for its own forward, so the split index is the
    # caller's batch size.
    squared = grad.pow(2).reshape(grad.shape[0], -1).sum(1)
    return squared[:split].mean(), squared[split:].mean()


def r1_penalty(discriminator, real, sample_rate=44100, branches=None,
               patch_reduction="mean"):
    """R1: :func:`gradient_penalty` on real samples. See there for the details."""
    return gradient_penalty(discriminator, real, sample_rate=sample_rate,
                            branches=branches, patch_reduction=patch_reduction)
