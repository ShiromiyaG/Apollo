###
# Author: Kai Li
# Date: 2021-06-09 16:43:09
# LastEditors: Please set LastEditors
# LastEditTime: 2024-01-24 00:00:52
###

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .perceptual import BleedFullPenaltyLoss, MelClarityLoss, WaveformL1Loss


# --------------------------------------------------------------- adversarial
#
# Three objectives, selected by name. Ported from
# `H:/Chouwa/SingingVocoders/modules/loss/chouwaloss.py`.
#
# * ``lsgan`` -- (D-1)^2 / D^2. The Apollo default and what every existing config
#   uses; kept as the default so nothing changes without asking.
# * ``hinge`` / ``soft_hinge`` -- the pairing used with SAN heads. ``soft_hinge``
#   is the non-saturating logistic generator term ``softplus(-D)``: unlike LSGAN
#   its gradient is bounded, so a critic that pulls ahead cannot hand the
#   generator an arbitrarily large step.
# * ``relativistic`` (RpGAN, R3GAN / NeurIPS 2024) -- judges the per-sample
#   *difference* D(real_i) - D(fake_i) rather than each against an absolute 0/1
#   boundary. This is only meaningful when real_i and fake_i are the same
#   utterance, which holds here: the generator's input is a degraded copy of the
#   very target it is scored against. Paired with R1+R2 it is the stability half
#   of the R3GAN recipe.


def d_branch_loss(dr, dg, loss_type="lsgan"):
    """One branch's discriminator loss. Real and fake are separate tensors so this
    works unchanged whether the caller ran them in one batched forward or two."""
    dr_f, dg_f = dr.float(), dg.float()
    if loss_type == "relativistic":
        return torch.mean(F.softplus(dg_f - dr_f))
    if loss_type == "lsgan":
        return torch.mean((dr_f - 1.0) ** 2) + torch.mean(dg_f ** 2)
    if loss_type == "softplus":
        return torch.mean(F.softplus(-dr_f)) + torch.mean(F.softplus(dg_f))
    # hinge and soft_hinge share the discriminator term
    return torch.mean(F.relu(1.0 - dr_f)) + torch.mean(F.relu(1.0 + dg_f))


def g_branch_loss(dg, loss_type="lsgan", dr=None):
    """One branch's generator adversarial term.

    ``dr`` is the corresponding real logit and is required only for
    ``relativistic``, where it enters as a no-grad target.
    """
    dg_f = dg.float()
    if loss_type == "relativistic":
        if dr is None:
            raise ValueError("the relativistic generator loss needs the real logits")
        return torch.mean(F.softplus(dr.float().detach() - dg_f))
    if loss_type == "lsgan":
        return torch.mean((dg_f - 1.0) ** 2)
    if loss_type == "hinge":
        return -torch.mean(dg_f)
    # soft_hinge / softplus: non-saturating, bounded gradient
    return torch.mean(F.softplus(-dg_f))


def freq_MAE(output, target):
    loss = 0.
    eps = torch.finfo(torch.float32).eps
    all_win = [32, 64, 128, 256, 512, 1024, 2048]
    for win in all_win:
        est_spec = torch.stft(output.view(-1, output.shape[-1]), n_fft=win, hop_length=win//2, 
                            window=torch.hann_window(win).to(output.device).float(),
                            return_complex=True)
        target_spec = torch.stft(target.view(-1, target.shape[-1]), n_fft=win, hop_length=win//2, 
                                window=torch.hann_window(win).to(target.device).float(),
                                return_complex=True)
        
        loss = loss + (est_spec.abs() - target_spec.abs()).abs().mean() / (target_spec.abs().mean() + eps)
    
    return loss / len(all_win)

class MultiFrequencyDisLoss(_Loss):
    """Discriminator loss, averaged over whichever branches were run.

    ``loss_type`` defaults to ``lsgan``, which is the original objective. See the
    module header for what the others buy.
    """

    def __init__(self, eps=1e-8, loss_type="lsgan"):
        super(MultiFrequencyDisLoss, self).__init__()
        self.loss_type = str(loss_type)

    def forward(self, target_outputs, est_outputs):
        n = max(len(target_outputs), 1)
        loss = 0
        for real, fake in zip(target_outputs, est_outputs):
            loss = loss + d_branch_loss(real, fake, self.loss_type) / n
        return loss
    
class MultiFrequencyGenLoss(_Loss):
    def __init__(self, eps=1e-8):
        super(MultiFrequencyGenLoss, self).__init__()
        self.eps = eps

    def forward(self, est_outputs, est_feature_maps, targets_feature_maps, output, ori_data,
                adv_scale=1.0, target_outputs=None):
        # target_outputs is accepted and ignored: the training loop passes the
        # critic's real logits unconditionally, because the relativistic objective
        # needs them. This loss is LSGAN only, which does not.
        G_fake = 0
        feature_matching = 0
        eps = self.eps

        for i in range(len(est_outputs)):
            G_fake = G_fake + (est_outputs[i] - 1).pow(2).mean() / len(est_outputs)
            for j in range(len(est_feature_maps[i])):
                feature_matching = feature_matching + (est_feature_maps[i][j] - targets_feature_maps[i][j].detach()).abs().mean() / (targets_feature_maps[i][j].detach().abs().mean() + eps)

        feature_matching = feature_matching / (len(est_outputs) * len(est_feature_maps[0]))
        freq_loss = freq_MAE(output, ori_data.unsqueeze(1))
        total_loss = freq_loss + adv_scale * (G_fake + feature_matching)

        return total_loss


class VocalGenLoss(_Loss):
    """Generator loss for the vocal/speech restoration mode.

    Extends the stock objective with three things the original lacks, each of which
    maps to a specific complaint about restored voice:

    * ``fullness`` -- penalises mel bins the model left quieter than the target, so
      it stops under-shooting the harmonics and detail a codec removed. This is the
      differentiable form of the metric of the same name.
    * ``bleedless`` -- penalises energy the model added that the target lacks. It is
      the counterweight to ``fullness``: without it, asking for a fuller voice also
      invites invented hiss.
    * ``waveform`` -- an L1 in the time domain. The stock loss compares magnitudes
      only and never constrains phase, which is a direct source of the diffuse
      noise in the output.

    All weights are exposed so the balance can be tuned per dataset. Setting every
    extra weight to 0 reproduces :class:`MultiFrequencyGenLoss` exactly.
    """

    def __init__(
        self,
        eps=1e-8,
        sr=44100,
        freq_weight=1.0,
        adv_weight=1.0,
        feature_weight=1.0,
        fullness_weight=0.02,
        bleedless_weight=0.01,
        clarity_weight=0.5,
        waveform_weight=0.5,
        fullness_band_weights=((300.0, 8000.0, 1.5),),
        mel_n_fft=4096,
        mel_hop=1024,
        mel_bins=512,
        gan_loss_type="lsgan",
    ):
        super().__init__()
        self.eps = eps
        self.gan_loss_type = str(gan_loss_type)
        self.freq_weight = freq_weight
        self.adv_weight = adv_weight
        self.feature_weight = feature_weight
        self.fullness_weight = fullness_weight
        self.bleedless_weight = bleedless_weight
        self.clarity_weight = clarity_weight
        self.waveform_weight = waveform_weight

        mel_kwargs = dict(sr=sr, n_fft=mel_n_fft, hop_length=mel_hop, n_mels=mel_bins)
        self.fullness = (BleedFullPenaltyLoss(mode="fullness", band_weights=fullness_band_weights,
                                              **mel_kwargs) if fullness_weight else None)
        self.bleedless = (BleedFullPenaltyLoss(mode="bleedless", **mel_kwargs)
                          if bleedless_weight else None)
        self.clarity = MelClarityLoss(sr=sr) if clarity_weight else None
        self.waveform = WaveformL1Loss() if waveform_weight else None

        self.last_terms = {}

    def reconstruction_terms(self, output, ori_data):
        """The non-adversarial half of the objective.

        Split out so the training loop can evaluate it once and then walk the
        critic's branches one at a time -- which is what keeps peak memory at
        max(branch) instead of sum(branches).
        """
        target = ori_data.unsqueeze(1) if ori_data.dim() == output.dim() - 1 else ori_data

        terms = {"freq": self.freq_weight * freq_MAE(output, target)}
        if self.fullness is not None:
            terms["fullness"] = self.fullness_weight * self.fullness(output, target)
        if self.bleedless is not None:
            terms["bleedless"] = self.bleedless_weight * self.bleedless(output, target)
        if self.clarity is not None:
            terms["clarity"] = self.clarity_weight * self.clarity(output, target)
        if self.waveform is not None:
            terms["waveform"] = self.waveform_weight * self.waveform(output, target)
        return terms

    def adversarial_terms(self, est_outputs, est_feature_maps, targets_feature_maps,
                          adv_scale=1.0, n_branches=None, target_outputs=None):
        """The adversarial half, for any subset of the critic's branches.

        ``n_branches`` is the size of the *full* bank; pass it when evaluating a
        subset so each branch keeps the weight it would have had in a full pass.

        ``target_outputs`` are the critic's real-side logits, needed only by the
        relativistic objective, where they enter as detached targets.
        """
        eps = self.eps
        total = n_branches or len(est_outputs)

        G_fake = 0.0
        feature_matching = 0.0
        for i in range(len(est_outputs)):
            real_logit = target_outputs[i] if target_outputs is not None else None
            G_fake = G_fake + g_branch_loss(
                est_outputs[i], self.gan_loss_type, dr=real_logit) / total
            per_layer = 0.0
            for j in range(len(est_feature_maps[i])):
                reference = targets_feature_maps[i][j].detach()
                per_layer = per_layer + (
                    (est_feature_maps[i][j] - reference).abs().mean()
                    / (reference.abs().mean() + eps)
                )
            if est_feature_maps[i]:
                feature_matching = feature_matching + per_layer / len(est_feature_maps[i])
        feature_matching = feature_matching / total

        return {
            "adv": adv_scale * self.adv_weight * G_fake,
            "fm": adv_scale * self.feature_weight * feature_matching,
        }

    def forward(self, est_outputs, est_feature_maps, targets_feature_maps, output, ori_data,
                adv_scale=1.0, target_outputs=None):
        eps = self.eps
        terms = self.reconstruction_terms(output, ori_data)

        G_fake = 0.0
        feature_matching = 0.0
        for i in range(len(est_outputs)):
            real_logit = target_outputs[i] if target_outputs is not None else None
            G_fake = G_fake + g_branch_loss(
                est_outputs[i], self.gan_loss_type, dr=real_logit) / len(est_outputs)
            for j in range(len(est_feature_maps[i])):
                reference = targets_feature_maps[i][j].detach()
                feature_matching = feature_matching + (
                    (est_feature_maps[i][j] - reference).abs().mean()
                    / (reference.abs().mean() + eps)
                )
        # Pooled over the bank's total layer count rather than averaged per
        # branch. For a bank whose branches share a depth the two agree; this is
        # the form the loss shipped with, and `adversarial_terms` (which has to
        # work on one branch at a time) carries the per-branch form.
        feature_matching = feature_matching / (len(est_outputs) * len(est_feature_maps[0]))

        # adv_scale is the warmup/ramp factor from the training loop: it gates the
        # two adversarial terms only, leaving reconstruction to establish a sane
        # solution before the critic gets a vote
        terms["adv"] = adv_scale * self.adv_weight * G_fake
        terms["fm"] = adv_scale * self.feature_weight * feature_matching

        # kept for logging; detached so it cannot hold on to the graph
        self.last_terms = {k: float(v.detach()) if torch.is_tensor(v) else float(v)
                           for k, v in terms.items()}

        return sum(terms.values())
