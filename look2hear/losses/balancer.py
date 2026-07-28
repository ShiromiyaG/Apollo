"""Gradient balancer for multi-term generator objectives (EnCodec / DAC).

Ported from `H:/Chouwa/SingingVocoders/modules/loss/balancer.py` (Défossez et al.,
2022; DAC uses the same idea).

`VocalGenLoss` sums seven terms whose raw magnitudes have nothing to do with each
other: `freq_MAE` is normalised to roughly O(1), the waveform L1 on a -20 dBFS
signal sits around 1e-2, and the mel penalties move with the filterbank size. The
configured weights therefore do not mean what they look like -- `waveform_weight:
0.5` next to `freq_weight: 1.0` is not "half as important", it is closer to "two
orders of magnitude less important". Every entry in `docs/gan.md`'s tuning table
exists because of that mismatch.

The balancer removes it. For each term it takes the gradient the term contributes
*to the generator output*, divides by a running estimate of that gradient's norm,
and recombines:

    out_grad = sum_i  total_norm * (w_i / sum_j w_j) / EMA||g_i||  *  g_i

so each term's share of the update is exactly `w_i / sum_j w_j` regardless of its
raw scale or how that scale drifts during training. The weights become fractions
of the update. It also removes the classic vocoder-GAN failure where the
reconstruction term dominates early (the critic is ignored) and the adversarial
term dominates later (the model collapses onto it) purely because their scales
crossed over.

Only terms that are genuinely a function of the output belong here. A regulariser
on an internal activation has no gradient w.r.t. the output and must be added to
the backward the ordinary way.
"""

import torch


class Balancer:
    """Rescale each loss term's gradient to a fixed share of the update.

    Args:
        weights: term name -> relative weight. A term that is absent or weighted 0
            is skipped entirely.
        total_norm: norm of the combined gradient handed to ``input.backward``.
            Acts as a global learning-rate multiplier for the balanced terms.
        ema_decay: smoothing for the per-term gradient-norm estimate.
    """

    def __init__(self, weights, total_norm=1.0, ema_decay=0.999, eps=1e-12):
        self.weights = {str(k): float(v) for k, v in dict(weights).items()}
        self.total_norm = float(total_norm)
        self.decay = float(ema_decay)
        self.eps = float(eps)

        # Kept as 0-dim device tensors: the norms are read straight off the
        # gradients, so materialising them as Python floats would sync the CPU
        # against the GPU once per term per step to run three multiplies the GPU
        # can do itself. Only state_dict() ever pulls them off the device.
        self._ema = {}
        # Terms become active at different times (adv/fm only after the warmup),
        # so the bias correction has to be tracked per term. Data-independent,
        # hence a plain float -- which is also what keeps the `fix > 0` test off
        # the device.
        self._fix = {}

    def state_dict(self):
        return {"ema": {k: float(v) for k, v in self._ema.items()},
                "fix": dict(self._fix)}

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            return
        # Restored on CPU; the first backward moves each entry onto the gradient's
        # device (see `backward`).
        self._ema = {str(k): torch.tensor(float(v))
                     for k, v in dict(state.get("ema", {})).items()}
        fix = state.get("fix", {})
        self._fix = ({str(k): float(v) for k, v in dict(fix).items()}
                     if isinstance(fix, dict) else {})

    def backward(self, losses, input, scale=1.0, weight_scale=None, pre_grads=None):
        """Balance, combine and backward through ``input``.

        Args:
            losses: name -> scalar tensor, each differentiable w.r.t. ``input``.
            input: the tensor every term is a function of (the generator output).
            scale: multiplies the final gradient, e.g. 1/N under accumulation.
            weight_scale: name -> multiplier applied to that term's weight for
                this call only. This is where the adversarial ramp goes.
            pre_grads: name -> an already-computed dL/d``input``, for terms whose
                graph was consumed elsewhere. The branch-wise generator step
                accumulates the adversarial and feature-matching gradients one
                critic branch at a time and hands the sums over here. Identical
                mathematically to passing the scalar loss, since the balancer only
                ever uses dL/dinput.

        Returns:
            name -> effective gradient norm, for logging.
        """
        ws = weight_scale or {}
        pre_grads = pre_grads or {}

        all_names = list(losses) + [n for n in pre_grads if n not in losses]
        eff_w = {n: self.weights.get(n, 0.0) * float(ws.get(n, 1.0)) for n in all_names}

        names = [n for n, l in losses.items()
                 if n not in pre_grads and eff_w.get(n, 0.0) != 0.0
                 and torch.is_tensor(l) and l.requires_grad]
        pre_names = [n for n in pre_grads
                     if eff_w.get(n, 0.0) != 0.0 and torch.is_tensor(pre_grads[n])]
        if not names and not pre_names:
            return {}

        # The norms are taken in fp32 whatever the gradients are. A decay of 0.999
        # updates the estimate by one part in a thousand, and bfloat16 carries
        # about three significant decimal digits -- the increment rounds away
        # entirely and the EMA freezes at its first value, silently, for the whole
        # run. Measured: 0 of 200 updates changed a bf16 EMA at decay 0.999, against
        # 200 of 200 in fp32. Today `input` happens to arrive as fp32 even under
        # bf16 autocast, so this costs nothing; it stops a change to the generator's
        # output op from quietly disabling the balancer.
        grads, norms = {}, {}
        for name in names:
            g, = torch.autograd.grad(losses[name], input, retain_graph=True)
            grads[name] = g
            norms[name] = g.detach().float().norm()
        for name in pre_names:
            grads[name] = pre_grads[name]
            norms[name] = pre_grads[name].detach().float().norm()
        names = names + pre_names

        # Zero-initialised per term, so the bias-corrected first estimate equals
        # the observed norm. Seeding the EMA with the first norm instead would
        # shrink the very first update by ~1/(1-decay).
        for name in names:
            prev = self._ema.get(name)
            if prev is None:
                prev = torch.zeros((), device=norms[name].device, dtype=torch.float32)
            else:
                prev = prev.to(device=norms[name].device, dtype=torch.float32)
            self._ema[name] = self.decay * prev + (1.0 - self.decay) * norms[name]
            self._fix[name] = self.decay * self._fix.get(name, 0.0) + (1.0 - self.decay)

        sum_w = sum(eff_w[n] for n in names)
        # Accumulated in fp32 for the same reason as the norms: the terms' scales
        # can differ by orders of magnitude, and summing them in bf16 would drop
        # the small ones outright. Cast back at the end so `backward` gets the
        # gradient in the dtype it expects.
        out_grad = torch.zeros_like(input, dtype=torch.float32)
        effective = {}
        for name in names:
            fix = self._fix.get(name, 0.0)
            ema_n = (self._ema[name] / fix) if fix > 0 else norms[name]
            s = self.total_norm * (eff_w[name] / sum_w) / (ema_n + self.eps)
            out_grad = out_grad + s * grads[name].float()
            effective[name] = s * norms[name]

        input.backward((out_grad * float(scale)).to(input.dtype))
        return effective
