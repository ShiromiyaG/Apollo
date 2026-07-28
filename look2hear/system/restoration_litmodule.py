"""Training loop for both restoration modes.

This is where the GAN itself is tuned, as opposed to papering over its output at
inference time.

The first generation of changes, in rough order of how much they move the noise
floor of the *trained* model:

1. **Generator EMA.** The exported weights are an average over the last ~1000
   steps rather than whatever the final adversarial step happened to produce.
   Validation runs on the averaged weights too, so checkpoint selection describes
   what ships.
2. **Lazy gradient penalty.** Bounds how sharp the critic is allowed to get around
   the data. Without it the generator chases a gradient field it cannot track, and
   the residual is high-frequency noise.
3. **Adversarial warmup and ramp.** Letting the reconstruction terms establish a
   sane solution before the critic gets a vote avoids the early-training noise
   that never fully washes out.
4. **Branch rotation.** The critic's resolution branches are independent, so they
   need not all run every step.
5. **No-grad target features.** The reference loop builds a full autograd graph for
   the target feature maps and then discards it, since the loss detaches them.

The second generation, ported from `H:/Chouwa/SingingVocoders/training/chouwa_gan_task.py`,
is opt-in per config and enabled in `configs/apollo_v2_restoration.yaml`:

6. **`gstep_output_grad`.** The branch-wise generator step used to run one full
   backward *through the generator* per critic branch -- and with gradient
   checkpointing on, each of those also re-ran the generator's forward. Taking the
   gradient with respect to the generated waveform instead, accumulating it across
   branches and traversing the generator exactly once is the same arithmetic at a
   fraction of the cost.
7. **`use_loss_balancer`.** Rescales every objective term's gradient to a fixed
   share of the update, so the configured weights stop meaning "whatever this
   term's raw scale happens to be". See `look2hear/losses/balancer.py`.
8. **`r2_gamma` and `gan_loss_type: relativistic`.** The R3GAN recipe: judge the
   paired real-vs-fake difference, and penalise the critic's gradient on both
   sides rather than only the real one. **`r1_r2_joined`** then takes both
   penalties out of one forward and one double backward instead of two, which
   matters because a step that fires the penalty is otherwise an order of
   magnitude more expensive than an ordinary one.
9. **`disc_branch_every`.** Per-branch schedules matched by glob, instead of a
   uniform rotation that treats a 128-window branch as costing the same as an
   envelope bank.
10. **`disc_concat_real_fake`.** One critic forward over `cat([real, fake])`
    rather than two, since every branch operation is per sample.

Two counting corrections apply to every config, old or new, because both were
straightforwardly wrong and neither touches a checkpoint:

* The gradient penalty is scheduled on **completed discriminator updates**. It used
  to be scheduled on `batch_idx`, which counts micro-batches -- so under
  `accumulate_grad_batches: 8` with `r1_every: 16` it fired every 2 updates while
  still multiplying by the full interval of 16, applying the penalty 8x harder than
  the config asked for.
* The warmup and ramp count **optimizer windows**. They used to read
  `self.global_step`, which under manual optimisation counts every
  `optimizer.step()` -- two per window once the critic is running, one per window
  during the warmup. `adv_ramp_steps` therefore completed in half the requested
  number of steps.

See `docs/gan.md`.
"""

import fnmatch
from collections.abc import Mapping
from contextlib import nullcontext

import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..discriminators.frequencydis import gradient_penalty, joint_gradient_penalty
from ..losses.balancer import Balancer
from ..losses.perceptual import BleedFullPenaltyLoss
from .ema import ModelEMA


def _as_module_dict(loss_func):
    """Wrap a {name: nn.Module} mapping in a ModuleDict so it moves with the parent."""
    if isinstance(loss_func, torch.nn.ModuleDict) or not isinstance(loss_func, Mapping):
        return loss_func

    items = {str(k): v for k, v in loss_func.items()}
    if items and all(isinstance(v, torch.nn.Module) for v in items.values()):
        return torch.nn.ModuleDict(items)
    return loss_func


class RestorationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model=None,
        discriminator=None,
        optimizer=None,
        loss_func=None,
        metrics=None,
        scheduler=None,
        sr: int = 44100,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = 5.0,
        autocast_dtype: str = "none",
        gradient_checkpointing: bool = False,
        # --- GAN controls ---
        ema_decay: float = 0.999,
        ema_warmup_steps: int = 1000,
        use_ema_for_validation: bool = True,
        r1_gamma: float = 0.0,
        r2_gamma: float = 0.0,
        r1_every: int = 16,
        r1_patch_reduction: str = "mean",
        r1_branchwise: bool = False,
        r1_r2_joined: bool = False,
        hold_disc_scheduler: bool = False,
        disc_start_step: int = 0,
        adv_ramp_steps: int = 0,
        disc_branches_per_step: int = 0,
        disc_branchwise: bool = False,
        disc_branch_every=None,
        disc_branch_offset=None,
        disc_concat_real_fake: bool = False,
        gstep_output_grad: bool = False,
        use_loss_balancer: bool = False,
        balancer_weights=None,
        balancer_total_norm: float = 1.0,
        balancer_ema_decay: float = 0.999,
        val_empty_cache: bool = False,
        log_every_n_steps: int = 50,
    ):
        super().__init__()
        self.audio_model = model
        self.discriminator = discriminator
        self.optimizer = list(optimizer)
        self.metrics = metrics
        self.scheduler = list(scheduler)

        # A bare mapping is invisible to nn.Module, so Lightning would never move
        # the losses onto the GPU -- and these ones carry mel-filterbank buffers.
        # Matched as a Mapping rather than a dict because Hydra hands this over as
        # an omegaconf DictConfig, which is not a dict subclass.
        self.loss_func = _as_module_dict(loss_func)

        self.sr = sr
        self.accumulate = max(1, int(accumulate_grad_batches))
        self.gradient_clip_val = gradient_clip_val
        self.log_every_n_steps = log_every_n_steps

        self.r1_gamma = float(r1_gamma)
        self.r2_gamma = float(r2_gamma)
        self.r1_every = max(1, int(r1_every))
        self.r1_patch_reduction = str(r1_patch_reduction)
        self.r1_branchwise = bool(r1_branchwise)
        self.r1_r2_joined = bool(r1_r2_joined)
        self.hold_disc_scheduler = bool(hold_disc_scheduler)
        self.disc_start_step = int(disc_start_step)
        self.adv_ramp_steps = max(0, int(adv_ramp_steps))
        self.disc_branches_per_step = int(disc_branches_per_step)
        self.disc_branchwise = bool(disc_branchwise)
        self.disc_concat_real_fake = bool(disc_concat_real_fake)
        self.gstep_output_grad = bool(gstep_output_grad)
        self.val_empty_cache = bool(val_empty_cache)

        # Per-branch schedules: glob pattern -> interval, matched against the
        # critic's branch names. The first matching pattern wins, and a branch
        # matching nothing runs every update.
        self.disc_branch_every = {str(k): max(1, int(v))
                                  for k, v in dict(disc_branch_every or {}).items()}
        self.disc_branch_offset = {str(k): int(v)
                                   for k, v in dict(disc_branch_offset or {}).items()}

        self._branch_cursor = 0
        self._active = None
        # Counted here rather than read off the trainer: `global_step` counts every
        # manual `optimizer.step()`, which is two per window with a critic running
        # and one during the warmup, so neither the schedules nor the lazy
        # regulariser can be expressed in it.
        self._d_update_count = 0
        self._batch_step = 0
        self._last_reg = {"r1": 0.0, "r2": 0.0}

        self.balancer = None
        if use_loss_balancer:
            if not balancer_weights:
                raise ValueError("use_loss_balancer needs balancer_weights: "
                                 "a {term: relative weight} mapping")
            self.balancer = Balancer(dict(balancer_weights),
                                     total_norm=float(balancer_total_norm),
                                     ema_decay=float(balancer_ema_decay))

        self.autocast_dtype = {
            "none": None, "fp32": None,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
        }[str(autocast_dtype).lower()]

        if gradient_checkpointing and hasattr(self.audio_model, "set_gradient_checkpointing"):
            self.audio_model.set_gradient_checkpointing(True)

        self.use_ema_for_validation = use_ema_for_validation
        self.ema = ModelEMA(self.audio_model, decay=ema_decay,
                            warmup_steps=ema_warmup_steps) if ema_decay else None

        if self.r1_gamma > 0 or self.r2_gamma > 0:
            # the double backward can hang forever under cuDNN autotuning
            torch.backends.cudnn.benchmark = False

        self.default_monitor = "val_loss"
        self.automatic_optimization = False
        self.validation_step_outputs = []

        self.val_fullness = BleedFullPenaltyLoss(mode="fullness", sr=sr)
        self.val_bleedless = BleedFullPenaltyLoss(mode="bleedless", sr=sr)

    # ------------------------------------------------------------------

    def on_fit_start(self):
        # the EMA shadow is allocated in __init__, before the trainer moves the
        # module onto the accelerator
        if self.ema is not None:
            self.ema.to(self.device)
        self._validate_config()

    def _validate_config(self):
        """Fail loudly on combinations where a configured feature would be built,
        logged, and then silently bypassed."""
        if self.balancer is not None and not hasattr(self.loss_func["g"], "reconstruction_terms"):
            raise ValueError(
                "use_loss_balancer needs a generator loss that can report its "
                "terms separately (VocalGenLoss). MultiFrequencyGenLoss returns "
                "only the sum, so there would be nothing to balance.")
        if self.disc_branch_every and not hasattr(self.discriminator, "names"):
            raise ValueError(
                f"disc_branch_every is set but {type(self.discriminator).__name__} "
                "does not expose branch names, so the patterns could never match "
                "and the schedule would be silently ignored. Use a "
                "CombinedDiscriminator, or drop disc_branch_every.")

    def forward(self, wav):
        return self.audio_model(wav)

    def _autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    # -------------------------------------------------------- branch selection

    def _branch_names(self):
        disc = self.discriminator
        if hasattr(disc, "names"):
            return list(disc.names())
        names = getattr(disc, "branch_names", None)
        if names:
            return list(names)
        return [f"branch_{i}" for i in range(len(disc.discriminators))]

    def _active_branches(self):
        """Rotate through the critic's branches instead of running them all."""
        total = len(self.discriminator.discriminators)
        k = self.disc_branches_per_step
        if k <= 0 or k >= total:
            return None

        picked = [(self._branch_cursor + i) % total for i in range(k)]
        self._branch_cursor = (self._branch_cursor + k) % total
        return sorted(picked)

    def _scheduled_branches(self):
        """Per-branch intervals, matched by glob against the branch names.

        A uniform rotation implicitly claims every branch costs the same. They do
        not: an envelope bank or a complex-STFT branch at 2048 is worth several of
        the cheap ones, and those are exactly the ones worth running at 1/N rate
        while the cheap branches keep supervising every update.
        """
        step = self._d_update_count
        active = []
        for i, name in enumerate(self._branch_names()):
            every, offset = 1, 0
            for pattern, n in self.disc_branch_every.items():
                if fnmatch.fnmatchcase(name, pattern):
                    every = max(1, int(n))
                    offset = int(self.disc_branch_offset.get(pattern, 0))
                    break
            if step % every == offset % every:
                active.append(i)
        # a set of offsets can leave a hole; running everything beats crashing on
        # an empty branch list
        return active or list(range(len(self.discriminator.discriminators)))

    def _refresh_branches(self):
        """Pick this window's branch set.

        Called once per accumulation window, not once per micro-batch, so the
        discriminator step, the gradient penalty and the generator step all agree
        on which branches exist. A generator pushed by a branch the discriminator
        step never updated is a different algorithm from the one configured.
        """
        if self.disc_branch_every:
            self._active = self._scheduled_branches()
            return
        picked = self._active_branches()
        self._active = (picked if picked is not None
                        else list(range(len(self.discriminator.discriminators))))

    def _adv_scale(self):
        """0 before the critic starts, then a linear ramp to 1.

        Counted in optimizer windows -- see the module header for why not
        `global_step`.
        """
        step = self._batch_step
        if step < self.disc_start_step:
            return 0.0
        if self.adv_ramp_steps <= 0:
            return 1.0
        return min(1.0, (step - self.disc_start_step) / self.adv_ramp_steps)

    # ------------------------------------------------------------- gradient clip

    def _clip(self, optimizer):
        """Clip gradients, working around fused optimizers.

        Lightning refuses its generic clipping helper for optimizers that
        advertise AMP scaling support, which fused AdamW does. Unscale explicitly
        and use PyTorch's own norm clipping in that case.
        """
        if not self.gradient_clip_val:
            return
        raw = getattr(optimizer, "optimizer", optimizer)
        if not any(bool(g.get("fused", False)) for g in raw.param_groups):
            self.clip_gradients(optimizer, gradient_clip_val=self.gradient_clip_val,
                                gradient_clip_algorithm="norm")
            return

        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            scaler.unscale_(raw)
        params = [p for g in raw.param_groups for p in g["params"] if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(params, self.gradient_clip_val)

    # ------------------------------------------------------------------ D step

    def _discriminator_outputs(self, real, fake, group):
        """The critic's real and fake logits for one group of branches.

        With `disc_concat_real_fake` the two sides go through as a single batch:
        every branch operation is per sample, so one forward covers both, halving
        the kernel launches and the STFT front-ends. The split is always exactly
        half, whatever channel folding the bank applies.
        """
        if self.disc_concat_real_fake:
            with self._autocast():
                outputs, _ = self.discriminator(torch.cat([real, fake], dim=0),
                                                sample_rate=self.sr, branches=group)
            half = [o.shape[0] // 2 for o in outputs]
            return ([o[h:] for o, h in zip(outputs, half)],
                    [o[:h] for o, h in zip(outputs, half)])

        with self._autocast():
            est, _ = self.discriminator(fake, sample_rate=self.sr, branches=group)
            target, _ = self.discriminator(real, sample_rate=self.sr, branches=group)
        return est, target

    def _gan_reg_penalty(self, real, fake, group, n_active, stats):
        """Lazy zero-centred gradient penalties (R1 on real, R2 on fake).

        Runs on the micro-batches of the window that *completes* each `r1_every`-th
        discriminator update, with the usual StyleGAN2 rescale by the interval.

        ``r1_branchwise`` decides which of two genuinely different quantities gets
        penalised, and the difference is large:

        * on:  ``mean_b ||d(score_b)/dx||^2`` -- each branch is its own critic, its
          own gradient bounded on its own. This is the standard multi-discriminator
          form.
        * off: ``||d(mean_b score_b)/dx||^2`` -- the norm of the *summed* gradient
          field, which expands to the sum of the branch terms plus every
          cross-branch inner product.

        Branch gradients are close to uncorrelated, so the cross terms cancel and
        the second form comes out about ``n_branches`` times smaller: measured on
        an 8-branch bank, 7.16e-4 against 5.72e-3, a ratio of exactly 8.0. So the
        two are not a memory trade with the same meaning -- ``r1_gamma`` means
        something ``n`` times weaker with it off, and that ``n`` moves whenever the
        bank changes.

        It also caps memory: each branch's double-backward graph is released by its
        own backward before the next one is built.
        """
        if self.r1_gamma <= 0 and self.r2_gamma <= 0:
            return
        if (self._d_update_count + 1) % self.r1_every != 0:
            return

        # One sub-group per branch, or the whole group at once. The weights add up
        # to the same total either way; what changes is the quantity being bounded.
        subgroups = [[b] for b in group] if self.r1_branchwise else [group]

        for sub in subgroups:
            share = len(sub) / max(n_active, 1)
            mult = self.r1_every * share / self.accumulate

            if self.r1_r2_joined and self.r1_gamma > 0 and self.r2_gamma > 0:
                # One forward and one double backward for both penalties. The
                # double backward is what makes a regularisation step expensive,
                # and running R1 and R2 separately pays for it twice over the same
                # critic.
                r1, r2 = joint_gradient_penalty(
                    self.discriminator, real, fake, self.sr, sub,
                    patch_reduction=self.r1_patch_reduction)
                self.manual_backward(
                    0.5 * mult * (self.r1_gamma * r1 + self.r2_gamma * r2))
                stats["r1"] = stats.get("r1", 0.0) + float(r1.detach()) * share
                stats["r2"] = stats.get("r2", 0.0) + float(r2.detach()) * share
                continue

            for name, gamma, sample in (("r1", self.r1_gamma, real),
                                        ("r2", self.r2_gamma, fake)):
                if gamma <= 0:
                    continue
                penalty = gradient_penalty(self.discriminator, sample, self.sr, sub,
                                           patch_reduction=self.r1_patch_reduction)
                self.manual_backward(0.5 * gamma * mult * penalty)
                stats[name] = stats.get(name, 0.0) + float(penalty.detach()) * share

    # ------------------------------------------------------------------ G step

    @staticmethod
    def _scalar(value):
        """A loss term may legitimately be a plain 0.0 -- a branch with no feature
        maps contributes no feature-matching tensor."""
        return float(value.detach()) if torch.is_tensor(value) else float(value)

    def _branch_adversarial(self, real, fake, active, loss_fn, adv_scale, inv_n,
                            collect, logged):
        """Walk the critic one branch at a time on the generator step.

        Only one branch's critic activations are alive at a time, so the peak is
        max(branch) rather than the sum -- and feature matching is what makes that
        peak the true one, since it holds every branch's feature maps at once.

        `collect` decides *where* the branch loop stops. Backwarding each branch
        directly walks the whole generator graph once per branch, and with
        gradient checkpointing on that also re-runs the generator's forward each
        time. Collecting instead stops the branch gradients at `fake`, costing one
        tensor the size of the output, and lets the caller traverse the generator
        exactly once. The gradients are identical: they are additive across
        branches either way.

        Returns `(g_adv, g_fm)` when collecting, `(None, None)` otherwise.
        """
        # fp32 accumulators: 18 branch gradients summed in bf16 would lose the
        # small ones, and the balancer needs an honest norm out of this.
        g_adv = torch.zeros_like(fake, dtype=torch.float32) if collect else None
        g_fm = torch.zeros_like(fake, dtype=torch.float32) if collect else None

        for position, branch in enumerate(active):
            last = position == len(active) - 1
            with self._autocast():
                est_outputs, est_fm = self.discriminator(fake, sample_rate=self.sr,
                                                         branches=[branch])
                # The loss detaches these, so building a graph for them is waste.
                with torch.no_grad():
                    target_outputs, target_fm = self.discriminator(
                        real, sample_rate=self.sr, branches=[branch])
                # Collected gradients carry the ramp later (the balancer applies it
                # as a weight scale), so the terms come out of the loss unramped.
                terms = loss_fn.adversarial_terms(
                    est_outputs, est_fm, target_fm,
                    adv_scale=1.0 if collect else adv_scale,
                    n_branches=len(active), target_outputs=target_outputs)

            if collect:
                grads = [(key, terms[key]) for key in ("adv", "fm")
                         if torch.is_tensor(terms.get(key)) and terms[key].requires_grad]
                for index, (key, value) in enumerate(grads):
                    # `autograd.grad` stops at `fake`, so this only ever frees the
                    # critic branch's graph -- the generator's is untouched.
                    keep = not (last and index == len(grads) - 1)
                    target = g_adv if key == "adv" else g_fm
                    target.add_(torch.autograd.grad(value, fake, retain_graph=keep)[0].float())
            else:
                self.manual_backward(sum(terms.values()) * inv_n, retain_graph=not last)

            for key, value in terms.items():
                logged[key] = logged.get(key, 0.0) + self._scalar(value)

        return g_adv, g_fm

    def _generator_step(self, real, fake, adv_scale, active, inv_n, logged):
        """Backward the generator objective. Returns the value to log."""
        loss_fn = self.loss_func["g"]
        supports_split = hasattr(loss_fn, "reconstruction_terms")
        collect = self.gstep_output_grad or self.balancer is not None
        branchwise = supports_split and (self.disc_branchwise or self.balancer is not None)

        if supports_split and adv_scale == 0:
            # During the warmup the adversarial terms are multiplied by zero, so
            # running the critic at all is pure waste -- and the warmup can be
            # thousands of steps.
            recon = loss_fn.reconstruction_terms(fake, real)
            logged.update({k: self._scalar(v) for k, v in recon.items()})
            if self.balancer is not None:
                self._log_balancer(self.balancer.backward(recon, fake, scale=inv_n), logged)
            else:
                self.manual_backward(sum(recon.values()) * inv_n)
            return sum(logged.values())

        if branchwise and adv_scale > 0:
            recon = loss_fn.reconstruction_terms(fake, real)
            logged.update({k: self._scalar(v) for k, v in recon.items()})

            if not collect:
                # Reconstruction first, keeping the generator graph alive across
                # the per-branch backwards; the last branch releases it.
                self.manual_backward(sum(recon.values()) * inv_n, retain_graph=True)

            g_adv, g_fm = self._branch_adversarial(
                real, fake, active, loss_fn, adv_scale, inv_n, collect, logged)

            if self.balancer is not None:
                # The balancer's backward through `fake` is what finally releases
                # the generator graph, so it has to come last.
                self._log_balancer(self.balancer.backward(
                    recon, fake, scale=inv_n,
                    weight_scale={"adv": adv_scale, "fm": adv_scale},
                    pre_grads={"adv": g_adv, "fm": g_fm}), logged)
            elif collect:
                out_grad = (g_adv + g_fm) * adv_scale
                out_grad += torch.autograd.grad(sum(recon.values()), fake)[0].float()
                out_grad = out_grad.to(fake.dtype)
                # Not `manual_backward`: the seed gradient makes this a
                # vector-Jacobian product rather than a scalar loss. There is no
                # GradScaler to bypass -- the configs run fp32 with manual
                # autocast -- and the balancer's own backward does the same.
                fake.backward(gradient=out_grad * inv_n)
            return sum(v for k, v in logged.items() if not k.startswith("bal_"))

        # Fused: one critic forward over the whole active set.
        with self._autocast():
            est_outputs, est_fm = self.discriminator(fake, sample_rate=self.sr,
                                                     branches=active)
            with torch.no_grad():
                target_outputs, target_fm = self.discriminator(
                    real, sample_rate=self.sr, branches=active)
            loss_g = loss_fn(est_outputs, est_fm, target_fm, fake, real,
                             adv_scale=adv_scale, target_outputs=target_outputs)
        self.manual_backward(loss_g / self.accumulate)
        logged.update(getattr(loss_fn, "last_terms", None) or {})
        return loss_g.detach()

    @staticmethod
    def _log_balancer(effective, logged):
        for key, value in effective.items():
            logged[f"bal_{key}"] = float(value)

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        ori_data, codec_data = batch
        optimizer_g, optimizer_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        win_start = batch_idx % self.accumulate == 0
        win_step = (batch_idx + 1) % self.accumulate == 0
        inv_n = 1.0 / self.accumulate

        if win_start or self._active is None:
            self._refresh_branches()
        active = self._active
        adv_scale = self._adv_scale()
        logged = {}
        stats = {}

        with self._autocast():
            output = self(codec_data)

        # ---------------- discriminator ----------------
        loss_d = torch.zeros((), device=self.device)

        if adv_scale > 0:
            if win_start:
                optimizer_d.zero_grad(set_to_none=True)

            groups = [[b] for b in active] if self.disc_branchwise else [active]
            for group in groups:
                est_outputs, target_outputs = self._discriminator_outputs(
                    ori_data, output.detach(), group)
                with self._autocast():
                    branch_loss = self.loss_func["d"](target_outputs, est_outputs)
                    branch_loss = branch_loss * (len(group) / len(active))

                # Backward per group: this group's critic activations are released
                # before the next one allocates, so peak tracks the largest single
                # branch rather than the sum of all of them.
                self.manual_backward(branch_loss * inv_n)
                loss_d = loss_d + branch_loss.detach()

                # Separate from the branch loss on purpose: its graph is gone by
                # the time the penalty's double-backward graph is built.
                self._gan_reg_penalty(ori_data, output.detach(), group, len(active), stats)

            if win_step:
                self._clip(optimizer_d)
                optimizer_d.step()
                optimizer_d.zero_grad(set_to_none=True)
                self._d_update_count += 1

        # ---------------- generator ----------------
        if win_start:
            optimizer_g.zero_grad(set_to_none=True)

        loss_g = self._generator_step(ori_data, output, adv_scale, active, inv_n, logged)

        if win_step:
            self._clip(optimizer_g)
            optimizer_g.step()
            optimizer_g.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.audio_model)
            self._batch_step += 1

        if self.trainer.is_last_batch:
            scheduler_g.step()
            # The critic's scheduler must not run ahead of the critic. While
            # `adv_scale` is 0 the discriminator optimizer never steps, so a StepLR
            # decaying once an epoch through a `disc_start_step: 5000` warmup hands
            # the critic an already-annealed learning rate on the step it finally
            # wakes up -- and PyTorch's "scheduler.step() before optimizer.step()"
            # warning is pointing at exactly this. Off by default so existing runs
            # are unchanged; the v2 configs, which have the long warmups, set it.
            if not self.hold_disc_scheduler or self._d_update_count > 0:
                scheduler_d.step()

        self.log("train_loss_g", loss_g, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss_d", loss_d, on_epoch=True, prog_bar=True, sync_dist=True)

        # The penalties are sparse while logging is periodic, so carry the most
        # recent measurement forward; otherwise a lazy interval that aliases with
        # the log interval never reaches TensorBoard at all.
        self._last_reg.update(stats)
        if batch_idx % self.log_every_n_steps == 0:
            extra = {"train/adv_scale": adv_scale,
                     "train/active_branches": float(len(active))}
            if self.r1_gamma > 0:
                extra["train/r1"] = self._last_reg["r1"]
            if self.r2_gamma > 0:
                extra["train/r2"] = self._last_reg["r2"]
            extra.update({f"train/{k}": v for k, v in logged.items()})
            self.log_dict(extra, on_step=True, on_epoch=False, sync_dist=True)

    # ------------------------------------------------------------------

    def _ema_context(self):
        if self.ema is None or not self.use_ema_for_validation:
            return nullcontext()
        return self.ema.averaged(self.audio_model)

    def validation_step(self, batch, batch_idx):
        ori_data, codec_data = batch

        with self._ema_context():
            est = self(codec_data)

        length = min(est.shape[-1], ori_data.shape[-1])
        est, ori = est[..., :length], ori_data[..., :length]

        loss = self.metrics(est, ori)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_fullness_penalty", self.val_fullness(est, ori),
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_bleedless_penalty", self.val_bleedless(est, ori),
                 on_epoch=True, sync_dist=True)

        self.validation_step_outputs.append(loss)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        try:
            if self.validation_step_outputs:
                avg = torch.stack(self.validation_step_outputs).mean()
                self.log("val_si_sdr", -torch.mean(self.all_gather(avg)),
                         on_epoch=True, sync_dist=True)
                self.log("lr", self.optimizer[0].param_groups[0]["lr"],
                         on_epoch=True, sync_dist=True)
                self.validation_step_outputs.clear()
        finally:
            self._release_validation_cache()

    def _release_validation_cache(self):
        """Hand the validation-shaped allocator blocks back to the driver.

        Validation runs longer segments than training (`eval_segments` is 6 s
        against 3 s) and every item carves its own block sizes. Those blocks stay
        reserved and are the wrong shape for training -- this is the VRAM that goes
        up after a validation pass and never comes back down. Costs a few
        milliseconds of re-allocation once per check interval.
        """
        if self.val_empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()
        if self.balancer is not None:
            checkpoint["loss_balancer"] = self.balancer.state_dict()
        checkpoint["gan_schedule"] = {
            "d_update_count": self._d_update_count,
            "batch_step": self._batch_step,
            "branch_cursor": self._branch_cursor,
        }

    def on_load_checkpoint(self, checkpoint):
        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
        if self.balancer is not None:
            self.balancer.load_state_dict(checkpoint.get("loss_balancer", {}))
        # Absent from checkpoints written before the schedules were counted here;
        # such a run resumes with the schedules restarted, which is the same
        # position it was in before this existed.
        schedule = checkpoint.get("gan_schedule", {})
        self._d_update_count = int(schedule.get("d_update_count", 0))
        self._batch_step = int(schedule.get("batch_step", 0))
        self._branch_cursor = int(schedule.get("branch_cursor", 0))

    def export_generator(self, path):
        """Write the EMA weights in the format inference.py loads.

        The architecture arguments travel with the weights: v2 has settings that
        leave no trace in the tensor shapes, so a bare state dict is not enough to
        rebuild it.
        """
        with self._ema_context():
            state = {k: v.detach().cpu().clone() for k, v in self.audio_model.state_dict().items()}

        payload = {
            "state_dict": state,
            "model_name": type(self.audio_model).__name__,
        }
        try:
            payload["model_args"] = self.audio_model.get_model_args()
        except NotImplementedError:
            pass

        torch.save(payload, path)
        return path

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        schedulers = []
        for sched in self.scheduler:
            if isinstance(sched, dict):
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                if sched.get("interval") == "batch":
                    sched["interval"] = "step"
                schedulers.append(sched)
            elif isinstance(sched, ReduceLROnPlateau):
                schedulers.append({"scheduler": sched, "monitor": self.default_monitor})
            else:
                schedulers.append(sched)

        return self.optimizer, schedulers


# the vocal config referred to this by its old name
VocalLightningModule = RestorationLightningModule
