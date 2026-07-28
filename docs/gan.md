# Apollo's GAN, and how to change it

Everything below is about the *trained* model. Post-processing at inference
(`--crossover`, `--gate`) hides artefacts; this is about not producing them.

## How it is wired

Three parts, in `look2hear/`:

| part | file | role |
| --- | --- | --- |
| generator | `models/apollo.py` | band-split network; the thing you ship |
| critic | `discriminators/frequencydis.py` | bank of 2D CNNs, one per STFT resolution |
| objective | `losses/gan_losses.py` | LSGAN + feature matching + reconstruction |
| loop | `system/restoration_litmodule.py` | who steps when, EMA, R1, warmup |

Per training step:

1. generator produces `output` from the degraded input;
2. critic scores `output.detach()` and the target → `loss_d` → critic steps;
3. critic scores `output` again, and the target under `no_grad` → `loss_g` →
   generator steps;
4. the generator's EMA is updated.

The critic works on **real and imaginary** STFT parts stacked as channels, not on
magnitudes. That matters: it is the only part of the whole objective that can
constrain phase, because `freq_MAE` takes `.abs()` of both spectra and throws
phase away.

## Why the stock setup produces noise

**Nothing bounds how sharp the critic gets.** With no gradient penalty, an LSGAN
critic is free to become arbitrarily confident in a thin shell around the real
data. The generator then chases a gradient field that changes faster than it can
follow, and the residual it leaves is broadband and noise-like.

**The exported weights are one adversarial step.** A GAN generator does not
converge to a point, it orbits one. Taking whatever the last optimiser step
produced bakes in that orbit's jitter.

**The critic votes from step 0.** Before the reconstruction terms have found a
sane solution, the adversarial gradient is pointing at noise, and early damage
does not fully wash out.

**Two of the seven default branches are degenerate.** `window: [32, 64, ...]` —
a 32-point FFT is 17 frequency bins over 0.7 ms. That branch cannot see spectral
structure; it behaves as a time-domain critic and is the one most likely to
reward high-frequency energy regardless of whether it belongs there.

## Two counting corrections

These are not options. Both were wrong, both apply to every config, and neither
touches a checkpoint — a v1 checkpoint loads and resumes exactly as before.

**The gradient penalty is scheduled on completed discriminator updates.** It used
to key off `batch_idx`, which counts *micro-batches*. Under
`accumulate_grad_batches: 8` the critic only steps once per 8 of those, so
`r1_every: 16` fired the penalty every **2** critic updates while still applying
the lazy rescale for an interval of 16 — a penalty **8x stronger** than the config
asked for. If you were tuning `r1_gamma` down to stop the output going soft, that
is why.

**The warmup and ramp count optimizer windows.** They used to read
`self.global_step`, which under manual optimisation counts every
`optimizer.step()`: one per window during the warmup (generator only), two per
window afterwards (generator and critic). So `adv_ramp_steps: 2000` actually
finished after 1000 windows. Both are now counted by the module itself and are
saved into the checkpoint under `gan_schedule`, so a resume picks the schedule up
where it left off.

There is a third correction that *does* change what a config value means, so read
this one before reusing an old `r1_gamma`: see `r1_patch_reduction` below.

## What changed

All of it is config, in the `system:` block. Set `ema_decay: 0`,
`r1_gamma: 0`, `disc_start_step: 0`, `adv_ramp_steps: 0`,
`disc_branches_per_step: 0` and restore the original `window` list to get the
upstream behaviour back exactly.

### `ema_decay` (0.999)

Exponential moving average of the generator weights. Validation runs on the
averaged weights and `best_model.pth` is written from them, so what gets selected
is what ships. This is the highest quality-per-risk change here: it costs one
extra copy of the parameters and a `lerp_` per step, and it cannot destabilise
anything because it does not touch the gradient path at all.

`ema_warmup_steps` ramps the decay up from 0 so the average is not anchored to
the weights it started from. Raise `ema_decay` to 0.9995 for long runs, lower it
to ~0.995 for short fine-tunes — an average over more steps than you train is
just the initial weights.

### `r1_gamma` (1.0) and `r1_every` (16)

R1 gradient penalty: the squared gradient of the critic with respect to *real*
input, which bounds how sharp it may become. Applied lazily — every `r1_every`
steps with the weight scaled by the same factor, so the time-averaged strength is
unchanged but the double backward is amortised. Measured cost at the default
interval: none I could distinguish from run-to-run noise.

Raise `r1_gamma` if the critic wins too easily (`loss_d` collapsing toward 0 while
`train/adv` climbs). Lower it if the output goes soft.

> The R1 double backward can stall indefinitely under `torch.backends.cudnn.benchmark`
> — the autotuner never finishes. The training module turns benchmark mode off
> whenever `r1_gamma > 0`. If you wire R1 in somewhere else, do the same.

### `r1_patch_reduction` (`mean`)

**This changes what `r1_gamma` means.** Each branch outputs a patch map, not a
scalar. The penalty differentiates a per-sample score derived from that map, and
how the map is collapsed decides the penalty's whole scale.

The original code summed the map. Summing `P` patches scales the gradient by `P`,
so the penalty comes out **`P²`** times larger — and `P` is a function of the
segment length and the branch's stride configuration, neither of which has
anything to do with how sharp the critic should be allowed to get. Measured on the
fast bank at 3 s: `~1e-5` under `mean` against `~5` under `sum`. So `r1_gamma: 1.0`
was applying a penalty roughly five orders of magnitude above the adversarial loss
it exists to temper, and the value could not be reasoned about at all.

`mean` averages over patches — the same score the adversarial objective is built
from — and then over the branches in the group.

Set `r1_patch_reduction: sum` only to reproduce a pre-fix run. If you carry an
`r1_gamma` over from one, it will now be far weaker; treat it as a fresh number
and tune from 1.0.

### `r1_branchwise` (false)

Which of two genuinely different quantities the penalty bounds:

| | penalised |
| --- | --- |
| `true` | `mean_b ‖∂score_b/∂x‖²` — each branch is its own critic, bounded on its own |
| `false` | `‖∂(mean_b score_b)/∂x‖²` — the norm of the *summed* gradient field |

These are not the same number and the gap is not small. Expanding the second gives
the sum of the branch terms **plus every cross-branch inner product**, and branch
gradients are close to uncorrelated, so those cancel and it lands about
`n_branches` times lower. Measured on an 8-branch bank: **7.16e-4 against 5.72e-3,
a ratio of exactly 8.0.**

That matters for two reasons. `r1_gamma` means something `n` times weaker with it
off, and that `n` moves every time you add or remove a branch — so a gamma tuned
on one bank is wrong on the next. And the per-branch form is the standard one for
a multi-discriminator setup; treating a 18-branch bank as a single critic whose
gradients may cancel against each other has no counterpart in the literature.

It also caps memory: each branch's double-backward graph is released by its own
backward before the next is built.

With `disc_branchwise: true` the penalty already ran one branch at a time, so this
changes nothing for the shipped v2 configs — it makes the choice explicit and
stops it from silently riding on an unrelated setting. It does change things if
you set `disc_branchwise: false`.

### `r2_gamma` (0)

The same penalty applied to *generated* samples. R1 alone is the StyleGAN2 recipe;
R1 + R2 with a relativistic objective is R3GAN, which is the pairing the v2 configs
use. It costs a second double backward on the same lazy schedule.

### `r1_r2_joined` (false)

Both penalties out of **one** critic forward and **one** double backward instead
of two. Samples are independent — `d(score_i)/dx_j` is zero for `i != j` — so a
single pass over `cat([real, fake])` produces both halves' per-sample gradients
without the two contaminating each other. Only applies when both gammas are set.

Worth having because a regularisation step is not a normal step. Measured on the
18-branch bank, 3 s stereo, branch-wise:

| | penalty alone | full step, penalty every update | peak on that step |
| --- | --- | --- | --- |
| separate | 6365 ms | 7098 ms | 2263 MB |
| joined | **3895 ms** | **4730 ms** | 3648 MB |

**1.63x on the penalty itself, 1.50x on the step that fires it.** Amortised over
`r1_every: 16` — fifteen ordinary steps at ~1131 ms plus one penalty step — that is
about **1.11x end to end**. The penalties agree to 2.2e-5 relative.

The trade is memory, and it is not small: that one forward carries twice the batch
plus its double-backward graph, so the peak on a regularisation step goes from
2263 to 3648 MB. Since that step is the run's high-water mark either way, enabling
this raises the whole run's peak by about 61%. At 3.6 GB on an 8 GB card there is
still plenty of headroom, which is why the v2 configs ship with it on — but if you
are close to the limit, this is the first thing to turn off, and it costs you 11%.

### `hold_disc_scheduler` (false)

Holds the critic's LR scheduler until the critic has actually taken a step.

While `adv_scale` is 0 the discriminator optimizer never steps, but the scheduler
was stepping at every epoch boundary regardless. With `disc_start_step: 5000` that
means a `StepLR` decays through the entire warmup and hands the critic an already
annealed learning rate on the step it finally wakes up. PyTorch's "called
`scheduler.step()` before `optimizer.step()`" warning is pointing straight at it.

Off by default so existing runs are unchanged. The v2 configs, which are the ones
with long warmups, set it.

### `disc_start_step` (2000) and `adv_ramp_steps` (2000)

The adversarial and feature-matching terms are multiplied by a scale that is 0
until `disc_start_step`, then ramps linearly to 1 over `adv_ramp_steps`. The
reconstruction terms are never scaled, so the model spends the opening of the run
solving the easy, well-posed part of the problem before the critic gets a vote.

Watch `train/adv_scale` in TensorBoard to confirm the schedule is doing what you
think.

### `disc_branches_per_step` (3)

Branches are independent, so they do not all have to run every step; the loop
rotates through them. Measured: 0.84x step time with the default narrow critic,
0.72x with a wide one. Every branch still gets updated, just less often — think of
it as the critic seeing each resolution at a lower rate, not as dropping
resolutions.

### `window` and `hidden_channels`

`hidden_channels` was hard-coded to 8 upstream (a branch of 8 → 256 channels).
That is a small critic. Raising it to 16 or 32 gives it more to say, at a real
cost: hidden 32 roughly doubles the step time, and branch rotation is what buys
that back.

The default `window` list now starts at 128 rather than 32. If transients go soft,
put 64 back before 32.

`min_freq_bin` skips the lowest bins entirely. A codec leaves them intact, so
asking the critic to judge them spends capacity on a solved problem. This pairs
naturally with `--crossover` at inference: if you are going to keep the input's
low band anyway, the critic need not police it.

## The v2 options

Everything in this section is off by default, so no existing config changes
behaviour. All of it is enabled in `configs/apollo_v2_restoration.yaml`. Ported
from `SingingVocoders/training/chouwa_gan_task.py`.

### `gstep_output_grad` (false)

The biggest compute win available here, and it only applies when
`disc_branchwise: true`.

The branch-wise generator step exists to bound memory: run one branch at a time so
the peak tracks the largest branch instead of the sum. But backwarding each
branch's loss walks the **whole generator graph**, once per branch — and with
`gradient_checkpointing: true` each of those also re-runs the generator's forward.
With the 16-branch v2 bank that is 16 generator backwards and 16 recomputes per
step.

Taking `∂L/∂output` for each branch instead, accumulating it into one tensor the
size of the output, and doing a single `output.backward(gradient=...)` at the end
gives **identical gradients** — they are additive across branches either way — and
traverses the generator exactly once. `tests/test_gan_v2.py` asserts both the
equivalence and the single traversal.

### `use_loss_balancer` (false) and `balancer_weights`

`VocalGenLoss` sums seven terms whose raw magnitudes have nothing to do with each
other: `freq_MAE` is normalised to roughly O(1), the waveform L1 on a -20 dBFS
signal sits near 1e-2, and the mel penalties move with the filterbank size. So
`waveform_weight: 0.5` next to `freq_weight: 1.0` never meant "half as important".
Most of the tuning table at the bottom of this document exists because of that.

The balancer (EnCodec / DAC) divides each term's gradient by a running estimate of
its own norm before combining, so `balancer_weights` are **fractions of the
update**. A term at 2.0 gets twice the gradient share of one at 1.0, whatever the
losses happen to read.

Two consequences worth knowing:

- The weights in the `loss_g:` block stop mattering. A constant prefactor cancels
  out of the balancer entirely, because it divides by the gradient's own norm and
  the norm carries the prefactor. The v2 config leaves them at 1.0 to make that
  explicit.
- It needs `VocalGenLoss`, which can report its terms separately.
  `MultiFrequencyGenLoss` returns only the sum, and the module raises at fit start
  rather than silently running unbalanced.

`train/bal_*` logs each term's effective gradient norm — that is the number to
watch, not the raw losses.

### `gan_loss_type` and `loss_d.loss_type` (`lsgan`)

`lsgan` is the original. `hinge` / `soft_hinge` are the pairing that goes with SAN
heads; `soft_hinge`'s generator term is `softplus(-D)`, whose gradient is bounded,
so a critic that pulls ahead cannot hand the generator an arbitrarily large step
the way LSGAN's quadratic does.

`relativistic` is RpGAN (R3GAN): the critic judges the per-sample **difference**
`D(real_i) - D(fake_i)` rather than each against an absolute 0/1 boundary. This is
only meaningful when `real_i` and `fake_i` are the same item, which holds here —
the generator's input is a degraded copy of the very target it is scored against.
Pair it with `r1_gamma` *and* `r2_gamma`; that combination is the whole R3GAN
recipe.

The two settings must match. `loss_d.loss_type: relativistic` with
`loss_g.gan_loss_type: lsgan` is not a valid game.

### `disc_branch_every` / `disc_branch_offset`

Per-branch schedules, matched by glob against the critic's branch names
(`disc.names()` — `mpd_2`, `mrd_2048`, `msstft_512`, `med_8b_64`, `transient`, …):

```yaml
disc_branch_every:
  'msstft_*': 2        # complex-STFT branches: every other update
  'periodicity': 4
disc_branch_offset:
  'periodicity': 2     # ...and on update 2 of every 4, not update 0
```

`disc_branches_per_step` rotates a fixed *count* of branches, which implicitly
claims every branch costs the same. They do not — the envelope bank and a 2048
complex-STFT branch are worth several of the cheap ones, and those are exactly the
ones worth running at 1/N rate while the cheap branches supervise every update.
Offsets let you stagger them so no single update pays for all the expensive ones.

The interval counts **discriminator updates**, so it means what it says under
gradient accumulation. First matching pattern wins; a branch matching nothing runs
every update; a set of offsets that would leave an update with no branches at all
falls back to running everything. If nothing matches any pattern the schedule is
silently a no-op, so the module raises at fit start when the critic exposes no
branch names.

Set `disc_branches_per_step: 0` when using this — the schedule takes precedence.

### `disc_concat_real_fake` (false)

The discriminator step runs `cat([real, fake])` through the critic as one batch
instead of two separate forwards. Every branch operation is per sample, so this is
the same arithmetic with half the kernel launches and half the STFT front-ends.
The split is taken at the midpoint of the output batch, so it is correct whatever
channel folding the bank applies.

Only the discriminator step. The generator step still needs the two sides
separated, because the real side runs under `no_grad`.

### `val_empty_cache` (false)

Validation runs `eval_segments` (6 s) against training's 3 s, and each item carves
its own allocator block sizes. Those blocks stay reserved and are the wrong shape
for training — this is the VRAM that climbs after a validation pass and never comes
back down. Costs a few milliseconds of re-allocation per check interval.

## Measured cost, v1 critic

Mono, 3 s segments, gradient checkpointing on, RTX 5060 8 GB. Each row adds to
the one above:

| configuration | step | peak VRAM |
| --- | --- | --- |
| upstream: 7 windows, all branches | 1768 ms | 2530 MB |
| + no-grad target features | 1784 ms | 2520 MB |
| + 5 windows (drop 32/64) | 1646 ms | 2421 MB |
| + rotate 3 of 5 branches | 1483 ms | 2346 MB |
| + lazy R1 (gamma 1, every 16) | 1477 ms | 2346 MB |

**1.20x faster overall, 7% less VRAM.** Modest, and worth being clear about why:
with gradient checkpointing on, the generator dominates the step, so changes to
the critic can only move so much. EMA, R1 and the warmup — the three that actually
target the noise — are free.

Run-to-run variance on this machine is roughly ±8%, which is larger than some of
the individual rows above. Do not read too much into a single measurement.

## Measured cost, v2 critic

**Stereo**, 3 s segments, gradient checkpointing on, RTX 5060 8 GB, the 18-branch
bank from `apollo_v2_restoration.yaml` (12.65M critic parameters). The generator
now stops dominating the step, so the numbers move much more than above.

> On Windows/WDDM an over-budget allocation does **not** raise — the driver spills
> into shared system memory and the run keeps going, very slowly, showing roughly
> twice the card in Task Manager. If a configuration looks like it fits but crawls,
> that is what is happening. `torch.cuda.set_per_process_memory_fraction(0.9)`
> turns it back into an honest OOM, which is how the table below was measured.

**Discriminator step, peak per branch** (real and fake concatenated):

| branch | peak | branch | peak |
| --- | --- | --- | --- |
| `mrd_512` | 1400 MB | `med_8b_64` | 443 MB |
| `mrd_1024` | 1389 MB | `transient` | 308 MB |
| `mrd_2048` | 1387 MB | `mel_2048` | 230 MB |
| `msstft_*` | ~740 MB | `periodicity` | 224 MB |
| `mpd_*` | ~670 MB | `med_16b_1024` | 223 MB |
| | | `med_8b_256` | 209 MB |
| | | `hf_modulation` | 180 MB |

Branch-wise, the step peaks at the largest single branch: **1400 MB**. Running all
18 in one forward needs roughly the sum, **11.6 GB** — more than the card. At this
branch count `disc_branchwise: true` is not an optimisation, it is a requirement.

Note where the cost is: the three `mrd_*` branches account for more peak than
every dynamics branch combined. The four dynamics families together add about
1.3 GB of *summed* branch cost and nothing at all to the branch-wise peak, because
none of them is the largest branch.

**Generator step**, all 18 branches:

| configuration | step | peak VRAM |
| --- | --- | --- |
| per-branch backward | 12304 ms | 4003 MB |
| `gstep_output_grad: true` | 1303 ms | 2565 MB |
| + `use_loss_balancer: true` | 1291 ms | 2638 MB |
| + `autocast_dtype: bf16` | 973 ms | 1860 MB |

**9.4x faster and 36% less memory** for the output-gradient step. That is far more
than the arithmetic suggests, and the reason is gradient checkpointing: each of the
18 per-branch backwards re-ran the generator's forward. The two multiply. The
balancer costs about 3% on top — it takes one extra `autograd.grad` per term, all
of which stop at the output.

The gradients are identical. On GPU that can only mean "within the same path's own
run-to-run noise", and it is: re-running the per-branch path twice gives a maximum
relative deviation of 6.1e-4 between its own two runs, against 6.3e-4 between the
two paths (8.4e-3 against 9.5e-3 under bf16). `tests/test_gan_v2.py` asserts the
exact equality on CPU, where the reduction order is deterministic.

**Gradient penalty**, per branch, branch-wise: peaks at 1562 MB (`mrd_512`), about
11% above that branch's ordinary discriminator step. It runs on one update in
`r1_every`, so this is not the step you budget for.

**Complete training step** — the number to actually budget against. Batch 1,
`accumulate_grad_batches: 8`, critic active, everything the config ships with:

Peak allocated, measured at both precisions because the answer differs:

| configuration | fp32 | bf16 (shipped) |
| --- | --- | --- |
| ordinary step | ~2950 MB | **2130 MB** |
| step that fires the penalty, separate R1/R2 | ~2900 MB | 2263 MB |
| step that fires the penalty, `r1_r2_joined` | — | 3648 MB |
| `gstep_output_grad: false` | 4194 MB | 2848 MB |
| `disc_branchwise: false` | **OOM** above 8 GB | 4744 MB |

Step time, same configuration: 1646 ms fp32, 1138 ms bf16, 1126 ms bf16 + fused
AdamW. Run-to-run variance on the memory figures is a few percent.

The shipped configuration sits at about 2.1 GB allocated on an ordinary step. The
run's true high-water mark is the regularisation step at 3.6 GB, one update in
`r1_every`; that is what to budget against, and it still leaves half the card.

**`disc_branchwise` is a hard requirement in fp32 and a strong preference in
bf16.** The per-branch table above is fp32: summed, those 18 branches want about
11.6 GB, which is more than the card, and the run OOMs. bf16 roughly halves the
critic's activations, and the all-at-once step then fits at 4744 MB. It is still
worth keeping on — 2130 against 4744 MB is 55% of the peak for no change in the
gradients — but it stops being the thing that decides whether the run starts at
all. If you go back to fp32, it is.

`gstep_output_grad: true` is worth 718 MB under bf16 (2130 against 2848) and, far
more importantly, 9.4x on the generator step.

### `autocast_dtype: bf16`

**1.45x faster and 26% less memory** on the full step. The v1 configs leave this
at `none` with a comment saying bf16 is "not faster here" — that was true of their
5-branch critic, where the generator dominated the step. With the 18-branch bank
the critic is a large share of it, and bf16 pays.

Four things make it safe for this recipe, and they are worth knowing because three
of them are properties of the code rather than of bf16:

- bf16 carries fp32's exponent range, so there is no `GradScaler` and no overflow
  path. What you trade is mantissa: about three significant decimal digits.
- The gradient penalty runs **outside** the autocast context by construction, so
  its double backward stays fp32. Forced into bf16 it gives 2.7217e-3 against
  2.7154e-3 — it survives, but there is no reason to take the risk.
- The dynamics branches disable autocast on their own front-ends, so the sinc
  filterbanks, envelope detectors and FFTs stay fp32 regardless.
- **The loss balancer keeps its statistics in fp32 regardless of the input dtype.**
  This one is not cosmetic. At `ema_decay: 0.999` each update moves the estimate by
  one part in a thousand, which rounds away entirely in three significant digits:
  measured, **0 of 200 updates moved a bf16 EMA, against 200 of 200 in fp32**. The
  balancer would have frozen at its first estimate, silently, for the whole run.
  Today the generator's output arrives as fp32 even under autocast so the hazard
  does not fire on its own, but that depends on the generator's last operation —
  which is exactly the kind of thing that changes.

What you do accept is gradient noise: about 1e-2 relative under bf16 against 6e-4
in fp32. Set `autocast_dtype: none` if the adversarial game destabilises.

### `fused: true` on the optimizers

One kernel for the whole parameter update instead of one per tensor. On this
model's 22.1M parameters: **4.74 ms → 1.65 ms per `optimizer.step()`**.

Be clear about what that is worth. With `accumulate_grad_batches: 8` the optimizers
step once per 8 micro-batches, so end to end it is **1138 ms against 1126 ms** —
about 0.07%, inside the run-to-run noise. It is on because it is free and correct,
not because it is a win at this accumulation setting. It starts to matter if you
lower `accumulate_grad_batches`.

Fused optimizers need one piece of support from the loop: Lightning refuses its
generic `clip_gradients` helper for optimizers that advertise AMP scaling, which
fused AdamW does. The module detects that and uses PyTorch's native norm clipping
instead, unscaling first when a scaler is present.

Note the gap between allocated and reserved — 2901 against 4440 MB. Reserved is
what the driver and Task Manager report. If you are reading VRAM from Task Manager
rather than from `torch.cuda.max_memory_allocated`, expect the larger number, and
expect it to grow further after a validation pass unless `val_empty_cache: true`.

## Tuning order

If the trained model still sounds wrong, change one thing at a time:

1. **Noisy / grainy** → confirm you are exporting EMA weights, then raise
   `r1_gamma` to 2-4, then drop the shortest `window` entries.
2. **Muffled / over-smooth** → lower `waveform_weight` (it is the term that
   trades detail for phase accuracy), then lower `r1_gamma`.
3. **Metallic / ringing** → usually the critic winning. Lower `optimizer_d.lr`,
   or raise `disc_start_step`.
4. **Unstable, loss spikes** → raise `accumulate_grad_batches` before touching
   the learning rate.
5. **Smeared attacks, static-sounding hiss in the pauses** → these are dynamics
   failures, and no weight above addresses them. Turn on `use_med` /
   `use_transient` / `use_hf_modulation`.

`train/adv`, `train/fm`, `train/r1` and `train/adv_scale` are logged separately
for exactly this. A six-term objective cannot be tuned from the sum.

**With `use_loss_balancer: true`, tune `balancer_weights`, not the loss weights.**
Steps 1 and 2 above become "raise or lower that term's entry in
`balancer_weights`", and the number to watch is `train/bal_<term>` — the effective
gradient norm — rather than the raw loss. The `loss_g:` weights no longer do
anything, because a constant prefactor cancels out of the balancer.

## Extending the critic

`MultiFrequencyDiscriminator.forward` returns `(outputs, feature_maps)` as lists,
one entry per active branch, and the losses only ever iterate those lists. So a
new critic needs to satisfy very little:

```python
class MyDiscriminator(nn.Module):
    def forward(self, est, sample_rate=44100, branches=None):
        # est: (B, nch, samples)
        return outputs, feature_maps   # list[Tensor], list[list[Tensor]]
```

Point `discriminator._target_` at it. Add a `branch_names` list (or a `names()`
method) if you want `disc_branch_every` to be able to address it.

Most of this list is now built. `build_fast_discriminator` covers the magnitude,
mel, period, complex-STFT and SAN entries; what follows is what the bank contains
and what each part is for.

### The dynamics branches (`dynamics.py`)

Every branch listed above judges a **static picture**: what a frame contains, or
what the waveform looks like folded by a period. None of them can express how
energy *moves* over tens or hundreds of milliseconds.

That is a strange gap for a restoration model, because dynamics is most of what a
low-bitrate codec destroys. Pre-echo spreads a transient backwards across its
window. Consonant attacks are smeared. Quantisation noise fills the pauses with a
hiss that has the right *level* and the wrong *movement*. A per-frame magnitude
critic accepts all three, and the reconstruction terms cannot object either —
`freq_MAE` is a magnitude comparison and `fullness`/`bleedless` are mel-domain
level penalties.

| flag | branch | what it sees |
| --- | --- | --- |
| `use_med` | 3x envelope, ~1.5 / 6 / 23 ms | multi-band envelope, its first difference, and its local contrast |
| `use_transient` | waveform derivative | attacks and pre-echo, at sample resolution |
| `use_hf_modulation` | 2-10 kHz modulation spectrum | whether the high band arrives in bursts or as static noise |
| `use_periodicity` | lag-domain ACF | the over-regular "plastic" harmonic comb |

They are cheap: the analysis front-ends (sinc filterbanks, envelope detectors) have
**no trainable parameters**, and everything after them runs at the envelope rate
rather than the sample rate.

Two of them are deliberately level-invariant by construction — the HF modulation
spectrum is divided by its own DC bin, and the ACF by its zero lag. The generator
cannot satisfy either by adding energy, only by moving it like the real thing.
That is what makes `use_hf_modulation` a sharper tool against invented hiss than
raising `bleedless_weight`, which can only ask for *less* energy rather than for
better-shaped energy.
