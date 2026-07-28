# Apollo v2: generator and critic

v1 stays exactly as it was and still loads the released checkpoint. Everything
here is a separate, from-scratch architecture — that is the trade: no pretrained
weights, so budget for a real training run.

## Where v1 spends its parameters

Measured, not estimated (`sum(p.numel())` per component, 44.1 kHz / 20 ms / 256 /
6 layers):

| component | params | share |
| --- | --- | --- |
| `seq_net` (ICB, temporal) | 9.50M | 57.4% |
| `band_net` MLP | 4.72M | 28.5% |
| `band_net` attention | 1.57M | 9.5% |
| band projections | 0.74M | 4.5% |
| **total** | **16.54M** | |

86% of the model is feed-forward. The largest single block — the ICB — is three
stacked 4x-expansion FFNs whose whole job is a kernel-7 convolution over time: 19
frames (190 ms) per layer, **109 frames (1.09 s)** across the six-layer stack.
9.5M parameters for a second of context is the inefficiency v2 attacks.

The band attention, which is the paper's actual contribution, is under 10% of the
parameters. It is not the problem and v2 keeps it.

## Generator changes

### Temporal path: 3 blocks → 2, kernel 7 → 11, dilations (1, 2)

The depthwise convolution is nearly free in parameters — the FFN either side is
what holds them — so kernel width and dilation are the cheapest knobs in the whole
architecture. Widening them and dropping a block gives **58% fewer parameters and
1.66x the context** (1.09 s → 1.81 s across the stack).

Do not push the context past the training segment length. Context the model never
saw during training is context it cannot use, and it forces a larger overlap at
chunked inference. At the defaults the stack reaches ~1.8 s against a 3 s segment.

### SwiGLU at conventional width

v1 builds its feed-forward as `Conv1d(N, N*8) → SiLU → chunk → SiLU(gate) * z`.
Two things are off: the expansion is roughly 3x wider than a standard SwiGLU, and
SiLU lands on the gate branch twice (once inside the `Sequential`, once in
`forward`). v2 uses the conventional 2.67x and applies the gate once.

### Residual output

This is the change aimed squarely at the noise.

v1 synthesises the entire complex spectrum from scratch, including the low band
the codec left intact, and `freq_MAE` compares magnitudes only — nothing in the
reconstruction loss constrains phase. So every small error in a band that was
already correct comes back as broadband, noise-like residual.

v2 predicts a *correction*: `est_spec = input_spec + net(input_spec)`. Leaving a
band alone is now the cheap default rather than something the network has to learn
to reproduce. `output_init_scale` starts the output head small so training begins
near pass-through, which also converges much faster than learning identity from a
standard init.

`output_mode: direct` reproduces v1's behaviour if you want to compare.

### Optional mid/side

`stereo_mode: ms` converts to mid/side before analysis. Lossy codecs joint-stereo
code, so their artefacts are structured in M/S rather than L/R — processing in the
same domain as the damage. Ignored for mono input.

### Measured

Stereo, 3 s, inference, 128 MB slice budget, RTX 5060:

| model | params | context | time | peak |
| --- | --- | --- | --- | --- |
| v1 (256 / 6 layers) | 16.54M | 1.09 s | 537 ms | 604 MB |
| v2 (same dims) | 9.46M | 1.81 s | 351 ms | 757 MB |

**43% fewer parameters, 1.53x faster.** The peak-memory figure is not directly
comparable — v2's slice budget is derived from its real layer widths, so it slices
less eagerly at the same nominal budget; lower `--vram_budget` to match.

## Critic changes

The critic is discarded after training, so there is no compatibility cost to
changing it at all. This is the freest part of the whole system and the place
where a swap pays off most.

### What was wrong with the stock one

`MultiFrequencyDiscriminator` is seven STFT resolutions of 2D convolutions on
real/imaginary input, with `hidden_channels` hard-coded to 8. Two problems: the
two shortest windows (32, 64) are 17 and 33 bins over well under a millisecond —
they cannot see spectral structure and mostly reward high-frequency energy — and
the critic is small enough that it has limited capacity to say anything.

### The families now available

Ported from the `SingingVocoders` recipe (`modules/fast_D`, `modules/chouwa_D`,
`modules/bigvgan_D`), all following the same `(outputs, feature_maps)` +
`branches` protocol so any of them can be swapped in from config:

| family | domain | sees phase? | what it catches |
| --- | --- | --- | --- |
| `FastMPD` | period (no windowing) | no | buzz, pitch-domain artefacts |
| `FastMRD` | multi-resolution magnitude | no | spectral balance across time scales |
| `FastMelBank` | log-mel | no | harmonic structure, CQT-like emphasis |
| `MultiScaleSTFTDiscriminator` | complex STFT | **yes** | phase, HF detail |
| `MultiFrequencyDiscriminator` | complex STFT | yes | the v1 critic, kept |
| `MultiEnvelopeDiscriminator` | band envelopes, 3 time constants | no | smeared attacks, pre-echo, plastic dynamics |
| `TransientDiscriminator` | waveform derivative | no | attacks at sample resolution |
| `HFModulationDiscriminator` | HF modulation spectrum | no | high band that is static noise at the right level |
| `PeriodicityACFDiscriminator` | lag domain | no | the over-regular harmonic comb |

The first five all judge a **static picture** — what a frame contains, or what the
waveform looks like folded by a period. The last four judge **dynamics**: how the
energy moves over tens to hundreds of milliseconds. That distinction matters more
for a restoration model than for a vocoder, because dynamics is most of what a
low-bitrate codec destroys, and nothing else in the objective can express it —
`freq_MAE` is a magnitude comparison and `fullness`/`bleedless` are level
penalties. See `look2hear/discriminators/dynamics.py`.

Two design choices carry most of the benefit:

**Space-to-depth instead of strided convolutions.** A textbook HiFi-GAN MPD
downsamples with stride-3 convolutions and doubles channels each time, ending at
1024 — **41M parameters** measured. `FastMPD` gets the same effect with a `view`:
time folds into the feature axis, which is free and discards nothing. Measured at
7.1M for the same five periods.

**RMSNorm instead of weight/spectral norm.** Spectral norm runs a power iteration
on every forward, and the critic is called four times per training step.

**SAN heads** (`use_san: true`, default) L2-normalise the final projection, making
each branch's logit a direction-only similarity. That bounds it without spectral
norm or a gradient penalty — the cheapest available answer to a critic that
overpowers the generator.

### Two traps worth knowing

**`freq_stride`.** A first pass at the complex-STFT critic strided only the time
axis, leaving every layer running over all `n_fft/2+1` frequency bins. The deep
wide layers then dominate everything: it measured **52 seconds per step and 5.3 GB**.
Striding frequency as well brought it to 145 ms and 1.4 GB — a 350x difference
from one parameter. Layer-1 feature maps still supervise at full resolution
through feature matching, so acuity is not lost.

**`mag_compression`.** With linear real/imaginary input, bins 30–50 dB below the
fundamental barely move the logit, so a nominally full-band branch behaves as a
low-band critic. For Apollo that is exactly backwards — the high band is what is
being reconstructed and where the artefacts are. Compressing to
`mag^alpha · e^(i·phase)` keeps phase intact while giving quiet bins real gradient
weight. Default 0.3; 1.0 disables it.

### Measured

Mono, 1.5 s, forward+backward, capped at ~3 GB so nothing can thrash:

| critic | params | branches | time | peak (all-at-once) | peak (branch-wise) | phase |
| --- | --- | --- | --- | --- | --- | --- |
| v1 MultiFrequency (7 win) | 2.77M | 7 | 85 ms | 192 MB | 74 MB | yes |
| v1 MultiFrequency (5 win) | 1.98M | 5 | 60 ms | 138 MB | 66 MB | yes |
| FastMPD | 7.10M | 5 | 65 ms | 377 MB | 140 MB | no |
| FastMRD | 1.23M | 3 | 32 ms | 547 MB | 227 MB | no |
| FastMel | 0.41M | 1 | 11 ms | 37 MB | 37 MB | no |
| MS-STFT-D | 1.10M | 3 | 26 ms | 388 MB | 182 MB | yes |
| **MPD+MRD+Mel+MS-STFT** | **9.84M** | **12** | **96 ms** | 1225 MB | **294 MB** | yes |

The full four-family bank costs about the same wall-clock as the stock v1 critic
and, with branch-wise stepping, less than 300 MB. There is no reason to run a
narrow critic here.

`FastMel` is the standout on value: 0.41M parameters and 11 ms for a
log-frequency view.

The four dynamics families are not in this table because they were added later and
measured under the config's real conditions (stereo, 3 s, 18 branches) rather than
these. Their per-branch peaks are in `docs/gan.md`; the short version is that the
most expensive of them (`med_8b_64`, 443 MB) is a third of the largest `FastMRD`
branch, and none of them is ever the branch that sets the branch-wise peak. They
are close to free at the peak that matters.

### Branch-wise stepping

`disc_branchwise: true` runs forward+backward one branch at a time, so peak memory
tracks the largest single branch rather than the sum: **1225 MB → 294 MB (−76%)**
for the full bank, at no measurable cost in time.

The gradients are identical — summing per-branch backwards is the same as one
backward over the sum. The loss is split into its reconstruction part (evaluated
once) and its adversarial part (per branch, weighted by `n_branches` so each keeps
the weight it would have had in a full pass). There is a test asserting the split
sums back to the combined value.

> A word on measuring this. An earlier version of this table was taken without a
> memory cap, and the same configuration measured 91 ms on one run and 9314 ms on
> another — the card was near its limit and the allocator was thrashing, which
> looks exactly like a compute cost until you check `max_memory_reserved` and free
> VRAM. If you benchmark on a small card, cap the process and leave headroom, or
> you will end up designing around an artefact.

## Tuning order

1. **Noisy / grainy** → confirm EMA weights are what you exported; raise
   `r1_gamma`; check `mag_compression` is below 1.0.
2. **Dull, missing high end** → the critic is probably acting low-band. Lower
   `mag_compression` further (0.2), or add the mel branch.
3. **Buzzy / metallic** → add or widen `FastMPD`; this is the period domain's job.
4. **Smeared attacks / static hiss in the pauses / plastic dynamics** → the
   dynamics families. `use_med` and `use_transient` for the first,
   `use_hf_modulation` for the second, `use_periodicity` for the third. No
   spectral branch and no loss weight can express any of them.
5. **Out of memory** → `autocast_dtype: bf16` first: it is the single largest win
   (−26% on the full step) and it is also faster. Then `disc_branchwise: true` —
   in fp32 this is not optional at 18 branches, since the all-at-once step needs
   about 11.6 GB against 1.4 GB branch-wise; in bf16 it is a 55% saving rather
   than a requirement. Then `gstep_output_grad: true`, then drop `use_mrd` (the
   branch family that overlaps most with the mel one and costs the most), then
   shorten `segments`. The dynamics families are the last thing worth cutting:
   together they add nothing to the branch-wise peak.
