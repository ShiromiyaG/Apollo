# Vocal / speech restoration mode

A training mode aimed at voice rather than full mixes. The goal is the one people
usually state as three separate complaints — *too many artefacts*, *not clear
enough*, *sounds thin* — which turn out to be three different failure modes of the
same objective.

```bash
python train.py --conf_dir=configs/apollo_v2_vocal.yaml
```

This config uses the v2 architecture and trains from scratch — the released v1
weights cannot be loaded into it, so there is no `--init_from`. Budget for a real
run.

To go back to fine-tuning the released checkpoint instead, point `model` at
`look2hear.models.apollo.Apollo`, drop `optimizer_g.lr` to 2e-4, and add
`--init_from apollo_model.ckpt`. Everything else in the config — the objective
below, the critic, and the GAN settings in `docs/gan.md` — applies unchanged to
either generator.

## Why the stock objective is not enough for voice

Apollo's generator loss is `freq_MAE + adversarial + feature_matching`. Two gaps
matter here:

**It only compares magnitudes.** `freq_MAE` takes `.abs()` of both spectra. Phase
is left entirely to the discriminator, which constrains it only as far as it can
tell real from fake. Whatever is left unconstrained comes out as a diffuse,
noise-like residual spread across the band. This is the single biggest structural
reason restored output sounds grainy.

**It is dominated by whatever is loudest.** An L1 on linear magnitudes is driven
by the high-energy bins. A voice's character lives in low-energy detail —
consonants, breath, the upper harmonics that make it sound present — and those
barely move the number. A model can score well and still sound dull and thin.

## The added terms

All weights live in `loss_g` in the config. Setting every added weight to `0`
reproduces the original `MultiFrequencyGenLoss` exactly (there is a test for
this), so you can always bisect back.

> **With `use_loss_balancer: true` — which `apollo_v2_vocal.yaml` ships with —
> these `loss_g` weights no longer set the balance.** `system.balancer_weights`
> does, and the defaults quoted in the headings below are the *unbalanced* ones.
> The distinction is not bookkeeping. A `loss_g` weight fights the term's raw
> gradient scale, and those scales differ by orders of magnitude: `fullness`
> measures a gradient norm around 3.2e-2 against `freq`'s 6.7e-4, roughly 50x.
> That is the whole reason `fullness_weight` had to be 0.02 while
> `clarity_weight` was 0.5 — it says nothing about which matters more. The
> balancer divides the scale out, so `balancer_weights` are pure priorities and
> carrying the old ratios across would demote `fullness` to a twentieth of
> `clarity` in the one mode whose purpose is a fuller voice. Read the headings
> below for what each term *does*; take the numbers from `balancer_weights`.

### `fullness_weight` (default 0.02)

Mel-dB penalty on bins where the **target is louder than the estimate** — content
the model failed to reproduce. This is the differentiable twin of the `fullness`
metric from Music-Source-Separation-Training, computed in the same mel-dB domain,
so what you optimise is what you measure.

Raising it pushes the model to fill in the harmonics and detail a codec stripped
out. That is what reads as a voice sounding *full* rather than thin.

### `bleedless_weight` (default 0.01)

The mirror image: penalises energy the model **added** that the target does not
have. That is the definition of an artefact.

These two are a matched pair and the balance between them is the knob that
actually matters. `fullness` alone will happily buy you a fuller voice by
inventing hiss — it cannot tell the difference between restored harmonics and
noise, because both add energy. `bleedless` is what makes the trade honest. Keep
`bleedless` at roughly half of `fullness` as a starting point:

- output sounds thin or dull → raise `fullness_weight` (try 0.04)
- output sounds hissy or grainy → raise `bleedless_weight` (try 0.02)
- both → raise both, keeping the ratio

Change one at a time. `train/fullness` and `train/bleedless` are logged
separately for exactly this reason, and `val_fullness_penalty` /
`val_bleedless_penalty` track it on the validation set.

### `clarity_weight` (default 0.5)

L1 on log-mel magnitudes, weighted 1.5x over 300 Hz – 8 kHz. The log domain gives
quiet detail the same say as loud detail, which is where consonant definition
lives. This is what moves "clear" as opposed to "full".

### `waveform_weight` (default 0.5)

Plain L1 in the time domain. This is the term that constrains phase, which
nothing else in the objective does. A small weight is enough to kill the diffuse
residual; a large one over-smooths and costs you the high-frequency detail the
adversarial term is there to generate. Do not push this past ~1.0.

## Fitting in 8 GB

Measured on an RTX 5060 (8 GB), one full training step including the
discriminator pass:

| configuration | peak VRAM | step time |
| --- | --- | --- |
| mono 3 s, no checkpointing | **out of memory** | — |
| mono 3 s, checkpointing | 2.5 GB | 1.8 s |
| mono 3 s, checkpointing + bf16 | 1.6 GB | 1.9 s |
| stereo 3 s, checkpointing | 4.4 GB | — |

`gradient_checkpointing: true` is what makes this possible at all — without it a
single mono 3-second step does not fit. It recomputes each band-split block during
the backward pass instead of storing its activations, costing roughly one extra
forward pass. The result is exact, not approximate (tested).

`autocast_dtype: bf16` saves a further 35% of memory but is **not faster** here —
with checkpointing on, recompute dominates and the casting overhead cancels the
gain. Treat it as a memory knob only, and A/B the output quality before trusting
it on a real run.

At 2.5 GB there is headroom on an 8 GB card. Spend it on longer segments before
stereo: `segments: 6.0` gives the sequence model more context, which matters more
for voice than a second channel does.

`accumulate_grad_batches: 8` gives an effective batch of 8 at the memory cost of
1. If training is unstable, raise it rather than the learning rate.

## Data

`datas.train_dir` is a folder of **clean** recordings; degraded copies are made on
the fly. Files must already be at `sr` (44100 by default) — mismatched files are
skipped rather than silently resampled, so check the log if your epoch looks
empty.

Segments are seeked, not loaded whole, so long files cost the same per sample as
short ones. Crops quieter than `min_active_db` are rejected and redrawn, which
keeps silent gaps out of the training set.

The degradation chain covers what actually happens to voice recordings:

| knob | default | note |
| --- | --- | --- |
| `codec_prob` | 0.9 | codec round-trip, drawn from `codec_formats` |
| `codec_formats` | `[voice]` | a bundle; see below |
| `codec_weights` | — | how often each codec is drawn; uniform if unset |
| `bitrates` | 16–96 kbps | clamped per codec into the range it is specified for |
| `bandlimit_prob` | 0.3 | direct low-pass, cheaper than a real encode |
| `clip_prob` | 0.1 | clipped recordings |
| `quantize_prob` | 0.1 | bit-depth reduction |

### Which codecs

A restoration model only learns to undo the artefacts it was shown. MP3 and
Vorbis alone teach the MDCT family's failure modes — pre-echo, spectral holes,
joint-stereo collapse — and leave the model guessing at everything else.
`look2hear/datas/codec.py` covers four families that fail in genuinely different
ways:

| family | codecs | failure mode |
| --- | --- | --- |
| MDCT transform | `mp3` `aac` `mp2` `ac3` `wma` | pre-echo, spectral holes |
| hybrid | `opus` `ogg` | mode-switching; one bitrate, two artefacts |
| narrowband speech | `amr_nb` `amr_wb` `gsm` `g722` `speex` | everything above 4–8 kHz is gone |
| ADPCM / companding | `g726` `mulaw` `alaw` | quantisation noise, not removal |

Name a bundle instead of a list: `voice` (everything a voice recording plausibly
went through, phones included), `music` (no narrowband — it forces 8–16 kHz mono,
which a music file does not suffer), `narrowband`, `mdct`, `hybrid`, `adpcm`,
`all`.

**Weight them.** Uniform sampling over the 12-entry `voice` bundle puts six
narrowband entries in the draw, so half of every epoch would be telephone audio
and the model would spend its capacity on bandwidth extension from 4 kHz while
treating ordinary 128 kbps MP3 as the edge case. The shipped `codec_weights` land
at roughly 30% narrowband: often enough to learn the hard case, rare enough that
it does not become the task.

Everything is probed at startup. A codec this install cannot encode is dropped
with a warning, and if *none* of them can be encoded the run fails immediately —
because `encode_decode` returns its input unchanged when no backend can handle the
format, which would pair every clean segment with an identical copy and quietly
teach the model the identity mapping. `mp3` and `ogg` need no ffmpeg; everything
else does, and the decode side goes through ffmpeg too, since libsndfile cannot
read AAC, Opus, WMA, AMR or G.722.

Cost, measured: ~21 ms for MP3 through `lameenc`, ~200 ms for anything through
ffmpeg (two subprocesses, encode then decode). At four dataloader workers against
a ~1.1 s training step that is not close to a bottleneck.

For evaluation, either set `eval_degraded_dir` to a folder of real degraded files
matched by relative path, or leave it `null` to degrade the clean set
deterministically with a fixed seed.

## Validation

`val_loss` is negative SI-SDR, so lower is better and checkpoint selection follows
it. `val_fullness_penalty` and `val_bleedless_penalty` are reported alongside it
because SI-SDR on its own will not tell you which of the two directions a
regression went.

## Notes

- `discriminator.nch` must match `datas.channels`.
- `disc_start_step` delays the adversarial term. Raise it if the GAN destabilises
  early in a fine-tune; the reconstruction terms alone are a safe warm-up.
- The music mode (`configs/apollo.yaml`) is untouched and still uses
  `MultiFrequencyGenLoss`.
