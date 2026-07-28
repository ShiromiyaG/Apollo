<p align="center">
  <img src="asserts/apollo-logo.png" alt="Logo" width="150"/>
</p>

<p align="center">
  <strong>Kai Li<sup>1,2</sup>, Yi Luo<sup>2</sup></strong><br>
    <strong><sup>1</sup>Tsinghua University, Beijing, China</strong><br>
    <strong><sup>2</sup>Tencent AI Lab, Shenzhen, China</strong><br>
  <a href="https://arxiv.org/abs/2409.08514">ArXiv</a> | <a href="https://cslikai.cn/Apollo/">Demo</a>

<p align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=JusperLee.Apollo" alt="访客统计" />
  <img src="https://img.shields.io/github/stars/JusperLee/Apollo?style=social" alt="GitHub stars" />
  <img alt="Static Badge" src="https://img.shields.io/badge/license-CC%20BY--SA%204.0-lightgrey">
</p>

<p align="center">

# Apollo: Band-sequence Modeling for High-Quality Audio Restoration

## 📖 Abstract

Audio restoration has become increasingly significant in modern society, not only due to the demand for high-quality auditory experiences enabled by advanced playback devices, but also because the growing capabilities of generative audio models necessitate high-fidelity audio. Typically, audio restoration is defined as a task of predicting undistorted audio from damaged input, often trained using a GAN framework to balance perception and distortion. Since audio degradation is primarily concentrated in mid- and high-frequency ranges, especially due to codecs, a key challenge lies in designing a generator capable of preserving low-frequency information while accurately reconstructing high-quality mid- and high-frequency content. Inspired by recent advancements in high-sample-rate music separation, speech enhancement, and audio codec models, we propose Apollo, a generative model designed for high-sample-rate audio restoration. Apollo employs an explicit **frequency band split module** to model the relationships between different frequency bands, allowing for **more coherent and higher-quality** restored audio. Evaluated on the MUSDB18-HQ and MoisesDB datasets, Apollo consistently outperforms existing SR-GAN models across various bit rates and music genres, particularly excelling in complex scenarios involving mixtures of multiple instruments and vocals. Apollo significantly improves music restoration quality while maintaining computational efficiency.

## 🔥 News

- [2025.03.07] We released the training data preprocessing code on [Apollo-data-preprocess](https://github.com/JusperLee/Apollo-data-preprocess).
- [2024.09.10] Apollo is now available on [ArXiv](#) and [Demo](https://cslikai.cn/Apollo/).
- [2024.09.10] Apollo checkpoints and pre-trained models are available for download.

## ⚡️ Installation

No conda required. Install PyTorch for your CUDA version first, then the rest:

```bash
git clone https://github.com/JusperLee/Apollo.git && cd Apollo

# pick the index-url that matches your GPU / driver
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt                       # inference
pip install -r requirements.txt -r requirements-train.txt   # + training
```

Inference needs only `torch`, `torchaudio`, `numpy`, `soundfile` and
`huggingface_hub`. Audio I/O goes through libsndfile rather than TorchCodec, so
WAV/FLAC/OGG/MP3 work out of the box with no ffmpeg install; ffmpeg is used only
as a fallback for containers libsndfile cannot read (m4a, aac).

The old `look2hear.yml` is a conda spec-list pinned to `linux-64` and does not
resolve on Windows or macOS. It is kept only for reference.

## 🖥️ Usage

### 🎧 Restoring audio

```bash
# one file
python inference.py --in_wav input.mp3 --out_wav restored.wav --ckpt apollo_model.ckpt

# a whole folder, recursively
python inference.py --in_dir ./lossy --out_dir ./restored --ckpt apollo_model.ckpt
```

`--ckpt` accepts a Lightning `.ckpt`, a serialised `.pth`, a bare state dict or a
`.safetensors` file; the architecture hyper-parameters are read off the weights,
so `--win`/`--feature_dim`/`--layer` never have to be guessed. Omit `--ckpt` to
download from `--repo_id`.

**Memory.** Peak VRAM is set by `--chunk` and `--vram_budget`, not by track
length, so a 4-minute song costs the same as a 10-second clip. The defaults stay
around 1.3 GB. If you are tight, lower `--chunk` first.

| flag | effect |
| --- | --- |
| `--chunk 10` | seconds per model pass; VRAM scales with this |
| `--vram_budget 128` | MB ceiling on the widest intermediate tensor; bit-exact |
| `--batch_size 1` | chunks per forward; raise only if VRAM allows |

**If the output sounds noisy.** Apollo regenerates the *whole* spectrum, low
frequencies included, and its training loss (`freq_MAE`) compares magnitudes
only — nothing ties the reconstructed phase to the input. What that leaves behind
is a diffuse, noise-like residual. Three flags address it directly:

| flag | what it does | when to use it |
| --- | --- | --- |
| `--crossover 3000` | keeps the input's own low band (magnitude *and* phase) below 3 kHz and the model's output above it | the usual first fix; the low band of a decent-bitrate file was already correct |
| `--gate` | fades the model's contribution out where the input is near-silent | hiss in silence, fade-outs, gaps between phrases |
| `--dry_wet 0.8` | linear blend back towards the input | when the restoration is too aggressive overall |

`--normalize chunk` (the default) peak-normalises each chunk before the model and
undoes the gain after. Apollo was trained exclusively on segments normalised to a
peak of 1.0, so quiet material fed at its natural level is out of distribution —
this is a correctness fix, not a preference, and it matters most on quiet tracks.

### 🗂️ Datasets

Apollo is trained on the MUSDB18-HQ and MoisesDB datasets. To download the datasets, run the following commands:

```bash
wget https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1
wget https://ds-website-downloads.55c2710389d9da776875002a7d018e59.r2.cloudflarestorage.com/moisesdb.zip
```
During data preprocessing, we drew inspiration from music separation techniques and implemented the following steps:

1. **Source Activity Detection (SAD):**  
   We used a Source Activity Detector (SAD) to remove silent regions from the audio tracks, retaining only the significant portions for training.

2. **Data Augmentation:**  
   We performed real-time data augmentation by mixing tracks from different songs. For each mix, we randomly selected between 1 and 8 stems from the 11 available tracks, extracting 3-second clips from each selected stem. These clips were scaled in energy by a random factor within the range of [-10, 10] dB relative to their original levels. The selected clips were then summed together to create simulated mixed music.

3. **Simulating Dynamic Bitrate Compression:**  
   We simulated various bitrate scenarios by applying MP3 codecs with bitrates of [24000, 32000, 48000, 64000, 96000, 128000]. 

4. **Rescaling:**  
   To ensure consistency across all samples, we rescaled both the target and the encoded audio based on their maximum absolute values.

5. **Saving as HDF5:**  
   After preprocessing, all data (including the source stems, mixed tracks, and compressed audio) was saved in HDF5 format, making it easy to load for training and evaluation purposes.

### 🚀 Training

Two domains, and for each one a fine-tune of the released v1 checkpoint or a v2
run from scratch. Prepare the data first:

```bash
# stereo, for lossy -> lossless music restoration
python scripts/preprocess.py --mode restoration --in_dir ./raw_music --out_dir ./data/music
python train.py --conf_dir=configs/apollo_restoration.yaml --init_from apollo_model.ckpt

# mono, for vocal / speech
python scripts/preprocess.py --mode vocal --in_dir ./raw_vocals --out_dir ./data/vocals
python train.py --conf_dir=configs/apollo_v2_vocal.yaml
```

The v2 configs (`apollo_v2_restoration.yaml`, `apollo_v2_vocal.yaml`) are a
different architecture, so the released weights cannot be loaded into them and
there is no `--init_from` shortcut — budget for a real training run.
`apollo_restoration.yaml` remains the fine-tune path.

`preprocess.py` resamples, fixes the channel count, drops files that are too
short / silent / already clipped, peak-normalises, splits train/valid and writes a
manifest of what it kept and why. It does **not** degrade anything — the degraded
pairs are generated on the fly during training so every epoch sees different codec
settings.

#### Sourcing the data

One criterion dominates every other: **the clean side must actually be lossless.**
A target that is itself an MP3 teaches the model to reproduce codec artefacts,
which is the exact opposite of the task, and nothing downstream can detect it —
the degradation stage will re-encode it and hand the trainer a pair that is damaged
on both sides. This rules out most of the large public music corpora, which
distribute MP3: **FMA**, **MTG-Jamendo**, anything derived from **AudioSet** or
YouTube, and **GTZAN**.

Lossless sources worth looking at (verify the licence yourself before use — several
are non-commercial or need a request form):

| corpus | size | notes |
| --- | --- | --- |
| **MUSDB18-HQ** | 150 tracks, ~10 h | stereo 44.1 kHz WAV, mixes + stems. The reference set; small |
| **MedleyDB** v1+v2 | ~200 multitracks | 44.1 kHz, request form |
| **MoisesDB** | 240 tracks | 44.1 kHz stems, registration |
| **Slakh2100** | ~145 h | synthesised from MIDI, so lossless by construction and *large* — but sample-library timbres, a real distribution shift. Good for volume, bad as the only source |
| **Cambridge "Mixing Secrets"** | ~500 songs | raw multitracks; you mix them yourself |
| **Live Music Archive** (etree) | tens of thousands of shows | genuinely lossless FLAC, CC-licensed. Live only, quality varies wildly |
| **Jamendo** (direct FLAC, not MTG-Jamendo) | large | CC-licensed; the *research* MTG-Jamendo release is MP3, the site's own downloads are not |

For the vocal config, prefer natively 44.1/48 kHz material: **VCTK 0.92** (48 kHz
FLAC), **DAPS**, **VocalSet**, **OpenSinger** / **M4Singer** / **Opencpop** for
sung voice, or MUSDB18-HQ's vocal stems for voice in context. Avoid **LibriTTS**
here — it is 24 kHz, and upsampling gives every target a hard 12 kHz wall, which
the model will faithfully learn to reproduce.

To check a corpus you already have, `preprocess.py --min_cutoff_hz` looks for the
brick wall a lossy encoder leaves behind. Measured against real encodes of the same
source: MP3 96k cuts at 15.0 kHz, 128k at 16.7, 192k at 18.7, 320k at 20.1; AAC 64k
at 12.4, 128k at 17.3. So `--min_cutoff_hz 19000` drops anything at or below 192
kbps. It is deliberately a **cliff** detector rather than a bandwidth measure, so a
solo piano or an analogue transfer is not mistaken for a transcode — and it is
report-only by default, because it is a heuristic and quietly shrinking someone's
corpus is not its job. It does not catch Vorbis, whose roll-off is gradual, and it
cannot distinguish a near-transparent encode from the real thing.

The original config still works unchanged:

```bash
python train.py --conf_dir=configs/apollo.yaml
```

### 🎛️ The GAN

Both new configs apply the same set of changes to the adversarial setup, aimed at
a *trained* model that is clean rather than one that needs cleaning up afterwards:

| control | default | what it addresses |
| --- | --- | --- |
| `ema_decay` | 0.999 | exported weights are averaged over ~1000 steps, not one adversarial step |
| `r1_gamma` / `r1_every` | 1.0 / 16 | bounds how sharp the critic can get; unbounded critics are what the noise chases |
| `disc_start_step` / `adv_ramp_steps` | 2000 / 2000 | reconstruction settles before the critic votes |
| `disc_branches_per_step` | 3 | rotates the critic's resolution branches; 0.84x step time |
| `window` | starts at 128 | drops two branches too short to see spectral structure |

Measured: 1.20x faster training, 7% less VRAM, with EMA/R1/warmup essentially
free. [docs/gan.md](docs/gan.md) explains how the pieces fit together, what each
knob trades off, and how to plug in a different critic.

### 🎤 Vocal / speech mode

A second training mode aimed at voice rather than full mixes: fewer artefacts, a
clearer and fuller-sounding vocal. It adds `fullness` and `bleedless` penalties
(the differentiable form of the metrics from
[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)),
a vocal-band-weighted clarity term, and a waveform term that constrains phase.

```bash
python train.py --conf_dir=configs/apollo_v2_vocal.yaml
```

Point `datas.train_dir` at a folder of clean vocal or speech recordings — any
nesting, wav/flac/ogg. Degraded copies are generated on the fly, so there is no
preprocessing step and no HDF5 conversion.

This config moved to the v2 architecture, so it trains from scratch rather than
fine-tuning the released checkpoint. To get the fine-tune path back, point `model`
at `look2hear.models.apollo.Apollo` and drop `optimizer_g.lr` to 2e-4 — every
other setting in the file applies unchanged.

Defaults are sized for a single 8 GB GPU: mono, 3 s segments, batch 1 with
gradient accumulation, gradient checkpointing on. Measured on an RTX 5060 (8 GB),
one full training step including the discriminator:

| configuration | peak VRAM | step |
| --- | --- | --- |
| mono 3 s, no checkpointing | out of memory | — |
| mono 3 s, checkpointing | 2.5 GB | 1.8 s |
| mono 3 s, checkpointing + bf16 | 1.6 GB | 1.9 s |
| stereo 3 s, checkpointing | 4.4 GB | — |

See [docs/vocal_training.md](docs/vocal_training.md) for what each loss weight
does and how to tune the fullness/bleedless balance.

### 🎨 Evaluation

```bash
python inference.py --in_wav=asserts/input_wav.wav --out_wav=output.wav --ckpt apollo_model.ckpt
```

### 🧪 Tests

```bash
pip install pytest && python -m pytest tests/ -q
```

## 📊 Results

*Here, you can include a brief overview of the performance metrics or results that Apollo achieves using different bitrates*

![](./asserts/bitrates.png)


*Different methods' SDR/SI-SNR/VISQOL scores for various types of music, as well as the number of model parameters and GPU inference time. For the GPU inference time test, a music signal with a sampling rate of 44.1 kHz and a length of 1 second was used.*
![](./asserts/types.png)

## License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

## Third Party

[Apollo-Colab-Inference](https://github.com/jarredou/Apollo-Colab-Inference)

## Acknowledgements

Apollo is developed by the **Look2Hear** at Tsinghua University.

## Citation

If you use Apollo in your research or project, please cite the following paper:

```bibtex
@inproceedings{li2025apollo,
  title={Apollo: Band-sequence Modeling for High-Quality Music Restoration in Compressed Audio},
  author={Li, Kai and Luo, Yi},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025},
  organization={IEEE}
}
```

## Contact

For any questions or feedback regarding Apollo, feel free to reach out to us via email: `tsinghua.kaili@gmail.com`
