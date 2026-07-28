"""Lossy codec simulation.

``torchaudio.functional.apply_codec`` was deprecated in torchaudio 2.0 and removed
in 2.2; ``torchaudio.io.AudioEffector`` that replaced it is itself gone in 2.9+.
The original data pipeline therefore cannot run on any current install.

This module re-implements the degradation with backends that install from PyPI:
``lameenc`` for MP3 (exact bitrate control, no system packages), ``soundfile`` for
Ogg/Vorbis, and an ffmpeg subprocess for everything else.

A restoration model only learns to undo the artefacts it was shown. Training on
MP3 and Vorbis alone teaches the MDCT family's failure modes -- pre-echo, band
zeroing, joint-stereo collapse -- and leaves the model guessing at everything
else. :data:`CODECS` therefore covers the families a real file is likely to have
been through, and they fail in genuinely different ways:

* **MDCT transform codecs** (mp3, aac, ac3, wma, mp2) -- pre-echo and spectral
  holes;
* **Hybrid speech/music codecs** (opus, vorbis) -- Opus in particular switches
  between a CELT and a SILK mode depending on content, so a single bitrate gives
  two very different artefacts;
* **Narrowband speech codecs** (amr_nb, amr_wb, gsm, g722, speex) -- these throw
  away everything above 4-8 kHz and are what a phone recording or a VoIP capture
  actually went through. For the vocal mode this is the most realistic degradation
  available and the hardest one to invert;
* **ADPCM / companding** (g726, mulaw, alaw) -- quantisation noise rather than
  spectral removal, a failure mode nothing else in the list produces.

Availability is probed once per process; see :func:`available_codecs`.
"""

import io
import logging
import math
import shutil
import subprocess
import tempfile
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

MP3_BITRATES = [24, 32, 48, 64, 96, 128]
SPEECH_BITRATES = [16, 24, 32, 48, 64, 96]


class CodecSpec:
    """How to drive one ffmpeg encoder.

    Args:
        encoder: ffmpeg ``-c:a`` value.
        container: output file extension.
        rates: sample rates the encoder accepts, or None for "anything". The
            closest one at or below the input rate is used, and the decode
            resamples back -- which is precisely the band limiting these codecs
            impose in the wild.
        mono: force a single channel. Every narrowband speech codec requires it.
        bitrates: the range this codec is actually specified for. A codec asked for
            a bitrate it cannot do either fails or silently ignores the request, so
            the caller's choice is clamped into this range.
        vbr_only: pass no ``-b:a`` at all (Vorbis via ffmpeg prefers ``-q:a``).
        fmt: explicit ffmpeg ``-f`` muxer, for containers it cannot infer from the
            extension.
    """

    __slots__ = ("encoder", "container", "rates", "mono", "bitrates", "vbr_only", "fmt")

    def __init__(self, encoder, container, rates=None, mono=False,
                 bitrates=(8, 320), vbr_only=False, fmt=None):
        self.encoder = encoder
        self.container = container
        self.rates = tuple(rates) if rates else None
        self.mono = mono
        self.bitrates = bitrates
        self.vbr_only = vbr_only
        self.fmt = fmt

    def target_rate(self, sr):
        """The rate this encoder will actually run at: the nearest one it accepts.

        Nearest rather than "the largest that fits" because for Opus the nearest
        rate to 44.1 kHz is 48 kHz, and resampling *up* to 48 is precisely what
        every real Opus encoder does. Picking the largest rate at or below the
        input would send it to 24 kHz and simulate a degradation that no Opus file
        ever suffered.
        """
        if not self.rates:
            return int(sr)
        return int(min(self.rates, key=lambda r: abs(r - int(sr))))

    def clamp_bitrate(self, kbps):
        low, high = self.bitrates
        return int(min(max(int(kbps), low), high))


# Registry of the ffmpeg-backed codecs. mp3 and ogg have PyPI backends and are
# handled before this table is consulted; they appear here as ffmpeg fallbacks.
CODECS = {
    # --- MDCT transform codecs -------------------------------------------
    "mp3": CodecSpec("libmp3lame", "mp3", bitrates=(8, 320)),
    "aac": CodecSpec("aac", "aac", fmt="adts", bitrates=(16, 320)),
    "m4a": CodecSpec("aac", "aac", fmt="adts", bitrates=(16, 320)),
    # AC-3 is specified for 32/44.1/48 kHz only; anything else is rejected outright
    "ac3": CodecSpec("ac3", "ac3", rates=(48000, 44100, 32000), bitrates=(32, 640)),
    "eac3": CodecSpec("eac3", "eac3", rates=(48000, 44100, 32000), bitrates=(32, 640)),
    "mp2": CodecSpec("mp2", "mp2", rates=(48000, 44100, 32000, 24000, 22050, 16000),
                     bitrates=(32, 384)),
    "wma": CodecSpec("wmav2", "wma", fmt="asf", bitrates=(32, 320)),
    # --- hybrid speech/music ---------------------------------------------
    # libopus accepts only 8/12/16/24/48 kHz; 44.1 kHz content goes to 48, which
    # is what a real Opus encoder does with it
    "opus": CodecSpec("libopus", "opus", rates=(48000, 24000, 16000, 12000, 8000),
                      bitrates=(6, 256)),
    "ogg": CodecSpec("libvorbis", "ogg", bitrates=(32, 500)),
    "vorbis": CodecSpec("libvorbis", "ogg", bitrates=(32, 500)),
    # --- narrowband speech: the realistic phone / VoIP degradation --------
    "amr_nb": CodecSpec("libopencore_amrnb", "amr", fmt="amr", rates=(8000,),
                        mono=True, bitrates=(5, 12)),
    "amr_wb": CodecSpec("libvo_amrwbenc", "amr", fmt="amr", rates=(16000,),
                        mono=True, bitrates=(7, 24)),
    "gsm": CodecSpec("libgsm", "gsm", rates=(8000,), mono=True, bitrates=(13, 13)),
    "g722": CodecSpec("g722", "wav", rates=(16000,), mono=True, bitrates=(64, 64)),
    "speex": CodecSpec("libspeex", "ogg", fmt="ogg", rates=(32000, 16000, 8000),
                       mono=True, bitrates=(4, 44)),
    # --- ADPCM / companding: quantisation noise, not spectral removal -----
    "g726": CodecSpec("g726", "wav", rates=(8000,), mono=True, bitrates=(16, 40)),
    "mulaw": CodecSpec("pcm_mulaw", "wav", rates=(8000,), mono=True),
    "alaw": CodecSpec("pcm_alaw", "wav", rates=(8000,), mono=True),
}

# Sensible bundles for the configs, so `codec_formats` does not have to be a wall
# of strings. Resolved by `resolve_formats`.
CODEC_BUNDLES = {
    "mdct": ["mp3", "aac", "mp2", "ac3", "wma"],
    "hybrid": ["opus", "ogg"],
    "narrowband": ["amr_nb", "amr_wb", "gsm", "g722", "speex"],
    "adpcm": ["g726", "mulaw", "alaw"],
    # what a music file plausibly went through
    "music": ["mp3", "aac", "ogg", "opus", "mp2", "ac3", "wma"],
    # everything a voice recording plausibly went through, phones included
    "voice": ["mp3", "aac", "ogg", "opus", "wma",
              "amr_nb", "amr_wb", "gsm", "g722", "speex", "g726", "mulaw"],
    "all": sorted(set(CODECS) - {"m4a", "vorbis", "eac3"}),
}


def estimate_delay(reference: torch.Tensor, degraded: torch.Tensor) -> int:
    """Samples by which ``degraded`` lags ``reference``, via GCC-PHAT.

    The phase transform whitens the cross-spectrum, which makes the peak sharp and
    independent of how the codec reshaped the spectrum.
    """
    length = min(reference.shape[-1], degraded.shape[-1])
    ref, deg = reference[..., :length].float(), degraded[..., :length].float()

    cross = torch.fft.rfft(deg, dim=-1) * torch.fft.rfft(ref, dim=-1).conj()
    cross = cross / (cross.abs() + 1e-3)
    cross[..., 0] = 0
    correlation = torch.fft.irfft(cross, dim=-1)

    shift = int(torch.argmax(correlation.abs().mean(0)))
    # the correlation is circular: the upper half represents negative delays
    if shift > length // 2:
        shift -= length
    return shift


def align_to_reference(reference: torch.Tensor, degraded: torch.Tensor) -> torch.Tensor:
    """Undo the encoder delay and return a tensor the same length as ``reference``.

    Lossy encoders prepend priming samples, so a naive decode is offset by up to a
    few hundred samples. Training on misaligned pairs teaches the model to smear
    transients, so this has to happen before the pair is used.

    The correction is a slice, not a roll -- rolling wraps the priming silence onto
    the tail and leaves a step discontinuity there, which shows up as broadband
    high-frequency splatter in exactly the band the model is supposed to rebuild.
    """
    shift = estimate_delay(reference, degraded)
    n = reference.shape[-1]

    if shift >= 0:
        out = degraded[..., shift:shift + n]
    else:
        out = torch.nn.functional.pad(degraded, (-shift, 0))[..., :n]

    if out.shape[-1] < n:
        out = torch.nn.functional.pad(out, (0, n - out.shape[-1]))
    return out


def _to_int16_bytes(wav: torch.Tensor) -> bytes:
    """(nch, n) float -> interleaved int16 PCM."""
    clipped = wav.transpose(0, 1).clamp(-1.0, 1.0).contiguous().numpy()
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _decode(buffer: io.BytesIO, nch: int, sr: int) -> torch.Tensor:
    """Decode back to ``sr``, undoing any rate switch the encoder made.

    Below ~32 kbps, LAME drops to MPEG-2/2.5 and halves or quarters the sample
    rate, so the decoded stream comes back at 22.05 or 16 kHz with a proportionally
    different length. Resampling it back up is what turns that into the
    band-limited 44.1 kHz signal the model has to extend.
    """
    import soundfile as sf

    buffer.seek(0)
    data, decoded_sr = sf.read(buffer, always_2d=True, dtype="float32")
    wav = torch.from_numpy(data.T.copy())

    if wav.shape[0] != nch:  # some decoders collapse or duplicate channels
        wav = wav[:1].repeat(nch, 1) if wav.shape[0] == 1 else wav[:nch]

    if int(decoded_sr) != int(sr):
        import torchaudio.functional as AF

        wav = AF.resample(wav, int(decoded_sr), int(sr))

    return wav


def _mp3_lameenc(wav, sr, bitrate_kbps, quality):
    import lameenc

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(int(bitrate_kbps))
    encoder.set_in_sample_rate(int(sr))
    encoder.set_channels(int(wav.shape[0]))
    encoder.set_quality(int(quality))
    encoder.silence()

    payload = encoder.encode(_to_int16_bytes(wav)) + encoder.flush()
    return _decode(io.BytesIO(bytes(payload)), wav.shape[0], sr)


def _via_soundfile(wav, sr, fmt, subtype):
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, wav.transpose(0, 1).numpy(), int(sr), format=fmt, subtype=subtype)
    return _decode(buffer, wav.shape[0], sr)


def _via_ffmpeg(wav, sr, spec, bitrate_kbps):
    """Round-trip through an ffmpeg encoder described by ``spec``.

    The sample-rate and channel handling is the whole reason this takes a spec
    rather than a codec name: a narrowband encoder rejects 44.1 kHz stereo
    outright, and forcing it down to 8 kHz mono *is* the degradation being
    simulated. The decode resamples back and re-expands the channels, so the
    caller always gets its own shape back -- band-limited and, for a stereo input,
    collapsed to a single channel, which is exactly what a phone recording did.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    import soundfile as sf
    import torchaudio.functional as AF

    nch = wav.shape[0]
    target_sr = spec.target_rate(sr)
    payload = wav.mean(0, keepdim=True) if (spec.mono and nch > 1) else wav
    if target_sr != sr:
        payload = AF.resample(payload, int(sr), target_sr)

    tmpdir = tempfile.mkdtemp(prefix="apollo_codec_")
    src = os.path.join(tmpdir, "in.wav")
    dst = os.path.join(tmpdir, f"out.{spec.container}")
    back = os.path.join(tmpdir, "back.wav")
    try:
        sf.write(src, payload.transpose(0, 1).numpy(), target_sr, subtype="PCM_16")
        encode = [ffmpeg, "-v", "error", "-y", "-i", src, "-c:a", spec.encoder,
                  "-ar", str(target_sr)]
        if spec.mono:
            encode += ["-ac", "1"]
        if not spec.vbr_only:
            encode += ["-b:a", f"{spec.clamp_bitrate(bitrate_kbps)}k"]
        if spec.fmt:
            encode += ["-f", spec.fmt]
        encode.append(dst)
        subprocess.run(encode, check=True, capture_output=True)

        # Decoded by ffmpeg as well, not by soundfile: libsndfile reads wav, flac,
        # ogg and (recent builds) mp3, and nothing else here. Every AAC, Opus, WMA,
        # AMR and G.722 file would fail at the *decode*, and because a failed
        # round-trip returns the input unchanged, the whole codec would look like
        # "a codec that did nothing" instead of like a missing feature.
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", dst,
                        "-c:a", "pcm_s16le", "-f", "wav", back],
                       check=True, capture_output=True)
        data, decoded_sr = sf.read(back, always_2d=True, dtype="float32")
    except Exception as exc:
        logger.debug("ffmpeg codec %s failed: %s", spec.encoder, exc)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    out = torch.from_numpy(data.T.copy())
    if int(decoded_sr) != int(sr):
        out = AF.resample(out, int(decoded_sr), int(sr))
    if out.shape[0] < nch:
        out = out[:1].repeat(nch, 1)
    return out[:nch]


_AVAILABILITY = {}


def available_codecs(names=None, sr=44100, force=False):
    """Which of ``names`` this install can actually encode.

    Probes each one on a short buffer, once per process. Worth doing up front: an
    encoder that is missing makes :func:`encode_decode` fall through to returning
    the input **unchanged**, which silently turns a degraded/clean training pair
    into a clean/clean one -- the model then has an identity mapping in its data
    and no error anywhere says so.
    """
    names = list(names or CODECS)
    probe = torch.zeros(1, int(sr) // 2)
    probe[0, ::64] = 0.5                     # broadband enough for any encoder

    usable = []
    for name in names:
        key = (name, int(sr))
        if force or key not in _AVAILABILITY:
            try:
                out = encode_decode(probe, sr, fmt=name, bitrate_kbps=64, strict=True)
                _AVAILABILITY[key] = out is not None
            except Exception:
                _AVAILABILITY[key] = False
        if _AVAILABILITY[key]:
            usable.append(name)
    return usable


def resolve_formats(formats, sr=44100, check=True):
    """Expand bundle names from :data:`CODEC_BUNDLES` and drop what is unavailable.

    Raises if nothing survives, rather than letting a run train on clean pairs.
    """
    wanted = []
    for entry in formats:
        wanted.extend(CODEC_BUNDLES.get(str(entry).lower(), [str(entry).lower()]))
    seen, ordered = set(), []
    for name in wanted:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    unknown = [n for n in ordered if n not in CODECS]
    if unknown:
        raise ValueError(f"unknown codec(s) {unknown}; known: {sorted(CODECS)}, "
                         f"bundles: {sorted(CODEC_BUNDLES)}")
    if not check:
        return ordered

    usable = available_codecs(ordered, sr=sr)
    missing = [n for n in ordered if n not in usable]
    if missing:
        logger.warning(
            "codec(s) %s are not available on this install (ffmpeg missing or built "
            "without them) and have been dropped from the degradation set", missing)
    if not usable:
        raise RuntimeError(
            f"none of the requested codecs {ordered} can be encoded here. Training "
            "would pair every clean segment with an identical 'degraded' copy. "
            "Install ffmpeg, or set codec_formats to something this install "
            "supports (mp3 and ogg need no ffmpeg).")
    return usable


def encode_decode(wav: torch.Tensor, sr: int, fmt: str = "mp3",
                  bitrate_kbps: int = 128, quality: int = 2,
                  strict: bool = False) -> torch.Tensor:
    """Round-trip ``wav`` (nch, n) through a lossy codec and return it aligned.

    Falls back through the available backends. If none can handle the format the
    input is returned unchanged, which keeps a training run alive but produces a
    clean/clean pair -- pass ``strict=True`` to get ``None`` instead and let the
    caller notice. :func:`resolve_formats` uses that to fail at startup rather
    than silently training on identity.
    """
    if wav.dim() != 2:
        raise ValueError(f"expected (nch, n), got {tuple(wav.shape)}")

    wav = wav.detach().cpu().float()
    fmt = fmt.lower()
    if fmt not in CODECS:
        raise ValueError(f"unsupported codec {fmt!r}; known: {sorted(CODECS)}")
    spec = CODECS[fmt]
    degraded = None

    # lameenc takes the requested bitrate exactly and is ~10x faster than shelling
    # out (24 ms against 200 ms), so mp3 prefers it.
    if fmt == "mp3":
        try:
            degraded = _mp3_lameenc(wav, sr, bitrate_kbps, quality)
        except Exception:
            degraded = None

    if degraded is None:
        degraded = _via_ffmpeg(wav, sr, spec, bitrate_kbps)

    # Last resort for the two formats libsndfile can write. Vorbis through
    # soundfile is VBR at libsndfile's default quality and ignores the requested
    # bitrate entirely -- it produces a much milder degradation than the config
    # asked for (96.9% of the band above 8 kHz survives at a nominal 48 kbps).
    # Fine as a no-ffmpeg fallback, wrong as a first choice, which is why it sits
    # below the ffmpeg path rather than above it.
    if degraded is None:
        try:
            if fmt in ("ogg", "vorbis"):
                degraded = _via_soundfile(wav, sr, "OGG", "VORBIS")
            elif fmt == "mp3":
                degraded = _via_soundfile(wav, sr, "MP3", "MPEG_LAYER_III")
        except Exception:
            degraded = None

    if degraded is None:
        return None if strict else wav

    # Align first, then trim -- the decoded stream is longer than the input by the
    # encoder delay, and that tail is exactly what compensates for the shift.
    return align_to_reference(wav, degraded)


def codec_simu(wav, sr=44100, options=None):
    """Drop-in replacement for the original ``codec_simu``.

    ``options`` accepts the same dict the configs already use, where any value may
    be the string ``"random"``.
    """
    options = dict(options or {})

    bitrate = options.get("bitrate", "random")
    if bitrate == "random":
        bitrate = random.choice(MP3_BITRATES) * 1000
    bitrate_kbps = int(bitrate) // 1000 if int(bitrate) > 1000 else int(bitrate)

    quality = options.get("quality", "random")
    quality = random.randint(0, 5) if quality == "random" else int(quality)

    fmt = options.get("format", "mp3")
    if fmt == "random":
        fmt = random.choice(["mp3", "ogg"])

    return encode_decode(wav, sr, fmt=fmt, bitrate_kbps=bitrate_kbps, quality=quality)


def lowpass_resample(wav: torch.Tensor, sr: int, cutoff_hz: float) -> torch.Tensor:
    """Band-limit by resampling down to ``2 * cutoff_hz`` and back.

    Cheap stand-in for the brick-wall low-pass every low-bitrate codec applies,
    useful for teaching bandwidth extension without paying for a real encode.
    """
    import torchaudio.functional as AF

    target = max(1000, int(2 * cutoff_hz))
    if target >= sr:
        return wav
    return AF.resample(AF.resample(wav, sr, target), target, sr)[..., : wav.shape[-1]]
