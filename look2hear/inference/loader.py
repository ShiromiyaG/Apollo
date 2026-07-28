"""Checkpoint loading for Apollo.

The upstream `inference.py` calls ``BaseModel.from_pretrain("JusperLee/Apollo", ...)``,
which routes into a ``torch.load`` of that string as a filesystem path and fails.
This module accepts what people actually have on disk -- a Lightning ``.ckpt``, a
serialised ``.pth``, a bare state dict, or a Hugging Face repo id -- and infers the
architecture hyper-parameters from the weights so they never have to be guessed.
"""

import os

import torch

from ..models.apollo import Apollo

DEFAULT_REPO = "JusperLee/Apollo"
_PREFIXES = ("audio_model.", "module.", "model.")


def _unwrap_state_dict(obj):
    """Pull a plain ``{param_name: tensor}`` mapping out of whatever was saved."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise ValueError("checkpoint does not contain a state dict")

    state = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    if not state:
        raise ValueError("checkpoint does not contain a state dict")

    # Lightning wraps the generator as `audio_model.*` and also stores the
    # discriminator; keep only the generator side.
    for prefix in _PREFIXES:
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            break

    return state


def infer_hparams(state, sr=44100):
    """Recover (win_ms, feature_dim, layer) from the shapes in a state dict."""
    feature_dim = state["BN.0.1.weight"].shape[0]

    layer = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("net."))

    band_indices = [int(k.split(".")[1]) for k in state if k.startswith("output.") and k.endswith(".1.bias")]
    nband = max(band_indices) + 1
    head_width = state["output.0.1.bias"].shape[0] // 4
    tail_width = state[f"output.{nband - 1}.1.bias"].shape[0] // 4

    enc_dim = (nband - 1) * head_width + tail_width
    win_samples = (enc_dim - 1) * 2
    win_ms = round(win_samples * 1000 / sr)

    return win_ms, int(feature_dim), int(layer)


def load_apollo(
    checkpoint=None,
    repo_id=DEFAULT_REPO,
    sr=44100,
    win=None,
    feature_dim=None,
    layer=None,
    device="cpu",
    vram_budget_mb=128,
):
    """Instantiate Apollo and load weights.

    Args:
        checkpoint: local ``.ckpt``/``.pth``/``.bin``/``.safetensors`` file. When
            omitted the weights are pulled from ``repo_id`` on the Hub.
        sr, win, feature_dim, layer: architecture overrides. Anything left as
            ``None`` is inferred from the checkpoint.
        vram_budget_mb: passed to :meth:`Apollo.set_vram_budget`.
    """
    if checkpoint is None:
        checkpoint = _download_from_hub(repo_id)

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    if str(checkpoint).endswith(".safetensors"):
        from safetensors.torch import load_file
        raw = load_file(checkpoint, device="cpu")
    else:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)

    state = _unwrap_state_dict(raw)

    if is_v2_state_dict(state):
        from ..models.apollo_v2 import ApolloV2

        # v2 settings such as the dilations and the output mode leave no trace in
        # the tensor shapes, so they have to come from the checkpoint itself
        saved_args = raw.get("model_args") if isinstance(raw, dict) else None
        if not saved_args:
            raise ValueError(
                "this looks like an Apollo v2 checkpoint but carries no `model_args`. "
                "v2 cannot be rebuilt from weight shapes alone -- re-export it with "
                "RestorationLightningModule.export_generator."
            )

        args = dict(saved_args)
        args.setdefault("sr", sr)
        if win is not None:
            args["win"] = win
        if feature_dim is not None:
            args["feature_dim"] = feature_dim
        if layer is not None:
            args["layer"] = layer

        model = ApolloV2(**args)
    else:
        inferred_win, inferred_dim, inferred_layer = infer_hparams(state, sr=sr)
        model = Apollo(
            sr=sr,
            win=win if win is not None else inferred_win,
            feature_dim=feature_dim if feature_dim is not None else inferred_dim,
            layer=layer if layer is not None else inferred_layer,
        )

    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    model.set_vram_budget(vram_budget_mb)

    return model


def is_v2_state_dict(state) -> bool:
    """v2 stores the band projections as stacked parameters, v1 as a ModuleList."""
    return any(k.startswith("bn_weight.") or k.startswith("out_weight.") for k in state)


def _download_from_hub(repo_id):
    from huggingface_hub import list_repo_files, hf_hub_download

    preferred = ("pytorch_model.bin", "model.safetensors", "apollo_model.ckpt", "best_model.pth")
    files = list_repo_files(repo_id)
    for name in preferred:
        if name in files:
            return hf_hub_download(repo_id, name)

    for name in files:
        if name.endswith((".ckpt", ".pth", ".bin", ".safetensors")):
            return hf_hub_download(repo_id, name)

    raise FileNotFoundError(f"no checkpoint file found in {repo_id}")
