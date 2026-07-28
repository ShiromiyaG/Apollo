###
# Author: Kai Li
# Date: 2024-01-22 01:16:22
# Email: lk21@mails.tsinghua.edu.cn
###
import json
from typing import Any, Dict, List, Optional, Tuple
import os
from omegaconf import OmegaConf
import argparse
import pytorch_lightning as pl
import torch
import hydra
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
import look2hear.system
import look2hear.datas
import look2hear.losses
from look2hear.utils import print_only
import warnings
warnings.filterwarnings("ignore")


def load_pretrained_weights(model: torch.nn.Module, checkpoint: str) -> None:
    """Warm-start the generator from an existing checkpoint.

    Fine-tuning the released weights is the realistic path for the vocal mode --
    training Apollo from scratch is a multi-GPU-week job. Accepts the same formats
    inference does (Lightning .ckpt, serialised .pth, bare state dict).
    """
    from look2hear.inference.loader import _unwrap_state_dict

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _unwrap_state_dict(raw)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print_only(f"Loaded pretrained weights from {checkpoint}")
    if missing:
        print_only(f"  {len(missing)} missing key(s), first few: {missing[:5]}")
    if unexpected:
        print_only(f"  {len(unexpected)} unexpected key(s), first few: {unexpected[:5]}")


def build_trainer(cfg: DictConfig, callbacks, logger) -> Trainer:
    """Instantiate the Trainer, using DDP only when there is more than one device."""
    devices = cfg.trainer.get("devices", 1)
    n_devices = len(devices) if isinstance(devices, (list, tuple)) else int(devices)

    extra = {}
    if n_devices > 1:
        extra["strategy"] = DDPStrategy(find_unused_parameters=True)

    return hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, **extra)


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    torch.set_float32_matmul_precision(cfg.get("matmul_precision", "highest"))

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    print_only(f"Instantiating datamodule <{cfg.datas._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datas)

    print_only(f"Instantiating AudioNet <{cfg.model._target_}>")
    model: torch.nn.Module = hydra.utils.instantiate(cfg.model)
    if cfg.get("init_from"):
        load_pretrained_weights(model, cfg.init_from)

    print_only(f"Instantiating Discriminator <{cfg.discriminator._target_}>")
    discriminator: torch.nn.Module = hydra.utils.instantiate(cfg.discriminator)

    print_only(f"Instantiating optimizer <{cfg.optimizer_g._target_}>")
    optimizer_g: torch.optim = hydra.utils.instantiate(cfg.optimizer_g, params=model.parameters())
    optimizer_d: torch.optim = hydra.utils.instantiate(cfg.optimizer_d, params=discriminator.parameters())

    print_only(f"Instantiating scheduler <{cfg.scheduler_g._target_}>")
    scheduler_g: torch.optim.lr_scheduler = hydra.utils.instantiate(cfg.scheduler_g, optimizer=optimizer_g)
    scheduler_d: torch.optim.lr_scheduler = hydra.utils.instantiate(cfg.scheduler_d, optimizer=optimizer_d)

    print_only(f"Instantiating loss <{cfg.loss_g._target_}>")
    losses = {
        "g": hydra.utils.instantiate(cfg.loss_g),
        "d": hydra.utils.instantiate(cfg.loss_d),
    }

    print_only(f"Instantiating metrics <{cfg.metrics._target_}>")
    metrics: torch.nn.Module = hydra.utils.instantiate(cfg.metrics)

    print_only(f"Instantiating system <{cfg.system._target_}>")
    system: LightningModule = hydra.utils.instantiate(
        cfg.system,
        model=model,
        discriminator=discriminator,
        loss_func=losses,
        metrics=metrics,
        optimizer=[optimizer_g, optimizer_d],
        scheduler=[scheduler_g, scheduler_d]
    )

    callbacks: List[Callback] = []
    if cfg.get("early_stopping"):
        print_only(f"Instantiating early_stopping <{cfg.early_stopping._target_}>")
        callbacks.append(hydra.utils.instantiate(cfg.early_stopping))

    checkpoint = None
    if cfg.get("checkpoint"):
        print_only(f"Instantiating checkpoint <{cfg.checkpoint._target_}>")
        checkpoint = hydra.utils.instantiate(cfg.checkpoint)
        callbacks.append(checkpoint)

    print_only(f"Instantiating logger <{cfg.logger._target_}>")
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "logs"), exist_ok=True)
    logger = hydra.utils.instantiate(cfg.logger)

    print_only(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = build_trainer(cfg, callbacks, logger)

    trainer.fit(system, datamodule=datamodule, ckpt_path=cfg.get("resume_from"))
    print_only("Training finished!")

    if checkpoint is not None and checkpoint.best_model_path:
        best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
        with open(os.path.join(cfg.exp.dir, cfg.exp.name, "best_k_models.json"), "w") as f:
            json.dump(best_k, f, indent=0)

        state_dict = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
        system.load_state_dict(state_dict=state_dict["state_dict"])
        if hasattr(system, "on_load_checkpoint"):
            system.on_load_checkpoint(state_dict)
        system.cpu()

        best_path = os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth")
        if getattr(system, "ema", None) is not None:
            # export the averaged weights, not the last adversarial step's
            system.ema.to("cpu")
            system.export_generator(best_path)
            print_only(f"Best model (EMA weights) written to {best_path}")
        else:
            torch.save(system.audio_model.serialize(), best_path)
            print_only(f"Best model written to {best_path}")

    try:  # wandb is optional; the default config logs to TensorBoard
        import wandb
        if wandb.run:
            print_only("Closing wandb!")
            wandb.finish()
    except ImportError:
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="configs/apollo.yaml",
        help="Path to the training config",
    )
    parser.add_argument(
        "--init_from",
        default=None,
        help="Checkpoint to warm-start the generator from (overrides the config)",
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Lightning checkpoint to resume a run from, optimiser state included",
    )

    args = parser.parse_args()
    cfg = OmegaConf.load(args.conf_dir)
    if args.init_from:
        cfg.init_from = args.init_from
    if args.resume_from:
        cfg.resume_from = args.resume_from

    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))

    train(cfg)
