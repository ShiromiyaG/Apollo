"""Backwards-compatible alias.

The vocal-specific module turned into the shared training loop once the stereo
restoration mode needed the same GAN improvements. Kept so existing configs that
name `look2hear.system.vocal_litmodule.VocalLightningModule` keep resolving.
"""

from .restoration_litmodule import (
    RestorationLightningModule,
    VocalLightningModule,
    _as_module_dict,
)

__all__ = ["RestorationLightningModule", "VocalLightningModule", "_as_module_dict"]
