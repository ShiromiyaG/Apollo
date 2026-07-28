"""Exponential moving average of the generator weights.

A GAN generator does not converge to a point, it orbits one. The weights at any
single step carry the high-frequency jitter of the adversarial game, and that
jitter is audible: it is a large part of why a checkpoint exported straight from
the last optimiser step sounds noisier than the training curves suggest.

Averaging the weights over a long window removes that jitter without touching the
architecture or the objective. It costs one extra copy of the parameters and a
`lerp_` per step, and it is the highest quality-per-risk change available here.
Validate and export from the EMA weights, not the raw ones.
"""

from contextlib import contextmanager

import torch


class ModelEMA:
    """Tracks a decayed average of a module's floating-point state.

    Args:
        decay: target smoothing factor. 0.999 averages over roughly the last 1000
            steps; raise it for long runs, lower it for short fine-tunes.
        warmup_steps: for the first steps the effective decay is ramped up from 0,
            so the average is not dominated by the randomly initialised weights it
            started from. Set to 0 to use ``decay`` from the very first update.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999, warmup_steps: int = 1000):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")

        self.decay = decay
        self.warmup_steps = max(0, int(warmup_steps))
        self.num_updates = 0
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

    def _current_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        # ramps 0 -> decay; the (1+n)/(10+n) form is the standard TF/timm schedule
        warmup = (1.0 + self.num_updates) / (10.0 + self.num_updates)
        return min(self.decay, warmup)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        decay = self._current_decay()
        for name, tensor in model.state_dict().items():
            shadow = self.shadow.get(name)
            if shadow is None:
                self.shadow[name] = tensor.detach().clone()
                continue
            if shadow.device != tensor.device:
                # the shadow is built at __init__, before the trainer moves the
                # model onto the accelerator; follow it rather than fail
                shadow = shadow.to(tensor.device)
                self.shadow[name] = shadow
            if tensor.dtype.is_floating_point:
                shadow.lerp_(tensor.detach().to(shadow.dtype), 1.0 - decay)
            else:
                # counters, integer buffers: an average is meaningless, track the value
                shadow.copy_(tensor.detach())
        self.num_updates += 1

    @contextmanager
    def averaged(self, model: torch.nn.Module):
        """Temporarily swap the EMA weights into ``model``.

        Used around validation and export so the numbers that drive checkpoint
        selection describe the weights that will actually ship.
        """
        backup = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        try:
            model.load_state_dict(self.shadow, strict=False)
            yield model
        finally:
            model.load_state_dict(backup, strict=False)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)

    def to(self, device) -> "ModelEMA":
        for name, tensor in self.shadow.items():
            self.shadow[name] = tensor.to(device)
        return self

    def state_dict(self):
        return {"decay": self.decay, "warmup_steps": self.warmup_steps,
                "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state.get("decay", self.decay)
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self.num_updates = state.get("num_updates", 0)
        saved = state.get("shadow", {})
        for name, tensor in saved.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor.to(self.shadow[name].device))
            else:
                self.shadow[name] = tensor.clone()
