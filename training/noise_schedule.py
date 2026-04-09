"""
Alpha noise schedules for Masked Diffusion Language Modeling (MDLM).

The forward diffusion process corrupts a clean token sequence by masking
tokens with probability p_mask(t) = 1 − α(t), where α(t) is the
"alpha schedule" — a monotonically decreasing function on [0, 1]:

    t = 0  →  α(0) = 1  →  p_mask = 0  (no masking, fully clean)
    t = 1  →  α(1) = 0  →  p_mask = 1  (fully masked / noised)

The loss weight for timestep t is:

    w(t) = −dα/dt / (1 − α(t) + ε)

This upweights training examples that are harder (high-t, heavily masked)
relative to easy ones (low-t, lightly masked).

Reference:
    Shi et al. (2024) — "Simplified and Effective Masked Diffusion Language Models"
    https://arxiv.org/abs/2406.07524
    (ported from dllm/dllm/core/schedulers/alpha.py)
"""

from __future__ import annotations

import math
from typing import Union

import torch

Number = Union[float, torch.Tensor]

_EPS = 1e-6   # numerical stability in weight denominator


# ------------------------------------------------------------------ #
# Base class
# ------------------------------------------------------------------ #

class BaseAlphaScheduler:
    """
    Abstract base for alpha schedules.

    Public API:
        scheduler.alpha(t)            → α(t)
        scheduler.alpha_derivative(t) → dα/dt
        scheduler.weight(t)           → w(t) = −dα/dt / (1−α(t)+ε)
        scheduler(t)                  → α(t)   (callable shorthand)
    """

    def __call__(self, t: Number) -> Number:
        return self.alpha(t)

    def alpha(self, t: Number) -> Number:
        """Return α(t) — fraction of tokens that remain unmasked."""
        t_t = self._to_tensor(t)
        self._check_range(t_t, 0.0, 1.0)
        out = self._alpha(t_t)
        return out.item() if isinstance(t, float) else out

    def alpha_derivative(self, t: Number) -> Number:
        """Return dα/dt — rate of change of the masking schedule."""
        t_t = self._to_tensor(t)
        self._check_range(t_t, 0.0, 1.0)
        out = self._alpha_derivative(t_t)
        return out.item() if isinstance(t, float) else out

    def weight(self, t: Number) -> Number:
        """
        Return the MDLM loss weight w(t) = −dα/dt / (1 − α(t) + ε).

        Shape: same as t.
        """
        return -self.alpha_derivative(t) / (1.0 - self.alpha(t) + _EPS)

    # ---- subclass hooks ----

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _alpha_derivative(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ---- helpers ----

    @staticmethod
    def _to_tensor(t: Number) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            return t.float()
        return torch.tensor(t, dtype=torch.float32)

    @staticmethod
    def _check_range(t: torch.Tensor, lo: float, hi: float) -> None:
        if not torch.all((lo <= t) & (t <= hi)):
            raise ValueError(f"Timesteps must be in [{lo}, {hi}]; got min={t.min().item():.4f}, max={t.max().item():.4f}")


# ------------------------------------------------------------------ #
# Linear schedule:  α(t) = 1 − t
# ------------------------------------------------------------------ #

class LinearAlphaScheduler(BaseAlphaScheduler):
    """
    Linear noise schedule.

        α(t)  = 1 − t
        dα/dt = −1
        w(t)  = 1 / (1 − (1−t) + ε)  =  1 / (t + ε)

    Simple and used as the default in the MDLM paper.
    """

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    def _alpha_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)


# ------------------------------------------------------------------ #
# Cosine schedule:  α(t) = 1 − cos(π/2 · (1−t))
# ------------------------------------------------------------------ #

class CosineAlphaScheduler(BaseAlphaScheduler):
    """
    Cosine noise schedule.

        α(t)  = 1 − cos(π/2 · (1−t))
        dα/dt = −(π/2) · sin(π/2 · (1−t))
        w(t)  = (π/2) · sin(π/2·(1−t)) / (cos(π/2·(1−t)) + ε)

    Smooth schedule; spends more time at intermediate noise levels.
    """

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.cos((math.pi / 2.0) * (1.0 - t))

    def _alpha_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return -(math.pi / 2.0) * torch.sin((math.pi / 2.0) * (1.0 - t))


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #

_REGISTRY = {
    "linear": LinearAlphaScheduler,
    "cosine": CosineAlphaScheduler,
}


def get_noise_scheduler(name: str) -> BaseAlphaScheduler:
    """
    Instantiate a noise scheduler by name.

    Args:
        name: "linear" or "cosine"

    Returns:
        BaseAlphaScheduler instance
    """
    name_l = name.lower()
    if name_l not in _REGISTRY:
        raise ValueError(f"Unknown noise schedule '{name}'. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name_l]()
