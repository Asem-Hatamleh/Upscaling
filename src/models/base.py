"""Abstract upscaler interface and a tiny registry.

Concrete models register themselves on import via ``@register(name)``.
The CLI builds them by name.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np


@dataclass
class UpscalerConfig:
    name: str
    quant: str = "none"          # none|int8_woq|nf4 (model-dependent)
    sage_attn: bool = False      # enable SageAttention SDPA monkeypatch
    scale: int = 4               # native scale produced by the model
    device: str = "cuda"
    dtype: str = "bf16"          # bf16|fp16|fp32
    extra: dict | None = None


class BaseUpscaler(abc.ABC):
    """Stateful upscaler. Load once, call ``upscale`` many times."""

    native_scale: int = 4

    def __init__(self, cfg: UpscalerConfig) -> None:
        self.cfg = cfg

    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def upscale(self, frames: np.ndarray) -> np.ndarray:
        """Upscale a chunk of frames.

        Args:
            frames: uint8 RGB array shape (N, H, W, 3).
        Returns:
            uint8 RGB array shape (N, H*native_scale, W*native_scale, 3).
        """

    def close(self) -> None:
        return None


# --- registry ---
_REGISTRY: dict[str, Callable[[UpscalerConfig], BaseUpscaler]] = {}


def register(name: str) -> Callable[[type[BaseUpscaler]], type[BaseUpscaler]]:
    def deco(cls: type[BaseUpscaler]) -> type[BaseUpscaler]:
        _REGISTRY[name] = cls
        return cls
    return deco


def build(cfg: UpscalerConfig) -> BaseUpscaler:
    if cfg.name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{cfg.name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[cfg.name](cfg)


def available() -> List[str]:
    return sorted(_REGISTRY)
