"""Real-ESRGAN lightweight alternative model wrapper.

We use the ``realesr-general-x4v3`` SRVGGNetCompact tiny model (~10 MB) — a
good real-time alt to FlashVSR. Quality is per-frame (no temporal model), so
prefer ``--frame-interp rife`` if you want temporally smooth output.

Auto-download: weights are fetched to ``<project>/weights/realesrgan/`` on
first load if missing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseUpscaler, UpscalerConfig, register


_WEIGHTS = {
    "realesr-general-x4v3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "realesr-general-x4v3.pth",
    ),
    "realesr-animevideov3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "realesr-animevideov3.pth",
    ),
}


def _weights_dir() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2] / "weights" / "realesrgan"


def _ensure_weight(name: str) -> Path:
    url, fname = _WEIGHTS[name]
    out = _weights_dir() / fname
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request as _r
    print(f"[realesrgan_lite] downloading {url} -> {out}")
    _r.urlretrieve(url, out)
    return out


@register("realesrgan_lite")
class RealESRGANLite(BaseUpscaler):
    native_scale = 4

    def __init__(self, cfg: UpscalerConfig) -> None:
        super().__init__(cfg)
        self.upsampler = None
        self.model_name = (cfg.extra or {}).get("model_name", "realesr-general-x4v3")
        self.denoise_strength = float((cfg.extra or {}).get("denoise_strength", 0.5))
        self.tile = int((cfg.extra or {}).get("tile", 0))
        self.tile_pad = int((cfg.extra or {}).get("tile_pad", 10))

    def load(self) -> None:
        import torch
        from realesrgan import RealESRGANer  # type: ignore
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # type: ignore

        weight = _ensure_weight(self.model_name)

        # SRVGGNetCompact ("Compact") — 4x model
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_conv=32, upscale=4, act_type="prelu")

        # x4v3 ships two weights blended via denoise_strength: clean + wdn (with noise).
        dni_weight = None
        model_path: object = str(weight)
        if self.model_name == "realesr-general-x4v3" and 0.0 < self.denoise_strength < 1.0:
            wdn_path = _ensure_weight_via_wdn()
            model_path = [str(weight), str(wdn_path)]
            dni_weight = [self.denoise_strength, 1.0 - self.denoise_strength]

        half = self.cfg.dtype in ("fp16",)  # Real-ESRGAN supports half only
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=0,
            half=half,
            device=self.cfg.device,
        )

    def upscale(self, frames: np.ndarray) -> np.ndarray:
        import cv2

        assert self.upsampler is not None, "call .load() first"
        out_frames: list[np.ndarray] = []
        for f in frames:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            sr_bgr, _ = self.upsampler.enhance(bgr, outscale=4)
            out_frames.append(cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB))
        return np.stack(out_frames, axis=0)

    def close(self) -> None:
        import torch
        self.upsampler = None
        torch.cuda.empty_cache()


def _ensure_weight_via_wdn() -> Path:
    """Fetch the realesr-general-wdn-x4v3 weight used for denoise blending."""
    url = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
           "realesr-general-wdn-x4v3.pth")
    out = _weights_dir() / "realesr-general-wdn-x4v3.pth"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request as _r
    print(f"[realesrgan_lite] downloading {url} -> {out}")
    _r.urlretrieve(url, out)
    return out
