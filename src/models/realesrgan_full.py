"""Full Real-ESRGAN RRDB backend (no face restoration).

This is the heavier `RealESRGAN_x4plus` model, not the Compact/Lite
`realesr-general-x4v3` backend. Use it to compare plain full Real-ESRGAN
against Lite, GFPGAN, and CodeFormer face-restoration variants.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseUpscaler, UpscalerConfig, register


_X4PLUS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)


def _weights_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "weights" / "realesrgan"


def _ensure_x4plus_weight() -> Path:
    out = _weights_dir() / "RealESRGAN_x4plus.pth"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request as _r
    print(f"[realesrgan_full] downloading {_X4PLUS_URL} -> {out}")
    _r.urlretrieve(_X4PLUS_URL, out)
    return out


@register("realesrgan")
@register("realesrgan_full")
class RealESRGANFull(BaseUpscaler):
    """Full RRDB Real-ESRGAN x4plus, no GFPGAN/CodeFormer."""

    native_scale = 4

    def __init__(self, cfg: UpscalerConfig) -> None:
        super().__init__(cfg)
        ex = cfg.extra or {}
        self.upsampler = None
        self.tile = int(ex.get("tile", 0))
        self.tile_pad = int(ex.get("tile_pad", 10))

    def load(self) -> None:
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
        from realesrgan import RealESRGANer  # type: ignore

        weight = _ensure_x4plus_weight()
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        half = self.cfg.dtype == "fp16"
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=str(weight),
            dni_weight=None,
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

