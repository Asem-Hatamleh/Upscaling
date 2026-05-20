"""Real-ESRGAN-Compact background + GFPGAN face-restore backend.

Targeted at the in-cabin driver-monitoring use case: most of the value lies
in the driver's face (eyes, mouth, head pose) — generic SR blurs faces, so
we route faces through GFPGAN-1.4 and the rest of the frame through the
lightweight Compact (``realesr-general-x4v3``) model.

This is `realesrgan_lite`'s pipeline plus a face stage, wired through
GFPGAN's built-in ``bg_upsampler`` plug. Same ``BaseUpscaler`` contract,
so CLI, wizard, and runner all work unchanged.

The face detector used inside GFPGAN is ``facexlib`` RetinaFace (ResNet50
by default). We surface a knob to switch to MobileNet for ~5× detector
speedup at a small accuracy cost. SCRFD swap is left as a TODO (handoff).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseUpscaler, UpscalerConfig, register


_GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/"
    "GFPGANv1.4.pth"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _weights_dir() -> Path:
    return _project_root() / "weights" / "gfpgan"


def _ensure_gfpgan_weight() -> Path:
    out = _weights_dir() / "GFPGANv1.4.pth"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request as _r
    print(f"[realesrgan_gfpgan] downloading {_GFPGAN_URL} -> {out}")
    _r.urlretrieve(_GFPGAN_URL, out)
    return out


@register("realesrgan_gfpgan")
class RealESRGANGFPGAN(BaseUpscaler):
    """Compact background upsampler + GFPGAN face restoration."""

    native_scale = 4

    def __init__(self, cfg: UpscalerConfig) -> None:
        super().__init__(cfg)
        self.restorer = None
        ex = cfg.extra or {}
        self.bg_variant = ex.get("model_name", "realesr-general-x4v3")
        self.denoise_strength = float(ex.get("denoise_strength", 0.5))
        self.bg_tile = int(ex.get("tile", 0))
        # Driver-monitoring: one face per frame; suppress background reflections
        # in the rear-view mirror / passenger seat that GFPGAN would otherwise
        # "restore" into eerie low-res face stamps.
        self.only_center_face = bool(ex.get("only_center_face", True))
        self.detector = str(ex.get("face_detector", "retinaface_resnet50"))

    # ------------- lifecycle -------------
    def load(self) -> None:
        from realesrgan import RealESRGANer  # type: ignore
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # type: ignore
        from gfpgan import GFPGANer  # type: ignore

        # 1) Compact background upsampler — reuse the weights / blending logic
        #    from ``realesrgan_lite``.
        from .realesrgan_lite import _ensure_weight, _ensure_weight_via_wdn
        bg_weight = _ensure_weight(self.bg_variant)
        bg_model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
            upscale=4, act_type="prelu",
        )
        model_path: object = str(bg_weight)
        dni_weight = None
        if self.bg_variant == "realesr-general-x4v3" and 0.0 < self.denoise_strength < 1.0:
            wdn_path = _ensure_weight_via_wdn()
            model_path = [str(bg_weight), str(wdn_path)]
            dni_weight = [self.denoise_strength, 1.0 - self.denoise_strength]
        half = self.cfg.dtype in ("fp16",)
        bg_upsampler = RealESRGANer(
            scale=4, model_path=model_path, dni_weight=dni_weight,
            model=bg_model, tile=self.bg_tile, tile_pad=10, pre_pad=0,
            half=half, device=self.cfg.device,
        )

        # 2) GFPGAN face restorer. ``upscale=4`` so face crops land at the
        #    same resolution as ``bg_upsampler``'s output before paste-back.
        gfpgan_path = _ensure_gfpgan_weight()
        self.restorer = GFPGANer(
            model_path=str(gfpgan_path),
            upscale=4,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
        )

        # GFPGANer instantiates facexlib's FaceRestoreHelper lazily on first
        # call. The detector backbone is picked up from
        # ``self.face_helper.face_det.name``; default is RetinaFace-ResNet50.
        if self.detector and self.detector != "retinaface_resnet50":
            # Force a different backbone before first enhance() call.
            from facexlib.detection import init_detection_model  # type: ignore
            self.restorer.face_helper.face_det = init_detection_model(
                self.detector, half=False, device=self.cfg.device,
            )

    # ------------- inference -------------
    def upscale(self, frames: np.ndarray) -> np.ndarray:
        import cv2

        assert self.restorer is not None, "call .load() first"
        out: list[np.ndarray] = []
        for f in frames:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            _, _, restored = self.restorer.enhance(
                bgr,
                has_aligned=False,
                only_center_face=self.only_center_face,
                paste_back=True,
            )
            if restored is None:
                # No face detected -> GFPGAN returns None for the paste-back
                # image; fall back to the bg upsampler's output.
                sr_bgr, _ = self.restorer.bg_upsampler.enhance(bgr, outscale=4)
                restored = sr_bgr
            out.append(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
        return np.stack(out, axis=0)

    def close(self) -> None:
        import torch
        self.restorer = None
        torch.cuda.empty_cache()
