"""CodeFormer face restoration + Compact (`realesr-general-x4v3`) background.

Drop-in alternative to `realesrgan_gfpgan`. CodeFormer uses a learned
codebook prior — perceptually different from GFPGAN-1.4 (less "plastic"
skin, often better on profile / occluded / low-light cabin faces, weaker
on extreme low-res). Same `BaseUpscaler` contract so the CLI / runner /
benchmark harness pick it up without changes.

We deliberately import CodeFormer's bundled `basicsr` + `facelib` from
``third_party/CodeFormer/`` (added to ``sys.path`` only inside
``load()``). The benchmark harness spawns a fresh subprocess per cell,
so the sys.path mutation does not leak across runs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseUpscaler, UpscalerConfig, register


_CODEFORMER_URL = (
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_codeformer_weight() -> Path:
    out = _project_root() / "weights" / "codeformer" / "codeformer.pth"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request as _r
    print(f"[codeformer_compact] downloading {_CODEFORMER_URL} -> {out}")
    _r.urlretrieve(_CODEFORMER_URL, out)
    return out


# `_ensure_repo_on_path` is no longer needed — CodeFormer arch files are
# vendored under ``src/models/_codeformer/`` so we don't depend on a clone of
# the upstream repo at runtime. Kept as a no-op for backwards compatibility
# with anyone who imported it.
def _ensure_repo_on_path() -> None:
    return None


@register("codeformer_compact")
class CodeFormerCompact(BaseUpscaler):
    native_scale = 4

    def __init__(self, cfg: UpscalerConfig) -> None:
        super().__init__(cfg)
        ex = cfg.extra or {}
        self.net = None
        self.face_helper = None
        self.bg_upsampler = None
        # fidelity_w trades off "stick to detected landmarks" (w=1) vs
        # "let CodeFormer's codebook hallucinate" (w=0). For driver
        # monitoring we want identity preservation: 0.9 keeps eyes / nose
        # geometry tight to the source, ~0.7 (CodeFormer default) tends to
        # invent eyes when the face crop is below ~80 px tall.
        self.fidelity_w = float(ex.get("codeformer_fidelity", 0.9))   # 0..1
        # Driver-monitoring footage typically has one face; tiny mirror /
        # passenger reflections would otherwise get "restored" into extra
        # garbled mini-faces. Default to center-face only.
        self.only_center_face = bool(ex.get("only_center_face", True))
        self.eye_dist_threshold = int(ex.get("eye_dist_threshold", 10))
        self.bg_variant = ex.get("model_name", "realesr-general-x4v3")
        self.denoise_strength = float(ex.get("denoise_strength", 0.5))
        self.bg_tile = int(ex.get("tile", 0))
        self.use_bg_upsampler = bool(ex.get("use_bg_upsampler", True))

    # ------------- lifecycle -------------
    def load(self) -> None:
        import torch

        # CodeFormer arch is vendored at src/models/_codeformer/ (we only need
        # the two arch files; pip basicsr serves utils + registry). Avoids the
        # sys.path conflict between CodeFormer's bundled basicsr and our pip
        # basicsr (which realesrgan + gfpgan rely on).
        from ._codeformer.codeformer_arch import CodeFormer as CodeFormerArch
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper  # type: ignore

        weight = _ensure_codeformer_weight()
        net = CodeFormerArch(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.cfg.device)
        ckpt = torch.load(str(weight), map_location=self.cfg.device,
                          weights_only=False)
        net.load_state_dict(ckpt["params_ema"])
        net.eval()
        self.net = net

        # Face helper: detect + align + paste-back (RetinaFace-ResNet50 inside).
        self.face_helper = FaceRestoreHelper(
            upscale_factor=4,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.cfg.device,
        )

        # Background: same Compact pipeline as realesrgan_lite / _gfpgan.
        if self.use_bg_upsampler:
            from realesrgan import RealESRGANer  # type: ignore
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # type: ignore
            from .realesrgan_lite import _ensure_weight, _ensure_weight_via_wdn
            bg_weight = _ensure_weight(self.bg_variant)
            bg_model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
                upscale=4, act_type="prelu",
            )
            model_path: object = str(bg_weight)
            dni_weight = None
            if (self.bg_variant == "realesr-general-x4v3"
                    and 0.0 < self.denoise_strength < 1.0):
                wdn_path = _ensure_weight_via_wdn()
                model_path = [str(bg_weight), str(wdn_path)]
                dni_weight = [self.denoise_strength, 1.0 - self.denoise_strength]
            half = self.cfg.dtype == "fp16"
            self.bg_upsampler = RealESRGANer(
                scale=4, model_path=model_path, dni_weight=dni_weight,
                model=bg_model, tile=self.bg_tile, tile_pad=10, pre_pad=0,
                half=half, device=self.cfg.device,
            )

    # ------------- inference -------------
    def _restore_one(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2
        import torch
        from torchvision.transforms.functional import normalize
        # img2tensor / tensor2img come from pip-installed basicsr (same logic).
        from basicsr.utils import img2tensor, tensor2img  # type: ignore

        face_helper = self.face_helper
        face_helper.clean_all()
        face_helper.read_image(frame_bgr)
        # eye_dist_threshold filters out detections whose eye centers are
        # closer than N source-pixels apart. The default of 5 picks up
        # postage-stamp faces that the restorer can only hallucinate; 10
        # cuts those out for cabin-distance subjects.
        face_helper.get_face_landmarks_5(
            only_center_face=self.only_center_face,
            resize=640, eye_dist_threshold=self.eye_dist_threshold,
        )
        face_helper.align_warp_face()

        for cropped in face_helper.cropped_faces:
            t = img2tensor(cropped / 255., bgr2rgb=True, float32=True)
            normalize(t, (0.5,) * 3, (0.5,) * 3, inplace=True)
            t = t.unsqueeze(0).to(self.cfg.device)
            with torch.no_grad():
                out = self.net(t, w=self.fidelity_w, adain=True)[0]
                restored = tensor2img(out, rgb2bgr=True, min_max=(-1, 1))
            face_helper.add_restored_face(restored.astype("uint8"))

        # Background
        if self.bg_upsampler is not None:
            bg_img, _ = self.bg_upsampler.enhance(frame_bgr, outscale=4)
        else:
            bg_img = cv2.resize(
                frame_bgr,
                (frame_bgr.shape[1] * 4, frame_bgr.shape[0] * 4),
                interpolation=cv2.INTER_LANCZOS4,
            )
        face_helper.get_inverse_affine(None)
        # Older facexlib lacks `draw_box` / `face_upsampler` kwargs; pass only
        # the upsampled background and let paste_back handle the rest.
        out = face_helper.paste_faces_to_input_image(upsample_img=bg_img)
        return out

    def upscale(self, frames: np.ndarray) -> np.ndarray:
        import cv2

        assert self.net is not None, "call .load() first"
        out: list[np.ndarray] = []
        for f in frames:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            restored = self._restore_one(bgr)
            out.append(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
        return np.stack(out, axis=0)

    def close(self) -> None:
        import torch
        self.net = None
        self.face_helper = None
        self.bg_upsampler = None
        torch.cuda.empty_cache()
