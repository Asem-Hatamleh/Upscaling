"""Face restoration backends for the live pipeline.

Two backends, selectable at runtime:

* ``gfpgan``     — GFPGAN v1.4 via the official ``gfpgan`` package.
                   Fast, lower identity drift, smaller download (~333 MB).
* ``codeformer`` — CodeFormer via ``codeformer-pip``. Slower, higher quality,
                   fidelity-vs-quality weight tunable.

Both expose the same in-process API: build once, call ``restore(bgr_frame)``
per frame to get a same-shape BGR frame with detected faces restored and
pasted back. Detection failures or zero-face frames return the input
untouched (no copy).

This module is import-side-effect free except for backend modules that pull
in their own weights on first use (GFPGAN downloads to ~/.cache, CodeFormer
downloads under ./CodeFormer/weights on first inference).
"""

from __future__ import annotations

import re
from typing import Literal, Optional

import numpy as np
import torch

Backend = Literal["gfpgan", "codeformer"]


def _mask_torch_version_for_basicsr():
    """codeformer-pip ships an old basicsr whose version regex rejects modern
    torch suffixes like ``+cu128``. The regex insists on ``+git`` only and
    raises ``IndexError`` at import. Mask the suffix during import only."""
    import torch as _t
    orig = _t.__version__
    masked = re.sub(r"\+.*$", "+git0", orig)
    if masked == orig:
        return None  # no suffix to strip
    _t.__version__ = masked
    return orig


def _unmask_torch_version(orig: Optional[str]):
    if orig is None:
        return
    import torch as _t
    _t.__version__ = orig


class FaceEnhancer:
    """Lazy-loaded face restorer. Backend chosen at construction time."""

    def __init__(
        self,
        backend: Backend,
        fidelity: float = 0.5,
        only_center: bool = False,
        device: str = "cuda",
    ):
        self.backend = backend
        self.fidelity = float(fidelity)
        self.only_center = bool(only_center)
        self.device = device
        self._gfp = None
        self._cf_app = None
        self._cf_helper = None
        if backend == "gfpgan":
            self._load_gfpgan()
        elif backend == "codeformer":
            self._load_codeformer()
        else:
            raise ValueError(f"unknown face backend: {backend!r}")

    # ----- backend loaders -----
    def _load_gfpgan(self):
        from gfpgan import GFPGANer  # type: ignore

        url = (
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/"
            "GFPGANv1.4.pth"
        )
        # upscale=1: input is already SR-upscaled, GFPGAN should restore at
        # native res, not double the size. bg_upsampler=None: we don't want
        # GFPGAN to touch the background (Real-ESRGAN already did).
        self._gfp = GFPGANer(
            model_path=url,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=torch.device(self.device),
        )
        print(f"[face] gfpgan v1.4 loaded device={self.device} fidelity={self.fidelity}")

    def _load_codeformer(self):
        orig = _mask_torch_version_for_basicsr()
        try:
            from codeformer import app as cf_app  # type: ignore
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper  # type: ignore
        finally:
            _unmask_torch_version(orig)
        self._cf_app = cf_app
        # FaceRestoreHelper rebuilt per call because it caches per-image state
        # (cropped_faces / restored_faces). Reuse the detector across builds
        # by letting facexlib's internal weight cache do its job.
        self._cf_helper_cls = FaceRestoreHelper
        print(f"[face] codeformer loaded device={cf_app.device} fidelity={self.fidelity}")

    # ----- per-frame entry -----
    def restore(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Detect + restore faces in ``frame_bgr``. Returns same-shape BGR frame.

        On any backend exception, logs once and returns input unchanged so the
        live pipeline keeps flowing.
        """
        try:
            if self.backend == "gfpgan":
                return self._restore_gfpgan(frame_bgr)
            return self._restore_codeformer(frame_bgr)
        except Exception as e:
            # one print is enough; flood-throttle by not printing repeated msgs
            if not getattr(self, "_warned", False):
                print(f"[face] {self.backend} runtime error -> bypass: "
                      f"{type(e).__name__}: {e}")
                self._warned = True
            return frame_bgr

    def _restore_gfpgan(self, frame_bgr: np.ndarray) -> np.ndarray:
        _, _, restored_img = self._gfp.enhance(
            frame_bgr,
            has_aligned=False,
            only_center_face=self.only_center,
            paste_back=True,
            weight=self.fidelity,
        )
        return restored_img if restored_img is not None else frame_bgr

    def _restore_codeformer(self, frame_bgr: np.ndarray) -> np.ndarray:
        cf = self._cf_app
        helper = self._cf_helper_cls(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=cf.device,
        )
        helper.read_image(frame_bgr)
        n_faces = helper.get_face_landmarks_5(
            only_center_face=self.only_center, resize=640, eye_dist_threshold=5
        )
        if not n_faces:
            return frame_bgr
        helper.align_warp_face()
        for cropped in helper.cropped_faces:
            t = cf.img2tensor(cropped / 255.0, bgr2rgb=True, float32=True)
            cf.normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            t = t.unsqueeze(0).to(cf.device)
            with torch.no_grad():
                out = cf.codeformer_net(t, w=self.fidelity, adain=True)[0]
                restored = cf.tensor2img(out, rgb2bgr=True, min_max=(-1, 1))
            helper.add_restored_face(restored.astype("uint8"))
        helper.get_inverse_affine(None)
        return helper.paste_faces_to_input_image()
