"""RIFE frame interpolation wrapper.

Uses the canonical RIFE_HDv3 implementation from the included ``RIFE_trained_v6``
weights folder. RIFE doubles frames by interpolating one mid-frame between every
pair of consecutive frames. Stacking k passes multiplies frame count by ``2**k``.

We accept frames in SR (already upscaled) space so interpolation cost stays low.
For real-time we use 1-pass interpolation by default (re-create the dropped
frames from frame skipping).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch


def _ensure_rife_on_path(weights_dir: str | os.PathLike) -> Path:
    p = Path(weights_dir).expanduser().resolve()
    train_log = p / "train_log"
    if not (train_log / "flownet.pkl").exists():
        raise FileNotFoundError(
            f"RIFE weights not found at {train_log}/flownet.pkl. "
            "Pass --rife-weights /path/to/RIFE_trained_v6 or download v6."
        )
    if str(train_log) not in sys.path:
        sys.path.insert(0, str(train_log))
    return train_log


class RifeInterpolator:
    """Thin wrapper around RIFE_HDv3.Model.

    Use:
        rife = RifeInterpolator(weights_dir, device="cuda").load()
        out  = rife.interpolate_pair(a, b)            # one mid-frame
        outs = rife.interpolate_run(frames, passes=1) # 2x frames
    """

    def __init__(self, weights_dir: str | os.PathLike, device: str = "cuda",
                 dtype: torch.dtype = torch.float16) -> None:
        self.weights_dir = Path(weights_dir)
        self.device = device
        self.dtype = dtype
        self.model = None

    def load(self) -> "RifeInterpolator":
        train_log = _ensure_rife_on_path(self.weights_dir)
        try:
            from RIFE_HDv3 import Model  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"failed to import RIFE_HDv3 from {train_log}: {e}"
            ) from e
        m = Model()
        m.load_model(str(train_log), -1)
        m.eval()
        m.device()
        self.model = m
        return self

    # ---- core ----
    @torch.no_grad()
    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        # H,W,3 uint8 -> 1,3,H,W float in [0,1]
        t = torch.from_numpy(frame).to(self.device).permute(2, 0, 1).unsqueeze(0)
        return t.float().div_(255.0)

    @torch.no_grad()
    def _from_tensor(self, t: torch.Tensor) -> np.ndarray:
        t = t.clamp(0, 1).mul(255.0).round().byte()
        return t.squeeze(0).permute(1, 2, 0).cpu().numpy()

    @torch.no_grad()
    def interpolate_pair(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ta = self._to_tensor(a)
        tb = self._to_tensor(b)
        # pad to 32-multiple
        _, _, h, w = ta.shape
        ph = (32 - h % 32) % 32
        pw = (32 - w % 32) % 32
        if ph or pw:
            ta = torch.nn.functional.pad(ta, (0, pw, 0, ph), mode="replicate")
            tb = torch.nn.functional.pad(tb, (0, pw, 0, ph), mode="replicate")
        mid = self.model.inference(ta, tb)
        if ph or pw:
            mid = mid[:, :, :h, :w]
        return self._from_tensor(mid)

    @torch.no_grad()
    def interpolate_run(self, frames: np.ndarray, passes: int = 1) -> np.ndarray:
        """Return a frame sequence with ``2**passes`` density."""
        out: List[np.ndarray] = list(frames)
        for _ in range(passes):
            nxt: List[np.ndarray] = []
            for i in range(len(out) - 1):
                nxt.append(out[i])
                nxt.append(self.interpolate_pair(out[i], out[i + 1]))
            nxt.append(out[-1])
            out = nxt
        return np.stack(out, axis=0)

    @torch.no_grad()
    def fill_gaps(self, anchors: np.ndarray, gap: int) -> np.ndarray:
        """Given anchor frames spaced ``gap`` apart in source space, return a
        dense sequence of length ``(N-1)*gap + 1`` with intermediates.

        For gap=2: insert 1 mid-frame between each pair (1 pass).
        For gap=4: insert 3 frames between each pair (2 passes -> 4x, then drop last).
        For gap=3: not a power of two; we do 2 passes then resample to gap+1 stride.
        """
        if gap <= 1 or len(anchors) < 2:
            return anchors
        passes = max(1, int(np.ceil(np.log2(gap))))
        dense = self.interpolate_run(anchors, passes=passes)  # (N-1)*2**p + 1
        # subsample to exact gap density
        stride = (2 ** passes) // gap if (2 ** passes) % gap == 0 else 1
        if stride > 1:
            idx = np.arange(0, len(dense), stride)
            dense = dense[idx]
        return dense
