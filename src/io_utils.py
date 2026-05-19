"""Video I/O, resizing, and side-by-side composition.

Reads with imageio (ffmpeg), writes with imageio. Composition is done in
numpy to avoid a second ffmpeg pass per video.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import imageio.v3 as iio
import numpy as np
from PIL import Image


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float


def probe(path: str | os.PathLike) -> VideoMeta:
    meta = iio.immeta(str(path), plugin="pyav")
    fps = float(meta.get("fps", 25.0))
    duration = float(meta.get("duration", 0.0))
    size = meta.get("size", None)
    if size is None:
        first = next(iio.imiter(str(path), plugin="pyav"))
        h, w = first.shape[:2]
    else:
        w, h = int(size[0]), int(size[1])
    n_frames = int(round(fps * duration)) if duration else 0
    return VideoMeta(w, h, fps, n_frames, duration)


def iter_frames(
    path: str | os.PathLike,
    max_seconds: float | None = None,
    fps_hint: float | None = None,
) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames. Stop after ``max_seconds`` if set."""
    fps = fps_hint
    limit = None
    if max_seconds is not None and max_seconds > 0:
        if fps is None:
            fps = probe(path).fps
        limit = max(1, int(round(fps * max_seconds)))
    for i, frame in enumerate(iio.imiter(str(path), plugin="pyav")):
        if limit is not None and i >= limit:
            return
        yield frame  # HxWx3 uint8 RGB


def list_videos(path: str | os.PathLike) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() in VIDEO_EXTS else []
    return sorted([q for q in p.rglob("*") if q.suffix.lower() in VIDEO_EXTS])


def resize_pil(frame: np.ndarray, w: int, h: int, method: str = "bicubic") -> np.ndarray:
    interp = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC,
              "lanczos": Image.LANCZOS, "nearest": Image.NEAREST}[method]
    img = Image.fromarray(frame)
    img = img.resize((w, h), interp)
    return np.asarray(img)


def resize_batch(frames: np.ndarray, w: int, h: int, method: str = "bicubic") -> np.ndarray:
    return np.stack([resize_pil(f, w, h, method) for f in frames], axis=0)


def round_to_multiple(x: int, m: int) -> int:
    return max(m, int(math.ceil(x / m) * m))


def resolve_preprocess(src_w: int, src_h: int, opt: str) -> Tuple[int, int]:
    """Resolve a pre-processing resize policy.

    Supported:
        ``none``                 -> no change
        ``vga``                  -> 640x480 (keep aspect, fit)
        ``qvga``                 -> 320x240 (fit)
        ``WxH``                  -> exact target, e.g. ``256x192``
        ``pct:NN`` / ``NN%``     -> percentage of source, e.g. ``50%``
    """
    opt = (opt or "none").strip().lower()
    if opt == "none":
        return src_w, src_h
    if opt == "vga":
        return _fit(src_w, src_h, 640, 480)
    if opt == "qvga":
        return _fit(src_w, src_h, 320, 240)
    if opt.startswith("pct:"):
        pct = float(opt.split(":", 1)[1]) / 100.0
        return _scale_pct(src_w, src_h, pct)
    if opt.endswith("%"):
        pct = float(opt[:-1]) / 100.0
        return _scale_pct(src_w, src_h, pct)
    if "x" in opt:
        a, b = opt.split("x", 1)
        return int(a), int(b)
    raise ValueError(f"unknown resize spec: {opt!r} (try 'none', 'vga', 'qvga', "
                     f"'WxH' like '160x128', or 'NN%' like '30%')")


def _scale_pct(src_w: int, src_h: int, pct: float) -> Tuple[int, int]:
    if pct <= 0:
        raise ValueError(f"pct must be > 0, got {pct}")
    return max(2, int(round(src_w * pct))), max(2, int(round(src_h * pct)))


def _fit(src_w: int, src_h: int, tgt_w: int, tgt_h: int) -> Tuple[int, int]:
    r = min(tgt_w / src_w, tgt_h / src_h)
    return max(2, int(round(src_w * r))), max(2, int(round(src_h * r)))


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose left|right at the LARGER of the two heights, preserving aspect.

    Both inputs are uint8 RGB (N,H,W,3). Heights are matched by resizing the
    shorter video up (we want the side-by-side to show the original at full
    resolution, not a downscale).
    """
    if left.shape[0] != right.shape[0]:
        n = min(left.shape[0], right.shape[0])
        left = left[:n]
        right = right[:n]
    h_l, w_l = left.shape[1:3]
    h_r, w_r = right.shape[1:3]
    target_h = max(h_l, h_r)
    if h_l != target_h:
        new_w = int(round(w_l * target_h / h_l))
        left = resize_batch(left, new_w, target_h, "bicubic")
    if h_r != target_h:
        new_w = int(round(w_r * target_h / h_r))
        right = resize_batch(right, new_w, target_h, "bicubic")
    return np.concatenate([left, right], axis=2)


class VideoWriter:
    """Lazy mp4 writer using imageio/pyav (libx264 yuv420p)."""

    def __init__(self, path: str | os.PathLike, fps: float, crf: int = 18) -> None:
        self.path = str(path)
        self.fps = float(fps)
        self.crf = int(crf)
        self._writer = None

    def _open(self, h: int, w: int) -> None:
        # pad to even dims for yuv420p
        self._even_h = h - (h % 2)
        self._even_w = w - (w % 2)
        self._writer = iio.imopen(self.path, "w", plugin="pyav")
        self._writer.init_video_stream(
            "libx264",
            fps=self.fps,
            pixel_format="yuv420p",
        )
        try:
            self._writer.container_metadata = {"comment": "UpScaling"}
        except Exception:
            pass

    def append(self, frames: np.ndarray) -> None:
        if frames.ndim == 3:
            frames = frames[None, ...]
        if self._writer is None:
            self._open(frames.shape[1], frames.shape[2])
        if frames.shape[1] != self._even_h or frames.shape[2] != self._even_w:
            frames = frames[:, : self._even_h, : self._even_w, :]
        for f in frames:
            self._writer.write_frame(np.ascontiguousarray(f))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def encode_video(path: str | os.PathLike, frames: np.ndarray, fps: float, crf: int = 18) -> None:
    """One-shot writer for small arrays (used for side-by-side)."""
    h, w = frames.shape[1:3]
    h2, w2 = h - (h % 2), w - (w % 2)
    if (h2, w2) != (h, w):
        frames = frames[:, :h2, :w2, :]
    with iio.imopen(str(path), "w", plugin="pyav") as f:
        f.init_video_stream("libx264", fps=float(fps), pixel_format="yuv420p")
        for fr in frames:
            f.write_frame(np.ascontiguousarray(fr))
