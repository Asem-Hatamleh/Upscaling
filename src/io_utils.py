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


def iter_frame_chunks(
    path: str | os.PathLike,
    chunk_size: int,
    max_seconds: float | None = None,
    fps_hint: float | None = None,
) -> Iterator[np.ndarray]:
    """Yield consecutive (N,H,W,3) uint8 RGB chunks of up to ``chunk_size``
    frames each. The last chunk may be smaller. Streams from disk without
    buffering the full video — used by the --live pipeline so RAM stays
    bounded and time-to-first-SR-frame is small."""
    buf: list[np.ndarray] = []
    for f in iter_frames(path, max_seconds=max_seconds, fps_hint=fps_hint):
        buf.append(f)
        if len(buf) >= chunk_size:
            yield np.stack(buf, axis=0)
            buf = []
    if buf:
        yield np.stack(buf, axis=0)


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


def _x264_options(crf: int, preset: str = "medium") -> dict[str, str]:
    """libx264 options. We tag colorspace explicitly because outputs without
    these tags get rendered with the wrong color matrix by players, which
    looks like a regular pattern of red/orange/green dots on faces and
    high-detail regions.
    """
    return {
        "crf": str(int(crf)),
        "preset": preset,
        # color metadata — bt709 is the right match for upscaled HD output.
        "colorprim": "bt709",
        "transfer": "bt709",
        "colormatrix": "bt709",
        # broadcast range (tv), not full (pc). Source 1.mp4 uses tv range.
        "x264-params": "colorprim=bt709:transfer=bt709:colormatrix=bt709:fullrange=off",
    }


def _nvenc_options(crf: int, preset: str = "p4") -> dict[str, str]:
    """h264_nvenc options. NVENC uses ``cq`` (constant-quality) instead of
    libx264's crf, but accepts a similar 0-51 range so we pass the same
    integer. Preset ``p4`` is 'medium' on the NVENC quality/speed curve
    (p1=fastest, p7=highest quality)."""
    return {
        "cq": str(int(crf)),
        "preset": preset,
        "rc": "vbr",
        "tune": "ll",          # low-latency tune — keeps GOP short for live
        "colorprim": "bt709",
        "transfer": "bt709",
        "colormatrix": "bt709",
    }


def nvenc_available() -> bool:
    """Return True if PyAV's ffmpeg build can construct an h264_nvenc
    encoder. Cheap probe — opens an in-memory mp4 container, tries to add
    the stream, closes it. Cached on first call."""
    cached = getattr(nvenc_available, "_cached", None)
    if cached is not None:
        return cached
    try:
        import av  # PyAV
        import io as _io
        buf = _io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        try:
            container.add_stream("h264_nvenc", rate=30)
            container.close()
            ok = True
        except Exception:
            try:
                container.close()
            except Exception:
                pass
            ok = False
    except Exception:
        ok = False
    setattr(nvenc_available, "_cached", ok)
    return ok


def resolve_encoder(choice: str) -> str:
    """Map the user's --encoder choice to a concrete PyAV codec name."""
    if choice in ("libx264", "h264_nvenc"):
        return choice
    if choice == "auto":
        return "h264_nvenc" if nvenc_available() else "libx264"
    raise ValueError(f"unknown encoder: {choice!r}")


class VideoWriter:
    """Lazy mp4 writer using PyAV. Tags colorspace bt709 so players don't
    misrender chroma as colored-dot artifacts.

    ``encoder`` accepts ``libx264`` (software, default) or ``h264_nvenc``
    (NVIDIA GPU NVENC, 5-10x faster but needs an NVENC-enabled PyAV/ffmpeg
    build and a Turing+ GPU)."""

    def __init__(self, path: str | os.PathLike, fps: float, crf: int = 18,
                 pixel_format: str = "yuv420p",
                 encoder: str = "libx264") -> None:
        self.path = str(path)
        self.fps = float(fps)
        self.crf = int(crf)
        self.pixel_format = pixel_format
        self.encoder = encoder
        self._container = None
        self._stream = None
        self._even_h = self._even_w = 0

    def _open(self, h: int, w: int) -> None:
        import av  # PyAV

        # libx264 + yuv420p requires both dims even; yuv444p does not but we
        # round anyway for safety.
        self._even_h = h - (h % 2)
        self._even_w = w - (w % 2)
        self._container = av.open(self.path, mode="w")
        try:
            self._stream = self._container.add_stream(self.encoder, rate=int(round(self.fps)))
        except Exception as e:
            if self.encoder != "libx264":
                # Encoder unavailable on this PyAV/ffmpeg build. Fall back to
                # libx264 with a warning rather than aborting the live run.
                print(f"[io_utils] encoder {self.encoder!r} unavailable ({e}); "
                      f"falling back to libx264.")
                self.encoder = "libx264"
                self._stream = self._container.add_stream("libx264", rate=int(round(self.fps)))
            else:
                raise
        self._stream.width = self._even_w
        self._stream.height = self._even_h
        self._stream.pix_fmt = self.pixel_format
        if self.encoder == "h264_nvenc":
            self._stream.options = _nvenc_options(self.crf)
        else:
            self._stream.options = _x264_options(self.crf)
        # Tag the bitstream colorspace, not only the encoder, so MP4 stream
        # metadata carries it through.
        try:
            self._stream.codec_context.color_range = "tv"
            self._stream.codec_context.color_primaries = "bt709"
            self._stream.codec_context.color_trc = "bt709"
            self._stream.codec_context.colorspace = "bt709"
        except Exception:
            pass

    def append(self, frames: np.ndarray) -> None:
        import av

        if frames.ndim == 3:
            frames = frames[None, ...]
        if self._container is None:
            self._open(frames.shape[1], frames.shape[2])
        if frames.shape[1] != self._even_h or frames.shape[2] != self._even_w:
            frames = frames[:, : self._even_h, : self._even_w, :]
        for f in frames:
            av_frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(f), format="rgb24"
            )
            for packet in self._stream.encode(av_frame):
                self._container.mux(packet)

    def close(self) -> None:
        if self._container is None:
            return
        # Flush.
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._container = None
        self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def encode_video(path: str | os.PathLike, frames: np.ndarray, fps: float,
                 crf: int = 18, pixel_format: str = "yuv420p",
                 encoder: str = "libx264") -> None:
    """One-shot writer. Tags bt709 colorspace to avoid the colored-dot
    artifact that uncolor-tagged x264 mp4 files show in some players."""
    with VideoWriter(path, fps=fps, crf=crf, pixel_format=pixel_format,
                     encoder=encoder) as w:
        w.append(frames)
