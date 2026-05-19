"""Per-video orchestration: read -> (resize) -> skip+SR -> (interp) -> write."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import io_utils, frame_skip, report
from .models.base import BaseUpscaler


@dataclass
class RunOptions:
    seconds: float = 0.0          # 0 = full video
    out_scale: int = 4            # final scale we want (2 or 4)
    pre_resize: str = "none"      # "none" | "vga" | "qvga" | "WxH" | "pct:NN"
    frame_skip: int = 1           # 1 = none
    frame_interp: str = "none"    # none|repeat|rife
    write_comparison: bool = True
    write_upscaled: bool = True
    chunk_frames: int = 32        # frames per inference chunk (CPU streaming side)
    rife_weights: Optional[str] = None
    comparison_height: int = 0    # 0 = use SR height
    crf: int = 18


@dataclass
class RunResult:
    out_dir: Path
    upscaled_path: Optional[Path]
    comparison_path: Optional[Path]
    n_source: int
    n_sr: int
    wall_time_s: float
    e2e_fps: float


def derive_out_dir(root: Path, model_id: str, video_path: Path,
                   opts: RunOptions, lr_w: int, lr_h: int, quant: str,
                   sage: bool) -> Path:
    stem = video_path.stem
    parts = [
        stem,
        f"scale{opts.out_scale}x",
        f"{lr_w}x{lr_h}",
        f"skip{opts.frame_skip}",
        f"interp-{opts.frame_interp}",
        f"q-{quant}",
        f"sage-{'on' if sage else 'off'}",
    ]
    if opts.seconds:
        parts.insert(1, f"sec{int(round(opts.seconds))}")
    return root / model_id / "_".join(parts)


def process_video(
    upscaler: BaseUpscaler,
    video_path: Path,
    out_root: Path,
    opts: RunOptions,
    model_id: str,
    quant: str,
    sage: bool,
    rife=None,
) -> RunResult:
    meta = io_utils.probe(video_path)
    fps = meta.fps or 25.0

    # 1) read source frames (full-res, used for side-by-side)
    src_frames = []
    for f in io_utils.iter_frames(video_path, max_seconds=opts.seconds, fps_hint=fps):
        src_frames.append(f)
    if not src_frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    src = np.stack(src_frames, axis=0)
    n_src = src.shape[0]

    # 2) decide low-res input dims for the SR model
    lr_w, lr_h = io_utils.resolve_preprocess(meta.width, meta.height, opts.pre_resize)
    if (lr_w, lr_h) != (meta.width, meta.height):
        lr = io_utils.resize_batch(src, lr_w, lr_h, "bicubic")
    else:
        lr = src

    # 3) pick anchor frames (frame skipping)
    skip = max(1, opts.frame_skip)
    anchor_idx = np.arange(0, len(lr), skip)
    anchors_lr = lr[anchor_idx]

    # 4) SR forward (delegated to model). Defer out_dir creation until after the
    #    SR step so OOM/error paths don't leave empty folders littering output/.
    out_dir = derive_out_dir(out_root, model_id, video_path, opts, lr_w, lr_h, quant, sage)
    t0 = time.perf_counter()
    sr_anchors = upscaler.upscale(anchors_lr)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) downscale SR to requested out_scale if model native_scale > requested
    nat = upscaler.native_scale
    if opts.out_scale != nat:
        if opts.out_scale > nat:
            raise ValueError(
                f"requested out_scale {opts.out_scale}x exceeds model native "
                f"{nat}x; FlashVSR/Real-ESRGAN here only go up to 4x."
            )
        ratio = opts.out_scale / nat
        new_w = int(round(sr_anchors.shape[2] * ratio))
        new_h = int(round(sr_anchors.shape[1] * ratio))
        sr_anchors = io_utils.resize_batch(sr_anchors, new_w, new_h, "bicubic")

    # 6) fill skipped frames
    if skip > 1 and opts.frame_interp == "rife":
        sr_full = frame_skip.fill_rife(sr_anchors, n_src, skip, rife)
    elif skip > 1:
        sr_full = frame_skip.fill_repeat(sr_anchors, n_src, skip)
    else:
        sr_full = sr_anchors

    wall = time.perf_counter() - t0
    fps_e2e = n_src / wall if wall > 0 else 0.0

    # 7) write outputs
    up_path = out_dir / "upscaled.mp4" if opts.write_upscaled else None
    cmp_path = out_dir / "comparison.mp4" if opts.write_comparison else None
    if up_path is not None:
        io_utils.encode_video(up_path, sr_full, fps=fps, crf=opts.crf)
    if cmp_path is not None:
        side = io_utils.side_by_side(src, sr_full)
        io_utils.encode_video(cmp_path, side, fps=fps, crf=opts.crf)

    # 8) report
    rep = report.RunReport(
        out_dir=out_dir,
        model=model_id,
        quant=quant,
        sage_attn=sage,
        args={
            "input": str(video_path),
            "duration_s": f"{opts.seconds}  (0 = full video)",
            "out_scale": f"{opts.out_scale}x",
            "lr_resize": f"{lr_w}x{lr_h}",
            "frame_skip": opts.frame_skip,
            "frame_interp": opts.frame_interp,
            "out_res": f"{sr_full.shape[2]}x{sr_full.shape[1]}",
        },
        timing={
            "source_frames": n_src,
            "sr_frames": int(len(sr_anchors)),
            "wall_time_s": f"{wall:.2f}",
            "e2e_fps": f"{fps_e2e:.2f}",
            "src_fps": f"{fps:.2f}",
        },
    )
    rep.write()

    return RunResult(out_dir, up_path, cmp_path, n_src, len(sr_anchors), wall, fps_e2e)
