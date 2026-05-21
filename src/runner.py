"""Per-video orchestration: read -> (resize) -> skip+SR -> (interp) -> write.

Two paths:

- Default: collect frames, chunk through SR, write outputs. Supports
  side-by-side comparison and frame-skip + RIFE gap-fill.
- ``opts.live=True``: streaming producer/consumer threading. Decode,
  SR, and encode run on their own threads so CPU I/O overlaps with GPU
  SR. Comparison output is force-disabled. Frame-skip is forced to 1.
"""
from __future__ import annotations

import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import io_utils, frame_skip, report
from .models.base import BaseUpscaler


def _inference_ctx():
    """Wrap SR forward in ``torch.inference_mode()`` when torch is loaded.
    Falls back to a no-op context manager if torch import fails (e.g. CPU
    smoke tests). ``inference_mode`` is faster than ``no_grad`` because it
    also disables version counter bookkeeping on tensors."""
    try:
        import torch
        return torch.inference_mode()
    except Exception:
        return nullcontext()


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
    run_tag: str = ""
    cli_args: dict = field(default_factory=dict)
    # Live-streaming knobs
    live: bool = False
    encoder: str = "libx264"      # libx264 | h264_nvenc | auto
    io_queue_depth: int = 2       # bounded queue depth between threads
    preview: bool = False         # open two real-time cv2 windows in live mode


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
    if opts.run_tag:
        safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in opts.run_tag)
        parts.append(safe)
    if opts.seconds:
        parts.insert(1, f"sec{int(round(opts.seconds))}")
    return root / model_id / "_".join(parts)


class _Slot:
    """Thread-safe single-slot latest-frame holder. ``put`` overwrites
    silently; ``peek`` returns the most recent value or None. Used to
    decouple the cv2 preview rendering rhythm from the SR throughput so
    the 'Original' window stays smooth even when SR can't keep up."""

    __slots__ = ("_lock", "_frame")

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None

    def put(self, f: np.ndarray) -> None:
        with self._lock:
            self._frame = f

    def peek(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame


def _live_stream(
    upscaler: BaseUpscaler,
    video_path: Path,
    out_root: Path,
    opts: RunOptions,
    model_id: str,
    quant: str,
    sage: bool,
    meta: io_utils.VideoMeta,
    fps: float,
) -> RunResult:
    """Threaded streaming path with optional dual-window preview.

    Pipeline (when ``opts.preview`` is on):

        decode_thread  -> orig_slot      (paced at source fps)
                       \\-> sr_q ----> sr_thread -> sr_slot, encode_q
                                                    -> encode_thread -> mp4

        main_thread (display): ticks at source fps,
                               cv2.imshow(orig_slot, sr_slot)

    The 'Original' window is decoupled from SR throughput — it always
    plays at the source frame rate. The 'Upscaled' window updates as
    soon as a new SR frame is ready; if SR can't keep up, it holds the
    last rendered SR until the next one lands. To keep that decoupling
    real, the decode worker paces itself to source fps and DROPS chunks
    going into the SR queue when it is full (orig_slot still updates).
    The mp4 contains every SR frame that completed, so a slow SR yields
    a sparser mp4 — that's the realistic 'live' tradeoff.

    With ``opts.preview`` off, the pipeline runs as fast as possible
    (decode + SR + encode each on their own thread, no pacing, no
    drops); behavior is identical to a normal batch run."""
    lr_w, lr_h = io_utils.resolve_preprocess(meta.width, meta.height, opts.pre_resize)
    out_dir = derive_out_dir(out_root, model_id, video_path, opts, lr_w, lr_h, quant, sage)
    out_dir.mkdir(parents=True, exist_ok=True)
    up_path = out_dir / "upscaled.mp4" if opts.write_upscaled else None
    encoder_name = io_utils.resolve_encoder(opts.encoder)
    up_writer = (io_utils.VideoWriter(up_path, fps=fps, crf=opts.crf,
                                      encoder=encoder_name)
                 if up_path else None)
    if up_path is None and not opts.preview:
        # Nothing to write and nothing to show — caller asked for both
        # --no-upscaled and --no-preview; bail rather than spin threads
        # for no reason.
        return RunResult(out_dir, None, None, 0, 0, 0.0, 0.0)

    depth = max(1, int(opts.io_queue_depth))
    sr_q: queue.Queue = queue.Queue(maxsize=depth)
    encode_q: queue.Queue = queue.Queue(maxsize=depth)
    SENTINEL = object()
    orig_slot = _Slot() if opts.preview else None
    sr_slot = _Slot() if opts.preview else None

    peak_vram_mb = 0
    out_w = out_h = 0
    n_src = 0
    n_dropped = 0
    error_box: list[BaseException] = []
    abort_event = threading.Event()
    target_dt = 1.0 / max(fps, 1.0)

    def _decode_worker():
        nonlocal n_dropped
        # Pacing wall-clock so we feed frames into SR at most at source
        # fps under preview. Without preview, no pacing and no drops.
        deadline = time.perf_counter()
        try:
            for chunk in io_utils.iter_frame_chunks(
                video_path,
                chunk_size=max(1, opts.chunk_frames),
                max_seconds=opts.seconds,
                fps_hint=fps,
            ):
                if abort_event.is_set():
                    break
                if (lr_w, lr_h) != (meta.width, meta.height):
                    lr_chunk = io_utils.resize_batch(chunk, lr_w, lr_h, "bicubic")
                else:
                    lr_chunk = chunk
                if opts.preview and orig_slot is not None:
                    # Update the slot per-frame, pacing so the Original
                    # window plays at source fps. This is intentionally
                    # done frame-by-frame (not chunk-by-chunk) so the
                    # display thread sees smooth motion even on chunk=4.
                    for f in chunk:
                        if abort_event.is_set():
                            break
                        now = time.perf_counter()
                        if deadline > now:
                            time.sleep(deadline - now)
                        orig_slot.put(f)
                        deadline += target_dt
                    # Drop the SR chunk if SR can't keep up — preview
                    # mode prioritizes smooth playback over completeness.
                    try:
                        sr_q.put_nowait(lr_chunk)
                    except queue.Full:
                        n_dropped += lr_chunk.shape[0]
                else:
                    sr_q.put(lr_chunk)
        except BaseException as e:
            error_box.append(e)
        finally:
            sr_q.put(SENTINEL)

    def _sr_worker():
        nonlocal n_src, out_w, out_h
        try:
            with _inference_ctx():
                while True:
                    item = sr_q.get()
                    if item is SENTINEL:
                        break
                    if abort_event.is_set():
                        continue
                    lr_chunk = item
                    sr_chunk = upscaler.upscale(lr_chunk)
                    nat = upscaler.native_scale
                    if opts.out_scale != nat:
                        if opts.out_scale > nat:
                            raise ValueError(
                                f"requested out_scale {opts.out_scale}x exceeds "
                                f"model native {nat}x"
                            )
                        ratio = opts.out_scale / nat
                        new_w = int(round(sr_chunk.shape[2] * ratio))
                        new_h = int(round(sr_chunk.shape[1] * ratio))
                        sr_chunk = io_utils.resize_batch(sr_chunk, new_w, new_h, "bicubic")
                    out_h, out_w = sr_chunk.shape[1:3]
                    n_src += sr_chunk.shape[0]
                    if opts.preview and sr_slot is not None:
                        sr_slot.put(sr_chunk[-1])
                    encode_q.put(sr_chunk)
        except BaseException as e:
            error_box.append(e)
        finally:
            encode_q.put(SENTINEL)

    def _encode_worker():
        try:
            while True:
                item = encode_q.get()
                if item is SENTINEL:
                    break
                if up_writer is not None:
                    up_writer.append(item)
        except BaseException as e:
            error_box.append(e)

    def _main_thread_display(workers_done: threading.Event):
        # Runs in the main thread. Tick at source fps via cv2.waitKey.
        # ESC or 'q' aborts. Exits when workers_done is set AND the
        # final tick has rendered the last SR frame.
        cv2 = None
        windows_ready = False
        try:
            import cv2  # type: ignore
            cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Upscaled", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Original", meta.width, meta.height)
            cv2.moveWindow("Original", 0, 0)
            cv2.resizeWindow("Upscaled", meta.width, meta.height)
            cv2.moveWindow("Upscaled", meta.width + 30, 0)
            windows_ready = True
        except Exception as e:
            print(f"[runner] preview disabled — cv2 setup failed: {e}")
            cv2 = None

        tick_ms = max(1, int(round(target_dt * 1000)))
        try:
            while True:
                if windows_ready and cv2 is not None:
                    orig = orig_slot.peek() if orig_slot is not None else None
                    sr = sr_slot.peek() if sr_slot is not None else None
                    if orig is not None:
                        cv2.imshow("Original",
                                   cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
                    if sr is not None:
                        cv2.imshow("Upscaled",
                                   cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))
                    k = cv2.waitKey(tick_ms) & 0xFF
                    if k in (ord("q"), 27):
                        abort_event.set()
                        break
                else:
                    time.sleep(target_dt)
                if workers_done.is_set():
                    # Final tick to flush the last SR onto screen.
                    if windows_ready and cv2 is not None:
                        cv2.waitKey(tick_ms)
                    break
        finally:
            if windows_ready and cv2 is not None:
                try:
                    cv2.destroyAllWindows()
                    for _ in range(3):
                        cv2.waitKey(1)
                except Exception:
                    pass

    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    t0 = time.perf_counter()
    workers_done = threading.Event()
    workers = [
        threading.Thread(target=_decode_worker, name="decode", daemon=True),
        threading.Thread(target=_sr_worker, name="sr", daemon=True),
        threading.Thread(target=_encode_worker, name="encode", daemon=True),
    ]
    for t in workers:
        t.start()

    if opts.preview:
        # Display loop owns the main thread until the workers finish, so
        # the cv2 event pump keeps ticking even while SR is busy.
        def _wait_then_signal():
            for t in workers:
                t.join()
            workers_done.set()
        joiner = threading.Thread(target=_wait_then_signal, daemon=True)
        joiner.start()
        _main_thread_display(workers_done)
        joiner.join()
    else:
        for t in workers:
            t.join()

    if up_writer is not None:
        up_writer.close()
    wall = time.perf_counter() - t0

    if error_box:
        raise error_box[0]

    try:
        import torch as _t
        if _t.cuda.is_available():
            peak_vram_mb = int(_t.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        pass

    fps_e2e = n_src / wall if wall > 0 else 0.0
    latency_src_ms = (wall / n_src * 1000.0) if n_src else 0.0

    rep = report.RunReport(
        out_dir=out_dir,
        model=model_id,
        quant=quant,
        sage_attn=sage,
        args={
            **opts.cli_args,
            "input": str(video_path),
            "duration_s": f"{opts.seconds}  (0 = full video)",
            "out_scale": f"{opts.out_scale}x",
            "lr_resize": f"{lr_w}x{lr_h}",
            "frame_skip": opts.frame_skip,
            "frame_interp": opts.frame_interp,
            "out_res": f"{out_w}x{out_h}",
            "live": True,
            "encoder": encoder_name,
            "preview": opts.preview,
        },
        timing={
            "source_frames": n_src,
            "sr_frames": n_src,
            "wall_time_s": f"{wall:.2f}",
            "e2e_fps": f"{fps_e2e:.2f}",
            "latency_ms_per_source_frame": f"{latency_src_ms:.2f}",
            "latency_ms_per_sr_frame": f"{latency_src_ms:.2f}",
            "peak_vram_mb": peak_vram_mb,
            "src_fps": f"{fps:.2f}",
        },
    )
    rep.write()
    print(f"[runner] live mode encoder={encoder_name} chunk={opts.chunk_frames} "
          f"queue_depth={depth} preview={'on' if opts.preview else 'off'} "
          f"dropped={n_dropped}")
    return RunResult(out_dir, up_path, None, n_src, n_src, wall, fps_e2e)


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

    # Live-streaming path: producer/consumer threads, no upfront frame
    # collection, no comparison output, no frame-skip.
    if opts.live:
        return _live_stream(
            upscaler=upscaler, video_path=video_path, out_root=out_root,
            opts=opts, model_id=model_id, quant=quant, sage=sage, meta=meta,
            fps=fps,
        )

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

    # Most face-quality sweeps use skip=1/interp=none. Stream that path in
    # chunks so 10s 4x outputs do not allocate multi-GB SR/comparison arrays.
    if skip == 1 and opts.frame_interp == "none":
        out_dir = derive_out_dir(out_root, model_id, video_path, opts, lr_w, lr_h, quant, sage)
        out_dir.mkdir(parents=True, exist_ok=True)
        up_path = out_dir / "upscaled.mp4" if opts.write_upscaled else None
        cmp_path = out_dir / "comparison.mp4" if opts.write_comparison else None
        encoder_name = io_utils.resolve_encoder(opts.encoder)
        up_writer = (io_utils.VideoWriter(up_path, fps=fps, crf=opts.crf,
                                          encoder=encoder_name)
                     if up_path else None)
        cmp_writer = (io_utils.VideoWriter(cmp_path, fps=fps, crf=opts.crf,
                                           encoder=encoder_name)
                      if cmp_path else None)
        peak_vram_mb = 0
        out_w = out_h = 0
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        t0 = time.perf_counter()
        try:
            with _inference_ctx():
                for start in range(0, len(lr), max(1, opts.chunk_frames)):
                    end = min(len(lr), start + max(1, opts.chunk_frames))
                    sr_chunk = upscaler.upscale(lr[start:end])
                    nat = upscaler.native_scale
                    if opts.out_scale != nat:
                        if opts.out_scale > nat:
                            raise ValueError(
                                f"requested out_scale {opts.out_scale}x exceeds model native "
                                f"{nat}x; Real-ESRGAN here only goes up to 4x."
                            )
                        ratio = opts.out_scale / nat
                        new_w = int(round(sr_chunk.shape[2] * ratio))
                        new_h = int(round(sr_chunk.shape[1] * ratio))
                        sr_chunk = io_utils.resize_batch(sr_chunk, new_w, new_h, "bicubic")
                    out_h, out_w = sr_chunk.shape[1:3]
                    if up_writer is not None:
                        up_writer.append(sr_chunk)
                    if cmp_writer is not None:
                        cmp_writer.append(io_utils.side_by_side(src[start:end], sr_chunk))
        finally:
            if up_writer is not None:
                up_writer.close()
            if cmp_writer is not None:
                cmp_writer.close()
        wall = time.perf_counter() - t0
        try:
            import torch as _t
            if _t.cuda.is_available():
                peak_vram_mb = int(_t.cuda.max_memory_allocated() / (1024 * 1024))
        except Exception:
            pass
        fps_e2e = n_src / wall if wall > 0 else 0.0
        latency_src_ms = (wall / n_src * 1000.0) if n_src else 0.0
        latency_sr_ms = latency_src_ms
        rep = report.RunReport(
            out_dir=out_dir,
            model=model_id,
            quant=quant,
            sage_attn=sage,
            args={
                **opts.cli_args,
                "input": str(video_path),
                "duration_s": f"{opts.seconds}  (0 = full video)",
                "out_scale": f"{opts.out_scale}x",
                "lr_resize": f"{lr_w}x{lr_h}",
                "frame_skip": opts.frame_skip,
                "frame_interp": opts.frame_interp,
                "out_res": f"{out_w}x{out_h}",
            },
            timing={
                "source_frames": n_src,
                "sr_frames": n_src,
                "wall_time_s": f"{wall:.2f}",
                "e2e_fps": f"{fps_e2e:.2f}",
                "latency_ms_per_source_frame": f"{latency_src_ms:.2f}",
                "latency_ms_per_sr_frame": f"{latency_sr_ms:.2f}",
                "peak_vram_mb": peak_vram_mb,
                "src_fps": f"{fps:.2f}",
            },
        )
        rep.write()
        return RunResult(out_dir, up_path, cmp_path, n_src, n_src, wall, fps_e2e)

    # 4) SR forward (delegated to model). Defer out_dir creation until after the
    #    SR step so OOM/error paths don't leave empty folders littering output/.
    out_dir = derive_out_dir(out_root, model_id, video_path, opts, lr_w, lr_h, quant, sage)
    peak_vram_mb = 0
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    t0 = time.perf_counter()
    with _inference_ctx():
        sr_anchors = upscaler.upscale(anchors_lr)
    try:
        import torch as _t
        if _t.cuda.is_available():
            peak_vram_mb = int(_t.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        pass
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
    latency_src_ms = (wall / n_src * 1000.0) if n_src else 0.0
    latency_sr_ms = (wall / len(sr_anchors) * 1000.0) if len(sr_anchors) else 0.0

    # 7) write outputs
    up_path = out_dir / "upscaled.mp4" if opts.write_upscaled else None
    cmp_path = out_dir / "comparison.mp4" if opts.write_comparison else None
    encoder_name = io_utils.resolve_encoder(opts.encoder)
    if up_path is not None:
        io_utils.encode_video(up_path, sr_full, fps=fps, crf=opts.crf,
                              encoder=encoder_name)
    if cmp_path is not None:
        side = io_utils.side_by_side(src, sr_full)
        io_utils.encode_video(cmp_path, side, fps=fps, crf=opts.crf,
                              encoder=encoder_name)

    # 8) report
    rep = report.RunReport(
        out_dir=out_dir,
        model=model_id,
        quant=quant,
        sage_attn=sage,
        args={
            **opts.cli_args,
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
            "latency_ms_per_source_frame": f"{latency_src_ms:.2f}",
            "latency_ms_per_sr_frame": f"{latency_sr_ms:.2f}",
            "peak_vram_mb": peak_vram_mb,
            "src_fps": f"{fps:.2f}",
        },
    )
    rep.write()

    return RunResult(out_dir, up_path, cmp_path, n_src, len(sr_anchors), wall, fps_e2e)
