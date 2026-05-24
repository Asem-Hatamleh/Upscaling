"""
Live stream face-blur pipeline.

Architecture:
    [LiveFeeder NodePlayer thread]
        produces (capture_ts, frame) -> capture_queue
    [Batch detector worker thread]
        pulls up to BATCH_SIZE frames -> detector.detect_batch (GPU)
        applies ghost-tracker, merge, blur -> out_queue
    [Main thread / display]
        consumes out_queue, measures end-to-end latency

Latency stages reported:
    capture -> queue   : frame sit in capture_queue
    detect (GPU)       : batch inference
    post (ghost+blur)  : CPU per-frame
    end-to-end         : capture_ts -> display_ts
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import platform
import queue
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from blur_faces import (
    GhostTracker,
    RetinaFaceDetector,
    YoloDetector,
    blur_region,
    expand_box,
    iou_with_roi,
    merge_overlapping_boxes,
    parse_roi,
    driver_side_to_roi,
)
from LiveFeeder import NodePlayer

# Upscaling src: sibling subdir (Acacus-FB layout) OR flat root (FullPipeline branch)
for _root in (Path(__file__).parent / "Upscaling", Path(__file__).parent):
    if (_root / "src" / "models" / "_perf.py").exists() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
        break


MEDIAMTX_VERSION = "1.9.3"
MEDIAMTX_DIR = Path(__file__).parent / "bin"


def _mediamtx_release() -> tuple[str, str, str]:
    """Return (download_url, archive_name, exe_name) for the current OS."""
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    base = f"https://github.com/bluenviron/mediamtx/releases/download/v{MEDIAMTX_VERSION}"
    if sys_name == "windows":
        return (
            f"{base}/mediamtx_v{MEDIAMTX_VERSION}_windows_amd64.zip",
            "mediamtx.zip",
            "mediamtx.exe",
        )
    if sys_name == "linux":
        arch = "arm64v8" if ("aarch64" in machine or "arm64" in machine) else "amd64"
        return (
            f"{base}/mediamtx_v{MEDIAMTX_VERSION}_linux_{arch}.tar.gz",
            "mediamtx.tar.gz",
            "mediamtx",
        )
    if sys_name == "darwin":
        arch = "arm64" if ("arm64" in machine or "aarch64" in machine) else "amd64"
        return (
            f"{base}/mediamtx_v{MEDIAMTX_VERSION}_darwin_{arch}.tar.gz",
            "mediamtx.tar.gz",
            "mediamtx",
        )
    raise RuntimeError(f"unsupported platform: {sys_name}/{machine}")


def ensure_mediamtx() -> Path:
    """Download mediamtx for the current OS on first run. Return path to executable."""
    url, archive_name, exe_name = _mediamtx_release()
    exe = MEDIAMTX_DIR / exe_name
    if exe.exists():
        return exe
    MEDIAMTX_DIR.mkdir(exist_ok=True)
    archive_path = MEDIAMTX_DIR / archive_name
    print(f"[mediamtx] downloading v{MEDIAMTX_VERSION} for {platform.system()} from github...")
    urllib.request.urlretrieve(url, archive_path)
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(MEDIAMTX_DIR)
    else:
        with tarfile.open(archive_path, "r:gz") as t:
            t.extractall(MEDIAMTX_DIR)
    archive_path.unlink(missing_ok=True)
    if not exe.exists():
        raise RuntimeError(f"mediamtx executable missing after extract: {exe}")
    try:
        exe.chmod(0o755)
    except Exception:
        pass
    return exe


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _kill_stale_mediamtx() -> bool:
    """SIGTERM any running mediamtx process. Returns True if any killed."""
    killed = False
    try:
        out = subprocess.run(["pgrep", "-f", "mediamtx"], capture_output=True, text=True, timeout=2)
        pids = [p for p in (out.stdout or "").split() if p.strip().isdigit()]
        if pids:
            subprocess.run(["kill", *pids], check=False)
            time.sleep(0.7)
            killed = True
            print(f"[mediamtx] killed stale pids: {pids}")
    except FileNotFoundError:
        pass
    return killed


def start_mediamtx() -> subprocess.Popen | None:
    """Start mediamtx. If RTMP :1935 already serving, reuse it (return None).
    If a stale mediamtx blocks ports, kill it and retry once."""
    if _port_open("127.0.0.1", 1935):
        print("[mediamtx] RTMP 1935 already up; reusing existing server")
        return None
    exe = ensure_mediamtx()

    def _spawn() -> subprocess.Popen:
        print(f"[mediamtx] starting {exe}")
        return subprocess.Popen(
            [str(exe)],
            cwd=str(MEDIAMTX_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    proc = _spawn()
    time.sleep(2.0)
    if proc.poll() is not None:
        out = proc.stdout.read() if proc.stdout else ""
        # one retry after killing stale mediamtx (port-conflict recovery)
        if "address already in use" in out.lower() or "bind" in out.lower():
            print("[mediamtx] port conflict; killing stale instance and retrying")
            _kill_stale_mediamtx()
            proc = _spawn()
            time.sleep(2.0)
        if proc.poll() is not None:
            out2 = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"mediamtx died:\nfirst:\n{out}\nretry:\n{out2}")
    print("[mediamtx] up: RTMP=1935 HLS=8888 WebRTC=8889 RTSP=8554")
    return proc


def _enable_torch_perf():
    """Global torch perf knobs: cudnn.benchmark + TF32."""
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[perf] cudnn.benchmark=True TF32 ON")
    except Exception as e:
        print(f"[perf] torch knobs skipped: {e}")


def build_upscaler(args):
    """Build realesrgan_lite. channels_last + torch.compile applied inside RealESRGANLite.load()."""
    from src.models.base import UpscalerConfig, build  # type: ignore
    compile_on = not args.no_compile
    extra = {
        "model_name": "realesr-general-x4v3",
        "denoise_strength": args.upscale_denoise,
        "tile": 0,
        "compile": compile_on,
    }
    cfg = UpscalerConfig(
        name=args.upscale_model,
        scale=args.upscale_scale,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=args.upscale_dtype,
        extra=extra,
    )
    up = build(cfg)
    up.load()
    print(
        f"[upscale] loaded {args.upscale_model} scale={args.upscale_scale} "
        f"dtype={args.upscale_dtype} denoise={args.upscale_denoise} "
        f"channels_last=True compile={compile_on}"
    )
    return up


def batched_esrgan_infer(rgan_obj, frames_bgr: np.ndarray, half: bool, device: str, scale: int = 4, mod_pad: int = 2) -> np.ndarray:
    """Batched forward w/ channels_last + reflect pad to mod_pad multiple.
    Input  (N,H,W,3) uint8 BGR. Output (N,H*scale,W*scale,3) uint8 BGR.
    """
    model = rgan_obj.upsampler.model
    bgr = np.ascontiguousarray(frames_bgr[..., ::-1])  # to RGB
    with torch.inference_mode():
        t = torch.from_numpy(bgr).to(device, non_blocking=True)
        t = t.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        t = (t.half() if half else t.float()) / 255.0
        h, w = t.shape[-2:]
        pad_h = (mod_pad - h % mod_pad) % mod_pad
        pad_w = (mod_pad - w % mod_pad) % mod_pad
        if pad_h or pad_w:
            t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
        sr = model(t)
        if pad_h or pad_w:
            sr = sr[..., : h * scale, : w * scale]
        sr = (sr.clamp(0, 1) * 255.0).to(torch.uint8)
        arr = sr.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return np.ascontiguousarray(arr[..., ::-1])  # back to BGR


_NVENC_PROBE_RESULT: bool | None = None


def _ffmpeg_has_nvenc() -> bool:
    """Probe ffmpeg AND the GPU/driver for working h264_nvenc.

    `ffmpeg -encoders` only tells us the encoder is compiled in -- it does
    not tell us the device can open an NVENC session (Laptop GPUs without
    NVENC silicon, missing driver bits, or hybrid-graphics setups all fail
    at OpenEncodeSessionEx). We do a real 64x64 encode to confirm.
    Cached after first call.
    """
    global _NVENC_PROBE_RESULT
    if _NVENC_PROBE_RESULT is not None:
        return _NVENC_PROBE_RESULT
    try:
        listed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=3,
        )
        if "h264_nvenc" not in (listed.stdout or ""):
            _NVENC_PROBE_RESULT = False
            return False
        # Real session probe: try a 256x256x1-frame nullsrc encode.
        # NVENC h264 min frame size is ~145x49 (Maxwell+); older 64x64 probe
        # tripped "Frame Dimension less than the minimum supported value".
        probe = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "nullsrc=s=256x256:r=15",
                "-t", "0.1",
                "-c:v", "h264_nvenc", "-preset", "p1",
                "-pix_fmt", "yuv420p",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=5,
        )
        ok = probe.returncode == 0 and "OpenEncodeSessionEx" not in (probe.stderr or "")
        if not ok:
            err = (probe.stderr or "").strip().splitlines()[:3]
            print(f"[nvenc] probe failed -> libx264 fallback. ffmpeg said: {' | '.join(err) or 'no stderr'}")
        _NVENC_PROBE_RESULT = ok
        return ok
    except Exception as e:
        print(f"[nvenc] probe error -> libx264 fallback: {e}")
        _NVENC_PROBE_RESULT = False
        return False


def start_ffmpeg_pusher(
    width: int,
    height: int,
    fps: float,
    rtmp_url: str,
    encoder: str = "auto",
    preset: str = "veryfast",
    crf: int = 20,
    bitrate: str | None = None,
) -> subprocess.Popen:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg first.")

    if encoder == "auto":
        encoder = "h264_nvenc" if _ffmpeg_has_nvenc() else "libx264"
    elif encoder == "h264_nvenc" and not _ffmpeg_has_nvenc():
        print("[ffmpeg] requested h264_nvenc but ffmpeg lacks it; falling back to libx264")
        encoder = "libx264"

    gop = str(max(2, int(fps)))
    if encoder == "h264_nvenc":
        # NVENC: prefer constqp w/ a quality target, or fall back to bitrate cap if user sets one.
        venc = [
            "-c:v", "h264_nvenc",
            "-preset", "p4",      # p1=fastest/worst, p7=slowest/best. p4 = balanced.
            "-tune", "ll",
            "-zerolatency", "1",
            "-pix_fmt", "yuv420p",
            "-g", gop,
        ]
        if bitrate:
            venc += ["-rc", "cbr", "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate]
        else:
            venc += ["-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    else:
        # libx264: CRF mode gives best quality-per-byte. veryfast preset balances CPU vs quality.
        venc = [
            "-c:v", "libx264",
            "-preset", preset,
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-g", gop,
            "-keyint_min", gop,
        ]
        if bitrate:
            venc += ["-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate]
        else:
            venc += ["-crf", str(crf)]

    cmd = [
        "ffmpeg",
        "-loglevel", "warning",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        *venc,
        "-f", "flv",
        rtmp_url,
    ]
    print(f"[ffmpeg] push -> {rtmp_url} encoder={encoder} input={width}x{height}@{fps:.1f}")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Drain stderr so the pipe doesn't fill + so we can see WHY ffmpeg dies.
    def _drain_stderr(p):
        try:
            for raw in iter(p.stderr.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    print(f"[ffmpeg] {line}", flush=True)
        except Exception:
            pass
    threading.Thread(target=_drain_stderr, args=(proc,), daemon=True).start()
    return proc


@dataclass
class FrameItem:
    capture_ts: float
    enqueue_ts: float
    frame: np.ndarray
    seq: int


@dataclass
class OutItem:
    capture_ts: float
    enqueue_ts: float
    detect_start_ts: float
    detect_end_ts: float
    post_end_ts: float
    upscale_end_ts: float
    frame: np.ndarray
    seq: int
    n_faces: int


def capture_thread(player: NodePlayer, capture_q: queue.Queue, stop_event: threading.Event, max_seq: int | None):
    """Run NodePlayer in its own event loop; push frames w/ timestamps."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def runner():
        task = loop.create_task(player.run())
        seq = 0
        while not task.done() and not stop_event.is_set():
            try:
                frame = player.frame_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue
            ts = time.time()
            item = FrameItem(capture_ts=ts, enqueue_ts=ts, frame=frame, seq=seq)
            try:
                capture_q.put_nowait(item)
            except queue.Full:
                # drop oldest, keep latest (live streaming policy)
                try:
                    capture_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    capture_q.put_nowait(item)
                except queue.Full:
                    pass
            seq += 1
            if max_seq is not None and seq >= max_seq:
                stop_event.set()
                break
        stop_event.set()
        try:
            await task
        except Exception:
            pass

    try:
        loop.run_until_complete(runner())
    finally:
        loop.close()


def detect_worker(
    detector,
    capture_q: queue.Queue,
    next_q: queue.Queue,
    out_q: queue.Queue,
    stop_event: threading.Event,
    args,
    ghost: GhostTracker | None,
    clahe,
    roi,
    run_up: bool,
):
    """Pull up to BATCH_SIZE frames, run batched detection, apply ghost+blur, enqueue out."""
    batch_size = args.batch
    batch_timeout = args.batch_timeout / 1000.0

    while not stop_event.is_set():
        batch: list[FrameItem] = []
        deadline = time.time() + batch_timeout
        try:
            first = capture_q.get(timeout=0.1)
            batch.append(first)
        except queue.Empty:
            continue

        while len(batch) < batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                batch.append(capture_q.get(timeout=remaining))
            except queue.Empty:
                break

        # preprocess (CLAHE) per frame
        det_frames = []
        for item in batch:
            f = item.frame
            if clahe is not None:
                lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                f = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            det_frames.append(f)

        # batched inference (skip when no detector)
        detect_start = time.time()
        if detector is None:
            results = [(np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32))] * len(det_frames)
        elif isinstance(detector, RetinaFaceDetector):
            results = detector.detect_batch(det_frames, conf=args.conf)
        else:
            results = [detector.detect(f, imgsz=args.imgsz, conf=args.conf, iou=args.iou) for f in det_frames]
        detect_end = time.time()

        # post: ghost + merge + blur (ghost is stateful, must be serial)
        run_blur = args.service in ("blur", "both")
        for item, (xyxy, _confs) in zip(batch, results):
            n_faces = 0
            if run_blur:
                if ghost is not None:
                    boxes = ghost.update(xyxy if len(xyxy) else np.empty((0, 4), dtype=np.float32))
                else:
                    boxes = xyxy
                h, w = item.frame.shape[:2]
                if len(boxes) > 0:
                    expanded = np.array([expand_box(b, args.expand, w, h) for b in boxes], dtype=np.float32)
                    expanded = merge_overlapping_boxes(expanded, iou_thresh=args.merge_iou)
                    for box in expanded:
                        if roi and iou_with_roi(box, roi) > 0.5:
                            continue
                        blur_region(item.frame, box, args.method, args.strength)
                        n_faces += 1
            post_end = time.time()

            out = OutItem(
                capture_ts=item.capture_ts,
                enqueue_ts=item.enqueue_ts,
                detect_start_ts=detect_start,
                detect_end_ts=detect_end,
                post_end_ts=post_end,
                upscale_end_ts=post_end,  # no upscale yet; set by upscale_worker if it runs
                frame=item.frame,
                seq=item.seq,
                n_faces=n_faces,
            )
            target_q = next_q if run_up else out_q
            try:
                target_q.put_nowait(out)
            except queue.Full:
                try:
                    target_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    target_q.put_nowait(out)
                except queue.Full:
                    pass


def upscale_worker(
    upscaler,
    mid_q: queue.Queue,
    out_q: queue.Queue,
    stop_event: threading.Event,
    args,
    device: str,
):
    """Batch-pull from mid_q, native batched ESRGAN forward, push to out_q."""
    batch_size = args.upscale_batch
    pre = args.upscale_pre_resize
    half = args.upscale_dtype == "fp16"

    # ---- adaptive batch-timeout state ----
    # If user passed a positive value, honor it as a fixed timeout.
    # Otherwise (default 0), derive from observed inter-arrival via EMA.
    use_adaptive = args.upscale_batch_timeout <= 0.0
    fixed_timeout_s = (args.upscale_batch_timeout / 1000.0) if not use_adaptive else None
    ema_dt = 1.0 / 25.0       # bootstrap: assume 25 fps source
    ema_alpha = 0.1
    last_arrival = None
    safety = 1.2
    min_timeout_s = 0.005
    max_timeout_s = 0.200
    last_log_t = 0.0

    def _record_arrival():
        nonlocal last_arrival, ema_dt
        now = time.time()
        if last_arrival is not None:
            dt = now - last_arrival
            if 0.001 < dt < 1.0:
                ema_dt = (1.0 - ema_alpha) * ema_dt + ema_alpha * dt
        last_arrival = now

    def _current_timeout_s() -> float:
        if not use_adaptive:
            return fixed_timeout_s  # type: ignore[return-value]
        t = batch_size * ema_dt * safety
        return max(min_timeout_s, min(max_timeout_s, t))

    if use_adaptive:
        print(f"[upscale] batch_timeout=adaptive (clamped {min_timeout_s*1000:.0f}-{max_timeout_s*1000:.0f}ms, "
              f"safety={safety}, bootstrap fps=25)")
    else:
        print(f"[upscale] batch_timeout=fixed {fixed_timeout_s*1000:.1f}ms (user override)")

    while not stop_event.is_set():
        timeout_s = _current_timeout_s()
        batch: list[OutItem] = []
        deadline = time.time() + timeout_s
        try:
            batch.append(mid_q.get(timeout=0.1))
            _record_arrival()
        except queue.Empty:
            continue
        while len(batch) < batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                batch.append(mid_q.get(timeout=remaining))
                _record_arrival()
            except queue.Empty:
                break

        # periodic visibility into adaptive state (every ~5s)
        if use_adaptive and (time.time() - last_log_t) > 5.0:
            last_log_t = time.time()
            fps_est = 1.0 / max(ema_dt, 1e-6)
            print(f"[upscale] adaptive timeout={timeout_s*1000:.0f}ms  "
                  f"observed_fps~{fps_est:.1f}  last_batch={len(batch)}/{batch_size}")

        # pre-resize all frames to common (h, w)
        if pre and abs(pre - 1.0) > 1e-3:
            srcs = []
            for it in batch:
                nh = int(it.frame.shape[0] * pre)
                nw = int(it.frame.shape[1] * pre)
                srcs.append(cv2.resize(it.frame, (nw, nh), interpolation=cv2.INTER_AREA))
        else:
            srcs = [it.frame for it in batch]

        # require uniform shape for stacking; group if mixed
        shape0 = srcs[0].shape
        try:
            if not all(s.shape == shape0 for s in srcs):
                outs = [batched_esrgan_infer(upscaler, s[None, ...], half, device)[0] for s in srcs]
            else:
                stacked = np.stack(srcs, axis=0)
                outs = batched_esrgan_infer(upscaler, stacked, half, device)
        except Exception as e:
            print(f"[ERR upscale] {type(e).__name__}: {e}", flush=True)
            # fall back to original frames so pipeline doesn't stall
            outs = [it.frame for it in batch]

        ts_end = time.time()
        for it, sr in zip(batch, outs):
            it.frame = sr
            it.upscale_end_ts = ts_end
            try:
                out_q.put_nowait(it)
            except queue.Full:
                try:
                    out_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    out_q.put_nowait(it)
                except queue.Full:
                    pass


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Live-stream face-blur pipeline w/ batched parallel GPU.")
    # stream (defaults preloaded for dev)
    ap.add_argument("--username", default="m.alawneh")
    ap.add_argument("--password", default="sgQZ9Oou3CsP")
    ap.add_argument("--imei", default="867395071670570") # 867395071656363
    ap.add_argument("--cam-id", type=int, default=4)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--stream-type", default="sub", choices=["sub", "main"])
    ap.add_argument("--live", action="store_true", default=True)
    # pipeline
    ap.add_argument("--batch", type=int, default=16, help="frames per detector batch") #### Asem 
    ap.add_argument("--batch-timeout", type=float, default=15.0, help="ms to wait filling batch")
    ap.add_argument("--capture-q-size", type=int, default=16)
    ap.add_argument("--out-q-size", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0=unlimited)")
    # service selector
    ap.add_argument("--service", choices=["blur", "upscale", "both"], default="blur",
                    help="blur=face-blur only, upscale=ESRGAN only, both=blur then upscale")
     ############################################################## ############################################################## ############################################################## ############################################################## ############################################################## ############################################################## ############################################################## ############################################################## ############################################################## ############################################################## ############################################################## ############################################################## ############################################################## ############################################################## ##############################################################                
    # upscaler defaults (tested best for realesrgan_lite per user) ##############################################################
    ap.add_argument("--upscale-model", default="realesrgan_lite")
    ap.add_argument("--upscale-scale", type=int, default=4, choices=[2, 4],
                    help="effective output scale. model is hardcoded 4x; "
                         "scale=2 auto-halves --upscale-pre-resize so net output is 2x")
    ap.add_argument("--upscale-pre-resize", type=float, default=0.8,
                    help="pre-resize ratio applied BEFORE SR (0.8 = shrink to 80 percent). "
                         "Combined w/ --upscale-scale to compute net scale = 4 * pre_resize")
    ap.add_argument("--upscale-dtype", choices=["fp16"], default="fp16")
    ap.add_argument("--no-compile", action="store_true", default=False,
                    help="disable torch.compile (set on Windows where triton may be missing)")
    ap.add_argument("--upscale-batch", type=int, default=2, help="frames per upscale GPU batch")
    ap.add_argument("--upscale-batch-timeout", type=float, default=0.0,
                    help="ms to wait filling upscale batch. 0=adaptive: timeout = batch_size / observed_fps * 1.2, "
                         "clamped [5..200] ms. Pass positive number to force a fixed timeout")
    ap.add_argument("--mid-q-size", type=int, default=16, help="queue between blur and upscale workers")
    ap.add_argument("--upscale-denoise", type=float, default=0, help="esrgan denoise strength 0..1") 
     ############################################################## ############################################################## ############################################################## ############################################################## ##############################################################
    # detector
    ap.add_argument("--detector", choices=["retinaface", "yolo"], default="retinaface")
    ap.add_argument("--retinaface-net", choices=["resnet50", "mobilenet"], default="resnet50")
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    # blur
    ap.add_argument("--method", choices=["gaussian", "pixelate"], default="pixelate")
    ap.add_argument("--strength", type=float, default=2.5)
    ap.add_argument("--expand", type=float, default=0.3)
    ap.add_argument("--merge-iou", type=float, default=0.1)
    ap.add_argument("--no-clahe", dest="clahe", action="store_false", default=True)
    # ghost
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--ghost-frames", type=int, default=75)
    ap.add_argument("--ghost-min-hits", type=int, default=1)
    ap.add_argument("--ghost-iou-match", type=float, default=0.3)
    # driver exclusion
    ap.add_argument("--exclude-roi", default=None)
    ap.add_argument("--driver-side", default=None, choices=["left", "right"])
    # output
    ap.add_argument("--display", action="store_true", help="show output window")
    ap.add_argument("--display-scale", type=float, default=1.0,
                    help="shrink factor for --display window (0.35 = 35 percent of SR frame; full SR is too big)")
    ap.add_argument("--display-max-w", type=int, default=1600,
                    help="hard cap on display window width in pixels (downscale if larger)")
    ap.add_argument("--show-raw", action="store_true", help="also show raw NodePlayer window")
    ap.add_argument("--rtmp-url", default="rtmp://localhost:1935/blur", help="RTMP push URL")
    ap.add_argument("--no-rtmp", action="store_true", help="disable RTMP push")
    ap.add_argument("--start-server", action="store_true", default=True, help="auto-launch mediamtx local server")
    ap.add_argument("--no-server", dest="start_server", action="store_false")
    ap.add_argument("--push-fps", type=float, default=15.0, help="RTMP encode fps (lower = less CPU)")
    ap.add_argument("--decode-fps", type=float, default=0.0,
                    help="cap ffmpeg WS decoder output fps. 0=auto (match --push-fps). Lower = fewer frames"
                         " entering pipeline = more consecutive frames, less queue churn.")
    ap.add_argument("--queue-size", type=int, default=8,
                    help="NodePlayer.frame_queue maxsize. Small keeps frames fresh, large absorbs jitter.")
    ap.add_argument("--push-max-w", type=int, default=0,
                    help="cap RTMP push width (downscale SR frame before push). 0=disable. NVENC on Laptop GPUs"
                         " often refuses >3840 wide; default 1920 keeps things sane")
    ap.add_argument("--encoder", choices=["auto", "h264_nvenc", "libx264"], default="auto",
                    help="ffmpeg video encoder. auto picks h264_nvenc when available, else libx264")
    ap.add_argument("--x264-preset", default="veryfast",
                    choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"],
                    help="libx264 preset. ultrafast=lowest CPU / worst quality, medium=best quality / most CPU. "
                         "veryfast = good live balance")
    ap.add_argument("--push-crf", type=int, default=20,
                    help="x264/NVENC quality target (18=visually lossless, 23=x264 default, 28=lossy). "
                         "Lower = better quality + bigger stream")
    ap.add_argument("--push-bitrate", default=None,
                    help="force constant bitrate cap (e.g. 6M, 10M). Overrides --push-crf. Useful when bandwidth-limited")
    ap.add_argument("--report-every", type=int, default=30, help="latency report every N processed frames")
    ap.add_argument("--debug-log", default=None,
                    help="write per-frame timing CSV to this path. Columns: "
                         "seq,capture_ts,enqueue_ts,detect_start,detect_end,post_end,upscale_end,"
                         "consume_ts,qwait_ms,detect_ms,post_ms,upscale_ms,e2e_ms,n_faces,ooo_flag")
    ap.add_argument("--debug-decode", action="store_true",
                    help="have NodePlayer log frame_count + temp-file POS every 30 emits (debug FLV replay)")
    args = ap.parse_args()

    # Resolve --upscale-scale: model is hardcoded 4x.
    # scale=2 means we halve pre-resize so net = 4x * (pre/2) = 2x effective.
    # scale=4 keeps pre-resize as-is so net = 4x * pre.
    NATIVE_SCALE = 4
    if args.upscale_scale == 2:
        args.upscale_pre_resize *= 0.5
    effective_scale = NATIVE_SCALE * args.upscale_pre_resize
    print(
        f"[upscale] scale_choice={args.upscale_scale}x  native=4x  "
        f"pre_resize={args.upscale_pre_resize:.3f}  -> effective ~{effective_scale:.2f}x"
    )

    _enable_torch_perf()
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"[init] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[init] CPU mode (slow)")

    # detector (skip if pure upscale)
    detector = None
    if args.service in ("blur", "both"):
        if args.detector == "retinaface":
            detector = RetinaFaceDetector(device=device, network=args.retinaface_net)
        else:
            detector = YoloDetector(model_path=args.model, device=device, half=device.startswith("cuda"))

    # upscaler
    upscaler = None
    if args.service in ("upscale", "both"):
        try:
            upscaler = build_upscaler(args)
        except Exception as e:
            print(f"[ERR] upscaler load failed: {e}")
            return 5

    ghost = GhostTracker(args.ghost_frames, args.ghost_min_hits, args.ghost_iou_match) if not args.no_track else None
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if args.clahe else None

    # roi resolved lazily once we know frame size; for now placeholder
    roi = None
    if args.exclude_roi or args.driver_side:
        print("[roi] will resolve on first frame")

    capture_q: queue.Queue = queue.Queue(maxsize=args.capture_q_size)
    out_q: queue.Queue = queue.Queue(maxsize=args.out_q_size)
    stop_event = threading.Event()

    # NodePlayer fills frame_queue only when show_video=True (decoder thread).
    # Force it ON, then suppress cv2.imshow so no raw window pops up.
    # Also suppress cv2.waitKey from non-main threads -- Qt timers cannot be
    # started off the main thread, and NodePlayer's display thread calls
    # cv2.waitKey(1) every iter. Without this guard we get
    # "QObject::startTimer: Timers cannot be started from another thread"
    # spam and a segfault on shutdown.
    if not args.show_raw:
        _orig_imshow = cv2.imshow
        _orig_waitkey = cv2.waitKey
        def _imshow_noop(win, img):
            if win == "NodePlayer":
                return
            _orig_imshow(win, img)
        def _waitkey_main_only(d):
            if threading.current_thread() is threading.main_thread():
                return _orig_waitkey(d)
            return -1
        cv2.imshow = _imshow_noop  # type: ignore
        cv2.waitKey = _waitkey_main_only  # type: ignore

    decode_fps = args.decode_fps if args.decode_fps > 0 else args.push_fps
    player = NodePlayer(
        username=args.username,
        password=args.password,
        imei=args.imei,
        cam_id=args.cam_id,
        duration=args.duration,
        live_stream=args.live,
        stream_type=args.stream_type,
        show_video=True,
        debug_decode=args.debug_decode,
        decode_fps=decode_fps,
        queue_maxsize=args.queue_size,
    )
    print(f"[ws] decode_fps={decode_fps:.1f} queue_maxsize={args.queue_size}")

    cap_t = threading.Thread(
        target=capture_thread,
        args=(player, capture_q, stop_event, args.max_frames or None),
        daemon=True,
    )
    run_up = args.service in ("upscale", "both") and upscaler is not None
    mid_q: queue.Queue = queue.Queue(maxsize=args.mid_q_size) if run_up else out_q
    det_t = threading.Thread(
        target=detect_worker,
        args=(detector, capture_q, mid_q, out_q, stop_event, args, ghost, clahe, roi, run_up),
        daemon=True,
    )
    up_t = None
    if run_up:
        up_t = threading.Thread(
            target=upscale_worker,
            args=(upscaler, mid_q, out_q, stop_event, args, device),
            daemon=True,
        )

    print(f"[pipeline] batch={args.batch} batch_timeout={args.batch_timeout}ms detector={args.detector}")

    # optional mediamtx server
    mtx_proc = None
    if not args.no_rtmp and args.start_server:
        try:
            mtx_proc = start_mediamtx()
        except Exception as e:
            print(f"[mediamtx] failed: {e}. Continuing without server (assume external).")

    # ffmpeg pusher lazy-init (need frame size from first frame)
    ffmpeg_proc: subprocess.Popen | None = None
    push_dims: tuple[int, int] = (0, 0)
    push_path = args.rtmp_url.rsplit("/", 1)[-1] if args.rtmp_url else "blur"

    cap_t.start()
    det_t.start()
    if up_t is not None:
        up_t.start()
        print(f"[pipeline] upscale worker batch={args.upscale_batch} timeout={args.upscale_batch_timeout}ms")

    _display_win_init = {"done": False}

    # latency stats
    lat_e2e: list[float] = []
    lat_qwait: list[float] = []
    lat_detect: list[float] = []
    lat_post: list[float] = []
    lat_upscale: list[float] = []
    batch_sizes: list[int] = []
    n_processed = 0
    t_first = None
    last_report = time.time()

    # ---- debug instrumentation ----
    debug_fh = None
    if args.debug_log:
        debug_fh = open(args.debug_log, "w", buffering=1)  # line-buffered for tail-f
        debug_fh.write(
            "seq,capture_ts,enqueue_ts,detect_start,detect_end,post_end,upscale_end,"
            "consume_ts,qwait_ms,detect_ms,post_ms,upscale_ms,e2e_ms,n_faces,ooo_flag\n"
        )
        print(f"[debug] per-frame timeline -> {args.debug_log}")
    last_seq = -1
    ooo_count = 0

    try:
        while not stop_event.is_set() or not out_q.empty():
            try:
                item: OutItem = out_q.get(timeout=0.2)
            except queue.Empty:
                if not cap_t.is_alive() and not det_t.is_alive():
                    break
                continue

            now = time.time()
            if t_first is None:
                t_first = now
                # resolve roi now (need frame size)
                fh, fw = item.frame.shape[:2]
                roi_resolved = parse_roi(args.exclude_roi, fw, fh) or driver_side_to_roi(args.driver_side, fw, fh)
                if roi_resolved:
                    roi = roi_resolved  # noqa: F841 — used in worker after this, but worker captured by ref already
                    print(f"[roi] {roi}")

            qwait_ms = (item.detect_start_ts - item.enqueue_ts) * 1000
            detect_ms = (item.detect_end_ts - item.detect_start_ts) * 1000
            post_ms = (item.post_end_ts - item.detect_end_ts) * 1000
            upscale_ms = (item.upscale_end_ts - item.post_end_ts) * 1000
            e2e_ms = (now - item.capture_ts) * 1000

            # out-of-order detector: pipeline should emit increasing seq.
            # Any decrease = either NodePlayer replayed (cap reopened mid-stream)
            # or batch reordering. Either is a bug worth knowing about.
            ooo = item.seq < last_seq
            if ooo:
                ooo_count += 1
                if ooo_count <= 20:  # cap log spam
                    print(f"[OOO] frame seq={item.seq} arrived after last_seq={last_seq} (count={ooo_count})", flush=True)
            last_seq = max(last_seq, item.seq)

            if debug_fh is not None:
                debug_fh.write(
                    f"{item.seq},{item.capture_ts:.6f},{item.enqueue_ts:.6f},"
                    f"{item.detect_start_ts:.6f},{item.detect_end_ts:.6f},"
                    f"{item.post_end_ts:.6f},{item.upscale_end_ts:.6f},"
                    f"{now:.6f},{qwait_ms:.2f},{detect_ms:.2f},{post_ms:.2f},"
                    f"{upscale_ms:.2f},{e2e_ms:.2f},{item.n_faces},{int(ooo)}\n"
                )

            lat_qwait.append(qwait_ms)
            lat_detect.append(detect_ms)
            lat_post.append(post_ms)
            lat_upscale.append(upscale_ms)
            lat_e2e.append(e2e_ms)

            n_processed += 1

            # init ffmpeg on first frame
            if ffmpeg_proc is None and not args.no_rtmp:
                fh, fw = item.frame.shape[:2]
                # Downscale SR frame before push: NVENC on Laptop GPUs often refuses
                # >3840 wide, and 4096x2304 H.264 burns NVENC sessions anyway.
                if args.push_max_w > 0 and fw > args.push_max_w:
                    ratio = args.push_max_w / float(fw)
                    fw = args.push_max_w
                    fh = int(fh * ratio) & ~1  # keep even (yuv420p requires)
                    print(f"[ffmpeg] downscaling SR -> {fw}x{fh} for push (--push-max-w)")
                push_dims = (fw, fh)
                try:
                    ffmpeg_proc = start_ffmpeg_pusher(
                        fw, fh, args.push_fps, args.rtmp_url,
                        encoder=args.encoder,
                        preset=args.x264_preset,
                        crf=args.push_crf,
                        bitrate=args.push_bitrate,
                    )
                    print(f"\n=== STREAM URLS  (push res {fw}x{fh}) ===")
                    print(f"  RTMP  : {args.rtmp_url}")
                    print(f"  HLS   : http://localhost:8888/{push_path}/index.m3u8")
                    print(f"  WebRTC: http://localhost:8889/{push_path}")
                    print(f"  RTSP  : rtsp://localhost:8554/{push_path}")
                    print(f"  Browser: http://localhost:8888/{push_path}")
                    print("===================\n", flush=True)
                except Exception as e:
                    print(f"[ffmpeg] init failed: {e}")
                    args.no_rtmp = True

            if ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
                fh, fw = item.frame.shape[:2]
                if (fw, fh) != push_dims:
                    # frame size changed → must resize to ffmpeg's locked input dims
                    item.frame = cv2.resize(item.frame, push_dims, interpolation=cv2.INTER_AREA)
                try:
                    ffmpeg_proc.stdin.write(item.frame.tobytes())
                except (BrokenPipeError, OSError) as e:
                    print(f"[ffmpeg] pipe broken: {e}")
                    ffmpeg_proc = None
                    args.no_rtmp = True

            if args.display:
                disp = item.frame
                fh, fw = disp.shape[:2]
                scale = args.display_scale if args.display_scale > 0 else 1.0
                tw = int(fw * scale)
                th = int(fh * scale)
                if args.display_max_w > 0 and tw > args.display_max_w:
                    r = args.display_max_w / float(tw)
                    tw = args.display_max_w
                    th = int(th * r)
                if (tw, th) != (fw, fh):
                    disp = cv2.resize(disp, (tw, th), interpolation=cv2.INTER_AREA)
                if not _display_win_init["done"]:
                    cv2.namedWindow("blur-live", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("blur-live", tw, th)
                    _display_win_init["done"] = True
                cv2.imshow("blur-live", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

            if n_processed % args.report_every == 0 or (time.time() - last_report) > 5:
                last_report = time.time()
                elapsed = time.time() - t_first
                fps = n_processed / max(elapsed, 1e-6)
                e2e_arr = np.array(lat_e2e[-args.report_every:], dtype=np.float32)
                det_arr = np.array(lat_detect[-args.report_every:], dtype=np.float32)
                qw_arr = np.array(lat_qwait[-args.report_every:], dtype=np.float32)
                po_arr = np.array(lat_post[-args.report_every:], dtype=np.float32)
                up_arr = np.array(lat_upscale[-args.report_every:], dtype=np.float32)
                print(
                    f"[prog] n={n_processed} fps={fps:.1f} "
                    f"qwait={qw_arr.mean():.1f} "
                    f"detect={det_arr.mean():.1f} "
                    f"post={po_arr.mean():.1f} "
                    f"upscale={up_arr.mean():.1f} "
                    f"e2e={e2e_arr.mean():.1f}ms (p95={np.percentile(e2e_arr,95):.1f}) "
                    f"faces_last={item.n_faces}"
                )
    except KeyboardInterrupt:
        print("[stop] keyboard")
        stop_event.set()

    cap_t.join(timeout=5)
    det_t.join(timeout=5)
    if up_t is not None:
        up_t.join(timeout=5)

    if ffmpeg_proc is not None:
        try:
            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
        except Exception:
            pass
        ffmpeg_proc.terminate()
        try:
            ffmpeg_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            ffmpeg_proc.kill()

    if mtx_proc is not None:
        mtx_proc.terminate()
        try:
            mtx_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            mtx_proc.kill()

    if args.display:
        cv2.destroyAllWindows()

    if n_processed == 0:
        print("[done] no frames processed")
        return 1

    e2e = np.array(lat_e2e, dtype=np.float32)
    qw = np.array(lat_qwait, dtype=np.float32)
    det = np.array(lat_detect, dtype=np.float32)
    po = np.array(lat_post, dtype=np.float32)
    elapsed = time.time() - t_first if t_first else 1e-6
    fps = n_processed / elapsed

    print("\n=== END-TO-END LATENCY REPORT ===")
    print(f"  frames processed : {n_processed}")
    print(f"  wall time        : {elapsed:.2f}s")
    print(f"  effective fps    : {fps:.2f}")
    print(f"  batch size cfg   : {args.batch}  (timeout {args.batch_timeout} ms)")
    print(f"  stage timings (mean / p50 / p95 / p99 ms):")
    up = np.array(lat_upscale, dtype=np.float32)
    stages = [("queue wait", qw), ("detect (GPU)", det), ("post (ghost+blur)", po), ("upscale (GPU)", up), ("end-to-end", e2e)]
    for name, arr in stages:
        print(f"    {name:<20} {arr.mean():6.1f} / {np.percentile(arr,50):6.1f} / {np.percentile(arr,95):6.1f} / {np.percentile(arr,99):6.1f}")
    print(f"  e2e max         : {e2e.max():.1f} ms")

    src_fps_assumed = 25.0
    budget = 1000.0 / src_fps_assumed
    headroom = (budget - e2e.mean()) / budget * 100
    verdict = "REALTIME ✓" if e2e.mean() <= budget else "OVER BUDGET ✗"
    print(f"  assuming {src_fps_assumed:.0f}fps source: budget={budget:.1f}ms, headroom={headroom:+.1f}% -> {verdict}")
    print(f"  out-of-order frames: {ooo_count} (frames whose seq < previous seq)")
    if debug_fh is not None:
        try:
            debug_fh.close()
            print(f"  debug-log written: {args.debug_log}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
