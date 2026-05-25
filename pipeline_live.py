"""
Live stream Real-ESRGAN upscaling pipeline.

Architecture:
    [LiveFeeder NodePlayer thread]
        produces frame -> capture_queue
    [Upscale worker thread]
        batches frames -> Real-ESRGAN forward (GPU) -> out_queue
    [Main thread]
        consumes out_queue -> ffmpeg pusher (NVENC/libx264) -> mediamtx
        -> RTMP / HLS / WebRTC / RTSP

Latency stages reported:
    queue wait     : frame sat in capture_queue
    upscale (GPU)  : batch SR forward
    end-to-end     : capture_ts -> consume_ts
"""

from __future__ import annotations

import argparse
import asyncio
import logging
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

from LiveFeeder import NodePlayer

# Upscaling src: sibling subdir (Acacus layout) OR flat root (Upscaling-Streaming branch)
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
    not tell us the device can open an NVENC session. Real 256x256 probe.
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
        # NVENC h264 min frame size ~145x49 (Maxwell+); 256x256 well above.
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

    # NVENC h264 hard caps at 4096 wide on consumer GPUs. Above that the encoder
    # bails at session-open ("Width 4608 exceeds 4096") and RTMP push dies.
    if encoder == "h264_nvenc" and width > 4096:
        print(f"[ffmpeg] width {width} > NVENC h264 max (4096); falling back to libx264"
              " — pass --push-max-w 4096 or --encoder libx264 to silence")
        encoder = "libx264"

    gop = str(max(2, int(fps)))
    if encoder == "h264_nvenc":
        venc = [
            "-c:v", "h264_nvenc",
            "-preset", "p4",
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
    upscale_start_ts: float = 0.0
    upscale_end_ts: float = 0.0
    face_ms: float = 0.0


def capture_thread(
    player: NodePlayer,
    capture_q: queue.Queue,
    stop_event: threading.Event,
    max_seq: int | None,
    raw_q: queue.Queue | None = None,
):
    """Run NodePlayer in its own event loop; push frames w/ timestamps.

    If ``raw_q`` is provided (used by --compare), also tee a copy of each
    raw pre-SR frame into it (drop-oldest) so the main thread can display
    the source feed alongside the upscaled output.
    """
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
            if raw_q is not None:
                # copy so upscale_worker replacing item.frame doesn't clobber the
                # raw view (numpy array is shared between FrameItem and source).
                raw_copy = frame.copy()
                try:
                    raw_q.put_nowait(raw_copy)
                except queue.Full:
                    try:
                        raw_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        raw_q.put_nowait(raw_copy)
                    except queue.Full:
                        pass
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


def upscale_worker(
    upscaler,
    capture_q: queue.Queue,
    out_q: queue.Queue,
    stop_event: threading.Event,
    args,
    device: str,
    face_enhancer=None,
):
    """Batch-pull from capture_q, run Real-ESRGAN forward, push to out_q.

    If ``face_enhancer`` is supplied, each SR frame is passed through it
    after the batch forward to detect+restore faces. Face restoration is
    per-frame (not batched) because facexlib's detector is per-image.
    """
    batch_size = args.upscale_batch
    pre = args.upscale_pre_resize
    half = args.upscale_dtype == "fp16"

    # adaptive batch-timeout: positive arg = fixed; 0 = derive from EMA.
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
        batch: list[FrameItem] = []
        deadline = time.time() + timeout_s
        try:
            batch.append(capture_q.get(timeout=0.1))
            _record_arrival()
        except queue.Empty:
            continue
        while len(batch) < batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                batch.append(capture_q.get(timeout=remaining))
                _record_arrival()
            except queue.Empty:
                break

        if use_adaptive and (time.time() - last_log_t) > 5.0:
            last_log_t = time.time()
            fps_est = 1.0 / max(ema_dt, 1e-6)
            print(f"[upscale] adaptive timeout={timeout_s*1000:.0f}ms  "
                  f"observed_fps~{fps_est:.1f}  last_batch={len(batch)}/{batch_size}")

        start_ts = time.time()
        for it in batch:
            it.upscale_start_ts = start_ts

        # pre-resize all frames to common (h, w)
        if pre and abs(pre - 1.0) > 1e-3:
            srcs = []
            for it in batch:
                nh = int(it.frame.shape[0] * pre)
                nw = int(it.frame.shape[1] * pre)
                srcs.append(cv2.resize(it.frame, (nw, nh), interpolation=cv2.INTER_AREA))
        else:
            srcs = [it.frame for it in batch]

        shape0 = srcs[0].shape
        try:
            if not all(s.shape == shape0 for s in srcs):
                outs = [batched_esrgan_infer(upscaler, s[None, ...], half, device)[0] for s in srcs]
            else:
                stacked = np.stack(srcs, axis=0)
                outs = batched_esrgan_infer(upscaler, stacked, half, device)
        except Exception as e:
            print(f"[ERR upscale] {type(e).__name__}: {e}", flush=True)
            outs = [it.frame for it in batch]

        ts_end = time.time()

        # face restoration runs AFTER SR (operates on hi-res output); cost
        # tracked per-frame because detector is per-image.
        if face_enhancer is not None:
            for i, it in enumerate(batch):
                t0 = time.time()
                outs[i] = face_enhancer.restore(outs[i])
                it.face_ms = (time.time() - t0) * 1000.0
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
    ap = argparse.ArgumentParser(description="Live-stream Real-ESRGAN upscaling pipeline.")
    # stream
    ap.add_argument("--username", default="m.alawneh")
    ap.add_argument("--password", default="sgQZ9Oou3CsP")
    ap.add_argument("--imei", default="867395071670570")
    ap.add_argument("--cam-id", type=int, default=4)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--stream-type", default="sub", choices=["sub", "main"])
    ap.add_argument("--live", action="store_true", default=True)
    # pipeline queues
    ap.add_argument("--capture-q-size", type=int, default=8)
    ap.add_argument("--out-q-size", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0=unlimited)")
    # upscaler
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
    ap.add_argument("--upscale-denoise", type=float, default=0.5, help="esrgan denoise strength 0..1")
    # output / display
    ap.add_argument("--display", action="store_true", help="show output window")
    ap.add_argument("--display-scale", type=float, default=1.0,
                    help="shrink factor for --display window (0.35 = 35 percent of SR frame)")
    ap.add_argument("--display-max-w", type=int, default=1600,
                    help="hard cap on display window width in pixels (downscale if larger)")
    ap.add_argument("--show-raw", action="store_true", help="also show raw NodePlayer window")
    ap.add_argument("--compare", action="store_true",
                    help="open a second window with the raw (pre-SR) source feed alongside the upscaled output. "
                         "Implies --display. Useful to A/B the SR result against the original.")
    ap.add_argument("--rtmp-url", default="rtmp://localhost:1935/stream", help="RTMP push URL")
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
                    help="cap RTMP push width (downscale SR frame before push). 0=disable. NVENC h264 caps at 4096.")
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
                    help="force constant bitrate cap (e.g. 6M, 10M). Overrides --push-crf.")
    # face enhancement (off by default; opt-in per run, like --target-latency)
    ap.add_argument("--face-enhance", choices=["none", "gfpgan", "codeformer"], default="none",
                    help="OPT-IN face restoration applied AFTER Real-ESRGAN. "
                         "gfpgan = GFPGAN v1.4 (fast, less identity drift). "
                         "codeformer = CodeFormer (slower, higher quality, --face-fidelity tunes "
                         "quality<->identity). none (default) = disabled.")
    ap.add_argument("--face-fidelity", type=float, default=0.5,
                    help="Restoration weight 0..1. CodeFormer: 0=max quality (more identity drift), "
                         "1=max identity preservation (less aggressive restore). GFPGAN: blend weight "
                         "of restored vs original face. Typical 0.5-0.7.")
    ap.add_argument("--face-only-center", action="store_true",
                    help="restore only the center-most detected face per frame (faster when many "
                         "background faces are present)")
    ap.add_argument("--target-latency", type=float, default=0.0,
                    help="OPT-IN jitter buffer. Hold each frame until capture_ts + this many "
                         "seconds before pushing. Trades real-time lag for smooth output: model "
                         "jitter, batch stalls, and brief GPU spikes are absorbed by the buffer "
                         "instead of causing stutter or frame skip. Typical 3-5s for "
                         "broadcast-style live. 0 (default) = disabled, lowest-latency mode.")
    ap.add_argument("--report-every", type=int, default=30, help="latency report every N processed frames")
    ap.add_argument("--debug-log", default=None,
                    help="write per-frame timing CSV to this path. Columns: "
                         "seq,capture_ts,enqueue_ts,upscale_start,upscale_end,"
                         "consume_ts,qwait_ms,upscale_ms,e2e_ms,ooo_flag")
    ap.add_argument("--debug-decode", action="store_true",
                    help="have NodePlayer log extra decoder info")
    args = ap.parse_args()

    # Resolve --upscale-scale: model is hardcoded 4x.
    NATIVE_SCALE = 4
    if args.upscale_scale == 2:
        args.upscale_pre_resize *= 0.5
    effective_scale = NATIVE_SCALE * args.upscale_pre_resize
    print(
        f"[upscale] scale_choice={args.upscale_scale}x  native=4x  "
        f"pre_resize={args.upscale_pre_resize:.3f}  -> effective ~{effective_scale:.2f}x"
    )

    # Jitter buffer sizing: if target-latency > 0 we hold each frame until
    # capture_ts + target_latency before pushing. The out queue (and capture
    # queue, since both feed the same buffer in different stages) must be big
    # enough to hold that many frames, otherwise the drop-oldest policy in the
    # workers will silently truncate the buffer and the latency target is lost.
    if args.target_latency > 0:
        need = int(args.target_latency * max(args.push_fps, 1.0)) + 8
        if args.out_q_size < need:
            args.out_q_size = need
        if args.capture_q_size < need:
            args.capture_q_size = need
        print(f"[latency] target_latency={args.target_latency:.2f}s push_fps={args.push_fps:.1f} "
              f"-> capture_q={args.capture_q_size} out_q={args.out_q_size}")
    else:
        print("[latency] target_latency=0 (jitter buffer disabled, lowest-latency mode)")

    _enable_torch_perf()
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"[init] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[init] CPU mode (slow)")

    # upscaler (always on — this is an upscaling-only pipeline)
    try:
        upscaler = build_upscaler(args)
    except Exception as e:
        print(f"[ERR] upscaler load failed: {e}")
        return 5

    # optional face restorer
    face_enhancer = None
    if args.face_enhance != "none":
        try:
            from src.face_enhance import FaceEnhancer  # type: ignore
            face_enhancer = FaceEnhancer(
                backend=args.face_enhance,
                fidelity=args.face_fidelity,
                only_center=args.face_only_center,
                device=device,
            )
        except Exception as e:
            print(f"[face] init failed ({type(e).__name__}: {e}) -> face restore disabled")
            face_enhancer = None
    else:
        print("[face] face_enhance=none (disabled)")

    capture_q: queue.Queue = queue.Queue(maxsize=args.capture_q_size)
    out_q: queue.Queue = queue.Queue(maxsize=args.out_q_size)
    stop_event = threading.Event()

    # NodePlayer fills frame_queue only when show_video=True. Force ON, suppress
    # cv2.imshow + non-main-thread cv2.waitKey to avoid Qt segfaults.
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

    # --compare: tee raw frames into a small queue so main thread can
    # display the source feed in a second window. Implies --display.
    if args.compare:
        args.display = True
    raw_q: queue.Queue | None = queue.Queue(maxsize=2) if args.compare else None

    cap_t = threading.Thread(
        target=capture_thread,
        args=(player, capture_q, stop_event, args.max_frames or None, raw_q),
        daemon=True,
    )
    up_t = threading.Thread(
        target=upscale_worker,
        args=(upscaler, capture_q, out_q, stop_event, args, device, face_enhancer),
        daemon=True,
    )

    print(f"[pipeline] upscale-only mode  upscale_batch={args.upscale_batch} "
          f"capture_q={args.capture_q_size} out_q={args.out_q_size}")

    mtx_proc = None
    if not args.no_rtmp and args.start_server:
        try:
            mtx_proc = start_mediamtx()
        except Exception as e:
            print(f"[mediamtx] failed: {e}. Continuing without server (assume external).")

    ffmpeg_proc: subprocess.Popen | None = None
    push_dims: tuple[int, int] = (0, 0)
    push_path = args.rtmp_url.rsplit("/", 1)[-1] if args.rtmp_url else "stream"

    cap_t.start()
    up_t.start()

    _display_win_init = {"done": False}
    _raw_win_init = {"done": False}

    lat_e2e: list[float] = []
    lat_qwait: list[float] = []
    lat_upscale: list[float] = []
    lat_face: list[float] = []
    n_processed = 0
    t_first = None
    last_report = time.time()

    debug_fh = None
    if args.debug_log:
        debug_fh = open(args.debug_log, "w", buffering=1)
        debug_fh.write(
            "seq,capture_ts,enqueue_ts,upscale_start,upscale_end,"
            "consume_ts,qwait_ms,upscale_ms,e2e_ms,ooo_flag\n"
        )
        print(f"[debug] per-frame timeline -> {args.debug_log}")
    last_seq = -1
    ooo_count = 0

    try:
        while not stop_event.is_set() or not out_q.empty():
            try:
                item: FrameItem = out_q.get(timeout=0.2)
            except queue.Empty:
                if not cap_t.is_alive() and not up_t.is_alive():
                    break
                continue

            now = time.time()
            if t_first is None:
                t_first = now

            # Jitter buffer: hold the frame until target_latency has elapsed
            # since capture. Source frames arrive at source-fps cadence, so
            # capture_ts increases monotonically at source-fps -> the deadline
            # stream is also paced at source-fps. ffmpeg therefore sees steady
            # output cadence even if the GPU model jittered upstream.
            if args.target_latency > 0:
                deadline = item.capture_ts + args.target_latency
                wait_s = deadline - time.time()
                if wait_s > 0:
                    time.sleep(wait_s)
                now = time.time()

            qwait_ms = (item.upscale_start_ts - item.enqueue_ts) * 1000
            upscale_ms = (item.upscale_end_ts - item.upscale_start_ts) * 1000
            e2e_ms = (now - item.capture_ts) * 1000

            ooo = item.seq < last_seq
            if ooo:
                ooo_count += 1
                if ooo_count <= 20:
                    print(f"[OOO] frame seq={item.seq} arrived after last_seq={last_seq} (count={ooo_count})", flush=True)
            last_seq = max(last_seq, item.seq)

            if debug_fh is not None:
                debug_fh.write(
                    f"{item.seq},{item.capture_ts:.6f},{item.enqueue_ts:.6f},"
                    f"{item.upscale_start_ts:.6f},{item.upscale_end_ts:.6f},"
                    f"{now:.6f},{qwait_ms:.2f},{upscale_ms:.2f},{e2e_ms:.2f},{int(ooo)}\n"
                )

            lat_qwait.append(qwait_ms)
            lat_upscale.append(upscale_ms)
            lat_face.append(item.face_ms)
            lat_e2e.append(e2e_ms)

            n_processed += 1

            # init ffmpeg on first frame
            if ffmpeg_proc is None and not args.no_rtmp:
                fh, fw = item.frame.shape[:2]
                if args.push_max_w > 0 and fw > args.push_max_w:
                    ratio = args.push_max_w / float(fw)
                    fw = args.push_max_w
                    fh = int(fh * ratio) & ~1  # keep even (yuv420p)
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
                    cv2.namedWindow("upscale-live", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("upscale-live", tw, th)
                    if args.compare:
                        # park SR window to the right; raw window will sit left.
                        try:
                            cv2.moveWindow("upscale-live", tw + 20, 0)
                        except cv2.error:
                            pass
                    _display_win_init["done"] = True
                cv2.imshow("upscale-live", disp)

                # --compare: also draw latest raw frame in a second window.
                if args.compare and raw_q is not None:
                    raw_frame = None
                    # drain the small raw_q to whatever's freshest
                    while True:
                        try:
                            raw_frame = raw_q.get_nowait()
                        except queue.Empty:
                            break
                    if raw_frame is not None:
                        rh, rw = raw_frame.shape[:2]
                        # match the SR display window's pixel size so the eye can
                        # diff them at equal scale. Raw frame is much smaller, so
                        # this stretches it -- exactly what the SR step does too,
                        # but with bicubic (so SR detail shows up against bicubic).
                        raw_disp = cv2.resize(raw_frame, (tw, th), interpolation=cv2.INTER_CUBIC)
                        if not _raw_win_init["done"]:
                            cv2.namedWindow("raw-source", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("raw-source", tw, th)
                            try:
                                cv2.moveWindow("raw-source", 0, 0)
                            except cv2.error:
                                pass
                            _raw_win_init["done"] = True
                        cv2.imshow("raw-source", raw_disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

            if n_processed % args.report_every == 0 or (time.time() - last_report) > 5:
                last_report = time.time()
                elapsed = time.time() - t_first
                fps = n_processed / max(elapsed, 1e-6)
                e2e_arr = np.array(lat_e2e[-args.report_every:], dtype=np.float32)
                qw_arr = np.array(lat_qwait[-args.report_every:], dtype=np.float32)
                up_arr = np.array(lat_upscale[-args.report_every:], dtype=np.float32)
                face_arr = np.array(lat_face[-args.report_every:], dtype=np.float32)
                face_seg = f" face={face_arr.mean():.1f}" if face_enhancer is not None else ""
                print(
                    f"[prog] n={n_processed} fps={fps:.1f} "
                    f"qwait={qw_arr.mean():.1f} "
                    f"upscale={up_arr.mean():.1f}"
                    f"{face_seg} "
                    f"e2e={e2e_arr.mean():.1f}ms (p95={np.percentile(e2e_arr,95):.1f})"
                )
    except KeyboardInterrupt:
        print("[stop] keyboard")
        stop_event.set()

    cap_t.join(timeout=5)
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
    up = np.array(lat_upscale, dtype=np.float32)
    elapsed = time.time() - t_first if t_first else 1e-6
    fps = n_processed / elapsed

    print("\n=== END-TO-END LATENCY REPORT ===")
    print(f"  frames processed : {n_processed}")
    print(f"  wall time        : {elapsed:.2f}s")
    print(f"  effective fps    : {fps:.2f}")
    print(f"  upscale batch    : {args.upscale_batch}  (timeout {args.upscale_batch_timeout} ms; 0=adaptive)")
    face = np.array(lat_face, dtype=np.float32)
    print(f"  stage timings (mean / p50 / p95 / p99 ms):")
    stages = [("queue wait", qw), ("upscale (GPU)", up)]
    if face_enhancer is not None:
        stages.append((f"face ({args.face_enhance})", face))
    stages.append(("end-to-end", e2e))
    for name, arr in stages:
        print(f"    {name:<20} {arr.mean():6.1f} / {np.percentile(arr,50):6.1f} / {np.percentile(arr,95):6.1f} / {np.percentile(arr,99):6.1f}")
    print(f"  e2e max         : {e2e.max():.1f} ms")

    if args.target_latency > 0:
        target_ms = args.target_latency * 1000.0
        # GPU-side cost only -- excludes the deliberate jitter-buffer wait.
        gpu_cost = qw.mean() + up.mean()
        gpu_budget = target_ms
        headroom = (gpu_budget - gpu_cost) / gpu_budget * 100
        verdict = "BUFFER HOLDS ✓" if gpu_cost <= gpu_budget else "BUFFER DRAINED ✗"
        print(f"  target_latency   : {args.target_latency:.2f}s (jitter buffer)")
        print(f"  GPU pipeline cost: {gpu_cost:.1f}ms (qwait+upscale)  vs buffer {gpu_budget:.0f}ms "
              f"-> headroom={headroom:+.1f}% -> {verdict}")
    else:
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
