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
import queue
import shutil
import subprocess
import sys
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
MEDIAMTX_WIN_URL = f"https://github.com/bluenviron/mediamtx/releases/download/v{MEDIAMTX_VERSION}/mediamtx_v{MEDIAMTX_VERSION}_windows_amd64.zip"
MEDIAMTX_DIR = Path(__file__).parent / "bin"


def ensure_mediamtx() -> Path:
    """Download mediamtx (Windows) on first run. Return path to exe."""
    exe = MEDIAMTX_DIR / "mediamtx.exe"
    if exe.exists():
        return exe
    MEDIAMTX_DIR.mkdir(exist_ok=True)
    zip_path = MEDIAMTX_DIR / "mediamtx.zip"
    print(f"[mediamtx] downloading v{MEDIAMTX_VERSION} from github...")
    urllib.request.urlretrieve(MEDIAMTX_WIN_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(MEDIAMTX_DIR)
    zip_path.unlink(missing_ok=True)
    if not exe.exists():
        raise RuntimeError(f"mediamtx exe missing after extract: {exe}")
    return exe


def start_mediamtx() -> subprocess.Popen:
    exe = ensure_mediamtx()
    print(f"[mediamtx] starting {exe}")
    proc = subprocess.Popen(
        [str(exe)],
        cwd=str(MEDIAMTX_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # wait for "RTMP listener opened" or similar
    time.sleep(2.0)
    if proc.poll() is not None:
        out = proc.stdout.read() if proc.stdout else ""
        raise RuntimeError(f"mediamtx died: {out}")
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
    """Build realesrgan_lite w/ channels_last + torch.compile perf opts."""
    from src.models.base import UpscalerConfig, build  # type: ignore
    from src.models._perf import apply_perf_opts  # type: ignore
    extra = {
        "model_name": "realesr-general-x4v3",
        "denoise_strength": args.upscale_denoise,
        "tile": 0,
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
    try:
        up.upsampler.model = apply_perf_opts(
            up.upsampler.model,
            compile=not args.no_compile,
            channels_last=True,
            compile_mode="default",
        )
        print(f"[upscale] perf opts: channels_last=True compile={not args.no_compile}")
    except Exception as e:
        print(f"[upscale] apply_perf_opts skipped: {e}")
    print(f"[upscale] loaded {args.upscale_model} scale={args.upscale_scale} dtype={args.upscale_dtype} denoise={args.upscale_denoise}")
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


def start_ffmpeg_pusher(width: int, height: int, fps: float, rtmp_url: str) -> subprocess.Popen:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg first.")
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
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-g", str(max(2, int(fps))),
        "-keyint_min", str(max(2, int(fps))),
        "-b:v", "2M",
        "-maxrate", "2M",
        "-bufsize", "1M",
        "-f", "flv",
        rtmp_url,
    ]
    print(f"[ffmpeg] push -> {rtmp_url}")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
    batch_timeout = args.upscale_batch_timeout / 1000.0
    pre = args.upscale_pre_resize
    half = args.upscale_dtype == "fp16"

    while not stop_event.is_set():
        batch: list[OutItem] = []
        deadline = time.time() + batch_timeout
        try:
            batch.append(mid_q.get(timeout=0.1))
        except queue.Empty:
            continue
        while len(batch) < batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                batch.append(mid_q.get(timeout=remaining))
            except queue.Empty:
                break

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
    ap.add_argument("--imei", default="867395071670570")
    ap.add_argument("--cam-id", type=int, default=4)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--stream-type", default="sub", choices=["sub", "main"])
    ap.add_argument("--live", action="store_true", default=True)
    # pipeline
    ap.add_argument("--batch", type=int, default=4, help="frames per detector batch")
    ap.add_argument("--batch-timeout", type=float, default=15.0, help="ms to wait filling batch")
    ap.add_argument("--capture-q-size", type=int, default=16)
    ap.add_argument("--out-q-size", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0=unlimited)")
    # service selector
    ap.add_argument("--service", choices=["blur", "upscale", "both"], default="blur",
                    help="blur=face-blur only, upscale=ESRGAN only, both=blur then upscale")
    # upscaler defaults (tested best for realesrgan_lite per user)
    ap.add_argument("--upscale-model", default="realesrgan_lite")
    ap.add_argument("--upscale-scale", type=int, default=4)
    ap.add_argument("--upscale-pre-resize", type=float, default=0.8, help="pre-resize ratio (0.8 = 80%)")
    ap.add_argument("--upscale-dtype", choices=["fp16"], default="fp16")
    ap.add_argument("--compile", dest="no_compile", action="store_false", help="enable torch.compile (needs triton)")
    ap.add_argument("--no-compile", action="store_true", default=True, help="disable torch.compile (default; triton missing on Win)")
    ap.add_argument("--upscale-batch", type=int, default=4, help="frames per upscale GPU batch")
    ap.add_argument("--upscale-batch-timeout", type=float, default=15.0, help="ms to wait filling upscale batch")
    ap.add_argument("--mid-q-size", type=int, default=16, help="queue between blur and upscale workers")
    ap.add_argument("--upscale-denoise", type=float, default=1.0, help="esrgan denoise strength 0..1")
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
    ap.add_argument("--display", action="store_true", help="show blurred output window")
    ap.add_argument("--show-raw", action="store_true", help="also show raw NodePlayer window")
    ap.add_argument("--rtmp-url", default="rtmp://localhost:1935/blur", help="RTMP push URL")
    ap.add_argument("--no-rtmp", action="store_true", help="disable RTMP push")
    ap.add_argument("--start-server", action="store_true", default=True, help="auto-launch mediamtx local server")
    ap.add_argument("--no-server", dest="start_server", action="store_false")
    ap.add_argument("--push-fps", type=float, default=15.0, help="RTMP encode fps (lower = less CPU)")
    ap.add_argument("--report-every", type=int, default=30, help="latency report every N processed frames")
    args = ap.parse_args()

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

    # NOTE: NodePlayer fills frame_queue ONLY when show_video=True (decoder thread).
    # We force it ON, then suppress cv2.imshow so no raw window pops up.
    if not args.show_raw:
        _orig_imshow = cv2.imshow
        _orig_waitkey = cv2.waitKey
        def _imshow_noop(win, img):
            if win == "NodePlayer":
                return
            _orig_imshow(win, img)
        def _waitkey_noop(d):
            return _orig_waitkey(d) if d > 0 else -1
        cv2.imshow = _imshow_noop  # type: ignore

    player = NodePlayer(
        username=args.username,
        password=args.password,
        imei=args.imei,
        cam_id=args.cam_id,
        duration=args.duration,
        live_stream=args.live,
        stream_type=args.stream_type,
        show_video=True,
    )

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

            lat_qwait.append(qwait_ms)
            lat_detect.append(detect_ms)
            lat_post.append(post_ms)
            lat_upscale.append(upscale_ms)
            lat_e2e.append(e2e_ms)

            n_processed += 1

            # init ffmpeg on first frame
            if ffmpeg_proc is None and not args.no_rtmp:
                fh, fw = item.frame.shape[:2]
                push_dims = (fw, fh)
                try:
                    ffmpeg_proc = start_ffmpeg_pusher(fw, fh, args.push_fps, args.rtmp_url)
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
                cv2.imshow("blur-live", item.frame)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
