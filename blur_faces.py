"""
Face blurring for in-cabin dashcam video.

Pipeline: decode -> YOLOv8-face detect -> ByteTrack -> expand bbox -> blur -> display/encode.
Optimized for A100. Uses TensorRT engine if present, otherwise PyTorch FP16.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # only required for --detector yolo

try:
    import supervision as sv
    HAS_SV = True
except ImportError:
    HAS_SV = False


DEFAULT_MODEL_REPO = "arnabdhar/YOLOv8-Face-Detection"
DEFAULT_MODEL_FILE = "model.pt"


def resolve_model_path(model_arg: str) -> str:
    p = Path(model_arg)
    if p.exists():
        return str(p)
    # try huggingface download
    try:
        from huggingface_hub import hf_hub_download
        print(f"[model] downloading {DEFAULT_MODEL_REPO}/{DEFAULT_MODEL_FILE}")
        return hf_hub_download(repo_id=DEFAULT_MODEL_REPO, filename=DEFAULT_MODEL_FILE)
    except Exception as e:
        print(f"[model] HF download failed: {e}. Falling back to ultralytics 'yolov8n.pt' (person detector, not face).", file=sys.stderr)
        return "yolov8n.pt"


class YoloDetector:
    def __init__(self, model_path: str, device: str, half: bool):
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. pip install ultralytics")
        self.model = YOLO(model_path)
        self.device = device
        self.half = half

    def detect(self, frame: np.ndarray, imgsz: int, conf: float, iou: float):
        r = self.model.predict(frame, imgsz=imgsz, conf=conf, iou=iou,
                               device=self.device, half=self.half, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()


class RetinaFaceDetector:
    """RetinaFace via batch-face. ResNet50 backbone, GPU, high recall on profile/occluded."""

    def __init__(self, device: str, network: str = "resnet50"):
        try:
            from batch_face import RetinaFace
        except ImportError as e:
            raise RuntimeError("pip install batch-face") from e
        gpu_id = 0 if device.startswith("cuda") else -1
        self.det = RetinaFace(gpu_id=gpu_id, network=network)
        print(f"[detector] RetinaFace ({network}) gpu_id={gpu_id}")

    def detect(self, frame: np.ndarray, imgsz: int, conf: float, iou: float):
        # batch-face: pass BGR np array directly. Lib handles preprocessing.
        try:
            faces = self.det(frame, threshold=conf, return_dict=False)
        except TypeError:
            faces = self.det(frame, threshold=conf)
        if not faces:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        boxes, confs = [], []
        for f in faces:
            if isinstance(f, dict):
                box = f.get("box")
                score = f.get("score", 1.0)
            else:
                box, _landmarks, score = f
            boxes.append(box)
            confs.append(score)
        return np.array(boxes, dtype=np.float32), np.array(confs, dtype=np.float32)

    def detect_batch(self, frames: list, conf: float):
        """Run detector on batch of frames. Returns list of (xyxy, confs) per frame."""
        try:
            results = self.det(frames, threshold=conf, return_dict=False)
        except TypeError:
            results = self.det(frames, threshold=conf)
        out = []
        for faces in results:
            if not faces:
                out.append((np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)))
                continue
            boxes, confs = [], []
            for f in faces:
                if isinstance(f, dict):
                    box = f.get("box")
                    score = f.get("score", 1.0)
                else:
                    box, _landmarks, score = f
                boxes.append(box)
                confs.append(score)
            out.append((np.array(boxes, dtype=np.float32), np.array(confs, dtype=np.float32)))
        return out


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def merge_overlapping_boxes(boxes: np.ndarray, iou_thresh: float = 0.1) -> np.ndarray:
    """Union-merge boxes whose IoU exceeds threshold. Iterates until stable."""
    if len(boxes) <= 1:
        return boxes
    boxes = boxes.astype(np.float32).copy()
    changed = True
    while changed:
        changed = False
        n = len(boxes)
        keep = [True] * n
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if not keep[j]:
                    continue
                if iou_xyxy(boxes[i], boxes[j]) >= iou_thresh:
                    boxes[i, 0] = min(boxes[i, 0], boxes[j, 0])
                    boxes[i, 1] = min(boxes[i, 1], boxes[j, 1])
                    boxes[i, 2] = max(boxes[i, 2], boxes[j, 2])
                    boxes[i, 3] = max(boxes[i, 3], boxes[j, 3])
                    keep[j] = False
                    changed = True
        boxes = boxes[keep]
    return boxes


class GhostTracker:
    """Persist blur boxes for N frames after detection drops.

    Match new detections to existing tracks by IoU. Confirmed tracks (>= min_hits)
    that miss detection continue ghosting at last bbox until max_age frames pass.
    """

    def __init__(self, max_age: int, min_hits: int, iou_thresh: float):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.tracks: list[dict] = []  # {box, hits, misses, confirmed, last_seen_box}
        self._next_id = 0

    def update(self, dets: np.ndarray) -> np.ndarray:
        # match dets to tracks greedy by IoU
        matched_track = set()
        matched_det = set()
        if len(self.tracks) and len(dets):
            pairs = []
            for ti, t in enumerate(self.tracks):
                for di, d in enumerate(dets):
                    iou = iou_xyxy(t["box"], d)
                    if iou >= self.iou_thresh:
                        pairs.append((iou, ti, di))
            pairs.sort(reverse=True)
            for _, ti, di in pairs:
                if ti in matched_track or di in matched_det:
                    continue
                matched_track.add(ti)
                matched_det.add(di)
                self.tracks[ti]["box"] = dets[di]
                self.tracks[ti]["hits"] += 1
                self.tracks[ti]["misses"] = 0
                if self.tracks[ti]["hits"] >= self.min_hits:
                    self.tracks[ti]["confirmed"] = True

        # new tracks for unmatched dets
        for di, d in enumerate(dets):
            if di in matched_det:
                continue
            self.tracks.append({
                "box": d.astype(np.float32),
                "hits": 1,
                "misses": 0,
                "confirmed": self.min_hits <= 1,
                "id": self._next_id,
            })
            self._next_id += 1

        # age unmatched tracks
        for ti, t in enumerate(self.tracks):
            if ti not in matched_track:
                t["misses"] += 1

        # drop dead tracks
        self.tracks = [t for t in self.tracks
                       if t["misses"] <= self.max_age and (t["confirmed"] or t["misses"] == 0)]

        # output: all confirmed tracks (active or ghosting)
        out = [t["box"] for t in self.tracks if t["confirmed"]]
        if not out:
            return np.empty((0, 4), dtype=np.float32)
        return np.array(out, dtype=np.float32)


def parse_roi(s: str | None, frame_w: int, frame_h: int) -> tuple[int, int, int, int] | None:
    if not s:
        return None
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--exclude-roi must be 'x,y,w,h'")
    x, y, w, h = parts
    return (x, y, x + w, y + h)


def driver_side_to_roi(side: str | None, frame_w: int, frame_h: int) -> tuple[int, int, int, int] | None:
    if not side:
        return None
    if side == "left":
        return (0, 0, frame_w // 2, frame_h)
    if side == "right":
        return (frame_w // 2, 0, frame_w, frame_h)
    raise ValueError("--driver-side must be 'left' or 'right'")


def iou_with_roi(box: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi
    ix1, iy1 = max(bx1, rx1), max(by1, ry1)
    ix2, iy2 = min(bx2, rx2), min(by2, ry2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    box_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / box_area  # fraction of box inside roi


def expand_box(box: np.ndarray, expand: float, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx, dy = bw * expand * 0.5, bh * expand * 0.5
    return np.array([
        max(0, int(x1 - dx)),
        max(0, int(y1 - dy)),
        min(w, int(x2 + dx)),
        min(h, int(y2 + dy)),
    ])


def blur_region(frame: np.ndarray, box: np.ndarray, method: str, strength: float = 1.0) -> None:
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if method == "pixelate":
        h, w = roi.shape[:2]
        # bigger block = stronger pixelation. scale with strength.
        k = max(10, int(min(w, h) / max(2.0, 6.0 / strength)))
        small = cv2.resize(roi, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_LINEAR)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = out
    else:
        # gaussian: sigma scales with face size + strength. two passes for heavy smear.
        sigma = max(12.0, min(roi.shape[:2]) / 2.5) * strength
        k = int(sigma * 2) | 1  # odd, wider kernel
        blurred = cv2.GaussianBlur(roi, (k, k), sigma)
        if strength >= 1.5:
            blurred = cv2.GaussianBlur(blurred, (k, k), sigma)
        frame[y1:y2, x1:x2] = blurred


def main() -> int:
    ap = argparse.ArgumentParser(description="Face-blur cabin dashcam video.")
    ap.add_argument("video", help="path to input video")
    ap.add_argument("--out", default=None, help="output mp4 path (omit to skip writing)")
    ap.add_argument("--side-by-side", action="store_true", help="write original|blurred side-by-side in output")
    ap.add_argument("--display", action="store_true", help="show window playback")
    ap.add_argument("--detector", choices=["yolo", "retinaface"], default="yolo", help="face detector backend")
    ap.add_argument("--retinaface-net", choices=["resnet50", "mobilenet"], default="resnet50", help="RetinaFace backbone")
    ap.add_argument("--model", default=DEFAULT_MODEL_FILE, help=".pt / .engine path or auto-download (yolo only)")
    ap.add_argument("--conf", type=float, default=0.5, help="detector confidence")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    ap.add_argument("--imgsz", type=int, default=1280, help="detector input size")
    ap.add_argument("--method", choices=["gaussian", "pixelate"], default="gaussian")
    ap.add_argument("--strength", type=float, default=2.5, help="blur intensity multiplier (1=base, 2=heavy, 3=opaque)")
    ap.add_argument("--expand", type=float, default=0.3, help="bbox expand ratio (covers hair/jaw)")
    ap.add_argument("--clahe", action="store_true", default=True, help="CLAHE preprocess to recover faces in glare")
    ap.add_argument("--no-clahe", dest="clahe", action="store_false")
    ap.add_argument("--retry-imgsz", type=int, default=1600, help="2nd-pass imgsz when no detections (0=off)")
    ap.add_argument("--track-buffer", type=int, default=60, help="ByteTrack lost frames buffer (covers glare miss)")
    ap.add_argument("--ghost-frames", type=int, default=75, help="keep blur on lost tracks for N frames (~2.5s @ 30fps)")
    ap.add_argument("--ghost-min-hits", type=int, default=1, help="track must be confirmed N detections before ghosting")
    ap.add_argument("--ghost-iou-match", type=float, default=0.3, help="IoU to match new det against ghost track")
    ap.add_argument("--merge-iou", type=float, default=0.1, help="IoU to union-merge overlapping blur boxes (prevents double-blur)")
    ap.add_argument("--exclude-roi", default=None, help="x,y,w,h region to never blur (driver)")
    ap.add_argument("--driver-side", default=None, choices=["left", "right"], help="quick driver-half exclusion")
    ap.add_argument("--no-track", action="store_true", help="disable ByteTrack temporal smoothing")
    ap.add_argument("--max-fps", type=float, default=0.0, help="cap throughput (0 = uncapped)")
    ap.add_argument("--half", action="store_true", help="force FP16 (auto on CUDA)")
    ap.add_argument("--cpu", action="store_true", help="allow CPU fallback (default: require CUDA)")
    ap.add_argument("--debug", action="store_true", help="print per-frame detection counts")
    ap.add_argument("--live-sim", action="store_true", help="simulate live stream: read frames at source fps, measure realtime lag")
    args = ap.parse_args()

    in_path = Path(args.video)
    if not in_path.exists():
        print(f"ERROR: video not found: {in_path}", file=sys.stderr)
        return 2

    if torch.cuda.is_available():
        device = "cuda:0"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[init] GPU: {gpu_name}")
    else:
        if not args.cpu:
            print("ERROR: CUDA not available. Pass --cpu to allow CPU fallback (slow).", file=sys.stderr)
            return 4
        device = "cpu"
    use_half = device.startswith("cuda") or args.half
    print(f"[init] device={device} half={use_half} imgsz={args.imgsz}")

    if args.detector == "retinaface":
        detector = RetinaFaceDetector(device=device, network=args.retinaface_net)
    else:
        model_path = resolve_model_path(args.model)
        detector = YoloDetector(model_path=model_path, device=device, half=use_half)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open {in_path}", file=sys.stderr)
        return 3
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {w}x{h} @ {fps:.2f}fps frames={nframes}")

    roi = parse_roi(args.exclude_roi, w, h) or driver_side_to_roi(args.driver_side, w, h)
    if roi:
        print(f"[roi] driver exclusion box: {roi}")

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = w * 2 if args.side_by_side else w
        writer = cv2.VideoWriter(args.out, fourcc, fps, (out_w, h))
        print(f"[out] writing {out_w}x{h} {'side-by-side' if args.side_by_side else 'blurred-only'} -> {args.out}")

    ghost = GhostTracker(max_age=args.ghost_frames,
                          min_hits=args.ghost_min_hits,
                          iou_thresh=args.ghost_iou_match) if not args.no_track else None
    if ghost is not None:
        print(f"[track] GhostTracker max_age={args.ghost_frames} min_hits={args.ghost_min_hits} iou={args.ghost_iou_match}")

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if args.clahe else None
    if clahe is not None:
        print("[pre] CLAHE ON (recovers glare-blown faces)")

    min_period = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    src_period = 1.0 / fps  # seconds per source frame
    if args.live_sim:
        print(f"[live-sim] simulating stream @ {fps:.2f}fps ({src_period*1000:.1f}ms/frame budget)")

    t_start = time.time()
    n = 0
    det_ms_total = 0.0
    total_ms_total = 0.0
    late_frames = 0
    max_lag_ms = 0.0
    latencies: list[float] = []
    while True:
        t_loop = time.time()
        # live-sim: skip to current "real" frame so we mimic a live stream that drops behind
        if args.live_sim and n > 0:
            target_idx = int((time.time() - t_start) / src_period)
            while n < target_idx:
                ok, _ = cap.read()
                if not ok:
                    break
                n += 1
        ok, frame = cap.read()
        if not ok:
            break

        original = frame.copy() if args.side_by_side and writer is not None else None

        # preprocess: CLAHE on L channel to recover faces hit by glare/headlights
        det_frame = frame
        if clahe is not None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            det_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # adaptive conf: bright frames drop conf to catch washed-out faces
        mean_lum = float(np.mean(det_frame[:, :, 1])) if clahe is not None else float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        conf_now = max(0.15, args.conf - 0.1) if mean_lum > 170 else args.conf

        t0 = time.time()
        xyxy, confs = detector.detect(det_frame, imgsz=args.imgsz, conf=conf_now, iou=args.iou)
        if args.debug:
            print(f"[debug] frame {n}: {len(xyxy)} dets conf_now={conf_now:.2f}", flush=True)

        # retry larger imgsz if nothing found (rare miss recovery, yolo only)
        if len(xyxy) == 0 and args.retry_imgsz > args.imgsz and args.detector == "yolo":
            xyxy, confs = detector.detect(det_frame, imgsz=args.retry_imgsz,
                                          conf=max(0.15, conf_now - 0.05), iou=args.iou)
        det_ms_total += (time.time() - t0) * 1000

        # ghost tracker: persist blur on lost dets, match new dets to existing tracks
        if ghost is not None:
            try:
                boxes_to_blur = ghost.update(xyxy if len(xyxy) else np.empty((0, 4), dtype=np.float32))
            except Exception as e:
                print(f"[ERR] ghost.update crashed: {type(e).__name__}: {e}")
                boxes_to_blur = xyxy
            if args.debug:
                print(f"[dbg2] frame {n}: ghost_out={len(boxes_to_blur)} tracks={len(ghost.tracks)} confirmed={sum(1 for t in ghost.tracks if t['confirmed'])}", flush=True)
        else:
            boxes_to_blur = xyxy
            if args.debug:
                print(f"[dbg2] frame {n}: NO GHOST raw={len(boxes_to_blur)}", flush=True)

        # expand all boxes first, then merge overlapping (post-expand IoU rises) → single blur per face
        if len(boxes_to_blur) > 0:
            expanded = np.array([expand_box(b, args.expand, w, h) for b in boxes_to_blur], dtype=np.float32)
            expanded = merge_overlapping_boxes(expanded, iou_thresh=args.merge_iou)
            for box in expanded:
                if roi and iou_with_roi(box, roi) > 0.5:
                    continue
                blur_region(frame, box, args.method, args.strength)

        if writer is not None:
            if args.side_by_side and original is not None:
                cv2.putText(original, "ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                blurred_label = frame.copy()
                cv2.putText(blurred_label, "BLURRED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                writer.write(np.hstack([original, blurred_label]))
            else:
                writer.write(frame)

        if args.display:
            cv2.imshow("blur", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_ms = (time.time() - t_loop) * 1000
        total_ms_total += frame_ms
        latencies.append(frame_ms)
        if frame_ms > src_period * 1000:
            late_frames += 1
        if frame_ms > max_lag_ms:
            max_lag_ms = frame_ms

        n += 1
        if min_period:
            dt = time.time() - t_loop
            if dt < min_period:
                time.sleep(min_period - dt)

        if n % 30 == 0:
            elapsed = time.time() - t_start
            print(f"[prog] {n}/{nframes} {n/elapsed:.1f} fps det={det_ms_total/n:.1f}ms/f frame={total_ms_total/n:.1f}ms late={late_frames}")

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    process_fps = n / max(elapsed, 1e-6)
    print(f"[done] {n} frames in {elapsed:.2f}s -> {process_fps:.1f} fps  det avg {det_ms_total/max(n,1):.1f} ms")

    if n > 0 and latencies:
        arr = np.array(latencies, dtype=np.float32)
        budget_ms = src_period * 1000
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        mean = float(arr.mean())
        late_pct = 100.0 * late_frames / n
        realtime = process_fps >= fps
        verdict = "REALTIME ✓" if realtime else "NOT REALTIME ✗"
        print("\n=== LATENCY REPORT ===")
        print(f"  source fps     : {fps:.2f} (budget {budget_ms:.1f} ms/frame)")
        print(f"  process fps    : {process_fps:.2f}")
        print(f"  per-frame ms   : mean={mean:.1f}  p50={p50:.1f}  p95={p95:.1f}  p99={p99:.1f}  max={max_lag_ms:.1f}")
        print(f"  frames > budget: {late_frames}/{n} ({late_pct:.1f}%)")
        print(f"  detector ms    : avg {det_ms_total/n:.1f}")
        print(f"  verdict        : {verdict}")
        if args.live_sim:
            headroom = (budget_ms - mean) / budget_ms * 100
            print(f"  headroom       : {headroom:+.1f}% (positive = spare time, negative = drops frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
