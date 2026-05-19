#!/usr/bin/env python3
"""Automated benchmark sweep for the UpScaling pipeline.

Spawns `python -m src.infer` as a subprocess for each config so that a CUDA
OOM (or any other fatal error) in one run does not kill the harness. After
each run we parse the per-run ``run_info.txt`` for FPS / latency / peak
VRAM, capture the subprocess return code + tail of stderr, and append the
row to CSV + JSON incrementally — so partial results survive a crash of the
harness itself.

Sweep strategy:
- Cartesian over the dimensions that drive throughput AND quality
  (``scale × pre-resize × frame-skip``) — that's the speed/quality
  trade-off the user mostly cares about.
- One-factor-at-a-time (OFAT) for every remaining dimension around a fixed
  ``BASE`` config, since a full Cartesian over 12 dimensions on 8 GB VRAM
  is intractable and most of the cells would OOM or be redundant.

Output:
    output/benchmark/<ts>/runs.csv
    output/benchmark/<ts>/runs.json
    output/benchmark/<ts>/summary.txt
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import itertools
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEO = ROOT / "Real Test Video" / "1.mp4"
SECONDS = 3.0


# ----------------------------- run config ----------------------------------

@dataclass
class RunCfg:
    label: str
    model: str
    scale: int = 4
    pre_resize: str = "25%"
    frame_skip: int = 2
    frame_interp: str = "rife"        # most-used path
    sage_attn: bool = True
    quant: str = "none"
    dtype: str = "bf16"
    device: str = "cuda"
    chunk_frames: int = 16
    topk_ratio: float = 2.0
    kv_ratio: float = 3.0
    local_range: int = 11
    color_fix: bool = True             # i.e. --no-color-fix not set
    esrgan_variant: str = "realesr-general-x4v3"
    esrgan_denoise: float = 0.5
    crf: int = 18
    video_path: str = ""               # filled in by the harness per-video

    def to_argv(self, output_root: Path) -> list[str]:
        in_path = self.video_path or str(INPUT_VIDEO)
        argv = [
            sys.executable, "-m", "src.infer",
            "--model", self.model,
            "--input", in_path,
            "--output", str(output_root),
            "--seconds", str(SECONDS),
            "--scale", str(self.scale),
            "--pre-resize", self.pre_resize,
            "--frame-skip", str(self.frame_skip),
            "--frame-interp", self.frame_interp,
            "--quant", self.quant,
            "--dtype", self.dtype,
            "--device", self.device,
            "--chunk-frames", str(self.chunk_frames),
            "--topk-ratio", str(self.topk_ratio),
            "--kv-ratio", str(self.kv_ratio),
            "--local-range", str(self.local_range),
            "--esrgan-variant", self.esrgan_variant,
            "--esrgan-denoise", str(self.esrgan_denoise),
            "--crf", str(self.crf),
        ]
        if self.sage_attn:
            argv.append("--sage-attn")
        if not self.color_fix:
            argv.append("--no-color-fix")
        return argv


# ----------------------------- sweep grids ---------------------------------

BASE_FLASH = RunCfg(label="flash_base", model="flashvsr_tiny",
                    pre_resize="25%", frame_skip=2, sage_attn=True,
                    quant="none", dtype="bf16", chunk_frames=16,
                    topk_ratio=2.0, kv_ratio=3.0, local_range=11,
                    color_fix=True, crf=18)

BASE_REAL = RunCfg(label="real_base", model="realesrgan_lite",
                   pre_resize="25%", frame_skip=2, sage_attn=False,
                   quant="none", dtype="fp16", crf=18,
                   frame_interp="rife")


SCALES = [2, 4]
# 8 GB Blackwell caps FlashVSR Tiny at ~640x512 SR (under the dense-SDPA
# fallback). Anything 35%+ of 704x576 lands at 896x768 SR and OOMs. Include
# a couple of doomed cells so the harness records the failure mode in the
# CSV, but anchor the sweep on the sizes that actually fit.
PRE_RESIZES_FLASH = ["15%", "20%", "25%", "30%", "35%", "50%"]
# RealESRGAN runs per-frame; it's bounded only by tile size, not seqlen.
PRE_RESIZES_REAL = ["35%", "50%", "75%", "100%"]
FRAME_SKIPS = [1, 2, 3, 4]
QUANTS = ["none", "int8_woq", "nf4"]
DTYPES = ["bf16", "fp16", "fp32"]
CHUNKS = [8, 16, 24, 32]
TOPK = [1.5, 2.0, 3.0]
KV = [1.5, 3.0, 5.0]
LOCAL = [9, 11]
CRFS = [16, 18, 20, 23]


def gen_flashvsr_runs() -> list[RunCfg]:
    runs: list[RunCfg] = []

    # 1) Cartesian over scale × pre-resize × frame-skip × sage_attn.
    #    `--scale 2` doesn't change VRAM (the model is still native 4x;
    #    we only post-downscale), but the user asked to test both, and the
    #    output file size will differ.
    for sc, pr, sk, sa in itertools.product(SCALES, PRE_RESIZES_FLASH, FRAME_SKIPS, [True, False]):
        runs.append(_with(BASE_FLASH,
            label=f"flash_grid_s{sc}_pr{pr}_sk{sk}_sa{int(sa)}",
            scale=sc, pre_resize=pr, frame_skip=sk, sage_attn=sa,
        ))

    # 2) OFAT for the remaining knobs, around BASE_FLASH
    ofat = [
        ("quant", QUANTS, "quant"),
        ("dtype", DTYPES, "dtype"),
        ("chunk_frames", CHUNKS, "chunk"),
        ("topk_ratio", TOPK, "topk"),
        ("kv_ratio", KV, "kv"),
        ("local_range", LOCAL, "local"),
        ("color_fix", [True, False], "cf"),
        ("crf", CRFS, "crf"),
    ]
    for field_name, values, short in ofat:
        for v in values:
            if getattr(BASE_FLASH, field_name) == v:
                continue   # already covered by base
            runs.append(_with(BASE_FLASH,
                label=f"flash_ofat_{short}{v}",
                **{field_name: v},
            ))

    return runs


def gen_realesrgan_runs() -> list[RunCfg]:
    runs: list[RunCfg] = []

    # Cartesian over scale × pre-resize × frame-skip × dtype
    for sc, pr, sk, dt in itertools.product(SCALES, PRE_RESIZES_REAL, FRAME_SKIPS, DTYPES):
        runs.append(_with(BASE_REAL,
            label=f"real_grid_s{sc}_pr{pr}_sk{sk}_{dt}",
            scale=sc, pre_resize=pr, frame_skip=sk, dtype=dt,
        ))

    # OFAT for CRF
    for crf in CRFS:
        if crf == BASE_REAL.crf:
            continue
        runs.append(_with(BASE_REAL, label=f"real_ofat_crf{crf}", crf=crf))

    return runs


BASE_GFPGAN = RunCfg(
    label="gfpgan_base", model="realesrgan_gfpgan",
    scale=4, pre_resize="50%", frame_skip=2, frame_interp="rife",
    sage_attn=False, quant="none", dtype="fp16", crf=18,
)


def gen_gfpgan_runs() -> list[RunCfg]:
    """Smaller targeted grid — the Compact+GFPGAN stack is slower per cell
    (each frame goes through detector + face-restore + bg-upsampler), so a
    211-style Cartesian is overkill. We sweep only the knobs that matter
    for the dev/A100 hand-off:
      - pre-resize: how aggressively we shrink before SR
      - frame-skip + interp: the throughput lever
      - scale: 2 vs 4 (just post-downscale at the writer)
      - dtype: fp16 vs fp32 (Compact half=True path vs full precision)
    Total: 2 scales × 3 pre-resizes × 3 skip values × 2 dtypes = 36 cells
    per video. With 3 videos that's ~108 runs — manageable in ~30 min on
    8 GB once weights are warm.
    """
    runs: list[RunCfg] = []
    scales = [4, 2]
    pre = ["35%", "50%", "none"]
    skips = [1, 2, 4]
    dtypes = ["fp16", "fp32"]
    for sc, pr, sk, dt in itertools.product(scales, pre, skips, dtypes):
        # skip 1 makes frame-interp irrelevant; force "none" for cleanliness
        fi = "none" if sk == 1 else "rife"
        runs.append(_with(BASE_GFPGAN,
            label=f"gfpgan_grid_s{sc}_pr{pr}_sk{sk}_{dt}",
            scale=sc, pre_resize=pr, frame_skip=sk,
            frame_interp=fi, dtype=dt,
        ))
    return runs


def _with(base: RunCfg, **kwargs) -> RunCfg:
    d = asdict(base)
    d.update(kwargs)
    return RunCfg(**d)


# ----------------------------- run executor --------------------------------

@dataclass
class RunResult:
    label: str
    model: str
    args: dict[str, Any]
    status: str = "pending"          # ok | oom | error | timeout
    exit_code: int = -1
    wall_time_s: float = 0.0
    e2e_fps: float = 0.0
    sr_frames: int = 0
    source_frames: int = 0
    out_res: str = ""
    lr_resize: str = ""
    peak_vram_mb: int = 0
    out_dir: str = ""
    upscaled_path: str = ""
    comparison_path: str = ""
    error: str = ""


def _parse_run_info(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.startswith("["):
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


_PEAK_RE = re.compile(r"peak_vram_mb=(\d+)")
_OUT_RE = re.compile(r"-> (\S*output/\S+)")


def run_one(cfg: RunCfg, output_root: Path, timeout: float = 360.0) -> RunResult:
    argv = cfg.to_argv(output_root)
    res = RunResult(label=cfg.label, model=cfg.model, args=asdict(cfg))
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            argv, cwd=str(ROOT), env=env,
            capture_output=True, text=True, timeout=timeout,
        )
        res.exit_code = proc.returncode
        elapsed = time.perf_counter() - t0
        res.wall_time_s = elapsed
        # Peak VRAM from the [infer] tag.
        m = _PEAK_RE.search(proc.stdout or "")
        if m:
            res.peak_vram_mb = int(m.group(1))
        # Out dir from stdout
        mo = _OUT_RE.search(proc.stdout or "")
        if mo:
            raw = mo.group(1)
            # store relative-to-project-root if possible
            try:
                rel = Path(raw).resolve().relative_to(ROOT)
                res.out_dir = str(rel)
            except Exception:
                res.out_dir = raw
            info_abs = Path(raw) if Path(raw).is_absolute() else (ROOT / raw)
            info = info_abs / "run_info.txt"
            if info.exists():
                info_map = _parse_run_info(info)
                # values shown in run_info.txt
                try:
                    res.e2e_fps = float(info_map.get("e2e_fps", "0").split()[0])
                    res.sr_frames = int(info_map.get("sr_frames", "0"))
                    res.source_frames = int(info_map.get("source_frames", "0"))
                    res.wall_time_s = float(info_map.get("wall_time_s", str(elapsed)))
                except Exception:
                    pass
                res.out_res = info_map.get("out_res", "")
                res.lr_resize = info_map.get("lr_resize", "")
                if not res.peak_vram_mb:
                    try:
                        res.peak_vram_mb = int(info_map.get("peak_vram_mb", "0"))
                    except Exception:
                        pass
                up = info_abs / "upscaled.mp4"
                cmp = info_abs / "comparison.mp4"
                if up.exists():
                    try:
                        res.upscaled_path = str(up.relative_to(ROOT))
                    except Exception:
                        res.upscaled_path = str(up)
                if cmp.exists():
                    try:
                        res.comparison_path = str(cmp.relative_to(ROOT))
                    except Exception:
                        res.comparison_path = str(cmp)
        if proc.returncode == 0 and res.e2e_fps > 0:
            res.status = "ok"
        else:
            # Filter to lines that actually carry signal (skip tqdm carriage-return spam).
            err_text = (proc.stderr or "") + "\n" + (proc.stdout or "")
            sig_lines = []
            for line in err_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if any(tag in line for tag in ("[infer]", "FAILED", "Error", "Traceback",
                                               "OutOfMemoryError", "Exception", "AssertionError",
                                               "RuntimeError", "raise ", "  File ")):
                    sig_lines.append(line)
            res.error = " | ".join(sig_lines[-5:]) if sig_lines else \
                       " | ".join(err_text.strip().splitlines()[-3:])
            if "out of memory" in res.error.lower():
                res.status = "oom"
            else:
                res.status = "error"
    except subprocess.TimeoutExpired:
        res.wall_time_s = time.perf_counter() - t0
        res.status = "timeout"
        res.error = f"timed out after {timeout}s"
    except Exception as e:
        res.wall_time_s = time.perf_counter() - t0
        res.status = "error"
        res.error = f"{type(e).__name__}: {e}"
    return res


# ----------------------------- IO ------------------------------------------

CSV_COLS = [
    "label", "model", "status", "exit_code",
    "scale", "pre_resize", "frame_skip", "frame_interp",
    "sage_attn", "quant", "dtype", "chunk_frames",
    "topk_ratio", "kv_ratio", "local_range", "color_fix",
    "esrgan_variant", "esrgan_denoise", "crf",
    "source_frames", "sr_frames", "lr_resize", "out_res",
    "wall_time_s", "e2e_fps", "peak_vram_mb",
    "upscaled_path", "comparison_path", "error",
]


def _flatten(r: RunResult) -> dict[str, Any]:
    d = {**r.args}
    d.update({
        "label": r.label,
        "model": r.model,
        "status": r.status,
        "exit_code": r.exit_code,
        "source_frames": r.source_frames,
        "sr_frames": r.sr_frames,
        "lr_resize": r.lr_resize,
        "out_res": r.out_res,
        "wall_time_s": round(r.wall_time_s, 3),
        "e2e_fps": round(r.e2e_fps, 3),
        "peak_vram_mb": r.peak_vram_mb,
        "upscaled_path": r.upscaled_path,
        "comparison_path": r.comparison_path,
        "error": r.error,
    })
    # drop fields not in CSV_COLS (e.g. "device", "label" key shadowing)
    return {k: d.get(k, "") for k in CSV_COLS}


def append_csv(path: Path, row: dict[str, Any], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with path.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _to_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _to_int(v) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    oom = [r for r in rows if r["status"] == "oom"]
    err = [r for r in rows if r["status"] in ("error", "timeout")]
    for r in ok:
        r["e2e_fps"] = _to_float(r["e2e_fps"])
        r["peak_vram_mb"] = _to_int(r["peak_vram_mb"])
    lines = []
    lines.append(f"runs: {len(rows)}  ok: {len(ok)}  oom: {len(oom)}  error: {len(err)}")
    lines.append("")
    if ok:
        ok_sorted = sorted(ok, key=lambda r: -r["e2e_fps"])
        lines.append("top 10 by e2e_fps:")
        for r in ok_sorted[:10]:
            lines.append(
                f"  {r['e2e_fps']:6.2f} fps  vram={r['peak_vram_mb']:>5} MB  "
                f"{r['model']:14s}  scale={r['scale']} pr={r['pre_resize']:<5} "
                f"sk={r['frame_skip']} sage={r['sage_attn']!s:5} quant={r['quant']:<8} "
                f"dt={r['dtype']:<4}  -> {r['upscaled_path']}"
            )
        lines.append("")
        lines.append("smallest VRAM among ok runs:")
        for r in sorted(ok, key=lambda r: r["peak_vram_mb"])[:5]:
            lines.append(
                f"  vram={r['peak_vram_mb']:>5} MB  fps={r['e2e_fps']:.2f}  "
                f"{r['model']}  {r['pre_resize']}  sk={r['frame_skip']}"
            )
    if oom:
        lines.append("")
        lines.append("OOM configs (first 10):")
        for r in oom[:10]:
            lines.append(f"  {r['label']}  pr={r['pre_resize']}  sk={r['frame_skip']}  quant={r['quant']}")
    if err:
        lines.append("")
        lines.append("non-OOM errors (first 10):")
        for r in err[:10]:
            lines.append(f"  {r['label']}: {r['error'][:120]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------- main ----------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/benchmark")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-run subprocess timeout (s)")
    ap.add_argument("--models", default="flashvsr_tiny,realesrgan_lite",
                    help="comma-sep subset to benchmark")
    ap.add_argument("--videos", default=None,
                    help="comma-sep list of video paths to run the grid against; "
                         "defaults to a single video (Real Test Video/1.mp4). "
                         "Paths are looked up relative to the project root, and "
                         "also under 'Real Test Video/' if a bare filename is given.")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap total runs (0 = no cap, useful for smoke)")
    ap.add_argument("--dry-run", action="store_true",
                    help="just print the plan, don't execute")
    ap.add_argument("--resume", default=None,
                    help="path to existing runs.csv; skip labels already recorded "
                         "and append remaining runs to the same directory")
    args = ap.parse_args()

    # Resolve the input videos.
    videos: list[Path] = []
    if args.videos:
        for raw in args.videos.split(","):
            raw = raw.strip()
            if not raw:
                continue
            p = Path(raw)
            if not p.is_absolute():
                if (ROOT / raw).exists():
                    p = ROOT / raw
                elif (ROOT / "Real Test Video" / raw).exists():
                    p = ROOT / "Real Test Video" / raw
                else:
                    p = ROOT / raw
            if not p.exists():
                print(f"video not found: {raw}", file=sys.stderr)
                return 1
            videos.append(p)
    else:
        if not INPUT_VIDEO.exists():
            print(f"input video not found: {INPUT_VIDEO}", file=sys.stderr)
            return 1
        videos.append(INPUT_VIDEO)

    done_labels: set[str] = set()
    if args.resume:
        resume_csv = Path(args.resume)
        if not resume_csv.exists():
            print(f"resume csv not found: {resume_csv}", file=sys.stderr)
            return 1
        out_root = resume_csv.parent
        csv_path = resume_csv
        with csv_path.open() as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("label"):
                    done_labels.add(row["label"])
        print(f"[bench] resume mode: {len(done_labels)} labels already in {csv_path}")
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(args.out) / ts
        out_root.mkdir(parents=True, exist_ok=True)
        csv_path = out_root / "runs.csv"
    json_path = out_root / "runs.json"
    summary_path = out_root / "summary.txt"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    base_runs: list[RunCfg] = []
    if "flashvsr_tiny" in models:
        base_runs.extend(gen_flashvsr_runs())
    if "realesrgan_lite" in models:
        base_runs.extend(gen_realesrgan_runs())
    if "realesrgan_gfpgan" in models:
        base_runs.extend(gen_gfpgan_runs())

    # Cross every config with every input video. Tag the label so resume + CSV
    # rows stay unique.
    runs: list[RunCfg] = []
    for cfg in base_runs:
        for vp in videos:
            stem = vp.stem
            runs.append(_with(cfg,
                label=f"{cfg.label}__v{stem}",
                video_path=str(vp),
            ))

    if done_labels:
        runs = [r for r in runs if r.label not in done_labels]
    if args.limit > 0:
        runs = runs[: args.limit]

    print(f"[bench] {len(runs)} runs queued over {len(videos)} video(s). out={out_root}")

    if args.dry_run:
        for i, r in enumerate(runs):
            print(f"  [{i:3d}] {r.label}")
        return 0

    rows: list[dict[str, Any]] = []
    if args.resume and csv_path.exists():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                # Re-cast numeric fields so write_summary's sort keys work.
                for k in ("e2e_fps", "wall_time_s", "esrgan_denoise",
                          "topk_ratio", "kv_ratio"):
                    if row.get(k):
                        try:
                            row[k] = float(row[k])
                        except Exception:
                            row[k] = 0.0
                for k in ("peak_vram_mb", "exit_code", "source_frames",
                          "sr_frames", "scale", "frame_skip", "chunk_frames",
                          "local_range", "crf"):
                    if row.get(k):
                        try:
                            row[k] = int(row[k])
                        except Exception:
                            row[k] = 0
                rows.append(row)
    started = time.perf_counter()
    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    for i, cfg in enumerate(runs):
        t0 = time.perf_counter()
        print(f"[bench] {i+1:3d}/{len(runs)} {cfg.label}  pr={cfg.pre_resize} "
              f"sk={cfg.frame_skip} sage={cfg.sage_attn} quant={cfg.quant} "
              f"dt={cfg.dtype} scale={cfg.scale}", flush=True)
        r = run_one(cfg, output_root=ROOT / "output", timeout=args.timeout)
        dt = time.perf_counter() - t0
        flat = _flatten(r)
        rows.append(flat)
        append_csv(csv_path, flat, write_header=not csv_exists)
        csv_exists = True
        write_json(json_path, rows)
        write_summary(summary_path, rows)
        print(f"        {r.status}  fps={r.e2e_fps:6.2f}  vram={r.peak_vram_mb:>5} MB  "
              f"wall={dt:6.2f}s  err={(r.error or '')[:70]}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"[bench] done in {elapsed:.1f}s. csv={csv_path}")
    write_summary(summary_path, rows)
    print(f"[bench] summary at {summary_path}")
    print("-" * 60)
    print(summary_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
