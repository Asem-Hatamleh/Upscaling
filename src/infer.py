"""Command-line inference entrypoint.

Usage examples:
    python -m src.infer --model flashvsr_tiny --input "Real Test Video/1.mp4"
    python -m src.infer --model flashvsr_tiny --input "Real Test Video" \\
        --seconds 5 --scale 4 --pre-resize vga --frame-skip 2 --frame-interp rife \\
        --sage-attn --quant int8_woq

Output layout: ``output/<model>/<stem>_<args>/{upscaled.mp4, comparison.mp4, run_info.txt}``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

from . import io_utils, sage_patch
from .models.base import UpscalerConfig, available, build
from .runner import RunOptions, process_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time video upscaling pipeline (FlashVSR Tiny + alt).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", required=True,
                   choices=sorted(available()) or ["flashvsr_tiny", "realesrgan_lite"],
                   help="Upscaler model")
    p.add_argument("--input", "-i", required=True,
                   help="Single video file OR directory containing videos")
    p.add_argument("--output", "-o", default="output",
                   help="Output root directory")

    p.add_argument("--seconds", type=float, default=0.0,
                   help="Seconds to process per video (0 = full)")
    p.add_argument("--scale", type=int, default=4, choices=[2, 4],
                   help="Final scale factor; 4 is native, 2 post-downscales SR by half")
    p.add_argument("--pre-resize", default="none",
                   help="Pre-process input resize: none | vga | qvga | WxH | pct:NN")

    p.add_argument("--frame-skip", type=int, default=1,
                   help="Run SR on every Nth frame (1 = no skip)")
    p.add_argument("--frame-interp", default="none",
                   choices=["none", "repeat", "rife"],
                   help="How to fill dropped frames after SR")
    p.add_argument("--rife-weights", default="RIFE_trained_v6",
                   help="Path to RIFE_trained_v6 (folder containing train_log/)")

    # Performance toggles
    p.add_argument("--sage-attn", action="store_true",
                   help="Enable SageAttention SDPA patch (Linux only)")
    p.add_argument("--quant", default="none",
                   choices=["none", "int8_woq", "nf4"],
                   help="Quantization scheme (model-dependent)")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="cuda")

    # FlashVSR specific
    p.add_argument("--chunk-frames", type=int, default=16,
                   help="FlashVSR: inference frames per chunk (lower = less VRAM)")
    p.add_argument("--topk-ratio", type=float, default=2.0,
                   help="FlashVSR: block-sparse top-k ratio (1.5/2.0/3.0)")
    p.add_argument("--kv-ratio", type=float, default=3.0)
    p.add_argument("--local-range", type=int, default=11)
    p.add_argument("--no-color-fix", action="store_true")

    # Real-ESRGAN specific
    p.add_argument("--esrgan-variant", default="realesr-general-x4v3",
                   choices=["realesr-general-x4v3", "realesr-animevideov3"])
    p.add_argument("--esrgan-denoise", type=float, default=0.5,
                   help="0..1 denoise strength for realesr-general-x4v3")
    p.add_argument("--esrgan-tile", type=int, default=0,
                   help="Tile size for VRAM-constrained inference (0 = no tile)")

    # Output toggles
    p.add_argument("--no-comparison", action="store_true",
                   help="Skip side-by-side comparison output")
    p.add_argument("--no-upscaled", action="store_true",
                   help="Skip upscaled-only output")
    p.add_argument("--crf", type=int, default=18,
                   help="x264 quality (lower = bigger/better, 18 ~ visually lossless)")

    return p.parse_args(argv)


def collect_videos(input_path: str) -> List[Path]:
    p = Path(input_path)
    if not p.exists():
        raise SystemExit(f"input not found: {p}")
    vids = io_utils.list_videos(p)
    if not vids:
        raise SystemExit(f"no video files under: {p}")
    return vids


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.sage_attn:
        ok = sage_patch.enable()
        if not ok:
            print("[infer] sage-attn requested but unavailable; continuing with torch SDPA.",
                  file=sys.stderr)

    extra: dict[str, object] = {
        "chunk_frames": args.chunk_frames,
        "topk_ratio": args.topk_ratio,
        "kv_ratio": args.kv_ratio,
        "local_range": args.local_range,
        "color_fix": not args.no_color_fix,
        # alt model
        "model_name": args.esrgan_variant,
        "denoise_strength": args.esrgan_denoise,
        "tile": args.esrgan_tile,
    }

    cfg = UpscalerConfig(
        name=args.model,
        quant=args.quant,
        sage_attn=args.sage_attn,
        device=args.device,
        dtype=args.dtype,
        extra=extra,
    )
    print(f"[infer] loading model: {cfg.name} (quant={cfg.quant} dtype={cfg.dtype})")
    upscaler = build(cfg)
    upscaler.load()

    # Optional RIFE
    rife = None
    if args.frame_interp == "rife":
        from . import rife_interp
        try:
            rife = rife_interp.RifeInterpolator(args.rife_weights, device=args.device).load()
            print("[infer] RIFE loaded")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[infer] RIFE unavailable ({e}); falling back to --frame-interp repeat.",
                  file=sys.stderr)
            args.frame_interp = "repeat"

    opts = RunOptions(
        seconds=args.seconds,
        out_scale=args.scale,
        pre_resize=args.pre_resize,
        frame_skip=args.frame_skip,
        frame_interp=args.frame_interp,
        write_comparison=not args.no_comparison,
        write_upscaled=not args.no_upscaled,
        chunk_frames=args.chunk_frames,
        rife_weights=args.rife_weights,
        crf=args.crf,
    )

    videos = collect_videos(args.input)
    print(f"[infer] {len(videos)} video(s) queued")

    out_root = Path(args.output)
    overall_t0 = __import__("time").perf_counter()
    total_src = total_sr = 0
    for vp in videos:
        print(f"[infer] -> {vp}")
        try:
            res = process_video(
                upscaler=upscaler,
                video_path=vp,
                out_root=out_root,
                opts=opts,
                model_id=cfg.name,
                quant=cfg.quant,
                sage=cfg.sage_attn,
                rife=rife,
            )
        except Exception as e:
            print(f"[infer] FAILED on {vp}: {e}", file=sys.stderr)
            continue
        print(f"      wall={res.wall_time_s:.2f}s  src={res.n_source}  sr={res.n_sr}  "
              f"e2e_fps={res.e2e_fps:.2f}  -> {res.out_dir}")
        total_src += res.n_source
        total_sr += res.n_sr

    elapsed = __import__("time").perf_counter() - overall_t0
    print(f"[infer] done in {elapsed:.1f}s  total_src={total_src}  total_sr={total_sr}")
    upscaler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
