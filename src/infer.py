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

    p.add_argument("--model", default=None,
                   choices=sorted(available()) or ["flashvsr_tiny", "realesrgan_lite"],
                   help="Upscaler model. If omitted, prompts interactively.")
    p.add_argument("--interactive", "-I", action="store_true",
                   help="Force the interactive wizard for every option, even if "
                        "they were passed on the command line.")
    p.add_argument("--input", "-i", default=None,
                   help="Single video file OR directory containing videos. "
                        "If omitted, prompts interactively.")
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


_DEFAULT_VIDEO_DIR = "Real Test Video"


# ----------------------------- interactive wizard ----------------------------

def _ask(prompt: str, default, choices=None, cast=str, allow_empty=False):
    """Prompt the user, return value of type ``cast``.

    Press Enter to accept ``default``. ``choices`` is an optional list/tuple
    of allowed string values (compared case-insensitively against the typed
    input). ``cast`` runs on the final string before return.
    """
    default_str = "" if default is None else str(default)
    suffix = f" [{default_str}]" if default_str != "" else ""
    hint = f" {{{', '.join(choices)}}}" if choices else ""
    while True:
        try:
            raw = input(f"{prompt}{hint}{suffix}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[infer] aborted.", file=sys.stderr)
            raise SystemExit(130)
        if raw == "":
            if default is None and not allow_empty:
                print("  (required)")
                continue
            return default
        if choices is not None:
            lower = raw.lower()
            for c in choices:
                if c.lower() == lower:
                    raw = c
                    break
            else:
                print(f"  must be one of {choices}")
                continue
        try:
            return cast(raw)
        except Exception as e:
            print(f"  invalid ({cast.__name__}): {e}")


def _ask_bool(prompt: str, default: bool) -> bool:
    while True:
        raw = input(f"{prompt} [{'y' if default else 'n'}]> ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes", "true", "1", "on"):
            return True
        if raw in ("n", "no", "false", "0", "off"):
            return False
        print("  enter y or n")


def run_wizard(args: argparse.Namespace) -> None:
    """Walk every option, prompting with the current value as default.

    Mutates ``args`` in place. Hides model-specific options that don't apply
    to the chosen backend.
    """
    print("=" * 60)
    print(" UpScaling — interactive run")
    print("=" * 60)

    # 1. Model
    models = sorted(available()) or ["flashvsr_tiny", "realesrgan_lite"]
    args.model = _ask("model", args.model or models[0], choices=models)

    # 2. Input (reuse the picker — it handles the index/filename/path UX)
    if args.input is None:
        args.input = prompt_for_input()
    else:
        keep = _ask_bool(f"keep input '{args.input}'?", True)
        if not keep:
            args.input = prompt_for_input()

    # 3. Output root
    args.output = _ask("output dir", args.output)

    # 4. Duration
    args.seconds = _ask("seconds to process (0 = full video)",
                        args.seconds, cast=float)

    # 5. Scale
    args.scale = _ask("output scale", args.scale, choices=["2", "4"], cast=int)

    # 6. Pre-resize
    print("  pre-resize choices: none | vga | qvga | WxH (e.g. 160x128) | NN% (e.g. 30%)")
    args.pre_resize = _ask("pre-resize", args.pre_resize)

    # 7. Frame skip
    args.frame_skip = _ask("frame skip (1 = no skip, 2 = every other, ...)",
                           args.frame_skip, cast=int)
    if args.frame_skip > 1:
        args.frame_interp = _ask("gap fill", args.frame_interp,
                                 choices=["none", "repeat", "rife"])
        if args.frame_interp == "rife":
            args.rife_weights = _ask("RIFE weights dir", args.rife_weights)
    else:
        args.frame_interp = "none"

    # 8. Perf
    args.sage_attn = _ask_bool("enable SageAttention?", args.sage_attn)
    args.quant = _ask("quantization", args.quant,
                      choices=["none", "int8_woq", "nf4"])
    args.dtype = _ask("dtype", args.dtype, choices=["bf16", "fp16", "fp32"])
    args.device = _ask("device", args.device)

    # 9. Model-specific
    if args.model == "flashvsr_tiny":
        args.chunk_frames = _ask("FlashVSR chunk_frames", args.chunk_frames, cast=int)
        args.topk_ratio = _ask("FlashVSR sparse top-k ratio (1.5/2.0/3.0)",
                               args.topk_ratio, cast=float)
        args.kv_ratio = _ask("FlashVSR kv_ratio", args.kv_ratio, cast=float)
        args.local_range = _ask("FlashVSR local_range (9 or 11)",
                                args.local_range, cast=int)
        args.no_color_fix = not _ask_bool("color-fix?", not args.no_color_fix)
    elif args.model == "realesrgan_lite":
        args.esrgan_variant = _ask("RealESRGAN variant", args.esrgan_variant,
                                   choices=["realesr-general-x4v3",
                                            "realesr-animevideov3"])
        args.esrgan_denoise = _ask("RealESRGAN denoise strength (0..1)",
                                   args.esrgan_denoise, cast=float)
        args.esrgan_tile = _ask("RealESRGAN tile size (0 = no tile)",
                                args.esrgan_tile, cast=int)

    # 10. Output toggles
    args.no_comparison = not _ask_bool("write side-by-side comparison.mp4?",
                                       not args.no_comparison)
    args.no_upscaled = not _ask_bool("write upscaled.mp4?",
                                     not args.no_upscaled)
    args.crf = _ask("x264 CRF (lower = bigger/better)", args.crf, cast=int)

    print("-" * 60)
    print(" Resolved args:")
    for k in sorted(vars(args)):
        if k == "interactive":
            continue
        print(f"   --{k.replace('_', '-')}: {getattr(args, k)}")
    print("-" * 60)
    if not _ask_bool("proceed?", True):
        print("[infer] cancelled by user.")
        raise SystemExit(0)


def prompt_for_input() -> str:
    """Interactive picker. Shows videos in ``Real Test Video/`` (if present)
    and lets the user pick by index, type a filename relative to that folder,
    or paste any absolute / relative path."""
    print("[infer] no --input given. Enter a video path, a directory, or pick an index.")
    candidates: List[Path] = []
    default_dir = Path(_DEFAULT_VIDEO_DIR)
    if default_dir.exists():
        candidates = io_utils.list_videos(default_dir)
        if candidates:
            print(f"[infer] videos in {default_dir}/:")
            for i, v in enumerate(candidates):
                print(f"   [{i}] {v.name}")
            print(f"   [a] all of the above (whole folder)")
    while True:
        raw = input("input> ").strip()
        if not raw:
            print("[infer] empty input, try again (Ctrl-C to quit)")
            continue
        if raw.lower() in ("a", "all") and candidates:
            return str(default_dir)
        if raw.isdigit() and candidates:
            idx = int(raw)
            if 0 <= idx < len(candidates):
                return str(candidates[idx])
            print(f"[infer] index out of range (0..{len(candidates) - 1})")
            continue
        # treat as a path; if it doesn't exist standalone, try under the default dir
        p = Path(raw)
        if p.exists():
            return str(p)
        alt = default_dir / raw
        if alt.exists():
            return str(alt)
        print(f"[infer] not found: {raw} (also tried {alt})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Trigger wizard when explicitly asked, or when any required arg is missing.
    needs_wizard = args.interactive or args.model is None or args.input is None
    if needs_wizard:
        run_wizard(args)

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
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                print("[infer] HINT: lower --pre-resize (e.g. 160x128), enable "
                      "--sage-attn, set --quant int8_woq, or "
                      "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.",
                      file=sys.stderr)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
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
