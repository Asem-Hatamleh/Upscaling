# UpScaling — Real-ESRGAN Lite Stack

Branch-scoped distribution of the UpScaling project that ships **only the
Real-ESRGAN family of backends** (no FlashVSR, no BasicVSR++).

Targeted at in-cabin driver-monitoring streams (704×576 @ ~30 FPS) where you
want a small, deterministic, per-frame model — optionally paired with a face
restorer for the driver's face.

| Model | Native scale | What it does | When to use |
|-------|--------------|--------------|-------------|
| **`realesrgan_lite`** (`realesr-general-x4v3`, SRVGGNetCompact) | 4× | Lightest (~10 MB) per-frame SR | Throughput baseline, low-VRAM target, multi-stream batching |
| **`realesrgan_full`** (RRDBNet `RealESRGAN_x4plus`, ~64 MB) | 4× | Heavier per-frame SR with better fine detail | Best non-face quality if VRAM allows |
| **`realesrgan_gfpgan`** (Compact bg + GFPGAN-1.4 faces) | 4× | Compact background + GFPGAN face restoration | Driver-face quality when "polished / plastic" skin is OK |
| **`codeformer_compact`** (Compact bg + CodeFormer-v1 faces) | 4× | Compact background + CodeFormer codebook face restorer | Driver-face quality with stronger identity preservation; better on profile / occluded / low-light cabin faces |

All four backends share the same CLI, frame-skip + RIFE pipeline, output
layout, and `run_info.txt` schema.

> The FlashVSR (diffusion / DiT) and BasicVSR++ (true temporal SR) backends
> live on `main`. This branch deliberately strips them — and their heavy
> deps (`bitsandbytes`, `sageattention`, `diffsynth`, `block-sparse-attn`)
> — so installation is fast and only requires the Real-ESRGAN toolchain.

---

## 1. Hardware & OS

| | Dev | Prod |
|---|---|---|
| GPU | RTX 5050 (sm_120 Blackwell) 8 GB | NVIDIA A100 80 GB |
| CUDA | 13.x | 12.x |
| Driver | 595.x | latest stable |
| OS | Linux (Ubuntu 24.04 tested) | Linux |
| Python | 3.11.x (recommended) | 3.11.x |

CPU-only is **not** supported — `realesrgan` / `gfpgan` / facexlib all assume
CUDA, and the per-frame throughput on CPU is unusable for video.

## 2. Quick install

```bash
git clone -b realesrgan-lite https://github.com/Asem-Hatamleh/Upscaling.git
cd Upscaling
./setup_linux.sh                  # creates ./venv, installs deps, downloads weights
source venv/bin/activate
```

`setup_linux.sh` flags:

- `--no-downloads` — skip pre-fetch of Real-ESRGAN / GFPGAN / CodeFormer /
  facexlib weights (they will still auto-download on first inference).
- `--python=python3.11` — pick a specific interpreter.

Manual setup:

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -U pip wheel
# PyTorch — pick one for your GPU:
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128   # Blackwell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121                  # A100 / Ampere
pip install -r requirements.txt
python scripts/download_weights.py --model all
```

## 3. Running inference

CLI entrypoint: `python -m src.infer`.

### `realesrgan_lite` — background only

```bash
python -m src.infer \
  --model realesrgan_lite \
  --input "Real Test Video/1.mp4" \
  --seconds 10 --scale 4 --pre-resize 80% \
  --frame-skip 1 --frame-interp none --dtype fp16 \
  --esrgan-denoise 1.0
```

Weights (~10 MB SRVGGNetCompact + ~10 MB WDN denoise companion) auto-download
to `weights/realesrgan/` on first use.

### `realesrgan_full` — heavier RRDB background

```bash
python -m src.infer \
  --model realesrgan_full \
  --input "Real Test Video/1.mp4" \
  --seconds 10 --scale 4 --pre-resize none \
  --frame-skip 1 --dtype fp16
```

### `realesrgan_gfpgan` — Compact + GFPGAN face

```bash
python -m src.infer \
  --model realesrgan_gfpgan \
  --input "Real Test Video/1.mp4" \
  --seconds 10 --scale 4 --pre-resize 80% \
  --frame-skip 1 --frame-interp none --dtype fp16 \
  --esrgan-denoise 1.0
```

GFPGAN-1.4 (~333 MB) auto-downloads to `weights/gfpgan/`. The face detector
backbone (RetinaFace-ResNet50 via facexlib, ~104 MB) auto-downloads to
`weights/facexlib/` on first call, or pre-fetch with
`python scripts/download_weights.py --model facexlib`.

### `codeformer_compact` — Compact + CodeFormer face (recommended in-cabin args)

```bash
python -m src.infer \
  --model codeformer_compact \
  --input "Test data/001.mp4" \
  --output output/final_run \
  --seconds 10 --scale 4 --pre-resize 80% \
  --frame-skip 1 --frame-interp none --dtype fp16 \
  --esrgan-denoise 1.0 \
  --cf-fidelity 1.0 --eye-dist-threshold 20
```

CodeFormer (~360 MB) auto-downloads to `weights/codeformer/`. The
CodeFormer arch is vendored under `src/models/_codeformer/`, so you don't
need a clone of the upstream CodeFormer repo at runtime.

In-cabin tuning notes:

- `--cf-fidelity 1.0` — max identity preservation. Stops the codebook from
  inventing eyes / mouth between frames (main source of inter-frame face
  flicker we saw at `0.7`–`0.9`).
- `--eye-dist-threshold 20` — drop face detections whose eye centers are
  closer than 20 source-pixels. Kills tiny / mirror-reflection detections
  that otherwise jitter the align matrix and trigger codebook flips.
- `--esrgan-denoise 1.0` — full denoise blend on the Compact background;
  cabin sensors are noisy.
- `--frame-skip 1` + `--frame-interp none` — keep RIFE out of the
  comparison so any remaining flicker is attributable to SR or the face
  stage, not to gap-fill.
- `only_center_face=True` is the hard-coded default for `codeformer_compact`
  and `realesrgan_gfpgan`, so the driver's face is restored while
  passenger-seat / rear-view reflections are ignored automatically.

### Folder of videos

`--input` accepts a directory; every `.mp4 / .mov / .mkv / .avi / .webm`
underneath is processed:

```bash
python -m src.infer --model realesrgan_lite \
  --input "Real Test Video" --seconds 3 --pre-resize 80% --dtype fp16
```

### Sweeping 10 sample clips (final-run pattern)

```bash
for i in 001 002 003 004 005 006 007 008 009 010; do
  python -m src.infer \
    --model codeformer_compact \
    --input "Test data/$i.mp4" \
    --output output/final_run \
    --seconds 10 --scale 4 --pre-resize 80% \
    --frame-skip 1 --frame-interp none --dtype fp16 \
    --esrgan-denoise 1.0 \
    --cf-fidelity 1.0 --eye-dist-threshold 20
done
```

## 4. All active CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--model` | *required* | `realesrgan_lite` / `realesrgan_full` / `realesrgan_gfpgan` / `codeformer_compact` |
| `--input` / `-i` | *required* | single video OR directory |
| `--output` / `-o` | `output` | output root |
| `--seconds` | `0` | seconds to process; `0` = full video |
| `--scale` | `4` | `2` or `4` (model is 4× native; `2` post-downscales) |
| `--pre-resize` | `none` | `none`/`vga`/`qvga`/`WxH`/`pct:NN`/`NN%` |
| `--frame-skip` | `1` | run SR every Nth frame |
| `--frame-interp` | `none` | `none`/`repeat`/`rife` — gap fill for dropped frames |
| `--rife-weights` | `RIFE_trained_v6` | folder containing `train_log/flownet.pkl` (only needed for `--frame-interp rife`) |
| `--dtype` | `bf16` | `bf16`/`fp16`/`fp32`. `fp16` is the practical default for Real-ESRGAN + face restorers |
| `--device` | `cuda` | CPU is not supported |
| `--esrgan-variant` | `realesr-general-x4v3` | also `realesr-animevideov3` for stylized inputs |
| `--esrgan-denoise` | `0.5` | 0..1 noise/sharpness balance. `1.0` for cabin sensor noise |
| `--esrgan-tile` | `0` | tile size for VRAM-tight runs (per-tile SR + paste-back) |
| `--cf-fidelity` | `0.9` | CodeFormer identity weight: `0` hallucinates, `1` locks landmarks. Use `1.0` for in-cabin to kill codebook flicker |
| `--face-detect-all` | off | restore every detected face. Default keeps only the largest/center (driver) and ignores mirror reflections |
| `--eye-dist-threshold` | `10` | drop face detections whose eye centers are closer than N source-pixels. `20` kills tiny/jittery cabin detections |
| `--no-comparison` | off | skip side-by-side output |
| `--no-upscaled` | off | skip upscaled-only output |
| `--crf` | `18` | x264 / NVENC `cq` quality (lower = larger/better) |
| `--encoder` | `libx264` | `libx264` / `h264_nvenc` / `auto`. NVENC is 5-10× faster than libx264 |
| `--live` | off | Live-streaming mode: producer/consumer threading (decode/SR/encode overlap), implies `--no-comparison`, picks NVENC if available, small chunk for low TTFB |
| `--io-threads` | `2` | bounded queue depth between live-mode pipeline threads |

### Dead flags (no effect on this branch)

These flags still appear in `--help` for backward compatibility with the
benchmark harness but are no-ops with the Real-ESRGAN family — they only
mattered for the FlashVSR backbone, which is not in this distribution:

`--sage-attn`, `--quant`, `--chunk-frames`, `--topk-ratio`, `--kv-ratio`,
`--local-range`, `--no-color-fix`.

## 5. Output layout

```
output/
└── codeformer_compact/
    └── 001_sec10_scale4x_512x288_skip1_interp-none_q-none_sage-off_dt-fp16_tile-0_dn-1_var-realesr-general-x4v3_cf-1_eye-20_all-0_crf-18/
        ├── upscaled.mp4
        ├── comparison.mp4     # full-res original | upscaled, side by side
        └── run_info.txt
```

`run_info.txt` carries `[run]`, `[args]`, `[timing]` sections — model,
SageAttention status (always `off` on this branch), input args, frame counts,
wall time, end-to-end FPS.

## 6. Real-time / live-streaming strategy

### Quick switch: `--live`

```bash
python -m src.infer --live \
  --model realesrgan_lite \
  --input "Real Test Video/1.mp4" \
  --pre-resize 80% --dtype fp16 --esrgan-denoise 1.0
```

`--live` enables a producer/consumer pipeline that **overlaps decode, SR,
and encode on three dedicated threads** — CPU and GPU work in parallel
instead of taking turns. It also:

- forces `--no-comparison` (the side-by-side mp4 doubles work);
- drops `--chunk-frames` to a small value (4) so time-to-first-output
  stays low;
- promotes `--encoder` to `auto`, which prefers `h264_nvenc` if PyAV's
  ffmpeg build supports it (5–10× faster than libx264).

Tune `--io-threads N` (default `2`) to deepen the bounded queues between
threads if your decoder is jittery (e.g. RTSP source).

### Other levers

1. **Pre-resize input** — `--pre-resize 80%` keeps detail while cutting
   compute ~36%. For tighter budgets, `50%` (352×288 → 1408×1152 SR) is
   usually enough for driver-monitoring downstream models.
2. **`--frame-skip 2`** + **`--frame-interp rife`** — SR runs every other
   frame; RIFE-HDv3 fills the gap. ~1.6× throughput at minor quality cost
   on non-face backbones. For face restorers, prefer `--frame-skip 1`
   because RIFE can interpolate face features inconsistently. Note:
   `--frame-skip` is ignored under `--live` (the streaming pipeline runs
   every frame to keep latency bounded).
3. **`--dtype fp16`** — Real-ESRGAN Compact and RRDB both support `half=True`
   end-to-end; GFPGAN and CodeFormer paths also benefit.
4. **`--esrgan-tile NN`** — tile size > 0 splits each frame into NN×NN
   patches with reflection padding for VRAM-tight machines. Costs throughput
   but lets you run native input on 4–6 GB GPUs.
5. **`--encoder h264_nvenc`** — NVIDIA NVENC hardware encoder. Needs a
   PyAV/ffmpeg build with NVENC support; the writer falls back to libx264
   with a warning if NVENC isn't available. Use `--encoder auto` to pick
   automatically.

Torch-side, the CLI also enables `cudnn.benchmark=True` and TF32 on
matmul/cudnn on every run — the SR forward pass runs inside
`torch.inference_mode()` (skips autograd bookkeeping, ~10-15% faster
than `no_grad`).

On A100 80 GB drop most of the tricks: `--live --pre-resize none
--dtype fp16 --encoder h264_nvenc`.

## 7. Test clips

Two local-only clip sets, both gitignored:

```
Real Test Video/      # short curated set, used by benchmark presets
├── 1.mp4   # 704x576 25 fps 32s   — primary benchmark clip
├── 2.mp4
├── 3.mp4
├── 4.mp4
└── 5.mp4 … 9.mp4

Test data/            # 100 in-cabin samples for final-run evaluation
├── 001.mp4
├── 002.mp4
└── … 100.mp4
```

A handful of `Test data/` clips ship with damaged H.264 NAL units
(`error while decoding MB X Y, bytestream -N`). The decoder error-conceals
and the upscaler runs through; the bad macroblock just becomes a brief
artifact in that region of the output. Inspect with
`ffmpeg -v error -i <clip> -f null -` before treating the result as ground
truth.

## 8. Project layout

```
UpScaling/
├── README.md
├── requirements.txt
├── setup_linux.sh
├── .gitignore
├── src/
│   ├── infer.py                  # CLI entrypoint + interactive wizard
│   ├── runner.py                 # per-video orchestration
│   ├── io_utils.py               # video I/O + side-by-side
│   ├── report.py                 # run_info.txt writer
│   ├── frame_skip.py             # skip + gap-fill
│   ├── rife_interp.py            # RIFE-HDv3 wrapper (used by --frame-interp rife)
│   └── models/
│       ├── base.py                 # Upscaler interface + registry
│       ├── realesrgan_lite.py      # Real-ESRGAN Compact baseline
│       ├── realesrgan_full.py      # Real-ESRGAN RRDB (x4plus) wrapper
│       ├── realesrgan_gfpgan.py    # Compact bg + GFPGAN-1.4 face restore
│       ├── codeformer_compact.py   # Compact bg + CodeFormer-v1 face restore
│       └── _codeformer/            # vendored CodeFormer arch files
├── scripts/
│   ├── download_weights.py
│   ├── benchmark.py              # parameter sweep harness (presets + OFAT)
│   └── analyze_benchmark.py      # CSV → Markdown report
├── benchmarks/latest/            # last sweep's CSV / JSON / REPORT.md
├── weights/realesrgan/           # downloaded by setup_linux.sh (gitignored)
├── weights/gfpgan/               # GFPGANv1.4.pth (gitignored)
├── weights/codeformer/           # codeformer.pth (gitignored)
├── weights/facexlib/             # RetinaFace ResNet50 + ParseNet (gitignored)
├── RIFE_trained_v6/              # checked-in RIFE-HDv3 weights + ECCV source
├── Real Test Video/              # local clips (gitignored)
└── Test data/                    # local in-cabin samples (gitignored)
```

## 9. Benchmark harness

`scripts/benchmark.py` exposes named sweep presets that target this stack:

```bash
# Quick A/B between lite / full / gfpgan / codeformer at fixed-res in-cabin args
python scripts/benchmark.py \
  --preset esrgan_target10 \
  --videos "Real Test Video/1.mp4,3.mp4" \
  --seconds 10 --scales 4 --pre-resizes 80% --frame-skips 1 \
  --dtypes fp16 --esrgan-denoise-values 1.0 \
  --timeout 2800 --auto-resume

# Sweep CodeFormer fidelity / eye-distance for temporal stability
python scripts/benchmark.py \
  --preset codeformer_temporal \
  --videos "Real Test Video/1.mp4,3.mp4" \
  --seconds 10 --scales 4 --pre-resizes 80% \
  --cf-fidelities 0.7,1.0 --eye-dist-thresholds 20 \
  --esrgan-denoise-values 1.0 --dtypes fp16 \
  --timeout 2800 --auto-resume
```

Outputs per sweep:

- `output/benchmark/<ts>/runs.csv` — one row per cell.
- `output/benchmark/<ts>/runs.json` — same data, machine-readable.
- `output/benchmark/<ts>/summary.txt` — top-N by FPS / VRAM / errors.
- `output/benchmark/<ts>/started.jsonl` — crash-resume log.

Use `--auto-resume` to pick up the most recent sweep with the same `--preset`
after a crash or interruption.

## 10. Troubleshooting

- **`CUDA error: no kernel image is available for execution`** — wrong torch
  for your GPU. Re-run `setup_linux.sh` so it picks the cu128 nightly for
  Blackwell.
- **`Disk quota exceeded` during pip install** — `/tmp` is tmpfs (~7 GB on
  this machine). `setup_linux.sh` exports `TMPDIR=$ROOT/.build_tmp` to
  side-step it.
- **`basicsr` import fails on `torchvision.transforms.functional_tensor`** —
  `setup_linux.sh` sed-patches the upstream import; if you bypassed setup,
  run that patch by hand.
- **OOM** — lower `--pre-resize` (try `50%` or `35%`), set `--esrgan-tile 256`
  or `--esrgan-tile 128`, set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Choppy output with high `--frame-skip`** — switch `--frame-interp` from
  `repeat` to `rife`.
- **`No module named 'RIFE_HDv3'`** — RIFE v6 weights need the ECCV2022-RIFE
  Python source. `setup_linux.sh` clones it and installs files into
  `RIFE_trained_v6/model/`. Only required if you use `--frame-interp rife`.
- **`error while decoding MB X Y, bytestream -N`** — source `.mp4` has
  damaged H.264 NAL units. ffmpeg error-conceals and the upscaler runs
  through; the bad macroblock just becomes a brief artifact in that region
  of the output. Source-side issue, not the pipeline.
- **`--sage-attn requested but unavailable`** — SageAttention is removed on
  this branch; the flag is a no-op and prints a warning. Strip it from your
  invocation.

## 11. License

Source under MIT. Real-ESRGAN weights under BSD-3. GFPGAN-1.4 weights under
the upstream Apache 2.0 license. CodeFormer weights under the upstream
S-Lab Non-Commercial License — review before any production deployment.
