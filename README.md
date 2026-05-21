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
| `--live` | off | Live-streaming mode: producer/consumer threading (decode/SR/encode overlap), implies `--no-comparison`, picks NVENC if available, small chunk for low TTFB, opens two preview windows |
| `--no-preview` | off | Disable the two real-time `Original` / `Upscaled` cv2 windows that `--live` opens by default. Use for headless servers |
| `--io-threads` | `2` | bounded queue depth between live-mode pipeline threads |
| `--no-compile` | off | Disable `torch.compile` on the SR + GFPGAN nets. Use for short runs (compile costs 30-60 s warmup) or to escape an Inductor crash on sm_120 + torch nightly |

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

### Production command (in-cabin driver-monitoring)

Two recommended invocations, depending on which face restorer you ship.

**`realesrgan_lite` (no face stage, fastest):**

```bash
python -m src.infer --live \
  --model realesrgan_lite \
  --input "<source — file path or RTSP url>" \
  --output output/final_run \
  --scale 4 --pre-resize 80% \
  --dtype fp16 --esrgan-denoise 1.0
```

**`codeformer_compact` (Compact bg + CodeFormer face, best identity preservation):**

```bash
python -m src.infer --live \
  --model codeformer_compact \
  --input "<source>" \
  --output output/final_run \
  --scale 4 --pre-resize 80% \
  --dtype fp16 --esrgan-denoise 1.0 \
  --cf-fidelity 1.0 --eye-dist-threshold 20
```

Every other flag stays at its default. The `--live` switch alone
turns on the latency-critical pipeline; below is what it activates and
how to tune further.

### What `--live` turns on automatically

| Activated by `--live` | Effect on latency |
|-----------------------|-------------------|
| **Threaded pipeline** (decode / SR / encode on separate threads) | CPU I/O overlaps with GPU SR — total wall time drops ~30 % |
| **`--encoder auto`** (auto-picks `h264_nvenc` when available) | NVENC encode is 5-10× faster than libx264; saves ~30-50 ms/frame |
| **`--chunk-frames 4`** (was 16 default) | Time-to-first-SR-frame drops from ~600 ms to ~150 ms |
| **`--no-comparison`** (suppresses side-by-side mp4) | Cuts encode work in half |
| **Two preview windows** (`Original` + `Upscaled`) | Independent of SR throughput — `Original` plays smooth at source fps regardless of SR speed (see "decoupled" subsection below) |
| **Decode pacing to source fps** | Drops SR frames if SR can't keep up — preview stays smooth, mp4 reflects what SR completed (`dropped=N` printed at end) |

### Model-level perf opts (always on, regardless of `--live`)

| Optimization | Effect | Disable with |
|--------------|--------|--------------|
| `torch.inference_mode()` wrap | ~10-15 % vs `no_grad` | n/a |
| `cudnn.benchmark = True` | Picks fastest conv algo for fixed input shape | n/a |
| TF32 on matmul + cuDNN | ~2× matmul throughput on Ampere+ | n/a |
| `channels_last` memory format on every Real-ESRGAN net + GFPGAN net | 5-15 % via Tensor Core NHWC kernels | n/a |
| `torch.compile(mode='default')` on every Real-ESRGAN net + GFPGAN net | 20-50 % steady-state; 30-60 s warmup on first inference | `--no-compile` |
| Batched RealESRGAN forward (`realesrgan_lite`, `realesrgan_full`) | 30-80 % via one big forward instead of N small ones | falls back automatically when `--esrgan-tile > 0` |

`torch.compile` is **not** applied to the CodeFormer face net — it
regressed on RTX 5050 + torch nightly. The CodeFormer bg upsampler is
still compiled, so 30-40 % of its wall time still benefits.

### Latency tuning checklist

If preview still feels laggy after `--live`:

1. **Confirm NVENC is being used.** Look for
   `encoder=h264_nvenc` in the `[runner] live mode encoder=...` line.
   If it says `libx264`, your PyAV/ffmpeg build lacks NVENC — either
   install a build with NVENC (`pip install av --force-reinstall
   --no-binary av` won't help; use a binary wheel from the NVENC-enabled
   PyAV channel, or build ffmpeg with `--enable-nvenc`).

2. **Check `dropped=N` at end of run.** If `dropped` is most of the
   source frames, SR can't keep up and the Upscaled window will lag.
   Drop pre-resize **down to your policy floor** (this branch's
   in-cabin floor is `--pre-resize 85%`). On RTX 5050:

   | `--pre-resize` | SR fps (lite) | Upscaled window |
   |---|---:|---|
   | `90%` | ~10 fps  | refreshes ~10 fps |
   | `85%` | ~11.5 fps | refreshes ~11.5 fps |
   | `80%` | ~13 fps  | refreshes ~13 fps |

3. **Drop `--chunk-frames 1` for the lowest possible TTFB.** Tradeoff:
   slightly lower throughput because each forward processes one frame
   instead of four. Use this only when first-frame latency matters
   more than steady-state fps.

4. **Drop `--io-threads 1`.** Bounded queues at depth 1 minimize the
   end-to-end pipeline latency at the cost of jitter tolerance. Use
   for clean local file sources; raise back to `2` or `3` for jittery
   network sources (RTSP).

5. **Add `--no-preview`** if you don't need the windows. Removes the
   cv2 GUI cost and unpins encode from the main thread; the encode
   stage moves back to its own dedicated worker for full three-stage
   overlap.

6. **First run feels stuck for ~60 s.** That's `torch.compile`
   warming up. Subsequent runs reuse the cached Inductor artifacts (in
   `~/.cache/torch/inductor/`). For short ad-hoc runs where you don't
   want to pay the warmup, add `--no-compile`.

7. **Switch model to `realesrgan_lite`** if you don't actually need
   face restoration. CodeFormer adds ~600 ms / face / frame on this
   GPU; lite alone runs ~13 fps at 80% pre-resize on RTX 5050,
   CodeFormer compact runs ~1.4 fps under the same conditions.

8. **Profile if numbers don't match expectations.** Run
   `python -m src.infer --no-preview --no-compile ...` to get a clean
   baseline without the GUI / compile noise, compare to
   `python -m src.infer --live ...` to see what the optimizations
   actually buy on your hardware.

### How the preview is decoupled from SR pace

The cv2 windows are driven from the main thread on a clock ticking at
the source video's fps (25 fps for our test clips). The decode worker
paces itself to source fps and updates a thread-safe "latest original
frame" slot on every decoded frame — so the `Original` window plays
smooth at source rate regardless of how slow SR is. The `Upscaled`
window shows the most recent completed SR frame (also from a slot);
when SR is slower than source, it holds the last SR frame until the
next one lands.

To keep that decoupling honest under heavy SR load, the decode worker
**drops chunks** going into the SR queue when the queue is full
(non-blocking `put_nowait`). The dropped count is printed at the end
of every run, e.g. `dropped=106`. The mp4 contains every SR frame that
completed, so a slow SR yields a sparser mp4 — that's the realistic
"live source" tradeoff. With `--no-preview` the pipeline runs as fast
as possible with no pacing and no drops; behavior matches a normal
batch run.

Measured on this RTX 5050 with `realesrgan_lite` after compile +
channels_last + batched forward kicked in (so steady-state, not first
inference), fp16, source 25 fps:

| `--pre-resize` | SR fps | dropped (6 s) | Original window | Upscaled window |
|----------------|-------:|--------------:|-----------------|-----------------|
| `90%`          | ~10    | most          | smooth 25 fps   | refreshes ~10 fps |
| `85%`          | ~11.5  | most          | smooth 25 fps   | refreshes ~11.5 fps |
| `80%`          | ~13    | most          | smooth 25 fps   | refreshes ~13 fps |

The **Original** window stays smooth because it's driven from the
source-fps clock on the main thread (independent of SR). The
**Upscaled** window updates only when a new SR frame completes — at
in-cabin pre-resize values (≥ 85 %) on this GPU it will still lag
behind the original by ~2× until the SR backbone is faster (TensorRT
export, deferred). On A100 the gap closes.

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

Three model-level perf opts are also active by default (disable with
`--no-compile` if needed):

1. **`channels_last` memory format** on every Real-ESRGAN net and on
   the GFPGAN restoration net. Tensor Cores on Ampere+ / Blackwell are
   most efficient on NHWC fp16 conv; cuDNN picks direct-NHWC kernels.
   Output is bit-equal to NCHW — pure layout change.
2. **`torch.compile(net, mode='default')`** on every Real-ESRGAN net,
   plus the GFPGAN restoration net. TorchDynamo + Inductor fuse
   adjacent conv+activation kernels and cut Python+CUDA dispatch
   overhead. ~30-60 s warm-up on first inference; ~20-50 % steady-state
   gain on SRVGGNetCompact / RRDBNet. **Not** applied to the CodeFormer
   face net — its codebook Transformer + AdaIN path triggers graph
   breaks on torch nightly that net a slowdown (measured -15 % on RTX
   5050). The bg upsampler under CodeFormer still gets compiled, so
   that 30-40 % of the codeformer wall time still benefits.
3. **Batched RealESRGAN forward** in `realesrgan_lite` and
   `realesrgan_full`. The default `RealESRGANer.enhance()` wrapper runs
   one PyTorch forward per frame with H2D + D2H copies each time; we
   bypass that and stack N frames into a single forward (one H2D, one
   forward, one D2H). GFPGAN and CodeFormer keep their per-frame
   restorer wrappers because the face restorer is structurally
   per-image; only their bg upsamplers see batching today.

Measured on RTX 5050, fp16, `--pre-resize 80%`, 10 s clip:

| backend | `--live` baseline | + compile + channels_last + batched | speedup |
|---------|------------------:|-----------------------------------:|--------:|
| `realesrgan_lite`     | 6.3 fps  | **13.0 fps** | **+106 %** |
| `realesrgan_full`     | (RRDB)   | **1.9 fps @ pr50%**          | n/a |
| `realesrgan_gfpgan`   | (n/a)    | **1.6 fps**                  | n/a |
| `codeformer_compact`  | 1.35 fps | **1.39 fps** (bg only — face net is the bottleneck) | +3 % |

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
