# UpScaling — Real-Time Video Super-Resolution

Real-time video upscaling pipeline targeted at in-cabin driver-monitoring streams
(704×576 @ ~30 FPS). Two interchangeable backends are provided:

| Model | Native scale | Strength | When to use |
|-------|--------------|----------|-------------|
| **`flashvsr_tiny`** (FlashVSR-v1.1 Tiny) | 4× | Best quality, temporal-aware | Driver of the project |
| **`realesrgan_lite`** (`realesr-general-x4v3`) | 4× | Lightweight, ~10 MB, faster on small GPUs | Fallback / baseline / sanity check |

Both go through the same pipeline (frame skipping, RIFE interpolation,
SageAttention SDPA, optional int8 quantization), the same CLI, and the same
output structure.

---

## 1. Hardware & OS

| | Dev (this laptop) | Prod |
|---|---|---|
| GPU | RTX 5050 (sm_120 Blackwell) 8 GB | NVIDIA A100 80 GB |
| CUDA | 13.x | 12.x |
| Driver | 595.x | latest stable |
| OS | Linux (Ubuntu 24.04 tested) | Linux |
| Python | 3.11.x (recommended) | 3.11.x |

> Why 3.11: the FlashVSR upstream targets 3.11.13, and several wheels we rely on
> (bitsandbytes, sageattention) do not yet ship for 3.13/3.14.

## 2. Quick install

```bash
git clone https://github.com/Asem-Hatamleh/Upscaling.git
cd Upscaling
./setup_linux.sh                  # creates ./venv, installs all deps + clones FlashVSR + downloads weights
source venv/bin/activate
```

`setup_linux.sh` flags:

- `--no-downloads` — skip cloning FlashVSR and downloading weights
- `--no-sage` — skip SageAttention build
- `--no-bsa` — skip Block-Sparse-Attention build (FlashVSR will fall back to dense attention; slower)
- `--python=python3.11` — pick a specific interpreter

If you prefer manual setup:

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -U pip wheel
# PyTorch (pick one)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128   # Blackwell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121                  # A100
pip install -r requirements.txt
python scripts/download_weights.py --model all
```

## 3. Running inference

CLI entrypoint: `python -m src.infer`.

### Single video

```bash
python -m src.infer \
  --model flashvsr_tiny \
  --input "Real Test Video/1.mp4" \
  --seconds 5 \
  --scale 4 \
  --pre-resize vga \
  --frame-skip 2 \
  --frame-interp rife \
  --sage-attn \
  --quant int8_woq
```

### Folder of videos

```bash
python -m src.infer --model flashvsr_tiny --input "Real Test Video" \
    --seconds 3 --pre-resize qvga --frame-skip 2 --frame-interp rife
```

### Real-ESRGAN alt

```bash
python -m src.infer --model realesrgan_lite --input "Real Test Video/1.mp4" \
    --esrgan-variant realesr-general-x4v3 --esrgan-denoise 0.5 \
    --frame-skip 2 --frame-interp rife
```

## 4. All CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--model` | *required* | `flashvsr_tiny` or `realesrgan_lite` |
| `--input` / `-i` | *required* | single video OR directory |
| `--output` / `-o` | `output` | output root |
| `--seconds` | `0` | seconds to process; `0` = full video |
| `--scale` | `4` | `2` or `4` (model is 4× native; `2` post-downscales) |
| `--pre-resize` | `none` | `none`/`vga`/`qvga`/`WxH`/`pct:NN` |
| `--frame-skip` | `1` | run SR every Nth frame |
| `--frame-interp` | `none` | `none`/`repeat`/`rife` — gap fill for dropped frames |
| `--rife-weights` | `RIFE_trained_v6` | folder containing `train_log/flownet.pkl` |
| `--sage-attn` | off | enable SageAttention SDPA monkeypatch |
| `--quant` | `none` | `none`/`int8_woq`/`nf4` (FlashVSR) |
| `--dtype` | `bf16` | `bf16`/`fp16`/`fp32` |
| `--device` | `cuda` |  |
| `--chunk-frames` | `16` | FlashVSR: frames per pipeline call (lower = less VRAM) |
| `--topk-ratio` | `2.0` | FlashVSR block-sparse top-k |
| `--kv-ratio` | `3.0` | FlashVSR KV reuse ratio |
| `--local-range` | `11` | FlashVSR local-window radius |
| `--no-color-fix` | off | disable FlashVSR color correction |
| `--esrgan-variant` | `realesr-general-x4v3` | `realesr-animevideov3` for stylized inputs |
| `--esrgan-denoise` | `0.5` | 0..1 noise/sharpness balance |
| `--esrgan-tile` | `0` | tile size for VRAM-tight runs |
| `--no-comparison` | off | skip side-by-side output |
| `--no-upscaled` | off | skip upscaled-only output |
| `--crf` | `18` | x264 quality (lower = larger/better) |

## 5. Output layout

```
output/
└── flashvsr_tiny/
    └── 1_scale4x_352x288_skip2_interp-rife_q-int8_woq_sage-on/
        ├── upscaled.mp4
        ├── comparison.mp4     # full-res original | upscaled, side by side
        └── run_info.txt
```

`run_info.txt` carries `[run]`, `[args]`, `[timing]` sections — model, quant,
SageAttention status, input args, frame counts, wall time, end-to-end FPS.

## 6. Real-time strategy

Real-time on 704×576 @ 30 FPS through FlashVSR Tiny on an 8 GB Blackwell GPU
requires combining several optimizations:

1. **Pre-resize input** to QVGA (320×240) or VGA (640×480) — the side-by-side
   comparison still uses the original full-resolution frame.
2. **`--frame-skip 2`** + **`--frame-interp rife`** — SR runs every other
   frame; RIFE-HDv3 fills the gap. ~1.6× throughput at minor quality cost.
3. **`--sage-attn`** — Sage-Attention beats Flash-Attention 2 on long
   sequences; expect 1.2–1.5× on FlashVSR transformer.
4. **`--quant int8_woq`** — bitsandbytes weight-only int8 cuts the FlashVSR DiT
   memory ~50% and gives ~1.15× on Ampere/Ada; smaller gain on Blackwell.
5. **`--chunk-frames`** smaller (e.g. 8) trades a bit of throughput for VRAM.

On A100 80 GB you can drop most of the tricks: `--frame-skip 1`,
`--frame-interp none`, `--quant none`, and rely on `--sage-attn` alone.

## 7. Real test videos

```
Real Test Video/
├── 1.mp4   # 704x576 25 fps 32s   — primary test
├── 2.mp4
├── 4.mp4
└── 5.mp4
```

## 8. Project layout

```
UpScaling/
├── README.md
├── HANDOFF.md
├── requirements.txt
├── setup_linux.sh
├── .gitignore
├── src/
│   ├── infer.py              # CLI entrypoint
│   ├── runner.py             # per-video orchestration
│   ├── io_utils.py           # video I/O + side-by-side
│   ├── report.py             # run_info.txt writer
│   ├── frame_skip.py         # skip + gap-fill
│   ├── rife_interp.py        # RIFE-HDv3 wrapper
│   ├── sage_patch.py         # SageAttention SDPA patch
│   └── models/
│       ├── base.py           # Upscaler interface + registry
│       ├── flashvsr_tiny.py  # FlashVSR-v1.1 Tiny wrapper
│       └── realesrgan_lite.py# Real-ESRGAN alt wrapper
├── scripts/download_weights.py
├── third_party/FlashVSR/     # cloned by setup_linux.sh (gitignored)
├── weights/realesrgan/       # downloaded by setup_linux.sh (gitignored)
├── RIFE_trained_v6/          # checked-in RIFE-HDv3 weights
└── Real Test Video/          # checked-in test clips
```

## 9. Troubleshooting

- **`CUDA error: no kernel image is available for execution`** — wrong torch
  for your GPU. Re-run `setup_linux.sh` so it picks the cu128 nightly for
  Blackwell.
- **`sageattention not available`** — `--sage-attn` is best-effort; the run
  continues with torch SDPA.
- **`block_sparse_attn` missing** — FlashVSR falls back to dense attention;
  ~1.5–2× slower but still correct.
- **OOM** — lower `--chunk-frames`, set `--pre-resize qvga`, use `--quant int8_woq`.
- **Choppy output with high `--frame-skip`** — switch `--frame-interp` from
  `repeat` to `rife`.

## 10. License

Source under MIT. FlashVSR weights are under the upstream license (see
`third_party/FlashVSR/LICENSE`). Real-ESRGAN weights under BSD-3.
