# UpScaling — Real-Time Video Super-Resolution

Real-time video upscaling pipeline targeted at in-cabin driver-monitoring streams
(704×576 @ ~30 FPS). Five interchangeable backends are provided so you can pick
the best speed / quality / VRAM trade-off for the deployment GPU:

| Model | Native scale | Strength | When to use |
|-------|--------------|----------|-------------|
| **`realesrgan_gfpgan`** (Compact bg + GFPGAN-1.4 faces) | 4× | Best perceptual quality for faces — driver eyes/mouth/head-pose come out crisp; background handled by Compact | **Recommended for driver monitoring on A100** |
| **`codeformer_compact`** (Compact bg + CodeFormer-v1 faces) | 4× | Same shape as the GFPGAN backend with a learned-codebook face prior — better on profile / occluded / low-light faces, less "plastic" skin | A/B against `realesrgan_gfpgan` to pick the face restorer |
| **`basicvsrpp`** (BasicVSR++) | 4× | True temporal SR — bidirectional flow propagation eliminates per-frame flicker without RIFE | Use when temporal coherence on cabin reflections / hair / motion matters more than face detail |
| **`flashvsr_tiny`** (FlashVSR-v1.1 Tiny) | 4× | Diffusion-class, temporal-aware | When you need diffusion-quality detail and can afford 4 GB+ VRAM |
| **`realesrgan_lite`** (`realesr-general-x4v3`) | 4× | Lightest (~10 MB), fastest per-frame | Throughput baseline / sanity check / multi-stream batching |

All five share the same CLI, frame-skip + RIFE pipeline, SageAttention SDPA
patch, output layout, and `run_info.txt` schema.

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

### Real-ESRGAN + GFPGAN face restoration (recommended for driver monitoring)

```bash
# Native input (best quality, slow on 8 GB dev GPU)
python -m src.infer --model realesrgan_gfpgan --input "Real Test Video/1.mp4" \
    --seconds 3 --dtype fp16

# Pre-resize + frame-skip for the laptop:
python -m src.infer --model realesrgan_gfpgan --input "Real Test Video/1.mp4" \
    --seconds 3 --pre-resize 50% --frame-skip 2 --frame-interp rife --dtype fp16
```

GFPGAN-1.4 weights (333 MB) and facexlib's RetinaFace-ResNet50 detector
(104 MB) auto-download to `weights/gfpgan/` and `weights/facexlib/` on
first use; pre-fetch with `python scripts/download_weights.py --model gfpgan`.

### CodeFormer + Compact (alternative face restorer)

Same pipeline shape as `realesrgan_gfpgan`, with CodeFormer-v1's
codebook-prior face restorer instead of GFPGAN-1.4. Useful when you want a
direct A/B test against GFPGAN on cabin faces.

```bash
python -m src.infer --model codeformer_compact --input "Real Test Video/1.mp4" \
    --seconds 3 --pre-resize 35% --frame-skip 2 --frame-interp rife --dtype fp16
```

Weights (~360 MB) auto-download to `weights/codeformer/` on first use, or
pre-fetch with `python scripts/download_weights.py --model codeformer`.

### BasicVSR++ (true temporal SR)

```bash
python -m src.infer --model basicvsrpp --input "Real Test Video/1.mp4" \
    --seconds 3 --pre-resize 35% --frame-skip 1 --dtype fp32
```

Sliding-window bidirectional model — eliminates flicker that per-frame
Compact has on cabin reflections / hair / clothing motion. Note: `--dtype
fp16` is silently kept at fp32 inside the wrapper because internal flow
ops don't autocast cleanly. Weights (~28 MB, mmediting REDS4 release)
auto-download to `weights/basicvsrpp/`, pre-fetch with
`python scripts/download_weights.py --model basicvsrpp`.

## 4. All CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--model` | *required* | `flashvsr_tiny`, `realesrgan_lite`, `realesrgan_gfpgan`, `codeformer_compact`, or `basicvsrpp` |
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
│   ├── infer.py                  # CLI entrypoint + interactive wizard
│   ├── runner.py                 # per-video orchestration
│   ├── io_utils.py               # video I/O + side-by-side
│   ├── report.py                 # run_info.txt writer
│   ├── frame_skip.py             # skip + gap-fill
│   ├── rife_interp.py            # RIFE-HDv3 wrapper
│   ├── sage_patch.py             # SageAttention SDPA patch
│   └── models/
│       ├── base.py               # Upscaler interface + registry
│       ├── flashvsr_tiny.py      # FlashVSR-v1.1 Tiny wrapper
│       ├── realesrgan_lite.py    # Real-ESRGAN baseline
│       └── realesrgan_gfpgan.py  # Real-ESRGAN Compact + GFPGAN-1.4 face restore
├── scripts/
│   ├── download_weights.py
│   ├── benchmark.py              # parameter sweep harness
│   └── analyze_benchmark.py      # CSV → Markdown report
├── benchmarks/latest/            # last sweep's CSV / JSON / REPORT.md
├── third_party/FlashVSR/         # cloned by setup_linux.sh (gitignored)
├── weights/realesrgan/           # downloaded by setup_linux.sh (gitignored)
├── weights/gfpgan/               # GFPGANv1.4.pth (gitignored)
├── weights/facexlib/             # RetinaFace ResNet50 + ParseNet (gitignored)
├── RIFE_trained_v6/              # checked-in RIFE-HDv3 weights + ECCV source
└── Real Test Video/              # checked-in test clips
```

## 9. Validated smoke-test configs (RTX 5050, 8 GB)

| Model | Pre-resize | Skip | Interp | SageAttn | Quant | e2e FPS | SR dims |
|-------|-----------|------|--------|----------|-------|--------:|---------|
| `realesrgan_lite` | `qvga` (293×240 fit) | 1 | none | – | none | **23.5** | 1172×960 |
| `flashvsr_tiny` | `160x128` | 1 | none | on | none | 8.25 | 640×512 |
| `flashvsr_tiny` | `160x128` | 2 | repeat | on | none | **14.4** | 640×512 |
| `flashvsr_tiny` | `160x128` | 2 | rife | on | none | 12.6 | 640×512 |
| `realesrgan_gfpgan` | `none` (704×576) | 1 | none | – | none | 1.78 | **2816×2304** |
| `codeformer_compact` | `35%` (246×202) | 1 | none | – | none | **3.63** | 984×808 |
| `basicvsrpp` | `35%` (246×202) | 1 | none | – | none | **8.01** | 984×808 |

Best per-knob results from the 211-run benchmark (`benchmarks/latest/`):

- **`realesrgan_lite` @ scale 2, pr 35 %, skip 4, fp16** — 48.5 fps, 175 MB
- **`flashvsr_tiny` @ scale 2, pr 15 %, skip 3, bf16, sage off** — 38.2 fps, 4.2 GB

A100 projections: each backend ~10-20× the laptop FPS at native resolution,
putting `realesrgan_gfpgan` comfortably above 30 fps for in-cabin
704×576 → 2816×2304 with the driver's face restored by GFPGAN.

Notes:
- FlashVSR Tiny is fixed 4× and the spatial input must be a multiple of 128
  *after* the 4× upscale, so the smallest practical LR on this GPU is
  ~160×128 → 640×512 SR. Larger inputs OOM the 8 GB VRAM under dense
  attention. Production A100 will lift this dramatically.
- FlashVSR Tiny processes frames in groups of `8n+1`, so a 50-frame clip
  yields 49 input frames → 45 produced SR frames; we crop to the original
  count.
- Block-Sparse-Attention is *not* installed (kernel needs full CUDA toolkit
  + nvcc; pip wheel only ships ptxas). Our patched `wan_video_dit.py`
  falls through to dense SDPA — slower but correct.

## 10. Troubleshooting

- **`CUDA error: no kernel image is available for execution`** — wrong torch
  for your GPU. Re-run `setup_linux.sh` so it picks the cu128 nightly for
  Blackwell.
- **`Disk quota exceeded` during pip install** — `/tmp` is tmpfs (~7 GB on
  this machine). `setup_linux.sh` exports `TMPDIR=$ROOT/.build_tmp` to side-step it.
- **`sageattention not available`** — `--sage-attn` is best-effort; the run
  continues with torch SDPA.
- **`block_sparse_attn` missing** — FlashVSR falls back to dense attention;
  ~1.5–2× slower but still correct.
- **OOM** — lower `--pre-resize` (try `160x128` or smaller), set
  `--quant int8_woq`, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Choppy output with high `--frame-skip`** — switch `--frame-interp` from
  `repeat` to `rife`.
- **`basicsr` import fails on `torchvision.transforms.functional_tensor`** —
  `setup_linux.sh` sed-patches the upstream import; if you bypassed setup,
  run that patch by hand.
- **`PretrainedConfig` import error from FlashVSR's diffsynth** — pin
  `transformers==4.46.2` (handled in `requirements.txt`).
- **`No module named 'RIFE_HDv3'`** — RIFE v6 weights need the ECCV2022-RIFE
  Python source. `setup_linux.sh` clones it and installs files into
  `RIFE_trained_v6/model/`.

## 10. License

Source under MIT. FlashVSR weights are under the upstream license (see
`third_party/FlashVSR/LICENSE`). Real-ESRGAN weights under BSD-3.
