# UpScaling — Handoff

This document is for whoever picks up this project next (yourself in three
months counts). It captures the **state**, **decisions**, **known unknowns**,
and **suggested next steps**.

---

## 1. Where things stand (2026-05-19, end of session)

### Done & validated end-to-end on the laptop (RTX 5050, 8 GB)

- venv + Python 3.11.15 + torch 2.12.0.dev cu128 + sageattention 1.0.6 + bitsandbytes 0.49.2.
- `realesrgan_lite` smoke test: 2 s of 1.mp4 → **23.5 FPS** e2e at 1172×960 SR.
- `flashvsr_tiny` smoke tests at 160×128 LR → 640×512 SR:
  - skip 1, interp none: **8.25 FPS**.
  - skip 2, interp repeat: **14.4 FPS**.
  - skip 2, interp rife: **12.6 FPS** (visibly smoother).
- `realesrgan_gfpgan` smoke test (the new face-aware backend): 3 s of 1.mp4 →
  **1.78 FPS** at full 704×576 → 2816×2304 SR, peak VRAM 1.3 GB. Slow on
  the laptop because both stages run at full output resolution (Compact
  upscales background 4×, GFPGAN restores 512×512 face crops, paste-back).
  On A100 this is expected to land at 30–60 FPS based on Compact-TRT
  benchmarks and the GFPGAN-1.4 published timings.
- 211-run automated benchmark sweep recorded under `benchmarks/latest/`:
  top realesrgan_lite **48.5 FPS** @ 175 MB VRAM, top flashvsr_tiny
  **38.2 FPS** @ 4.2 GB. See `benchmarks/latest/REPORT.md`.
- All four outputs (`upscaled.mp4`, `comparison.mp4`, `run_info.txt`,
  per-folder naming convention) land in
  `output/<model>/<stem>_<args>/`.

### Discovered & fixed during this session

1. **`/tmp` is tmpfs ~7 GB** on this laptop; pip build isolation blew through
   it pulling torch+CUDA. Fix: `setup_linux.sh` now sets
   `TMPDIR="$ROOT/.build_tmp"` on the 396 GB root partition.
2. **`basicsr` imports a removed torchvision API** (`functional_tensor`).
   `setup_linux.sh` sed-patches `degradations.py` post-install.
3. **`encode_video` passed `pixel_format=` to `iio.imwrite`** — newer PyAV
   plugin rejects it. Switched to `imopen` + `init_video_stream` pattern.
4. **FlashVSR's diffsynth fork hard-imports `block_sparse_attn`**. The
   kernel needs full nvcc; pip wheel `nvidia-cuda-nvcc-cu12` only ships
   ptxas, not the driver. `setup_linux.sh` patches the import to be
   optional and adds a dense-SDPA fallback in `flash_attention()`. Slower
   but correct; SageAttention SDPA patch claws back ~1.3× of the loss.
5. **Old FlashVSR wrapper was wrong.** Real upstream contract:
   - LQ tensor must be in `[-1, 1]`, not `[0, 1]`.
   - Spatial multiple is **128**, not 16.
   - LQ is bicubic-upsampled to the SR target dims *before* the pipeline
     call (the pipeline refines, doesn't upscale).
   - Frame count must be `8n+1`; pad with last frame ×4 then crop output.
   - `topk_ratio = sparse_ratio * 768 * 1280 / (tH * tW)`.
   - Init sequence is `mm.load_models([diffusion.safetensors])`,
     `FlashVSRTinyPipeline.from_model_manager`, manual
     `pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(...)`,
     manual `pipe.TCDecoder = build_tcdecoder(...)`, then
     `enable_vram_management`, `init_cross_kv(context_tensor=...)`,
     `load_models_to_device(["dit","vae"])`. (No VAE checkpoint loaded
     for the Tiny variant — `mm.fetch_model("wan_video_vae")` returns
     None and TCDecoder takes over.)
6. **`transformers==4.46.2` pin.** Newer `transformers` (5.x) moved
   `PretrainedConfig`, breaking FlashVSR's `stepvideo_text_encoder` import.
7. **RIFE-HDv3 source was missing.** Only v6 `flownet.pkl` was checked in;
   the Python source comes from the ECCV2022-RIFE repo. `setup_linux.sh`
   clones it and copies `model/{IFNet,RIFE,refine,warplayer,IFNet_m,laplacian}.py`
   into `RIFE_trained_v6/model/`, with a `loss.py` stub so `Model()` can
   construct without the training stack.

### Still open / known unknowns

- **`realesrgan_gfpgan` on RTX 5050 is slow** (1.78 FPS at native input).
  Two bottlenecks: GFPGAN-1.4's RetinaFace-ResNet50 detector (~25 ms/frame)
  and Compact at full 4× output (~150 ms on RTX 5050). On A100, swap
  Compact for the TensorRT INT8 export (~8 ms) and the detector for
  SCRFD-2.5G via InsightFace (~5 ms with five landmarks bundled) — the
  rest of the pipeline is unchanged. Expected ~30 FPS single-stream and
  ~50 FPS batched.
- **SCRFD detector swap** for `realesrgan_gfpgan` is a TODO. The wrapper
  takes a `face_detector` extra (defaults to RetinaFace ResNet50, falls
  back to anything `facexlib.init_detection_model` accepts). Hooking up
  `insightface.app.FaceAnalysis(name="buffalo_l")` would give us SCRFD +
  ArcFace ID + age/gender from a single detector, but that's an A100
  branch — laptop wins nothing.
- **Compact + RIFE at 2816×2304 OOMs the laptop.** The
  `realesrgan_gfpgan` smoke run at `--frame-skip 2 --frame-interp rife`
  was killed (exit 137). For the laptop, run with `--pre-resize 50%` or
  smaller when stacking RIFE on top of the face pipeline.
- **`nf4` quantization is a stub.** Only `int8_woq` (bitsandbytes Linear8bitLt)
  is wired up. `--quant nf4` is silently ignored.
- **Block-Sparse-Attention not built.** Would need `apt install nvidia-cuda-toolkit`
  (or the NVIDIA run-file) for nvcc. On A100 this is trivial; on the laptop
  it's not. SageAttention covers most of the perf gap.
- **VAE not loaded** for FlashVSR Tiny. The upstream Tiny script also omits
  it; TCDecoder handles the decode. If you ever call `pipe.encode_video()` /
  `pipe.decode_video()` directly, load Wan2.1_VAE.pth into the ModelManager
  too.
- **A100 path not yet exercised.** The wrapper hard-codes 128-multiple and
  the 8n+1 padding; both are FlashVSR-Tiny constraints and should carry
  over. Expect a big jump (~17 FPS at 768×1408 per the upstream README).

---

## 2. Architecture in one paragraph

`infer.py` parses args, optionally patches SDPA (`sage_patch.enable()`),
builds an `Upscaler` from the registry, then iterates videos. For each video,
`runner.process_video()` reads source frames (full-res, kept for side-by-side),
optionally resizes them to a low-res input (`io_utils.resolve_preprocess`),
selects anchor frames every `--frame-skip` steps, calls `upscaler.upscale()`
on the anchors, optionally downscales 4× SR to 2× if requested, fills the
skipped frames via `repeat` or RIFE, and writes `upscaled.mp4`,
`comparison.mp4`, and `run_info.txt` to a deterministically-named subfolder.

All model-specific code lives behind the `BaseUpscaler` interface
(`load() / upscale(frames) / close()`) so swapping in a third backend is a
single file + one import in `src/models/__init__.py`.

## 3. Why these design choices

- **Per-video output folder name is the args** — easy to glance at output
  tree and tell what produced each file. Matches the prior benchmark layout
  on disk so prior comparisons stay valid.
- **Side-by-side uses the original, not the LR** — the user explicitly asked
  for that. The SR is bicubic-resampled up to source height for the join, not
  down.
- **Model registry, not configs** — only two models so far. Configs would be
  premature abstraction; a registry is one line per model.
- **PyAV via imageio rather than shell ffmpeg** — avoids per-video subprocess
  overhead and gives us a consistent encoder across machines.
- **`bitsandbytes` Linear8bitLt for int8 WOQ** — works on the FlashVSR DiT
  transformer linears; safer than NF4 for activations that are already
  bfloat16 in the cached KVs.

## 4. Known unknowns / things to verify

1. **DiffSynth FlashVSR Tiny pipeline kwargs.** The v1.1 inference script wires
   `topk_ratio`, `kv_ratio`, `local_range`, `color_fix`. If upstream renames
   these between releases, fix `src/models/flashvsr_tiny.py`. The wrapper
   tries to be robust to the `Causal_LQ4x_Proj` / `TCDecoder` aux blocks
   moving inside the pipeline constructor in newer revisions.
2. **Blackwell + Block-Sparse-Attention.** Upstream BSA kernels are written
   for sm_80/86/89 mostly. If `pip install -e .` builds but inference crashes
   inside the sparse kernel, monkey-patch FlashVSR to use dense attention
   (commenting the sparse path inside `diffsynth/.../streaming.py`).
3. **RIFE v6 import path.** We append `RIFE_trained_v6/train_log` to `sys.path`
   to grab `RIFE_HDv3.py`. Some RIFE distributions ship the Python files under
   a different folder; if the import fails, set `--rife-weights` to the
   directory that *contains* `train_log/`.
4. **Real-ESRGAN + Python 3.11.** `basicsr` historically pinned
   `torchvision.transforms.functional_tensor` which was removed in
   torchvision 0.17+. If `import realesrgan` errors on
   `functional_tensor`, monkey-patch:

    ```python
    import torchvision.transforms.functional as F
    import sys, types
    m = types.ModuleType("torchvision.transforms.functional_tensor")
    m.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = m
    ```

   Add this to the top of `src/models/realesrgan_lite.py` if needed.

## 5. Next steps (suggested order)

1. **Get the env up.** `./setup_linux.sh`. Smoke test:
   `python -m src.infer --model realesrgan_lite --input "Real Test Video/1.mp4" --seconds 2`.
2. **First FlashVSR run.** `--model flashvsr_tiny --pre-resize qvga --frame-skip 2 --frame-interp rife --sage-attn --quant int8_woq`.
3. **Benchmark sweep.** Loop over `--pre-resize {qvga,vga,none}` × `--frame-skip {1,2,4}` × `--quant {none,int8_woq}`. Use `run_info.txt`'s `e2e_fps`.
4. **Real-time deployment.** Replace `runner.process_video()` file reader with a streaming source (RTSP / V4L2) and `VideoWriter` with the streaming sink (`io_utils.VideoWriter` is already chunk-friendly). FlashVSR's pipeline is already streaming-friendly internally.
5. **Quality eval.** Add LPIPS / DISTS / VMAF on a held-out subset of `Real Test Video/`. (`pyiqa` has both.)
6. **Move to A100.** Same code; bump `--chunk-frames` to 64, drop `--frame-skip`/`--frame-interp`, leave `--sage-attn` on, skip int8 quant.

## 6. Where to look first when something breaks

| Symptom | Likely file | Likely cause |
|---|---|---|
| Wrong output resolution | `src/runner.py:process_video` | `--scale` mismatch with `native_scale` |
| Import error on `diffsynth` | `src/models/flashvsr_tiny.py:_ensure_repo_on_path` | FlashVSR repo not cloned / path wrong |
| `block_sparse_attn` crash | `third_party/FlashVSR/diffsynth/.../attention.py` | sm_120 incompatibility |
| `sageattention` crash | `src/sage_patch.py:enable` | fallback wraps it; if still crashing, revert SDPA |
| RIFE import error | `src/rife_interp.py:_ensure_rife_on_path` | wrong `--rife-weights` path |
| OOM on FlashVSR | `src/models/flashvsr_tiny.py` | lower `--chunk-frames` or `--pre-resize` |

## 7. Contact / repo

GitHub: <https://github.com/Asem-Hatamleh/Upscaling>
Owner: Asem Hatamleh (Acacus Group) — `a.hattamleh@acacusgroup.com`
