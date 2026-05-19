# UpScaling — Handoff

This document is for whoever picks up this project next (yourself in three
months counts). It captures the **state**, **decisions**, **known unknowns**,
and **suggested next steps**.

---

## 1. Where things stand (2026-05-19)

### Done

- Project scaffold (`src/`, `scripts/`, `configs/`, `third_party/` placeholder).
- CLI (`src/infer.py`) supports both single-video and folder input.
- Model registry pattern in `src/models/base.py`; models register themselves on
  import.
- Two backends wired:
  - **`flashvsr_tiny`** — official FlashVSR-v1.1 Tiny via `FlashVSRTinyPipeline`.
  - **`realesrgan_lite`** — `realesr-general-x4v3` SRVGGNetCompact via RealESRGAN.
- Performance modules: `sage_patch.py` (SageAttention SDPA monkey-patch),
  `rife_interp.py` (RIFE-HDv3 wrapper using checked-in v6 weights),
  `frame_skip.py` (gap fill: `repeat` / `rife`).
- I/O via PyAV through imageio (no ffmpeg shell-out per video).
- Side-by-side composition keeps the **full-resolution** source on the left.
- `run_info.txt` writer matches the original benchmark format.
- `setup_linux.sh` chooses PyTorch wheel based on detected GPU (Blackwell -> cu128
  nightly; A100 -> cu121 stable), then builds Block-Sparse-Attention and
  SageAttention from source.
- `scripts/download_weights.py` clones FlashVSR repo + pulls v1.1 safetensors +
  Real-ESRGAN weights.

### Not yet done / not yet validated

- Nothing has been run end-to-end on the laptop. PyTorch is not installed yet —
  `setup_linux.sh` was authored but not executed. Run it first.
- `flashvsr_tiny.py` assumes the upstream `diffsynth.FlashVSRTinyPipeline` API
  matches the v1.1 inference script. If upstream renames a kwarg
  (`topk_ratio`, `kv_ratio`, `local_range`, `color_fix`), patch it in the
  `pipe(...)` call inside `FlashVSRTiny.upscale`.
- `nf4` quantization branch is *not implemented* — only `int8_woq` is. Treat
  `--quant nf4` as a known TODO; current code will simply not quantize.
- Block-Sparse-Attention build can fail on Blackwell because some kernels need
  patches for sm_120. If the build fails, `--sage-attn` is the next-best lever.

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
