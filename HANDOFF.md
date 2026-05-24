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
- `codeformer_compact` smoke (alt face restorer): 2 s of 1.mp4 @
  pre-resize 35 % → **3.63 FPS**, 1.15 GB VRAM. CodeFormer is a 94 M-param
  model (vs GFPGAN's 76 M) so per-face latency is ~30 % higher, but the
  codebook prior tends to produce more natural skin and tolerates profile
  / occluded faces better.
- `basicvsrpp` smoke (true temporal SR): 2 s of 1.mp4 @ pre-resize 35 % →
  **8.01 FPS**, 1.27 GB VRAM. Bidirectional flow propagation gives
  frame-to-frame coherence that the per-frame Compact path lacks; no
  RIFE post-EMA needed. Model is fp32-only on this build (internal flow
  ops mismatch fp16 autocast).
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

- **`basicvsrpp` weights are mmediting-format and remapped on load** —
  upstream BasicVSR++ release isn't on GitHub; we fetch from openmmlab's
  CDN and strip the `generator.` prefix + unwrap mmediting's spynet
  ConvModule indices to basicsr's Sequential layout. If openmmlab moves
  the CDN, swap the URL in `scripts/download_weights.py`.
- **`codeformer_compact` vendors two arch files** under
  ``src/models/_codeformer/`` — copying from upstream CodeFormer was
  necessary because their bundled `basicsr` conflicts with pip's
  `basicsr` (which `realesrgan` and `gfpgan` rely on). The vendored
  files import their own `vqgan_arch` relatively and use pip basicsr's
  `ARCH_REGISTRY` / `get_root_logger`. Re-vendor if upstream CodeFormer
  ever ships a meaningful architecture change.
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

---

## 8. FullPipeline branch — live-streaming SR pipeline (2026-05-24)

This section covers work done on the `FullPipeline` branch on top of
`realesrgan-lite`. Goal: pull a live WebSocket FLV stream from an Iotistic
MNVR IP camera, run Real-ESRGAN Lite (SR-only mode), push the processed
stream out via mediamtx for RTMP / HLS / WebRTC / RTSP playback.

### 8.1 Files

| File | Purpose |
|---|---|
| `pipeline_live.py` | Main entrypoint. Threaded capture → upscale → ffmpeg push pipeline. (face-blur removed) |
| `LiveFeeder.py` | `NodePlayer`: auth to Iotistic API, opens WS/HLS stream, decodes via cv2 to `frame_queue`. |
| `bin/mediamtx*` | mediamtx binary (auto-downloaded on first run). |
| `src/models/realesrgan_lite.py` | SR model wrapper (channels_last + torch.compile applied at load). |
| `src/models/_perf.py` | `apply_perf_opts`, `batched_realesrganer_enhance` (offline path). |

### 8.2 Architecture

```
[NodePlayer ws://...flv -> ffmpeg pipe decode -> frame_queue (size 8)]
    -> [capture_thread: drop-oldest -> capture_q (16)]
        -> [upscale_worker: batched SR -> out_q (16)]
            -> [main: imshow + ffmpeg pipe + RTMP push]
                    -> [mediamtx 1935/8888/8889/8554]
```

### 8.3 Known issues & decisions

1. **NodePlayer decode is fragile**. `_display_loop` runs `cv2.VideoCapture`
   on a growing FLV temp file written by the WS receiver. After ~2 s of
   stalled reads we release + reopen the cap; cv2 then restarts at byte 0
   of the file. We **seek** to `frames_emitted - 1` after reopen and also
   skip frames whose reported `CAP_PROP_POS_FRAMES` is ≤ `frame_count` to
   absorb replays. Still occasional micro-replay possible on FLV. A real
   fix is to bypass NodePlayer's temp-file design and pipe WS bytes
   directly into an ffmpeg subprocess (`-i pipe:0 -f rawvideo`). Not done.
2. **NVENC unavailable on RTX 5050 Laptop**. The GPU lists `h264_nvenc`
   via `ffmpeg -encoders` but `OpenEncodeSessionEx` returns `unsupported
   device (2)`. `_ffmpeg_has_nvenc()` now does a **real** 64×64 test encode
   and caches the result; falls back to `libx264`.
3. **`--upscale-scale` was cosmetic**. Model arch (`SRVGGNetCompact upscale=4`)
   is hardcoded 4×. Now restricted to `{2, 4}`: `2` auto-halves
   `--upscale-pre-resize` for effective 2× net scale.
4. **Adaptive batch timeout**. `--upscale-batch-timeout 0` (default) →
   worker tracks inter-arrival EMA, computes
   `timeout = batch_size * ema_dt * 1.2`, clamped `[5 ms, 200 ms]`.
   Pass positive ms to force fixed.
5. **Big batches hurt live**. batch=16 with 25 fps source = 640 ms to fill
   → e2e ≥ 1 s before SR even starts. Default is `--upscale-batch 4`.
   Per-frame throughput is the same; latency much better.
6. **Push downscale removed by default**. `--push-max-w 0` = no downscale
   before ffmpeg. Earlier default 1920 was hiding SR detail from browser.
7. **libx264 quality bumped**. `ultrafast` → `veryfast`, bitrate-cap-2M
   → CRF 20. Encode CPU +~5–10 ms/frame; quality much better at the
   same wire bitrate.
8. **Qt cross-thread crash fixed**. NodePlayer's `_display_loop` no longer
   calls `cv2.imshow` / `cv2.waitKey`. `pipeline_live.main` overrides
   `cv2.waitKey` so any leftover background call returns `-1` instead of
   touching a Qt timer (which segfaults).
9. **mediamtx stale-instance handling**. `start_mediamtx` probes `:1935`
   first; if already serving, reuses. If new launch fails with
   `address already in use`, kills stale pids via pgrep + retries once.
10. **_perf.batched_realesrganer_enhance had wrong channel order** for
    the offline path. RealESRGAN's `enhance()` does
    `cvtColor(BGR2RGB)` before the model, so the net is trained on RGB.
    The old helper flipped twice (RGB→BGR before, BGR→RGB after); now
    feeds RGB directly. Offline outputs may shift slightly from prior
    `realesrgan-lite` branch runs — visually similar but not bit-exact.

### 8.4 Debugging

Two flags added:

- `--debug-log /tmp/sr_timeline.csv` — per-frame CSV (seq, all stage
  timestamps, latencies, n_faces, ooo_flag). Append-only, line-buffered.
- `--debug-decode` — NodePlayer logs `[decode] emit#N pos=N temp_file=...
  qsize=...` every 30 emits.

Useful analysis snippets in §8.5.

### 8.5 Triage runbook for the live pipeline

| Symptom | Check | Likely cause |
|---|---|---|
| "stream not found" in browser/VLC | watch `[ffmpeg]` stderr at start | NVENC failed; libx264 fallback didn't kick in |
| `pipe broken` early | same | ffmpeg died — read `[ffmpeg] [...]` lines |
| `cap is none, waiting` forever | initial `Packet mismatch` count | stream too lossy, raise `initial_buffer_bytes` in `_display_loop` |
| "going backward" feel | `awk -F, '$15==1' /tmp/sr_timeline.csv \| wc -l` | OOO inside pipeline — should be 0; if > 0, NodePlayer cap-reopen issue |
| e2e > 3 s | `awk -F, 'NR>1 {sum+=$13;n++} END {print sum/n}'` | batch too big; lower `--upscale-batch` |
| seq gaps > 50 | `awk -F, 'NR<2{next} NR==2{p=$1;next} $1-p>1{print p"->"$1}'` | drop-oldest discarding source faster than SR; expected on RTX 5050 Laptop @ 4× SR |
| Qt segfault | run with `--no-display`, view via RTSP / WebRTC | Qt thread bug, missing `_waitkey_main_only` patch |

### 8.6 Recommended live config (RTX 5050 Laptop)

```
python pipeline_live.py \
  --decode-fps 8 --queue-size 8 \
  --upscale-batch 2 --upscale-batch-timeout 0 \
  --upscale-pre-resize 0.8 --upscale-scale 4 \
  --upscale-denoise 0.3 \
  --push-max-w 4096 --x264-preset veryfast --push-crf 20 \
  --display --display-scale 0.5
```

This is the configuration that converged in the 2026-05-24 debugging
session. Trade-offs:
- ~7 fps effective; e2e ~1.5 s. Not real-time but acceptable for
  monitoring on this GPU.
- Source feeds ~8 fps from sub stream — pipeline keeps up.
- batch=2 keeps torch.compile shape cache hot.

### 8.7 Next steps for this branch

1. **Replace NodePlayer's temp-file decode with a piped ffmpeg subprocess**
   (most robust fix to lingering replay micro-jitter; the cv2 / growing-FLV
   design is fundamentally racy). Sketch: WS bytes → `ffmpeg.stdin`; read
   raw `bgr24` frames from `ffmpeg.stdout` based on width/height parsed
   from stderr.
2. **Add `--upscale-stride N`** to drop every other (or every Nth)
   source frame deliberately rather than relying on drop-oldest. Smoother
   output at high source rates.
3. **CRF 18 + medium preset on the production A100**. Latency budget
   there easily handles it.
4. **Move authentication out of CLI defaults**. `m.alawneh` / pwd
   currently hardcoded in argparse. Pull from env vars.
5. **Optional: PyAV-based decoder + encoder** to avoid the
   ffmpeg-subprocess round trip entirely.

### 8.8 Latest measured run (2026-05-24 12:07, RTX 5050 Laptop, sub stream)

`python pipeline_live.py --debug-log /tmp/sr_timeline.csv --debug-decode`

```
frames processed : 641
out-of-order     : 0
mean qwait_ms    : 307
mean upscale_ms  : 2794   (batch=16 default at the time -> very high)
mean e2e_ms      : 3295
max  e2e_ms      : 6385
source seq skipped: 1323  (drop-oldest -> ~67% of source frames discarded)
```

These numbers are from the *pre-fix* configuration. After applying §8.6
config, expect `upscale_ms` to drop to ~400 ms, `e2e_ms` to ~700 ms.

### 8.9 Pipeline rework (2026-05-24, later)

Three issues from the §8.8 run were addressed:

1. **NVENC probe (`[h264_nvenc] InitializeEncoder failed: invalid param (8):
   Frame Dimension less than the minimum supported value.`)** — the old
   probe used `nullsrc=s=64x64`, below the NVENC h264 minimum (~145×49 on
   Maxwell+). Probe now uses `s=256x256:r=15` + `-pix_fmt yuv420p`. RTX 5050
   Laptop now selects `h264_nvenc` instead of falling back to `libx264`.
   See `pipeline_live.py:_ffmpeg_has_nvenc`.

2. **FLV `Packet mismatch ... 11 ...` flood + `error while decoding MB`
   spam** — the source FLV gateway emits bogus `PreviousTagSize` fields;
   `cv2.VideoCapture` had no way to ignore them and produced thousands of
   resync errors, silent frame drops, and visible decoder MB corruption.
   Replaced the cv2-on-growing-tempfile decoder with an ffmpeg subprocess:

   ```python
   ffmpeg -hide_banner -loglevel info
          -fflags +discardcorrupt+genpts -err_detect ignore_err
          -f flv -i pipe:0 -map 0:v:0 -an
          [-vsync cfr -r <decode_fps>]
          -f rawvideo -pix_fmt bgr24 pipe:1
   ```

   WS bytes go to ffmpeg's stdin; stderr is parsed for the
   `Stream … Video: …, WxH` line; stdout reader reads `W*H*3` bytes per
   frame, reshapes to numpy BGR, and pushes to `player.frame_queue`.
   Implementation: `LiveFeeder._save_and_show_video_ws`. `_FlvAligner`
   (a tag-boundary buffer added earlier in the session) was removed —
   the alignment was unnecessary once ffmpeg replaced cv2, since ffmpeg
   resyncs internally.

3. **Jerky motion from ffmpeg outpacing pipeline** — ffmpeg decode now
   delivers ~28 fps, pipeline sustains ~8 fps; `frame_queue` evicted
   ~75% of frames with drop-oldest. Added `-vsync cfr -r <decode_fps>`
   so ffmpeg drops evenly at source rate. Also shrunk `frame_queue` from
   300 → 8 to keep frames fresh. Exposed via two new CLI flags:

   - `--decode-fps N` (default = `--push-fps` = 15). Set to ≈ pipeline
     sustainable fps (8 on RTX 5050 @ 4×).
   - `--queue-size N` (default 8).

   Wired in `pipeline_live.py:817-832` (NodePlayer init) and
   `LiveFeeder.py:36-43, 354-362` (ffmpeg cmd build).

`requirements-pipeline.txt` now declares the previously-implicit
`websockets`, `pytz`, `requests` deps. `numpy` was already pulled in by
SR stack but is now used directly by `LiveFeeder._stdout_reader`.

Files touched: `pipeline_live.py`, `LiveFeeder.py`,
`requirements-pipeline.txt`, `README.md` (§7 rewrite), `.gitignore`
(`bin/`, `recordings/`, `terminal.txt`).

Owner: Asem Hatamleh (Acacus Group) — `a.hattamleh@acacusgroup.com`
