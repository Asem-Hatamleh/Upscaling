# Benchmark Run Commands

## Face Enhancement Test

Use this command for the current GFPGAN / CodeFormer face-quality test:

```bash
venv/bin/python scripts/benchmark.py \
  --preset face_enhance_best \
  --videos "Real Test Video/9.mp4,6.mp4,3.mp4" \
  --seconds 10 \
  --scales 4 \
  --timeout 1800 \
  --auto-resume
```

If VS Code or the terminal crashes, run the exact same command again.

## Fresh Run After Fix

If an old run already skipped too many configs and you want to start clean:

```bash
venv/bin/python scripts/benchmark.py \
  --preset face_enhance_best \
  --videos "Real Test Video/9.mp4,6.mp4,3.mp4" \
  --seconds 10 \
  --scales 4 \
  --timeout 1800
```

Use `--auto-resume` only after that new run exists.

## Crash Fix

The crash was likely RAM pressure.

Old path kept full 10-second 4x output arrays in memory before writing video.
For 640x360 -> 2560x1440, 10 seconds can become several GB, and comparison
video doubles memory pressure.

The `skip=1` / `interp=none` path now streams in chunks:

- process chunk
- write chunk
- release chunk
- continue

This should stop VS Code / system OOM crashes for the face-best preset.

## What The Command Does

`venv/bin/python scripts/benchmark.py`

Runs the benchmark script using the project virtual environment.

`--preset face_enhance_best`

Runs the focused face-enhancement sweep:

- `realesrgan_lite`
- `realesrgan_full`
- `realesrgan_gfpgan`
- `codeformer_compact`
- CodeFormer fidelity values: `0.7`, `0.9`, `1.0`
- best found args: `fp16`, `denoise=1.0`, `frame_skip=1`, `frame_interp=none`

`--videos "Real Test Video/9.mp4,6.mp4,3.mp4"`

Runs only videos `9.mp4`, `6.mp4`, and `3.mp4`.

`--seconds 10`

Processes first 10 seconds of each video.

`--scales 4`

Runs only 4x output scale.

Use this if you also want 2x:

```bash
--scales 4,2
```

`--timeout 1800`

Allows each single config up to 1800 seconds before marking it as timeout.

`--auto-resume`

Resumes the newest matching `face_enhance_best` run.

If a config started but VS Code crashed before it finished, resume skips that config by default and continues with the next one.

## Important

Do not use this unless you want to rerun the crashed config:

```bash
--retry-incomplete
```

No `--retry-incomplete` means:

- crashed/incomplete config is skipped
- run continues from next config
- skipped config is written to `skipped_after_crash.txt`

## Why It Started At `v6`

If output says:

```txt
[bench] skipping 1 incomplete label(s) from previous crash
[bench]   1/14 facebest_lite_s4_prnone_dn1_fp16__v6
```

That means:

- `v9` crashed during previous run
- benchmark skipped `v9`
- next remaining config is `v6`

This is expected.

## Output Files

Each run writes to:

```txt
output/benchmark/<timestamp>/
```

Important files:

- `runs.csv` - all completed results
- `summary.txt` - best FPS / latency / VRAM summary
- `runs.json` - full result data
- `started.jsonl` - configs that started
- `skipped_after_crash.txt` - configs skipped after crash

Each video output writes:

```txt
output/<model>/<run_name>/
```

Important file:

- `run_info.txt`

Key metrics:

- `e2e_fps`
- `latency_ms_per_source_frame`
- `latency_ms_per_sr_frame`
- `peak_vram_mb`

## Models In `face_enhance_best`

With:

```bash
--videos "Real Test Video/9.mp4,6.mp4,3.mp4" --scales 4
```

the preset runs:

- `realesrgan_lite`: 3 runs
- `realesrgan_full`: 3 runs
- `realesrgan_gfpgan`: 3 runs
- `codeformer_compact`: 9 runs (`cf_fidelity` = `0.7`, `0.9`, `1.0`)

Total: 18 runs.

`realesrgan_full` is full RRDB `RealESRGAN_x4plus`, no GFPGAN, no CodeFormer.
It is heavier than `realesrgan_lite`.
First run may download:

```txt
weights/realesrgan/RealESRGAN_x4plus.pth
```

## Generate Report

After benchmark finishes:

```bash
venv/bin/python scripts/analyze_benchmark.py output/benchmark/<timestamp>/runs.csv
```

This writes:

```txt
output/benchmark/<timestamp>/REPORT.md
```
