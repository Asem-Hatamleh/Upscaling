# Benchmark report — 20260519_145614

- runs: **108**
- ok: 108

## Top 20 by end-to-end FPS

| FPS | VRAM MB | model | scale | pr | skip | sage | quant | dtype | out_res |
|---:|---:|---|---:|---|---:|---|---|---|---|
| **19.57** | 1312 | realesrgan_gfpgan | 2 | 35% | 4 | False | none | fp16 | 448x252 |
| **19.47** | 1317 | realesrgan_gfpgan | 2 | 35% | 4 | False | none | fp32 | 448x252 |
| **15.64** | 1314 | realesrgan_gfpgan | 2 | 35% | 4 | False | none | fp16 | 492x404 |
| **15.57** | 1314 | realesrgan_gfpgan | 2 | 35% | 4 | False | none | fp16 | 492x404 |
| **15.34** | 1321 | realesrgan_gfpgan | 2 | 35% | 4 | False | none | fp32 | 492x404 |
| **15.33** | 1321 | realesrgan_gfpgan | 2 | 35% | 4 | False | none | fp32 | 492x404 |
| **14.90** | 1315 | realesrgan_gfpgan | 2 | 50% | 4 | False | none | fp16 | 640x360 |
| **14.79** | 1312 | realesrgan_gfpgan | 4 | 35% | 4 | False | none | fp16 | 896x504 |
| **14.70** | 1317 | realesrgan_gfpgan | 4 | 35% | 4 | False | none | fp32 | 896x504 |
| **14.43** | 1323 | realesrgan_gfpgan | 2 | 50% | 4 | False | none | fp32 | 640x360 |
| **11.66** | 1319 | realesrgan_gfpgan | 2 | 50% | 4 | False | none | fp16 | 704x576 |
| **11.64** | 1319 | realesrgan_gfpgan | 2 | 50% | 4 | False | none | fp16 | 704x576 |
| **11.21** | 1312 | realesrgan_gfpgan | 2 | 35% | 2 | False | none | fp16 | 448x252 |
| **11.10** | 1331 | realesrgan_gfpgan | 2 | 50% | 4 | False | none | fp32 | 704x576 |
| **10.99** | 1317 | realesrgan_gfpgan | 2 | 35% | 2 | False | none | fp32 | 448x252 |
| **10.93** | 1331 | realesrgan_gfpgan | 2 | 50% | 4 | False | none | fp32 | 704x576 |
| **10.62** | 1314 | realesrgan_gfpgan | 4 | 35% | 4 | False | none | fp16 | 984x808 |
| **10.46** | 1314 | realesrgan_gfpgan | 4 | 35% | 4 | False | none | fp16 | 984x808 |
| **10.45** | 1321 | realesrgan_gfpgan | 4 | 35% | 4 | False | none | fp32 | 984x808 |
| **10.43** | 1321 | realesrgan_gfpgan | 4 | 35% | 4 | False | none | fp32 | 984x808 |

## Best per model

### realesrgan_gfpgan

| FPS | VRAM MB | scale | pr | skip | sage | quant | dt | out_res |
|---:|---:|---:|---|---:|---|---|---|---|
| **19.57** | 1312 | 2 | 35% | 4 | False | none | fp16 | 448x252 |
| **19.47** | 1317 | 2 | 35% | 4 | False | none | fp32 | 448x252 |
| **15.64** | 1314 | 2 | 35% | 4 | False | none | fp16 | 492x404 |
| **15.57** | 1314 | 2 | 35% | 4 | False | none | fp16 | 492x404 |
| **15.34** | 1321 | 2 | 35% | 4 | False | none | fp32 | 492x404 |
| **15.33** | 1321 | 2 | 35% | 4 | False | none | fp32 | 492x404 |
| **14.90** | 1315 | 2 | 50% | 4 | False | none | fp16 | 640x360 |
| **14.79** | 1312 | 4 | 35% | 4 | False | none | fp16 | 896x504 |
| **14.70** | 1317 | 4 | 35% | 4 | False | none | fp32 | 896x504 |
| **14.43** | 1323 | 2 | 50% | 4 | False | none | fp32 | 640x360 |

## VRAM-efficient frontier (highest FPS at each VRAM band)

| ≤ VRAM MB | best FPS | model | pr | sk | quant | dt |
|---:|---:|---|---|---:|---|---|
| 1280 | 6.17 | realesrgan_gfpgan | 35% | 1 | none | fp16 |
| 1536 | 19.57 | realesrgan_gfpgan | 35% | 4 | none | fp16 |
| 1792 | 6.77 | realesrgan_gfpgan | 50% | 4 | none | fp16 |
| 3072 | 3.36 | realesrgan_gfpgan | none | 4 | none | fp16 |
| 4608 | 2.07 | realesrgan_gfpgan | none | 4 | none | fp16 |

## Per-knob sensitivity (mean FPS over `ok` runs)

### realesrgan_gfpgan

**pre_resize**
  - `35%`: mean=9.50  max=19.57  n=36
  - `50%`: mean=6.90  max=14.90  n=36
  - `none`: mean=2.73  max=6.69  n=36

**frame_skip**
  - `4`: mean=9.38  max=19.57  n=36
  - `2`: mean=6.05  max=11.21  n=36
  - `1`: mean=3.71  max=6.17  n=36

**dtype**
  - `fp16`: mean=6.48  max=19.57  n=54
  - `fp32`: mean=6.27  max=19.47  n=54

**scale**
  - `2`: mean=7.25  max=19.57  n=54
  - `4`: mean=5.50  max=14.79  n=54
