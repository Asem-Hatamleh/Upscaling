# Benchmark report — 20260519_122220

- runs: **211**
- ok: 153
- too_few_frames: 58

## Top 20 by end-to-end FPS

| FPS | VRAM MB | model | scale | pr | skip | sage | quant | dtype | out_res |
|---:|---:|---|---:|---|---:|---|---|---|---|
| **48.52** | 175 | realesrgan_lite | 2 | 35% | 4 | False | none | fp16 | 492x404 |
| **46.56** | 182 | realesrgan_lite | 2 | 35% | 4 | False | none | bf16 | 492x404 |
| **46.56** | 182 | realesrgan_lite | 2 | 35% | 4 | False | none | fp32 | 492x404 |
| **43.81** | 288 | realesrgan_lite | 4 | 25% | 2 | False | none | fp16 | 704x576 |
| **42.89** | 288 | realesrgan_lite | 4 | 25% | 2 | False | none | fp16 | 704x576 |
| **40.33** | 175 | realesrgan_lite | 2 | 35% | 2 | False | none | fp16 | 492x404 |
| **39.67** | 175 | realesrgan_lite | 2 | 35% | 3 | False | none | fp16 | 492x404 |
| **39.02** | 288 | realesrgan_lite | 4 | 25% | 2 | False | none | fp16 | 704x576 |
| **38.77** | 72 | realesrgan_lite | 4 | 35% | 1 | False | none | fp16 | 984x808 |
| **38.17** | 4218 | flashvsr_tiny | 2 | 15% | 3 | False | none | bf16 | 192x128 |
| **37.72** | 182 | realesrgan_lite | 2 | 35% | 2 | False | none | bf16 | 492x404 |
| **37.66** | 182 | realesrgan_lite | 2 | 35% | 2 | False | none | fp32 | 492x404 |
| **36.66** | 182 | realesrgan_lite | 2 | 35% | 3 | False | none | bf16 | 492x404 |
| **36.61** | 182 | realesrgan_lite | 2 | 35% | 3 | False | none | fp32 | 492x404 |
| **36.30** | 4218 | flashvsr_tiny | 2 | 15% | 3 | True | none | bf16 | 192x128 |
| **35.14** | 4218 | flashvsr_tiny | 4 | 15% | 3 | True | none | bf16 | 384x256 |
| **34.16** | 103 | realesrgan_lite | 4 | 35% | 1 | False | none | fp32 | 984x808 |
| **33.92** | 72 | realesrgan_lite | 2 | 35% | 1 | False | none | fp16 | 492x404 |
| **33.33** | 103 | realesrgan_lite | 4 | 35% | 1 | False | none | bf16 | 984x808 |
| **31.75** | 4218 | flashvsr_tiny | 4 | 15% | 3 | False | none | bf16 | 384x256 |

## Best per model

### realesrgan_lite

| FPS | VRAM MB | scale | pr | skip | sage | quant | dt | out_res |
|---:|---:|---:|---|---:|---|---|---|---|
| **48.52** | 175 | 2 | 35% | 4 | False | none | fp16 | 492x404 |
| **46.56** | 182 | 2 | 35% | 4 | False | none | bf16 | 492x404 |
| **46.56** | 182 | 2 | 35% | 4 | False | none | fp32 | 492x404 |
| **43.81** | 288 | 4 | 25% | 2 | False | none | fp16 | 704x576 |
| **42.89** | 288 | 4 | 25% | 2 | False | none | fp16 | 704x576 |
| **40.33** | 175 | 2 | 35% | 2 | False | none | fp16 | 492x404 |
| **39.67** | 175 | 2 | 35% | 3 | False | none | fp16 | 492x404 |
| **39.02** | 288 | 4 | 25% | 2 | False | none | fp16 | 704x576 |
| **38.77** | 72 | 4 | 35% | 1 | False | none | fp16 | 984x808 |
| **37.72** | 182 | 2 | 35% | 2 | False | none | bf16 | 492x404 |

### flashvsr_tiny

| FPS | VRAM MB | scale | pr | skip | sage | quant | dt | out_res |
|---:|---:|---:|---|---:|---|---|---|---|
| **38.17** | 4218 | 2 | 15% | 3 | False | none | bf16 | 192x128 |
| **36.30** | 4218 | 2 | 15% | 3 | True | none | bf16 | 192x128 |
| **35.14** | 4218 | 4 | 15% | 3 | True | none | bf16 | 384x256 |
| **31.75** | 4218 | 4 | 15% | 3 | False | none | bf16 | 384x256 |
| **30.38** | 4606 | 2 | 15% | 2 | False | none | bf16 | 192x128 |
| **29.96** | 4606 | 4 | 15% | 2 | False | none | bf16 | 384x256 |
| **29.69** | 4606 | 4 | 15% | 2 | True | none | bf16 | 384x256 |
| **29.52** | 4606 | 2 | 15% | 2 | True | none | bf16 | 192x128 |
| **27.23** | 4895 | 2 | 20% | 3 | False | none | bf16 | 256x192 |
| **27.10** | 4895 | 2 | 20% | 3 | True | none | bf16 | 256x192 |

## VRAM-efficient frontier (highest FPS at each VRAM band)

| ≤ VRAM MB | best FPS | model | pr | sk | quant | dt |
|---:|---:|---|---|---:|---|---|
| 256 | 48.52 | realesrgan_lite | 35% | 4 | none | fp16 |
| 512 | 43.81 | realesrgan_lite | 25% | 2 | none | fp16 |
| 768 | 24.05 | realesrgan_lite | 35% | 2 | none | fp16 |
| 1024 | 12.21 | realesrgan_lite | 50% | 2 | none | fp16 |
| 1280 | 10.88 | realesrgan_lite | 50% | 2 | none | bf16 |
| 2304 | 5.04 | realesrgan_lite | 75% | 2 | none | fp16 |
| 4096 | 2.86 | realesrgan_lite | 100% | 2 | none | fp16 |
| 4352 | 38.17 | flashvsr_tiny | 15% | 3 | none | bf16 |
| 4608 | 30.38 | flashvsr_tiny | 15% | 2 | none | bf16 |
| 4864 | 23.36 | flashvsr_tiny | 15% | 1 | none | bf16 |
| 5120 | 27.23 | flashvsr_tiny | 20% | 3 | none | bf16 |
| 5376 | 20.00 | flashvsr_tiny | 20% | 2 | none | bf16 |
| 5632 | 13.73 | flashvsr_tiny | 20% | 1 | none | bf16 |
| 5888 | 18.86 | flashvsr_tiny | 25% | 3 | none | bf16 |
| 6144 | 12.51 | flashvsr_tiny | 25% | 2 | none | bf16 |
| 6400 | 13.27 | flashvsr_tiny | 25% | 2 | none | bf16 |
| 6656 | 8.56 | flashvsr_tiny | 25% | 1 | none | bf16 |
| 6912 | 12.14 | flashvsr_tiny | 25% | 2 | none | bf16 |
| 7168 | 13.24 | flashvsr_tiny | 30% | 3 | none | bf16 |

## Per-knob sensitivity (mean FPS over `ok` runs)

### realesrgan_lite

**pre_resize**
  - `25%`: mean=41.91  max=43.81  n=3
  - `35%`: mean=30.99  max=48.52  n=24
  - `50%`: mean=15.66  max=27.85  n=24
  - `75%`: mean=6.49  max=11.58  n=24
  - `100%`: mean=3.59  max=6.39  n=24

**frame_skip**
  - `2`: mean=17.20  max=43.81  n=27
  - `4`: mean=15.86  max=48.52  n=24
  - `1`: mean=14.36  max=38.77  n=24
  - `3`: mean=12.40  max=39.67  n=24

**dtype**
  - `fp16`: mean=17.58  max=48.52  n=35
  - `bf16`: mean=13.65  max=46.56  n=32
  - `fp32`: mean=13.60  max=46.56  n=32

**crf**
  - `23`: mean=43.81  max=43.81  n=1
  - `20`: mean=42.89  max=42.89  n=1
  - `16`: mean=39.02  max=39.02  n=1
  - `18`: mean=14.18  max=48.52  n=96

**scale**
  - `2`: mean=17.71  max=48.52  n=48
  - `4`: mean=12.49  max=43.81  n=51

### flashvsr_tiny

**pre_resize**
  - `15%`: mean=29.34  max=38.17  n=12
  - `20%`: mean=19.31  max=27.23  n=12
  - `25%`: mean=12.44  max=18.86  n=26
  - `30%`: mean=11.81  max=13.24  n=4

**frame_skip**
  - `3`: mean=22.31  max=38.17  n=16
  - `2`: mean=16.14  max=30.38  n=26
  - `1`: mean=14.81  max=23.36  n=12

**sage_attn**
  - `False`: mean=19.31  max=38.17  n=20
  - `True`: mean=16.71  max=36.30  n=34

**quant**
  - `none`: mean=17.89  max=38.17  n=52
  - `nf4`: mean=12.39  max=12.39  n=1
  - `int8_woq`: mean=11.80  max=11.80  n=1

**chunk_frames**
  - `16`: mean=17.98  max=38.17  n=51
  - `24`: mean=12.55  max=12.55  n=1
  - `32`: mean=12.53  max=12.53  n=1
  - `8`: mean=12.22  max=12.22  n=1

**topk_ratio**
  - `2.0`: mean=17.88  max=38.17  n=52
  - `3.0`: mean=12.51  max=12.51  n=1
  - `1.5`: mean=12.31  max=12.31  n=1

**kv_ratio**
  - `3.0`: mean=17.88  max=38.17  n=52
  - `1.5`: mean=12.51  max=12.51  n=1
  - `5.0`: mean=12.14  max=12.14  n=1

**local_range**
  - `11`: mean=17.77  max=38.17  n=53
  - `9`: mean=12.32  max=12.32  n=1

**color_fix**
  - `True`: mean=17.77  max=38.17  n=53
  - `False`: mean=12.42  max=12.42  n=1

**crf**
  - `18`: mean=17.99  max=38.17  n=51
  - `20`: mean=12.44  max=12.44  n=1
  - `16`: mean=12.33  max=12.33  n=1
  - `23`: mean=12.30  max=12.30  n=1

**scale**
  - `2`: mean=20.25  max=38.17  n=20
  - `4`: mean=16.16  max=35.14  n=34

## Failure breakdown

### too_few_frames (58)
  - flash_grid_s2_pr15%_sk4_sa1
  - flash_grid_s2_pr15%_sk4_sa0
  - flash_grid_s2_pr20%_sk4_sa1
  - flash_grid_s2_pr20%_sk4_sa0
  - flash_grid_s2_pr25%_sk4_sa1
  - flash_grid_s2_pr25%_sk4_sa0
  - flash_grid_s2_pr30%_sk1_sa1
  - flash_grid_s2_pr30%_sk1_sa0
  - flash_grid_s2_pr30%_sk2_sa1
  - flash_grid_s2_pr30%_sk2_sa0
  - flash_grid_s2_pr30%_sk4_sa1
  - flash_grid_s2_pr30%_sk4_sa0
  - flash_grid_s2_pr35%_sk1_sa1
  - flash_grid_s2_pr35%_sk1_sa0
  - flash_grid_s2_pr35%_sk2_sa1
  - flash_grid_s2_pr35%_sk2_sa0
  - flash_grid_s2_pr35%_sk3_sa1
  - flash_grid_s2_pr35%_sk3_sa0
  - flash_grid_s2_pr35%_sk4_sa1
  - flash_grid_s2_pr35%_sk4_sa0
  - flash_grid_s2_pr50%_sk1_sa1
  - flash_grid_s2_pr50%_sk1_sa0
  - flash_grid_s2_pr50%_sk2_sa1
  - flash_grid_s2_pr50%_sk2_sa0
  - flash_grid_s2_pr50%_sk3_sa1
  - flash_grid_s2_pr50%_sk3_sa0
  - flash_grid_s2_pr50%_sk4_sa1
  - flash_grid_s2_pr50%_sk4_sa0
  - flash_grid_s4_pr15%_sk4_sa1
  - flash_grid_s4_pr15%_sk4_sa0
  - ... and 28 more
