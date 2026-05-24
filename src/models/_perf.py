"""Shared inference-time perf helpers for the Real-ESRGAN family.

Two utilities:

- ``apply_perf_opts`` flips a torch ``nn.Module`` into ``channels_last``
  memory format and wraps its forward in ``torch.compile``. Silent no-op
  on any failure — perf opts must never break correctness.

- ``batched_realesrganer_enhance`` is a drop-in batched alternative to
  ``RealESRGANer.enhance`` called in a per-frame Python loop. Stacks N
  RGB uint8 frames into one (N, 3, H, W) tensor and runs a single GPU
  forward — same math as the per-frame path, fewer kernel launches /
  fewer PCIe round-trips.

Both are best-effort. Callers should keep the per-frame ``.enhance()``
loop as a fallback path so a perf-opt failure (rare, but possible on
sm_120 + torch nightly) never blocks a run.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def apply_perf_opts(
    net: Any,
    *,
    compile: bool = True,
    channels_last: bool = True,
    compile_mode: str = "default",
) -> Any:
    """Return ``net`` flipped to channels_last and/or wrapped in
    ``torch.compile``. Returns the original module if any step fails."""
    import torch  # local import: keeps non-torch tooling clean.

    if net is None:
        return net

    if channels_last:
        try:
            net = net.to(memory_format=torch.channels_last)
        except Exception as e:
            print(f"[_perf] channels_last skipped ({type(e).__name__}: {e})")

    if compile and hasattr(torch, "compile"):
        try:
            net = torch.compile(
                net,
                mode=compile_mode,   # "default" = no CUDA graphs, ~20-25% gain
                fullgraph=False,     # tolerate graph breaks
                dynamic=False,       # fixed input shape across stream
            )
        except Exception as e:
            print(f"[_perf] torch.compile skipped ({type(e).__name__}: {e})")

    return net


def batched_realesrganer_enhance(
    upsampler: Any,
    frames_rgb: np.ndarray,
    *,
    device: str,
    half: bool,
    scale: int = 4,
    mod_pad: int = 2,
) -> np.ndarray:
    """Batched forward through ``upsampler.model``.

    ``frames_rgb``: (N, H, W, 3) uint8 RGB.
    Returns:     (N, H*scale, W*scale, 3) uint8 RGB.

    Preprocessing replicates ``RealESRGANer.enhance`` (which internally
    runs ``cvtColor(BGR2RGB)`` before the model forward, so the model
    sees and outputs RGB tensors):

    - permute HWC -> CHW, add batch
    - cast to fp16 (or fp32) and divide by 255
    - reflect-pad H/W to multiple of ``mod_pad`` (default 2, same as
      RealESRGANer's default for the Compact / RRDB nets)
    - forward through ``upsampler.model``
    - crop output by ``pad * scale``
    - clamp, *255, uint8, permute back to HWC

    The DNI denoise blend (when ``0 < denoise < 1`` for x4v3) is baked
    into ``upsampler.model`` at load time, so we only need a single
    forward. Skip this helper when ``upsampler.tile > 0`` because the
    tile path needs RealESRGANer's per-tile logic.
    """
    import torch

    rgb = np.ascontiguousarray(frames_rgb)
    t = torch.from_numpy(rgb).to(device, non_blocking=True)
    t = t.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    t = t.half() if half else t.float()
    t = t / 255.0

    h, w = t.shape[-2:]
    pad_h = (mod_pad - h % mod_pad) % mod_pad
    pad_w = (mod_pad - w % mod_pad) % mod_pad
    if pad_h or pad_w:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.inference_mode():
        sr = upsampler.model(t)

    if pad_h or pad_w:
        sr = sr[..., : h * scale, : w * scale]

    sr = sr.clamp(0, 1).mul(255.0).to(torch.uint8)
    sr = sr.permute(0, 2, 3, 1).contiguous()
    return sr.cpu().numpy()
