"""BasicVSR++ temporal video super-resolution backend.

Pip-installed ``basicsr`` ships ``BasicVSRPlusPlus``; the REDS4-trained
weights live on the upstream repo's GitHub release. The model takes a
sequence of frames (B, T, C, H, W) and produces (B, T, C, 4H, 4W) — true
temporal SR with bidirectional flow propagation, not per-frame.

We chunk the input into non-overlapping windows of ``window`` frames
(default 7) to bound VRAM. Quality near window seams is slightly worse
than a fully sliding window but the seam isn't visible in 30 fps content
because RIFE-style temporal coherence inside each window is already
strong.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np

from .base import BaseUpscaler, UpscalerConfig, register


# Official BasicVSR++ REDS4 release; matches the arch we instantiate below.
_WEIGHT_URL = (
    "https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/"
    "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_weight() -> Path:
    out = _project_root() / "weights" / "basicvsrpp" / "basicvsr_plusplus_reds4.pth"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request as _r
    print(f"[basicvsrpp] downloading {_WEIGHT_URL} -> {out}")
    _r.urlretrieve(_WEIGHT_URL, out)
    return out


@register("basicvsrpp")
class BasicVSRPP(BaseUpscaler):
    """Temporal SR via BasicVSR++. Native 4x. Sliding-window inference."""

    native_scale = 4

    def __init__(self, cfg: UpscalerConfig) -> None:
        super().__init__(cfg)
        ex = cfg.extra or {}
        self.net = None
        # Window of frames passed to the network at once. Lower this if OOM.
        self.window = int(ex.get("bvsrpp_window", 7))
        # Run inference in fp16 if requested via cfg.dtype.
        self.use_half = cfg.dtype in ("fp16",)

    # ------------- lifecycle -------------
    def load(self) -> None:
        import re

        import torch
        from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus  # type: ignore

        # Default REDS configuration: 64 mid channels, 7 blocks, spynet flow.
        net = BasicVSRPlusPlus(
            mid_channels=64,
            num_blocks=7,
            max_residue_magnitude=10,
            is_low_res_input=True,
            spynet_path=None,    # spynet weight ships inside the checkpoint
            cpu_cache_length=100,
        )
        weight_path = _ensure_weight()
        ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)

        # mmediting-format checkpoint: state is nested under `state_dict` with a
        # `generator.` prefix on every key, the spynet ConvModule wrapper exposes
        # a `.conv` subkey, and the final upsamplers are named
        # `upsample1/upsample_conv.*`. basicsr's `BasicVSRPlusPlus` uses
        # `upconv1/upconv2.*` and unwrapped spynet Sequential indices. Remap
        # transparently so we can use the public mmediting weights.
        raw = ckpt.get("state_dict", ckpt)

        def _remap(k: str) -> str:
            if k.startswith("generator."):
                k = k[len("generator."):]
            m = re.match(
                r"^(spynet\.basic_module\.\d+\.basic_module\.)(\d+)\.conv\.(weight|bias)$", k,
            )
            if m:
                return f"{m.group(1)}{int(m.group(2)) * 2}.{m.group(3)}"
            if k.startswith("upsample1.upsample_conv."):
                return "upconv1." + k.split("upsample1.upsample_conv.")[1]
            if k.startswith("upsample2.upsample_conv."):
                return "upconv2." + k.split("upsample2.upsample_conv.")[1]
            return k

        state = {_remap(k): v for k, v in raw.items() if k != "step_counter"}
        # strict=False to absorb any harmless extras (e.g. spynet meta tensors).
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[basicvsrpp] load: missing={len(missing)} unexpected={len(unexpected)}")
            if missing[:3]:
                print(f"  missing sample: {missing[:3]}")
            if unexpected[:3]:
                print(f"  unexpected sample: {unexpected[:3]}")
        net.eval()
        net = net.to(self.cfg.device)
        # BasicVSR++ uses internal float32 mean/std buffers and flow ops that
        # don't autocast cleanly to fp16; keep the model in fp32 even if the
        # user requested ``--dtype fp16``. Memory is dominated by the cpu_cache
        # in any case.
        self.net = net

    # ------------- inference -------------
    def upscale(self, frames: np.ndarray) -> np.ndarray:
        import torch

        assert self.net is not None, "call .load() first"
        n, h, w, _ = frames.shape
        if n == 0:
            return np.zeros((0, h * 4, w * 4, 3), dtype=np.uint8)

        # (N, H, W, 3) uint8 RGB -> (1, N, 3, H, W) float32 in [0, 1]
        t = torch.from_numpy(np.ascontiguousarray(frames)).to(self.cfg.device)
        t = t.permute(0, 3, 1, 2).float().div_(255.0).unsqueeze(0)

        # BasicVSR++ requires spatial dims divisible by 4 (downsample stack).
        # F.pad doesn't accept 4-element specs for 5-D tensors, so reshape the
        # batch and time dims together while padding.
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h or pad_w:
            b, tn, c, hh, ww = t.shape
            flat = t.reshape(b * tn, c, hh, ww)
            flat = torch.nn.functional.pad(flat, (0, pad_w, 0, pad_h), mode="replicate")
            t = flat.reshape(b, tn, c, hh + pad_h, ww + pad_w)

        out_chunks: List[np.ndarray] = []
        win = max(2, int(self.window))
        with torch.inference_mode():
            for start in range(0, n, win):
                end = min(n, start + win)
                clip = t[:, start:end]
                sr = self.net(clip)               # (1, T, 3, 4H, 4W)
                sr = sr.float().clamp_(0, 1).mul_(255.0).round_().byte()
                sr = sr[0].permute(0, 2, 3, 1).cpu().numpy()
                if pad_h or pad_w:
                    sr = sr[:, : h * 4, : w * 4, :]
                out_chunks.append(sr)

        return np.concatenate(out_chunks, axis=0)

    def close(self) -> None:
        import torch
        self.net = None
        torch.cuda.empty_cache()
