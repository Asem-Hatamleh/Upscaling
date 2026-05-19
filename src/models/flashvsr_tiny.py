"""FlashVSR Tiny wrapper.

This is a thin adapter over the official ``FlashVSRTinyPipeline`` from the
DiffSynth-Studio fork at https://github.com/OpenImagingLab/FlashVSR .

We expect the FlashVSR repo to be cloned next to this project (path is
configurable via ``UPSCALING_FLASHVSR_DIR`` or ``cfg.extra["repo_dir"]``)
and the v1.1 weights downloaded to ``<repo>/examples/WanVSR/FlashVSR-v1.1/``.

Layout assumed:
  FlashVSR/
    examples/WanVSR/
      FlashVSR-v1.1/
        LQ_proj_in.ckpt
        TCDecoder.ckpt
        Wan2.1_VAE.pth
        diffusion_pytorch_model_streaming_dmd.safetensors

The pipeline accepts ``num_frames`` shaped (1,3,F,H,W) and produces a 4x
output. We chunk frames so VRAM stays bounded.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseUpscaler, UpscalerConfig, register

# Multiples FlashVSR expects for spatial dims (Wan VAE downsamples /8 + sliding window).
_DIM_MULTIPLE = 16


def _resolve_repo_dir(cfg_extra: dict | None) -> Path:
    cand = (cfg_extra or {}).get("repo_dir") or os.environ.get("UPSCALING_FLASHVSR_DIR")
    if cand:
        return Path(cand).expanduser().resolve()
    # default sibling of this repo
    here = Path(__file__).resolve()
    proj_root = here.parents[2]
    default = proj_root / "third_party" / "FlashVSR"
    return default


def _ensure_repo_on_path(repo: Path) -> Path:
    wan = repo / "examples" / "WanVSR"
    if not wan.exists():
        raise FileNotFoundError(
            f"FlashVSR repo not found at {repo}. Set UPSCALING_FLASHVSR_DIR "
            "or pass extra.repo_dir to the model config."
        )
    for p in (str(repo), str(wan)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return wan


@register("flashvsr_tiny")
class FlashVSRTiny(BaseUpscaler):
    native_scale = 4

    def __init__(self, cfg: UpscalerConfig) -> None:
        super().__init__(cfg)
        self.pipe = None
        self.repo_dir: Path | None = None
        self.weights_dir: Path | None = None
        self.chunk_frames: int = int((cfg.extra or {}).get("chunk_frames", 16))
        self.topk_ratio: float = float((cfg.extra or {}).get("topk_ratio", 2.0))
        self.kv_ratio: float = float((cfg.extra or {}).get("kv_ratio", 3.0))
        self.local_range: int = int((cfg.extra or {}).get("local_range", 11))
        self.color_fix: bool = bool((cfg.extra or {}).get("color_fix", True))

    # ------------- lifecycle -------------
    def load(self) -> None:
        import torch

        repo = _resolve_repo_dir(self.cfg.extra)
        wan = _ensure_repo_on_path(repo)
        self.repo_dir = repo

        # locate weight folder (v1.1 preferred)
        for cand in ("FlashVSR-v1.1", "FlashVSR"):
            wdir = wan / cand
            if (wdir / "diffusion_pytorch_model_streaming_dmd.safetensors").exists():
                self.weights_dir = wdir
                break
        if self.weights_dir is None:
            raise FileNotFoundError(
                f"FlashVSR weights not found under {wan}. Download with "
                "`python scripts/download_weights.py --model flashvsr`."
            )

        from diffsynth import ModelManager, FlashVSRTinyPipeline  # type: ignore

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(self.cfg.dtype, torch.bfloat16)

        mm = ModelManager(torch_dtype=torch_dtype, device=self.cfg.device)
        mm.load_models([
            str(self.weights_dir / "diffusion_pytorch_model_streaming_dmd.safetensors"),
            str(self.weights_dir / "Wan2.1_VAE.pth"),
        ])
        self.pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=self.cfg.device)

        # optional auxiliary blocks
        try:
            from utils.utils import Causal_LQ4x_Proj  # type: ignore
            from utils.TCDecoder import build_tcdecoder  # type: ignore
            self.pipe.lq_proj = Causal_LQ4x_Proj.from_checkpoint(
                str(self.weights_dir / "LQ_proj_in.ckpt"),
                device=self.cfg.device, dtype=torch_dtype,
            )
            self.pipe.tcdecoder = build_tcdecoder(
                str(self.weights_dir / "TCDecoder.ckpt"),
                device=self.cfg.device, dtype=torch_dtype,
            )
        except Exception as e:  # pragma: no cover - varies by repo revision
            # Newer pipeline revisions wire these inside the constructor.
            self.pipe._aux_status = f"aux wiring fallback: {e}"

        # Optional weight-only int8 quantization of the diffusion transformer.
        if self.cfg.quant in ("int8_woq", "int8"):
            self._apply_int8_woq()

        self.pipe.eval = lambda *a, **k: None  # FlashVSRTinyPipeline has no .eval

    def _apply_int8_woq(self) -> None:
        try:
            import torch
            import bitsandbytes as bnb  # noqa: F401
            from bitsandbytes.nn import Linear8bitLt
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"int8_woq requires bitsandbytes (pip install bitsandbytes). {e}"
            )
        import torch.nn as nn

        def _swap(module: nn.Module) -> None:
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear) and child.in_features >= 256:
                    q = Linear8bitLt(child.in_features, child.out_features,
                                     bias=child.bias is not None,
                                     has_fp16_weights=False, threshold=6.0)
                    q.load_state_dict(child.state_dict())
                    q = q.to(self.cfg.device)
                    setattr(module, name, q)
                else:
                    _swap(child)

        for blk_name in ("dit", "transformer", "denoiser", "unet"):
            blk = getattr(self.pipe, blk_name, None)
            if blk is not None:
                _swap(blk)

    # ------------- inference -------------
    def _target_shape(self, h: int, w: int) -> tuple[int, int]:
        # FlashVSR expects spatial dims divisible by 16 (after model's own padding).
        th = ((h + _DIM_MULTIPLE - 1) // _DIM_MULTIPLE) * _DIM_MULTIPLE
        tw = ((w + _DIM_MULTIPLE - 1) // _DIM_MULTIPLE) * _DIM_MULTIPLE
        return th, tw

    def upscale(self, frames: np.ndarray) -> np.ndarray:
        import torch
        from einops import rearrange

        assert self.pipe is not None, "call .load() first"
        n, h, w, _ = frames.shape
        th, tw = self._target_shape(h, w)

        # pad input to (th, tw) by reflection so model accepts dims
        pad_h, pad_w = th - h, tw - w
        if pad_h or pad_w:
            frames_pad = np.pad(frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        else:
            frames_pad = frames

        out_chunks: list[np.ndarray] = []
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(self.cfg.dtype, torch.bfloat16)

        for start in range(0, n, self.chunk_frames):
            chunk = frames_pad[start : start + self.chunk_frames]
            if len(chunk) == 0:
                break
            t = torch.from_numpy(chunk).to(self.cfg.device)
            t = t.float().div_(255.0)
            t = rearrange(t, "f h w c -> 1 c f h w").to(torch_dtype)
            with torch.inference_mode():
                sr = self.pipe(
                    prompt="",
                    negative_prompt="",
                    cfg_scale=1.0,
                    num_inference_steps=1,
                    LQ_video=t,
                    num_frames=t.shape[2],
                    height=th * self.native_scale,
                    width=tw * self.native_scale,
                    topk_ratio=self.topk_ratio,
                    kv_ratio=self.kv_ratio,
                    local_range=self.local_range,
                    color_fix=self.color_fix,
                )
            sr = sr.detach().float().clamp_(0, 1).mul_(255.0).round_().byte()
            sr = rearrange(sr, "1 c f h w -> f h w c").cpu().numpy()
            if pad_h or pad_w:
                sr = sr[:, : h * self.native_scale, : w * self.native_scale, :]
            out_chunks.append(sr)

        return np.concatenate(out_chunks, axis=0)

    def close(self) -> None:
        import torch
        self.pipe = None
        torch.cuda.empty_cache()
