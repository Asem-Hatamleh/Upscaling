"""FlashVSR Tiny wrapper.

Thin adapter over the official ``FlashVSRTinyPipeline`` from the DiffSynth-Studio
fork shipped inside https://github.com/OpenImagingLab/FlashVSR . Mirrors the
upstream ``infer_flashvsr_v1.1_tiny.py`` init sequence so we get the same
quality, but exposes it through ``BaseUpscaler.upscale(np.uint8 RGB frames)``.

Layout assumed:
  third_party/FlashVSR/
    diffsynth/                            # pip-installed-equivalent (added to sys.path)
    examples/WanVSR/
      utils/{utils.py, TCDecoder.py}
      prompt_tensor/posi_prompt.pth
      FlashVSR-v1.1/
        LQ_proj_in.ckpt
        TCDecoder.ckpt
        Wan2.1_VAE.pth
        diffusion_pytorch_model_streaming_dmd.safetensors

Notes / gotchas baked into the upstream tiny script:
- LQ video tensor must be in ``[-1, 1]``, shape ``(1, C, F, H, W)``.
- Spatial dims must be a multiple of 128. The LQ video is pre-upsampled
  (bicubic) to the target SR resolution before being fed to the pipeline.
- Frame count must be ``8n + 1``. We pad with the last frame (matches
  upstream) and crop the SR output to the original frame count.
- ``topk_ratio`` is ``sparse_ratio * 768 * 1280 / (tH * tW)`` — scale-invariant.
- The sparse-attention path is only taken when ``block_sparse_attn`` is
  importable. We stub it so import succeeds even when the kernel isn't built;
  FlashVSR will fall back to its SageAttention / SDPA path automatically.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseUpscaler, UpscalerConfig, register

_SPATIAL_MULTIPLE = 128


# ----------------------------- helpers ------------------------------

def _resolve_repo_dir(cfg_extra: dict | None) -> Path:
    cand = (cfg_extra or {}).get("repo_dir") or os.environ.get("UPSCALING_FLASHVSR_DIR")
    if cand:
        return Path(cand).expanduser().resolve()
    here = Path(__file__).resolve()
    proj_root = here.parents[2]
    return proj_root / "third_party" / "FlashVSR"


def _stub_block_sparse_attn() -> None:
    """Stand in for ``block_sparse_attn`` when the CUDA kernel isn't built.

    FlashVSR's ``wan_video_dit.py`` hard-imports ``block_sparse_attn_func``.
    The function itself is only called from the sparse branch we don't take
    by default. Providing a stub lets the module import. If the sparse path
    is invoked at runtime we raise a clear error.
    """
    import types

    if "block_sparse_attn" in sys.modules:
        return
    mod = types.ModuleType("block_sparse_attn")

    def block_sparse_attn_func(*_a, **_kw):
        raise RuntimeError(
            "block_sparse_attn is not installed; FlashVSR sparse-attention path "
            "was invoked. Build Block-Sparse-Attention from source (needs full "
            "CUDA toolkit) or keep --sage-attn so the SageAttention path is taken."
        )

    mod.block_sparse_attn_func = block_sparse_attn_func  # type: ignore[attr-defined]
    sys.modules["block_sparse_attn"] = mod


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
    # Don't stub block_sparse_attn: our patched wan_video_dit.py handles
    # ImportError directly and falls back to dense SDPA when BSA is missing.
    return wan


def _compute_target_dims(lr_w: int, lr_h: int, scale: float = 4.0,
                         multiple: int = _SPATIAL_MULTIPLE) -> tuple[int, int]:
    """Return (tW, tH): largest 128-multiples ≤ lr * scale."""
    sw = int(round(lr_w * scale))
    sh = int(round(lr_h * scale))
    tw = (sw // multiple) * multiple
    th = (sh // multiple) * multiple
    if tw <= 0 or th <= 0:
        raise ValueError(
            f"Input too small ({lr_w}x{lr_h}) for scale={scale} and "
            f"multiple={multiple}. Increase --pre-resize."
        )
    return tw, th


def _largest_8np1_leq(n: int) -> int:
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


# ----------------------------- model --------------------------------

@register("flashvsr_tiny")
class FlashVSRTiny(BaseUpscaler):
    native_scale = 4

    def __init__(self, cfg: UpscalerConfig) -> None:
        super().__init__(cfg)
        self.pipe = None
        self.repo_dir: Path | None = None
        self.weights_dir: Path | None = None
        ex = cfg.extra or {}
        self.sparse_ratio = float(ex.get("topk_ratio", 2.0))  # 1.5 / 2.0 / 3.0
        self.kv_ratio = float(ex.get("kv_ratio", 3.0))
        self.local_range = int(ex.get("local_range", 11))
        self.color_fix = bool(ex.get("color_fix", True))
        self.is_full_block = bool(ex.get("is_full_block", False))
        self.if_buffer = bool(ex.get("if_buffer", True))

    # ---------------- lifecycle ----------------
    def load(self) -> None:
        import torch

        repo = _resolve_repo_dir(self.cfg.extra)
        wan = _ensure_repo_on_path(repo)
        self.repo_dir = repo

        for cand in ("FlashVSR-v1.1", "FlashVSR"):
            wdir = wan / cand
            if (wdir / "diffusion_pytorch_model_streaming_dmd.safetensors").exists():
                self.weights_dir = wdir
                break
        if self.weights_dir is None:
            raise FileNotFoundError(
                f"FlashVSR weights not found under {wan}. Run "
                "`python scripts/download_weights.py --model flashvsr`."
            )

        from diffsynth import ModelManager, FlashVSRTinyPipeline  # type: ignore
        from utils.utils import Causal_LQ4x_Proj  # type: ignore
        from utils.TCDecoder import build_tcdecoder  # type: ignore

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(self.cfg.dtype, torch.bfloat16)

        mm = ModelManager(torch_dtype=torch_dtype, device="cpu")
        mm.load_models([str(self.weights_dir / "diffusion_pytorch_model_streaming_dmd.safetensors")])
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=self.cfg.device)

        # LQ projection
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
            in_dim=3, out_dim=1536, layer_num=1
        ).to(self.cfg.device, dtype=torch_dtype)
        lq_proj_state = torch.load(
            str(self.weights_dir / "LQ_proj_in.ckpt"), map_location="cpu"
        )
        pipe.denoising_model().LQ_proj_in.load_state_dict(lq_proj_state, strict=True)
        pipe.denoising_model().LQ_proj_in.to(self.cfg.device)

        # TCDecoder (replaces VAE in tiny mode)
        pipe.TCDecoder = build_tcdecoder(
            new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768
        )
        tc_state = torch.load(str(self.weights_dir / "TCDecoder.ckpt"))
        pipe.TCDecoder.load_state_dict(tc_state, strict=False)

        pipe.to(self.cfg.device)
        pipe.enable_vram_management(num_persistent_param_in_dit=None)

        ctx = torch.load(
            str(wan / "prompt_tensor" / "posi_prompt.pth"),
            map_location=self.cfg.device,
        )
        pipe.init_cross_kv(context_tensor=ctx)
        pipe.load_models_to_device(["dit", "vae"])

        if self.cfg.quant in ("int8_woq", "int8"):
            self._apply_int8_woq(pipe)

        self.pipe = pipe

    def _apply_int8_woq(self, pipe) -> None:
        try:
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
                    q = Linear8bitLt(
                        child.in_features, child.out_features,
                        bias=child.bias is not None,
                        has_fp16_weights=False, threshold=6.0,
                    )
                    q.load_state_dict(child.state_dict())
                    q = q.to(self.cfg.device)
                    setattr(module, name, q)
                else:
                    _swap(child)

        for blk_name in ("dit", "transformer", "denoiser"):
            blk = getattr(pipe, blk_name, None)
            if blk is not None:
                _swap(blk)

    # ---------------- inference ----------------
    def upscale(self, frames: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        assert self.pipe is not None, "call .load() first"
        n, h, w, _ = frames.shape
        device = self.cfg.device
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(self.cfg.dtype, torch.bfloat16)

        # 1) Compute SR target dims (128-multiple)
        tw, th = _compute_target_dims(w, h, scale=self.native_scale,
                                      multiple=_SPATIAL_MULTIPLE)

        # 2) Bicubic-upsample each LR frame to (tw, th), convert to [-1, 1]
        upsampled = []
        for f in frames:
            img = Image.fromarray(f).resize((tw, th), Image.BICUBIC)
            arr = np.asarray(img, np.uint8)
            t = torch.from_numpy(arr).to(device, dtype=torch.float32)
            t = t.permute(2, 0, 1).div_(255.0).mul_(2.0).sub_(1.0).to(torch_dtype)
            upsampled.append(t)

        # 3) Pad to 8n+1 frame count (matches upstream)
        if len(upsampled) == 0:
            return np.zeros((0, th, tw, 3), dtype=np.uint8)
        padded = upsampled + [upsampled[-1]] * 4
        F = _largest_8np1_leq(len(padded))
        if F == 0:
            raise RuntimeError("not enough frames after pad to reach 8n+1")
        padded = padded[:F]

        # 4) Pack and call pipeline
        vid = torch.stack(padded, 0).permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, F, H, W)
        topk = self.sparse_ratio * 768 * 1280 / (th * tw)
        with torch.inference_mode():
            out = self.pipe(
                prompt="", negative_prompt="", cfg_scale=1.0,
                num_inference_steps=1, seed=0,
                LQ_video=vid, num_frames=F, height=th, width=tw,
                is_full_block=self.is_full_block, if_buffer=self.if_buffer,
                topk_ratio=topk, kv_ratio=self.kv_ratio,
                local_range=self.local_range, color_fix=self.color_fix,
            )

        # 5) Convert (C, T, H, W) in [-1,1] -> (T, H, W, C) uint8
        out = out.detach().float().add_(1.0).mul_(127.5).clamp_(0, 255).byte()
        out = out.permute(1, 2, 3, 0).cpu().numpy()
        # 6) Crop to original frame count
        return out[:n]

    def close(self) -> None:
        import torch

        self.pipe = None
        torch.cuda.empty_cache()
