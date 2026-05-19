"""Optional SageAttention SDPA monkeypatch.

SageAttention provides a drop-in replacement for ``torch.nn.functional.
scaled_dot_product_attention`` that is meaningfully faster on Ampere /
Ada / Blackwell. Linux only.

We patch lazily inside ``enable()`` so importing this module never crashes
on machines without the kernel installed. If the import fails we leave the
default SDPA in place and return ``False``.
"""
from __future__ import annotations

import os
import warnings
from typing import Callable, Optional

import torch

_ORIG_SDPA: Optional[Callable] = None
_PATCHED: bool = False


def is_available() -> bool:
    try:
        import sageattention  # noqa: F401
        return True
    except Exception:
        return False


def enable(force: bool = False) -> bool:
    """Monkeypatch SDPA -> SageAttention. Idempotent. Returns success."""
    global _ORIG_SDPA, _PATCHED
    if _PATCHED and not force:
        return True
    try:
        from sageattention import sageattn  # type: ignore
    except Exception as e:  # pragma: no cover - env-dependent
        warnings.warn(f"SageAttention not available ({e}); using torch SDPA.")
        return False

    F = torch.nn.functional
    if _ORIG_SDPA is None:
        _ORIG_SDPA = F.scaled_dot_product_attention

    def _sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        if attn_mask is not None or dropout_p:
            return _ORIG_SDPA(q, k, v, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        try:
            return sageattn(q, k, v, is_causal=is_causal, sm_scale=scale)
        except Exception:
            return _ORIG_SDPA(q, k, v, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    F.scaled_dot_product_attention = _sdpa  # type: ignore[assignment]
    os.environ.setdefault("SAGE_ATTN_ACTIVE", "1")
    _PATCHED = True
    return True


def disable() -> None:
    global _PATCHED
    if _ORIG_SDPA is None:
        return
    torch.nn.functional.scaled_dot_product_attention = _ORIG_SDPA  # type: ignore
    _PATCHED = False
    os.environ.pop("SAGE_ATTN_ACTIVE", None)
