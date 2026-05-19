"""Frame skipping + (optional) RIFE-based gap fill.

Strategy:
- Run the SR model on every Nth frame (N = ``skip``).
- For the dropped frames, either repeat the last anchor (cheapest) or
  interpolate via RIFE in SR space (better quality at modest cost).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import numpy as np


@dataclass
class SkipConfig:
    skip: int = 1                # 1 = no skip; 2 = every 2nd; ...
    interp: str = "none"         # none|repeat|rife


def select_anchors(idx: int, skip: int) -> bool:
    return (idx % max(1, skip)) == 0


def fill_repeat(anchors_sr: np.ndarray, total: int, skip: int) -> np.ndarray:
    """Repeat each SR anchor ``skip`` times until ``total`` frames produced."""
    out: List[np.ndarray] = []
    for a in anchors_sr:
        out.extend([a] * skip)
        if len(out) >= total:
            return np.stack(out[:total], axis=0)
    while len(out) < total:
        out.append(anchors_sr[-1])
    return np.stack(out[:total], axis=0)


def fill_rife(anchors_sr: np.ndarray, total: int, skip: int, rife) -> np.ndarray:
    """Use RIFE to interpolate the dropped frames in SR space."""
    if rife is None or skip <= 1 or len(anchors_sr) < 2:
        return fill_repeat(anchors_sr, total, skip)
    dense = rife.fill_gaps(anchors_sr, gap=skip)
    if len(dense) >= total:
        return dense[:total]
    pad = np.repeat(dense[-1:], total - len(dense), axis=0)
    return np.concatenate([dense, pad], axis=0)
