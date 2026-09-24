"""The trainable adversarial patch: sigmoid-parameterized pixels + a soft circular mask."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


def make_circular_mask(size: int, feather_px: float = 2.0) -> torch.Tensor:
    """Returns (1, size, size) mask in [0,1]: ~1 inside the circle, ~0 outside,
    with a soft linear feather of width feather_px at the boundary."""
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    center = (size - 1) / 2.0
    radius = size / 2.0
    dist = torch.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    mask = torch.clamp((radius - dist) / feather_px + 0.5, 0.0, 1.0)
    return mask.unsqueeze(0)


class Patch(nn.Module):
    def __init__(
        self,
        size: int = 200,
        shape: Literal["circle", "square"] = "circle",
        init: Literal["random", "gray"] = "random",
        marker: bool = False,
    ):
        super().__init__()
        self.size = size
        self.shape = shape
        if init == "random":
            raw = torch.randn(3, size, size) * 0.5
        else:
            raw = torch.zeros(3, size, size)
        if marker:
            _bake_orientation_marker(raw, size)
        self.raw = nn.Parameter(raw)

        if shape == "circle":
            mask = make_circular_mask(size)
        else:
            mask = torch.ones(1, size, size)
        self.register_buffer("mask", mask)

    def pixels(self) -> torch.Tensor:
        return torch.sigmoid(self.raw)


def _bake_orientation_marker(raw: torch.Tensor, size: int) -> None:
    """Paints a bright arrow-like mark pointing 'up' and off-center, purely so the
    placement/compositing pipeline can be visually sanity-checked (see scripts/sanity_check.py)."""
    raw.fill_(-3.0)  # dark background after sigmoid
    top = size // 6
    bottom = size - size // 6
    left = size // 2 - 2
    right = size // 2 + 2
    raw[:, top:bottom, left:right] = 3.0  # bright vertical bar -> "up" marker
    off_top = size // 6
    off_bottom = size // 6 + size // 8
    off_left = size // 2 + size // 6
    off_right = off_left + size // 8
    raw[0, off_top:off_bottom, off_left:off_right] = 3.0  # red-ish blob, off-center to the right
    raw[1, off_top:off_bottom, off_left:off_right] = -3.0
    raw[2, off_top:off_bottom, off_left:off_right] = -3.0
