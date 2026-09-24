"""Inverted attack: instead of optimizing a small patch placed on a frozen background,
freeze a real item image in the center of the frame and optimize the surrounding "mat"
(everything outside the item) so the whole scene reads as the target class. Motivated by
a real-world detectability problem: a small adversarial sticker on the item itself gets
flagged as "an extra object that shouldn't be there" -- a mat the item sits on is a much
more innocuous, plausible physical object (a placemat/tray liner), so it's a more
deployable attack surface even though the optimized region is much larger.

Two constraints requested for physical plausibility/stealth:
- Grayscale: enforced by construction (parameterize a single channel, broadcast to RGB)
  rather than as a soft penalty -- guarantees exact R=G=B at every pixel, no drift.
- Light/subtle bias: a soft penalty in the training loss (not enforced here) pushing pixel
  values toward white rather than dark/saturated, so the mat looks like a plain, unremarkable
  light-colored surface rather than a conspicuous pattern.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn


class Mat(nn.Module):
    def __init__(self, canonical_size: int, init: str = "light", min_lightness: float = 0.0):
        super().__init__()
        self.canonical_size = canonical_size
        # hard floor on pixel brightness -- e.g. 0.5 means the mat can never render darker
        # than mid-gray, capping worst-case darkness by construction (same reasoning as the
        # grayscale constraint: a guaranteed range beats a soft penalty the optimizer can
        # still trade off against attack loss).
        self.min_lightness = min_lightness
        if init == "light":
            # start near-white (raw such that sigmoid(raw) ~= 0.85) rather than random noise,
            # since we want the optimizer to stay in the light region, not fight its way there.
            raw = torch.full((1, canonical_size, canonical_size), 1.7)
        else:
            raw = torch.randn(1, canonical_size, canonical_size) * 0.1
        self.raw = nn.Parameter(raw)

    def pixels(self) -> torch.Tensor:
        gray = torch.sigmoid(self.raw)  # (1, H, W) in (0,1), single channel
        gray = self.min_lightness + (1.0 - self.min_lightness) * gray  # (1, H, W) in [min_lightness, 1)
        return gray.expand(3, -1, -1)  # (3, H, W), R=G=B by construction

    def lightness_penalty(self) -> torch.Tensor:
        """Mean distance from white -- 0 when the mat is pure white, 1 when pure black."""
        return (1.0 - self.pixels()).mean()


def apply_mat(mat: Mat, item_image: torch.Tensor, item_frac: float) -> torch.Tensor:
    """item_image: (3,h,w) frozen real photo (no grad needed), in [0,1].
    item_frac: fraction of canonical_size the item's longer side occupies, centered.
    Returns (3, canonical_size, canonical_size) composite, differentiable w.r.t. mat only."""
    import torch.nn.functional as F

    canonical_size = mat.canonical_size
    mat_pixels = mat.pixels()

    item_size = int(round(canonical_size * item_frac))
    item_resized = F.interpolate(
        item_image.unsqueeze(0), size=(item_size, item_size), mode="bilinear",
        align_corners=False, antialias=True,
    ).squeeze(0)

    item_padded = torch.zeros(3, canonical_size, canonical_size, device=mat_pixels.device)
    mask = torch.zeros(1, canonical_size, canonical_size, device=mat_pixels.device)
    y0 = (canonical_size - item_size) // 2
    x0 = (canonical_size - item_size) // 2
    item_padded[:, y0:y0 + item_size, x0:x0 + item_size] = item_resized
    mask[:, y0:y0 + item_size, x0:x0 + item_size] = 1.0

    return mask * item_padded + (1.0 - mask) * mat_pixels


def apply_random_occlusion(
    mat: Mat, area_frac_range: tuple[float, float] = (0.03, 0.30), device=None,
) -> torch.Tensor:
    """Composite the mat with ONE random solid-color square placed at a random in-bounds
    position/size, simulating an unknown foreign object placed somewhere on it -- used
    during training (a fresh random occlusion sampled every step) to force the mat to
    read as the target class from its visible parts alone, not just as a single fixed
    frozen scene. No real object dataset needed: a random flat color square is a simple,
    content-agnostic stand-in for "something unknown is covering part of the mat"."""
    canonical_size = mat.canonical_size
    mat_pixels = mat.pixels()
    device = device or mat_pixels.device

    area_frac = math.exp(random.uniform(math.log(area_frac_range[0]), math.log(area_frac_range[1])))
    side = int(round(canonical_size * math.sqrt(area_frac)))
    side = max(4, min(side, canonical_size))
    x0 = random.randint(0, canonical_size - side)
    y0 = random.randint(0, canonical_size - side)
    color = torch.rand(3, device=device)

    mask = torch.zeros(1, canonical_size, canonical_size, device=device)
    mask[:, y0:y0 + side, x0:x0 + side] = 1.0
    fill = torch.zeros(3, canonical_size, canonical_size, device=device)
    fill[:, y0:y0 + side, x0:x0 + side] = color.view(3, 1, 1)

    return mask * fill + (1.0 - mask) * mat_pixels
