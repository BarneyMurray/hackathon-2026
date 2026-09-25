"""EOT (Expectation over Transformation) sampling + differentiable patch compositing.

Patch pixels are treated as an "input" image; the canonical background is the
"output" of a grid_sample warp. `apply_patch` builds an affine theta that maps
canonical (output) normalized coordinates back into the patch's (input) normalized
coordinate frame, so a small patch tensor can be rotated, scaled, and translated
into a large canvas in one differentiable op -- both the patch RGB and its mask
are warped with the identical grid, so compositing is a simple alpha-blend after.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from patchattack.patch import Patch


@dataclass
class EOTConfig:
    canonical_size: int = 518
    rotation_deg: tuple[float, float] = (-45.0, 45.0)
    area_frac_range: tuple[float, float] = (0.01, 0.20)  # of canonical image area
    log_uniform_area: bool = True
    # optional corner anchor (fx, fy), each in [0, 1]: fx=0/1 -> left/right edge,
    # fy=0/1 -> top/bottom edge. The patch center is placed `corner_margin_px` + its own
    # half-extent in from that corner -- i.e. tangent to the corner, fully in-frame -- so as
    # area_frac grows the center slides diagonally toward the image center automatically,
    # rather than clipping or (worse) collapsing to one fixed pixel. A small amount of jitter
    # (a fraction of half_extent) is added around that anchor for placement realism.
    corner: tuple[float, float] | None = None
    corner_margin_px: float = 8.0
    corner_jitter_frac: float = 0.15
    # physical-world EOT (print + camera): off by default so existing runs/evals are unchanged.
    # squash_min < 1 foreshortens the patch along a random in-plane axis (a cheap affine stand-in
    # for viewing a flat sticker at an angle); print_color_jitter randomizes a per-sample
    # gain/bias on the patch's own pixels (printer gamut, paper white, ink density).
    squash_min: float = 1.0
    print_color_jitter: float = 0.0


BOTTOM_LEFT_CORNER = (0.0, 1.0)  # fx=0 (left edge), fy=1 (bottom edge)


def sample_eot_params(batch_size: int, cfg: EOTConfig, device) -> dict[str, torch.Tensor]:
    lo_deg, hi_deg = cfg.rotation_deg
    angle_deg = torch.empty(batch_size, device=device).uniform_(lo_deg, hi_deg)
    angle = angle_deg * math.pi / 180.0

    lo_a, hi_a = cfg.area_frac_range
    if cfg.log_uniform_area:
        log_lo, log_hi = math.log(lo_a), math.log(hi_a)
        area_frac = torch.exp(torch.empty(batch_size, device=device).uniform_(log_lo, log_hi))
    else:
        area_frac = torch.empty(batch_size, device=device).uniform_(lo_a, hi_a)

    # rendered circular-patch diameter in canonical pixels, s.t. circle area = area_frac * canvas area
    patch_side_px = cfg.canonical_size * torch.sqrt(4.0 * area_frac / math.pi)

    cos_a = torch.cos(angle).abs()
    sin_a = torch.sin(angle).abs()
    half_extent = (patch_side_px / 2.0) * (cos_a + sin_a)
    half_extent = torch.clamp(half_extent, max=cfg.canonical_size / 2.0 - 1e-3)

    lo_c = half_extent
    hi_c = cfg.canonical_size - half_extent

    if cfg.corner is not None:
        fx, fy = cfg.corner
        margin = cfg.corner_margin_px
        # anchor tangent to the corner + margin, inset by half_extent so the whole patch
        # stays in-frame; as half_extent grows this slides diagonally toward image center.
        cx_anchor = margin + half_extent if fx < 0.5 else cfg.canonical_size - margin - half_extent
        cy_anchor = margin + half_extent if fy < 0.5 else cfg.canonical_size - margin - half_extent

        jitter_x = (torch.rand(batch_size, device=device) * 2 - 1) * cfg.corner_jitter_frac * half_extent
        jitter_y = (torch.rand(batch_size, device=device) * 2 - 1) * cfg.corner_jitter_frac * half_extent
        cx = torch.clamp(cx_anchor + jitter_x, lo_c, hi_c)
        cy = torch.clamp(cy_anchor + jitter_y, lo_c, hi_c)
    else:
        u = torch.rand(batch_size, device=device)
        cx = lo_c + u * (hi_c - lo_c)
        u = torch.rand(batch_size, device=device)
        cy = lo_c + u * (hi_c - lo_c)

    params = {
        "angle": angle,
        "patch_side_px": patch_side_px,
        "cx": cx,
        "cy": cy,
        "area_frac": area_frac,
    }
    if cfg.squash_min < 1.0:
        params["squash"] = torch.empty(batch_size, device=device).uniform_(cfg.squash_min, 1.0)
    if cfg.print_color_jitter > 0:
        j = cfg.print_color_jitter
        # printed ink never hits pure black/white: compress contrast, then per-channel tint
        params["color_gain"] = torch.empty(batch_size, 3, 1, 1, device=device).uniform_(1 - 2 * j, 1 - j)
        params["color_bias"] = torch.empty(batch_size, 3, 1, 1, device=device).uniform_(0.0, j)
    return params


def fixed_eot_params(
    batch_size: int, canonical_size: int, area_frac: float, angle_deg: float, cx: float, cy: float, device
) -> dict[str, torch.Tensor]:
    """Deterministic params for the visual placement sanity check / fixed-size eval sweeps."""
    angle = torch.full((batch_size,), angle_deg * math.pi / 180.0, device=device)
    area_frac_t = torch.full((batch_size,), area_frac, device=device)
    patch_side_px = canonical_size * torch.sqrt(4.0 * area_frac_t / math.pi)
    cx_t = torch.full((batch_size,), float(cx), device=device)
    cy_t = torch.full((batch_size,), float(cy), device=device)
    return {"angle": angle, "patch_side_px": patch_side_px, "cx": cx_t, "cy": cy_t, "area_frac": area_frac_t}


def build_affine_theta(params: dict[str, torch.Tensor], canonical_size: int) -> torch.Tensor:
    angle = params["angle"]
    patch_side_px = params["patch_side_px"]
    cx, cy = params["cx"], params["cy"]
    B = angle.shape[0]

    s = patch_side_px / canonical_size  # patch-normalized-unit -> canonical-normalized-unit scale
    inv_s = 1.0 / s
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # canonical-normalized center in [-1, 1]
    ncx = 2.0 * cx / canonical_size - 1.0
    ncy = 2.0 * cy / canonical_size - 1.0

    theta = torch.zeros(B, 2, 3, device=angle.device, dtype=angle.dtype)
    theta[:, 0, 0] = inv_s * cos_a
    theta[:, 0, 1] = inv_s * sin_a
    theta[:, 1, 0] = -inv_s * sin_a
    theta[:, 1, 1] = inv_s * cos_a
    if "squash" in params:
        # shrink the patch along its own x axis: input-x row of the output->input map grows by 1/squash
        theta[:, 0, :2] = theta[:, 0, :2] / params["squash"].unsqueeze(-1)
    theta[:, 0, 2] = -(theta[:, 0, 0] * ncx + theta[:, 0, 1] * ncy)
    theta[:, 1, 2] = -(theta[:, 1, 0] * ncx + theta[:, 1, 1] * ncy)
    return theta


def apply_patch(patch: Patch, background: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """background: (B,3,canonical_size,canonical_size) in [0,1]. Returns composite, same shape,
    differentiable w.r.t. patch.raw only (background carries no grad)."""
    B, _, canonical_size, _ = background.shape
    pixels = patch.pixels()  # (3,H,W)
    mask = patch.mask  # (1,H,W)

    patch_rgb = pixels.unsqueeze(0).expand(B, -1, -1, -1)
    patch_mask = mask.unsqueeze(0).expand(B, -1, -1, -1)
    if "color_gain" in params:
        patch_rgb = patch_rgb * params["color_gain"] + params["color_bias"]

    theta = build_affine_theta(params, canonical_size)
    grid = F.affine_grid(theta, size=(B, 3, canonical_size, canonical_size), align_corners=False)

    warped_rgb = F.grid_sample(patch_rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    warped_mask = F.grid_sample(patch_mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    composite = warped_mask * warped_rgb + (1.0 - warped_mask) * background
    return composite


def _gaussian_kernel1d(sigma: float, device) -> torch.Tensor:
    radius = max(1, math.ceil(2.5 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def camera_augment(images: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Differentiable whole-frame camera model applied after compositing: per-sample exposure,
    contrast, white balance, a random defocus blur, and sensor noise. Makes the patch survive
    being printed and re-photographed by a webcam instead of only working as exact pixels.
    images: (B,3,H,W) in [0,1]."""
    if strength <= 0:
        return images
    B, device = images.shape[0], images.device
    s = strength

    def u(lo, hi, *shape):
        return torch.empty(B, *shape, device=device).uniform_(lo, hi)

    brightness = u(1 - 0.3 * s, 1 + 0.3 * s, 1, 1, 1)
    contrast = u(1 - 0.3 * s, 1 + 0.2 * s, 1, 1, 1)
    white_balance = u(1 - 0.08 * s, 1 + 0.08 * s, 3, 1, 1)
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    x = ((images - mean) * contrast + mean) * brightness * white_balance

    # one blur sigma per batch keeps it a single separable conv; the batch is re-sampled every step
    sigma = float(torch.empty(1).uniform_(0.0, 1.5 * s))
    if sigma > 0.3:
        k = _gaussian_kernel1d(sigma, device)
        r = k.numel() // 2
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
        x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)

    x = x + torch.randn_like(x) * u(0.0, 0.03 * s, 1, 1, 1)
    return x.clamp(0, 1)
