"""Small visualization helpers: save patch/composite tensors as PNGs, plot training curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.transforms.functional import to_pil_image

from patchattack.patch import Patch


def save_patch_preview(patch: Patch, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = patch.pixels().detach().cpu()
    mask = patch.mask.detach().cpu()
    rgba = torch.cat([pixels, mask], dim=0)  # (4,H,W)
    to_pil_image(rgba).save(path)


def save_tensor_image(x: torch.Tensor, path: Path) -> None:
    """x: (3,H,W) in [0,1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(x.detach().cpu().clamp(0, 1)).save(path)


def plot_training_curves(csv_path: Path, out_path: Path, val_csv_path: Path | None = None) -> None:
    df = pd.read_csv(csv_path)
    val_df = None
    if val_csv_path is not None and Path(val_csv_path).exists():
        val_df = pd.read_csv(val_csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["step"], df["loss"], label="train")
    if val_df is not None:
        axes[0].plot(val_df["step"], val_df["val_loss"], label="val", marker="o", linestyle="--")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("mean loss (1 - cos sim)")
    axes[0].set_title("training loss" + (" (train vs val)" if val_df is not None else ""))

    cos_cols = [c for c in df.columns if c.startswith("cos_sim_")]
    for c in cos_cols:
        axes[1].plot(df["step"], df[c], label=c.replace("cos_sim_", ""))
    if val_df is not None:
        val_cos_cols = [c for c in val_df.columns if c.startswith("val_cos_sim_")]
        colors = [line.get_color() for line in axes[1].lines]
        for c, color in zip(val_cos_cols, colors):
            axes[1].plot(val_df["step"], val_df[c], linestyle="--", marker="o", color=color,
                          alpha=0.6, markersize=4)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("cosine similarity to target")
    axes[1].set_title("per-model similarity (dashed = val)" if val_df is not None else "per-model similarity")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_success_vs_area(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, g in df.groupby("model"):
        g = g.sort_values("area_frac")
        ax.plot(g["area_frac"] * 100, g["success_rate"] * 100, marker="o", label=model_name)
    ax.set_xlabel("patch area as % of image")
    ax.set_ylabel("attack success rate (%)")
    ax.set_title("Attack success rate by patch size")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="~3cm on checkout plate (est.)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_vlm_hit_rate(df: pd.DataFrame, out_path: Path, target: str) -> None:
    """One line per VLM: solid = adversarial patch, dashed = random-noise patch of the
    same size/placement, dotted horizontal = clean-image baseline."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for vlm, g in df.groupby("vlm"):
        label = vlm.split("/")[-1]
        adv = g[g["condition"] == "adversarial"].sort_values("area_frac")
        (line,) = ax.plot(adv["area_frac"] * 100, adv["hit_rate"] * 100, marker="o", label=label)
        rnd = g[g["condition"] == "random_patch"].sort_values("area_frac")
        ax.plot(rnd["area_frac"] * 100, rnd["hit_rate"] * 100, marker="x", linestyle="--",
                color=line.get_color(), alpha=0.6)
        clean = g[g["condition"] == "clean"]["hit_rate"]
        if len(clean):
            ax.axhline(clean.iloc[0] * 100, color=line.get_color(), linestyle=":", linewidth=1)
    ax.set_xlabel("patch area as % of image")
    ax.set_ylabel(f"answers mentioning '{target}' (%)")
    ax.set_title("VLM transfer (solid = adversarial, dashed = random patch, dotted = clean)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
