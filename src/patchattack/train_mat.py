"""Train an optimized grayscale "mat" that a frozen real item photo sits on, so the whole
scene reads as a target class -- the inverse of train.py's patch-on-background attack.
Target is a single reference image's embedding (not a centroid over many references).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from patchattack.mat import Mat, apply_mat, apply_random_occlusion
from patchattack.models import load_ensemble
from patchattack.viz import save_tensor_image

OUT_ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "mats"
ALL_MODELS = ["resnet18", "mobilenet_v3_large", "efficientnet_b0", "dinov2_vits14", "dinov2_vitb14"]


def load_square(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tf = v2.Compose([
        v2.Resize(size, antialias=True),
        v2.CenterCrop(size),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return tf(v2.functional.pil_to_tensor(img))


def train_mat(
    run_name: str,
    item_path: Path | None,
    target_path: Path,
    ensemble_names: list[str],
    canonical_size: int = 518,
    item_frac: float = 0.6,
    steps: int = 2000,
    lr: float = 0.02,
    lightness_weight: float = 0.3,
    min_lightness: float = 0.0,
    robust_overlay: bool = False,
    overlay_area_range: tuple[float, float] = (0.03, 0.30),
    log_every: int = 20,
    preview_every: int = 20,
    device: str = "cuda",
) -> Path:
    device = device if torch.cuda.is_available() else "cpu"
    out_dir = OUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{run_name}] loading ensemble: {ensemble_names}")
    models = load_ensemble(ensemble_names, device=device)

    item_image = load_square(item_path, canonical_size).to(device) if item_path is not None else None
    target_image = load_square(target_path, canonical_size).to(device)

    print(f"[{run_name}] computing target embedding from single reference image: {target_path.name}")
    targets = {}
    with torch.no_grad():
        for name, model in models.items():
            emb = model.embed(target_image.unsqueeze(0))
            targets[name] = F.normalize(emb, dim=-1).squeeze(0)

    mat = Mat(canonical_size, init="light", min_lightness=min_lightness).to(device)
    opt = torch.optim.Adam([mat.raw], lr=lr)

    print(f"[{run_name}] mode: {'robust random-overlay (no frozen item)' if robust_overlay else 'frozen item'}")

    log_rows = []
    for step in range(steps):
        if robust_overlay:
            composite = apply_random_occlusion(mat, overlay_area_range, device=device).unsqueeze(0)
        else:
            composite = apply_mat(mat, item_image, item_frac).unsqueeze(0)  # (1,3,H,W)

        per_model_loss = {}
        for name, model in models.items():
            emb = model.embed(composite)
            emb = F.normalize(emb, dim=-1)
            cos_sim = (emb * targets[name]).sum(dim=-1).clamp(-1, 1)
            per_model_loss[name] = 1.0 - cos_sim.mean()

        attack_loss = sum(per_model_loss.values()) / len(per_model_loss)
        light_loss = mat.lightness_penalty()
        loss = attack_loss + lightness_weight * light_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            row = {"step": step, "loss": loss.item(), "attack_loss": attack_loss.item(),
                   "light_loss": light_loss.item()}
            for name, l in per_model_loss.items():
                row[f"cos_sim_{name}"] = 1.0 - l.item()
            log_rows.append(row)
            print(f"[{run_name}] step {step:5d}  loss={loss.item():.4f}  "
                  f"attack={attack_loss.item():.4f}  light={light_loss.item():.4f}  " +
                  "  ".join(f"{n}={1 - l.item():.3f}" for n, l in per_model_loss.items()))

        if step % preview_every == 0 or step == steps - 1:
            save_tensor_image(mat.pixels(), out_dir / f"mat_step{step:05d}.png")
            save_tensor_image(composite[0], out_dir / f"composite_step{step:05d}.png")
            if robust_overlay:
                # also save a couple of fresh random-overlay samples so previews show
                # variety, not just whatever the last training-step overlay happened to be
                with torch.no_grad():
                    for k in range(2):
                        sample = apply_random_occlusion(mat, overlay_area_range, device=device)
                        save_tensor_image(sample, out_dir / f"overlay_sample_step{step:05d}_{k}.png")

    df = pd.DataFrame(log_rows)
    df.to_csv(out_dir / "train_log.csv", index=False)
    torch.save({
        "raw": mat.raw.detach().cpu(),
        "canonical_size": canonical_size,
        "item_frac": item_frac,
        "item_path": str(item_path),
        "target_path": str(target_path),
        "ensemble": ensemble_names,
        "min_lightness": min_lightness,
    }, out_dir / "mat.pt")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["step"], df["attack_loss"], label="attack loss")
    axes[0].plot(df["step"], df["light_loss"], label="lightness penalty")
    axes[0].plot(df["step"], df["loss"], label="total loss", linestyle="--")
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("step")
    axes[0].set_title("training loss")
    cos_cols = [c for c in df.columns if c.startswith("cos_sim_")]
    for c in cos_cols:
        axes[1].plot(df["step"], df[c], label=c.replace("cos_sim_", ""))
    axes[1].legend(fontsize=7)
    axes[1].set_xlabel("step")
    axes[1].set_title("per-model similarity to target")
    fig.tight_layout()
    fig.savefig(out_dir / "train_curves.png", dpi=130)
    plt.close(fig)

    print(f"[{run_name}] done -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="mat_run")
    ap.add_argument("--item-image", type=Path, default=None,
                     help="frozen item photo to place in the center (omit with --robust-overlay)")
    ap.add_argument("--target-image", type=Path, required=True)
    ap.add_argument("--ensemble", nargs="+", default=ALL_MODELS)
    ap.add_argument("--canonical-size", type=int, default=518)
    ap.add_argument("--item-frac", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lightness-weight", type=float, default=0.3)
    ap.add_argument("--min-lightness", type=float, default=0.0,
                     help="hard floor on pixel brightness in [0,1) -- e.g. 0.5 makes the mat "
                          "unable to render darker than mid-gray, by construction")
    ap.add_argument("--robust-overlay", action="store_true",
                     help="no frozen item -- instead optimize the full canvas and, every "
                          "step, composite a fresh random solid-color square occlusion at a "
                          "random position/size, so the mat is robust to something unknown "
                          "being placed on it, rather than tuned to one fixed scene")
    ap.add_argument("--overlay-area-min", type=float, default=0.03)
    ap.add_argument("--overlay-area-max", type=float, default=0.30)
    args = ap.parse_args()

    if not args.robust_overlay and args.item_image is None:
        ap.error("--item-image is required unless --robust-overlay is set")

    train_mat(
        run_name=args.run_name,
        item_path=args.item_image,
        target_path=args.target_image,
        ensemble_names=args.ensemble,
        canonical_size=args.canonical_size,
        item_frac=args.item_frac,
        steps=args.steps,
        lr=args.lr,
        lightness_weight=args.lightness_weight,
        min_lightness=args.min_lightness,
        robust_overlay=args.robust_overlay,
        overlay_area_range=(args.overlay_area_min, args.overlay_area_max),
    )


if __name__ == "__main__":
    main()
