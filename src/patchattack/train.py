"""EOT training loop for the embedding-space universal adversarial patch."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from patchattack.data import make_background_loader
from patchattack.models import load_ensemble
from patchattack.patch import Patch
from patchattack.reference_embeddings import BANANA_REFS, load_or_compute_training_target
from patchattack.transforms import BOTTOM_LEFT_CORNER, EOTConfig, apply_patch, sample_eot_params
from patchattack.viz import plot_training_curves, save_patch_preview

OUT_ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "patches"

ALL_MODELS = ["resnet18", "mobilenet_v3_large", "efficientnet_b0", "dinov2_vits14", "dinov2_vitb14"]


@dataclass
class TrainConfig:
    run_name: str = "run"
    ensemble: list[str] = field(default_factory=lambda: list(ALL_MODELS))
    patch_size: int = 200
    patch_shape: str = "circle"
    canonical_size: int = 518
    batch_size: int = 16
    steps: int = 2000
    lr: float = 0.01
    eot: EOTConfig = field(default_factory=EOTConfig)
    log_every: int = 20
    preview_every: int = 200
    device: str = "cuda"
    fixed_area_frac: float | None = None  # sanity-check mode: pin area_frac instead of EOT sweep
    target_name: str = "banana"
    target_refs_dir: str = str(BANANA_REFS)
    background_dataset: str = "imagenette"
    val_every: int = 0  # 0 disables held-out validation checks; e.g. 600 to check every 600 steps
    val_batches: int = 4  # batches averaged per validation check (variance reduction on small pools)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def train(cfg: TrainConfig) -> Path:
    device = cfg.device if torch.cuda.is_available() else "cpu"
    out_dir = OUT_ROOT / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{cfg.run_name}] loading ensemble: {cfg.ensemble}")
    models = load_ensemble(cfg.ensemble, device=device)

    print(f"[{cfg.run_name}] computing/loading training-target centroids for target={cfg.target_name!r}")
    targets = {
        name: load_or_compute_training_target(
            name, model, device, target_name=cfg.target_name, target_refs_dir=Path(cfg.target_refs_dir)
        )
        for name, model in models.items()
    }

    patch = Patch(cfg.patch_size, cfg.patch_shape).to(device)
    opt = torch.optim.Adam([patch.raw], lr=cfg.lr)

    loader = make_background_loader(cfg.canonical_size, cfg.batch_size, split="train",
                                     dataset=cfg.background_dataset)
    loader_iter = cycle(loader)

    val_loader_iter = None
    if cfg.val_every > 0:
        val_loader = make_background_loader(cfg.canonical_size, cfg.batch_size, split="val",
                                             dataset=cfg.background_dataset, num_workers=2)
        val_loader_iter = cycle(val_loader)

    eot_cfg = cfg.eot
    log_rows = []
    val_log_rows = []

    for step in range(cfg.steps):
        bg = next(loader_iter).to(device)
        B = bg.shape[0]

        if cfg.fixed_area_frac is not None:
            # sanity-check mode: keep random rotation + translation from the normal EOT
            # sampler, but pin area_frac to a fixed value (translation bounds are derived
            # from this fixed area too, so placement stays correctly in-frame) so convergence
            # is faster/easier to read than under the full log-uniform sweep.
            fixed_cfg = EOTConfig(
                canonical_size=eot_cfg.canonical_size,
                rotation_deg=eot_cfg.rotation_deg,
                area_frac_range=(cfg.fixed_area_frac, cfg.fixed_area_frac),
                log_uniform_area=False,
                corner=eot_cfg.corner, corner_margin_px=eot_cfg.corner_margin_px, corner_jitter_frac=eot_cfg.corner_jitter_frac,
            )
            params = sample_eot_params(B, fixed_cfg, device)
        else:
            params = sample_eot_params(B, eot_cfg, device)

        composite = apply_patch(patch, bg, params)

        per_model_loss = {}
        for name, model in models.items():
            emb = model.embed(composite)
            emb = F.normalize(emb, dim=-1)
            cos_sim = (emb * targets[name]).sum(dim=-1).clamp(-1, 1)
            per_model_loss[name] = 1.0 - cos_sim.mean()

        loss = sum(per_model_loss.values()) / len(per_model_loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            row = {"step": step, "loss": loss.item()}
            for name, l in per_model_loss.items():
                row[f"cos_sim_{name}"] = 1.0 - l.item()
            log_rows.append(row)
            print(f"[{cfg.run_name}] step {step:5d}  loss {loss.item():.4f}  " +
                  "  ".join(f"{n}={1 - l.item():.3f}" for n, l in per_model_loss.items()))

        if step % cfg.preview_every == 0 or step == cfg.steps - 1:
            save_patch_preview(patch, out_dir / f"preview_step{step:05d}.png")

        if val_loader_iter is not None and (step % cfg.val_every == 0 or step == cfg.steps - 1):
            with torch.no_grad():
                val_per_model_loss = {name: 0.0 for name in models}
                for _ in range(cfg.val_batches):
                    val_bg = next(val_loader_iter).to(device)
                    val_params = sample_eot_params(val_bg.shape[0], eot_cfg, device)
                    val_composite = apply_patch(patch, val_bg, val_params)
                    for name, model in models.items():
                        emb = model.embed(val_composite)
                        emb = F.normalize(emb, dim=-1)
                        cos_sim = (emb * targets[name]).sum(dim=-1).clamp(-1, 1)
                        val_per_model_loss[name] += (1.0 - cos_sim.mean()).item() / cfg.val_batches
                val_loss = sum(val_per_model_loss.values()) / len(val_per_model_loss)
            val_row = {"step": step, "val_loss": val_loss}
            for name, l in val_per_model_loss.items():
                val_row[f"val_cos_sim_{name}"] = 1.0 - l
            val_log_rows.append(val_row)
            print(f"[{cfg.run_name}] step {step:5d}  VAL  loss {val_loss:.4f}  " +
                  "  ".join(f"{n}={1 - l:.3f}" for n, l in val_per_model_loss.items()))

    df = pd.DataFrame(log_rows)
    df.to_csv(out_dir / "train_log.csv", index=False)
    if val_log_rows:
        pd.DataFrame(val_log_rows).to_csv(out_dir / "val_log.csv", index=False)
    torch.save({"raw": patch.raw.detach().cpu(), "cfg": asdict(cfg)}, out_dir / "patch.pt")
    plot_training_curves(out_dir / "train_log.csv", out_dir / "train_curves.png",
                          val_csv_path=out_dir / "val_log.csv" if val_log_rows else None)
    print(f"[{cfg.run_name}] done -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--ensemble", nargs="+", default=ALL_MODELS)
    ap.add_argument("--patch-size", type=int, default=200)
    ap.add_argument("--canonical-size", type=int, default=518)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--area-min", type=float, default=0.01)
    ap.add_argument("--area-max", type=float, default=0.20)
    ap.add_argument("--fixed-area-frac", type=float, default=None)
    ap.add_argument("--corner", action="store_true",
                     help="constrain patch placement to a fixed bottom-left region instead of "
                          "anywhere in frame (e.g. sticker placed on the checkout surface next "
                          "to items, not on the items themselves)")
    ap.add_argument("--target-name", default="banana")
    ap.add_argument("--target-refs-dir", type=str, default=str(BANANA_REFS))
    ap.add_argument("--background-dataset", choices=["imagenette", "coco", "food_items"], default="imagenette")
    ap.add_argument("--val-every", type=int, default=0,
                     help="run a held-out validation check every N steps (0 disables); "
                          "useful to catch overfitting on small background pools")
    ap.add_argument("--val-batches", type=int, default=4)
    args = ap.parse_args()

    cfg = TrainConfig(
        run_name=args.run_name,
        ensemble=args.ensemble,
        patch_size=args.patch_size,
        canonical_size=args.canonical_size,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        eot=EOTConfig(
            canonical_size=args.canonical_size,
            area_frac_range=(args.area_min, args.area_max),
            corner=BOTTOM_LEFT_CORNER if args.corner else None,
        ),
        fixed_area_frac=args.fixed_area_frac,
        target_name=args.target_name,
        target_refs_dir=args.target_refs_dir,
        background_dataset=args.background_dataset,
        val_every=args.val_every,
        val_batches=args.val_batches,
    )
    train(cfg)


if __name__ == "__main__":
    main()
