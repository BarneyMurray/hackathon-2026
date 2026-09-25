"""Nearest-centroid attack-success evaluation + patch-area sweep, mirroring the
paper's Figure 3 (success rate vs. patch size as % of image)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from patchattack.models import default_device, load_ensemble
from patchattack.patch import Patch
from patchattack.reference_embeddings import build_eval_gallery
from patchattack.transforms import EOTConfig, apply_patch, sample_eot_params

OUT_ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "plots"

DEFAULT_AREA_FRACS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]


def load_patch(patch_path: Path, device: str) -> tuple[Patch, dict]:
    # weights_only=False: these are checkpoints this project generated itself locally
    # (not downloaded from an untrusted source), and the cfg dict can contain plain
    # Python objects (e.g. pathlib.Path) that the default weights_only=True loader
    # refuses to unpickle.
    ckpt = torch.load(patch_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    patch = Patch(cfg["patch_size"], cfg["patch_shape"]).to(device)
    with torch.no_grad():
        patch.raw.copy_(ckpt["raw"].to(device))
    return patch, cfg


@torch.no_grad()
def classify_nearest_centroid(embedding: torch.Tensor, gallery: dict[str, torch.Tensor]) -> list[str]:
    """embedding: (B, D) L2-normalized. Returns list of predicted class names, len B."""
    names = list(gallery.keys())
    centroids = torch.stack([gallery[n] for n in names], dim=0)  # (C, D)
    sims = embedding @ centroids.T  # (B, C)
    idx = sims.argmax(dim=-1)
    return [names[i] for i in idx.tolist()]


@torch.no_grad()
def attack_success_rate(
    model,
    gallery: dict[str, torch.Tensor],
    patch: Patch,
    test_loader,
    area_frac: float,
    canonical_size: int,
    device: str,
    max_images: int | None = None,
    corner=None,
    target_name: str = "banana",
) -> float:
    eot_cfg = EOTConfig(
        canonical_size=canonical_size, area_frac_range=(area_frac, area_frac),
        log_uniform_area=False, corner=corner,
    )
    n_success = 0
    n_total = 0
    # cycle the loader: a small background pool (e.g. the ~19-image food_items val split)
    # would otherwise yield only one batch and stop well short of max_images. Each pass
    # over the same images still contributes new information since patch position/rotation
    # is re-sampled independently per batch (see sample_eot_params).
    import itertools
    batch_iter = itertools.cycle(test_loader) if max_images is not None else test_loader
    for bg in batch_iter:
        bg = bg.to(device)
        B = bg.shape[0]
        params = sample_eot_params(B, eot_cfg, device)
        composite = apply_patch(patch, bg, params)
        emb = model.embed(composite)
        emb = F.normalize(emb, dim=-1)
        preds = classify_nearest_centroid(emb, gallery)
        n_success += sum(1 for p in preds if p == target_name)
        n_total += B
        if max_images is not None and n_total >= max_images:
            break
    return n_success / max(1, n_total)


def sweep_patch_size(
    patch: Patch,
    models: dict,
    galleries: dict[str, dict[str, torch.Tensor]],
    canonical_size: int,
    device: str,
    area_fracs: list[float] = DEFAULT_AREA_FRACS,
    batch_size: int = 16,
    max_images: int = 400,
    corner=None,
    target_name: str = "banana",
    background_dataset: str = "imagenette",
) -> pd.DataFrame:
    from patchattack.data import make_background_loader

    rows = []
    for area_frac in area_fracs:
        for name, model in models.items():
            # fresh loader each time: DataLoader iterators are single-pass
            loader = make_background_loader(canonical_size, batch_size, split="val", num_workers=2,
                                             dataset=background_dataset)
            rate = attack_success_rate(
                model, galleries[name], patch, loader, area_frac, canonical_size, device, max_images,
                corner=corner, target_name=target_name,
            )
            rows.append({"model": name, "area_frac": area_frac, "success_rate": rate})
            print(f"  area={area_frac:.3f}  {name:20s}  success={rate:.3f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-path", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=400)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--eval-ensemble", nargs="+", default=None,
                     help="evaluate against these models instead of the training ensemble "
                          "(e.g. to test black-box transfer to models the patch never saw)")
    ap.add_argument("--background-dataset", choices=["imagenette", "coco", "food_items"], default=None,
                     help="override the eval background dataset; defaults to whatever the patch "
                          "was trained with")
    args = ap.parse_args()

    patch, cfg = load_patch(args.patch_path, args.device)
    ensemble_names = args.eval_ensemble if args.eval_ensemble is not None else cfg["ensemble"]
    print(f"trained on: {cfg['ensemble']}  |  evaluating against: {ensemble_names}")
    canonical_size = cfg["canonical_size"]
    corner = cfg.get("eot", {}).get("corner")
    target_name = cfg.get("target_name", "banana")
    target_refs_dir = cfg.get("target_refs_dir", None)
    background_dataset = args.background_dataset or cfg.get("background_dataset", "imagenette")
    print(f"eval background dataset: {background_dataset!r}")
    if corner is not None:
        print(f"patch was trained with a corner-anchored placement: {corner} -- "
              f"evaluating with the same constraint")
    print(f"target class: {target_name!r}")

    print(f"loading ensemble {ensemble_names}")
    models = load_ensemble(ensemble_names, device=args.device)

    print("building eval galleries")
    gallery_kwargs = {"target_name": target_name}
    if target_refs_dir is not None:
        gallery_kwargs["target_refs_dir"] = Path(target_refs_dir)
    galleries = {
        name: build_eval_gallery(name, model, args.device, **gallery_kwargs)
        for name, model in models.items()
    }

    print("sweeping patch area fractions")
    df = sweep_patch_size(patch, models, galleries, canonical_size, args.device,
                           max_images=args.max_images, corner=corner, target_name=target_name,
                           background_dataset=background_dataset)

    run_name = Path(args.patch_path).parent.name
    if args.eval_ensemble is not None:
        run_name += "_evalall"
    if args.background_dataset is not None:
        run_name += f"_on_{args.background_dataset}"
    out_csv = OUT_ROOT / f"{run_name}_sweep.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    from patchattack.viz import plot_success_vs_area
    plot_success_vs_area(df, OUT_ROOT / f"{run_name}_sweep.png")
    print(f"saved -> {out_csv} and {OUT_ROOT / f'{run_name}_sweep.png'}")


if __name__ == "__main__":
    main()
