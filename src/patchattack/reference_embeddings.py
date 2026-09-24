"""Per-model target/gallery centroid embeddings, with on-disk caching.

A centroid is: L2-normalize each image's embedding, mean-pool, L2-normalize the
mean. Normalizing before AND after pooling matters -- pooling raw (unnormalized)
embeddings lets a few high-norm outlier images dominate the centroid.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from patchattack.data import DATA_ROOT, IMAGENETTE_CLASSES
from patchattack.models import ModelWrapper

OUT_ROOT = Path(__file__).resolve().parent.parent.parent / "outputs" / "embeddings"
DATA_REFS_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
BANANA_REFS = DATA_REFS_ROOT / "banana_refs"

_LOAD_TF = v2.Compose([
    v2.Resize(256, antialias=True),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
])


def _paths_hash(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for p in sorted(str(p) for p in paths):
        h.update(p.encode())
    return h.hexdigest()[:12]


def _load_batch(paths: list[Path], device) -> torch.Tensor:
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = v2.functional.pil_to_tensor(img)
        imgs.append(_LOAD_TF(img))
    return torch.stack(imgs).to(device)


@torch.no_grad()
def compute_centroid(
    model: ModelWrapper, image_paths: list[Path], device, batch_size: int = 16
) -> torch.Tensor:
    if not image_paths:
        raise ValueError("no image paths given for centroid computation")
    embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        x = _load_batch(batch_paths, device)
        e = model.embed(x)
        e = F.normalize(e, dim=-1)
        embs.append(e)
    embs = torch.cat(embs, dim=0)
    centroid = embs.mean(dim=0)
    return F.normalize(centroid, dim=0)


def _cached(cache_path: Path, key_paths: list[Path], compute_fn):
    tag = _paths_hash(key_paths)
    tagged_path = cache_path.with_name(cache_path.stem + f"_{tag}.pt")
    if tagged_path.exists():
        return torch.load(tagged_path, map_location="cpu")
    result = compute_fn()
    tagged_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.cpu(), tagged_path)
    return result


def load_or_compute_training_target(
    model_name: str, model: ModelWrapper, device,
    target_name: str = "banana", target_refs_dir: Path = BANANA_REFS,
) -> torch.Tensor:
    paths = sorted((target_refs_dir / "train").glob("*.jpg"))
    cache_path = OUT_ROOT / f"{model_name}_target_{target_name}"
    return _cached(cache_path, paths, lambda: compute_centroid(model, paths, device)).to(device)


def build_eval_gallery(
    model_name: str, model: ModelWrapper, device,
    target_name: str = "banana", target_refs_dir: Path = BANANA_REFS,
) -> dict[str, torch.Tensor]:
    """11-way gallery: target class (from disjoint eval refs) + 10 Imagenette classes
    (from Imagenette's own train split)."""
    gallery = {}

    target_paths = sorted((target_refs_dir / "eval").glob("*.jpg"))
    cache_path = OUT_ROOT / f"{model_name}_gallery_{target_name}"
    gallery[target_name] = _cached(
        cache_path, target_paths, lambda: compute_centroid(model, target_paths, device)
    ).to(device)

    train_root = DATA_ROOT / "train"
    for wnid, class_name in IMAGENETTE_CLASSES.items():
        class_paths = sorted((train_root / wnid).glob("*.JPEG"))[:40]
        cache_path = OUT_ROOT / f"{model_name}_gallery_{class_name.replace(' ', '_')}"
        gallery[class_name] = _cached(
            cache_path, class_paths, lambda cp=class_paths: compute_centroid(model, cp, device)
        ).to(device)

    return gallery
