"""Background image loading: Imagenette2-320 used purely as varied natural-image
backdrops for EOT training / eval -- labels are irrelevant to the attack itself,
but Imagenette's 10 classes double as the eval gallery's 10 "distractor" classes."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "imagenette2-320"
COCO_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "coco2017" / "train2017"
FOOD_ITEMS_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "food_items"

IMAGENETTE_CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chainsaw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


class ImagenetteBackgrounds(Dataset):
    def __init__(self, split: str, canonical_size: int, augment: bool = True):
        assert split in ("train", "val")
        self.root = DATA_ROOT / split
        self.paths = sorted(self.root.glob("*/*.JPEG"))
        if not self.paths:
            raise FileNotFoundError(
                f"no images found under {self.root} -- run scripts/download_imagenette.sh first"
            )
        if augment and split == "train":
            self.tf = v2.Compose([
                v2.RandomResizedCrop(canonical_size, scale=(0.8, 1.0), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.tf = v2.Compose([
                v2.Resize(canonical_size, antialias=True),
                v2.CenterCrop(canonical_size),
                v2.ToDtype(torch.float32, scale=True),
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = v2.functional.pil_to_tensor(img)
        return self.tf(img)

    def class_dirs(self) -> dict[str, list[Path]]:
        out = {}
        for wnid, name in IMAGENETTE_CLASSES.items():
            out[name] = sorted((self.root / wnid).glob("*.JPEG"))
        return out


class CocoBackgrounds(Dataset):
    """COCO train2017 as a flat pool of ~118k diverse, cluttered real-world photos
    (no label use -- same "backgrounds are content-agnostic" logic as Imagenette).
    COCO ships one big folder, so we carve our own deterministic train/val split
    (first 90% / last 10% of the sorted file list) to keep train and eval images
    disjoint, mirroring Imagenette's own train/val split."""

    def __init__(self, split: str, canonical_size: int, augment: bool = True):
        assert split in ("train", "val")
        all_paths = sorted(COCO_ROOT.glob("*.jpg"))
        if not all_paths:
            raise FileNotFoundError(
                f"no images found under {COCO_ROOT} -- download+unzip COCO train2017 first"
            )
        n_val = max(1, len(all_paths) // 10)
        self.paths = all_paths[:-n_val] if split == "train" else all_paths[-n_val:]
        if augment and split == "train":
            self.tf = v2.Compose([
                v2.RandomResizedCrop(canonical_size, scale=(0.8, 1.0), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.tf = v2.Compose([
                v2.Resize(canonical_size, antialias=True),
                v2.CenterCrop(canonical_size),
                v2.ToDtype(torch.float32, scale=True),
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = v2.functional.pil_to_tensor(img)
        return self.tf(img)


class FoodItemBackgrounds(Dataset):
    """Isolated grocery/food product photos (apple, chocolate bar, coke can, orange, bread,
    tomato -- curated from Wikimedia Commons), much closer to what a checkout scanner camera
    actually sees (single item, plain/white background) than generic natural-scene photos.
    Small pool (~130 images) relative to Imagenette/COCO, but domain-matched rather than
    scaled -- our own COCO-vs-Imagenette experiment found raw scale barely mattered, so
    domain match is the more promising lever here."""

    def __init__(self, split: str, canonical_size: int, augment: bool = True):
        assert split in ("train", "val")
        self.paths = sorted((FOOD_ITEMS_ROOT / split).glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(
                f"no images found under {FOOD_ITEMS_ROOT / split} -- build the food_items "
                "background pool first (see conversation / scripts for the curation steps)"
            )
        if augment and split == "train":
            self.tf = v2.Compose([
                v2.RandomResizedCrop(canonical_size, scale=(0.8, 1.0), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.tf = v2.Compose([
                v2.Resize(canonical_size, antialias=True),
                v2.CenterCrop(canonical_size),
                v2.ToDtype(torch.float32, scale=True),
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = v2.functional.pil_to_tensor(img)
        return self.tf(img)


def make_background_loader(
    canonical_size: int, batch_size: int, split: str = "train", num_workers: int = 4,
    dataset: str = "imagenette",
) -> DataLoader:
    if dataset == "imagenette":
        ds = ImagenetteBackgrounds(split, canonical_size, augment=(split == "train"))
    elif dataset == "coco":
        ds = CocoBackgrounds(split, canonical_size, augment=(split == "train"))
    elif dataset == "food_items":
        ds = FoodItemBackgrounds(split, canonical_size, augment=(split == "train"))
    else:
        raise ValueError(
            f"unknown background dataset {dataset!r}, expected 'imagenette', 'coco', or 'food_items'"
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
