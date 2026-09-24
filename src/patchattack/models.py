"""Unified embedding-extractor interface over the attack ensemble.

Every model is wrapped to expose `.embed(x)` where x is (B,3,H,W) in [0,1]:
the wrapper owns its own input resize + normalization, and returns a raw
(unnormalized) (B, embed_dim) feature vector. Callers L2-normalize as needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ModelWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, input_size: int, mean, std, embed_dim: int):
        super().__init__()
        self.backbone = backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        return self.backbone(x)

    def train(self, mode: bool = True):
        # keep backbone frozen in eval() regardless of outer module .train() calls
        super().train(mode)
        self.backbone.eval()
        return self


class DinoV2Embedder(nn.Module):
    def __init__(self, hub_model: nn.Module):
        super().__init__()
        self.m = hub_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m.forward_features(x)["x_norm_clstoken"]


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DINO_MEAN = [0.5, 0.5, 0.5]
DINO_STD = [0.5, 0.5, 0.5]


def _load_resnet18() -> tuple[nn.Module, int]:
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    return m, 512


def _load_mobilenet_v3_large() -> tuple[nn.Module, int]:
    m = torchvision.models.mobilenet_v3_large(
        weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )
    # classifier: Sequential(Linear(960,1280), Hardswish, Dropout, Linear(1280,1000))
    # keep the 960->1280 projection + activation, drop only the final classification Linear.
    assert isinstance(m.classifier[-1], nn.Linear), (
        f"unexpected classifier tail: {m.classifier}"
    )
    embed_dim = m.classifier[-1].in_features
    m.classifier[-1] = nn.Identity()
    # also drop the Dropout before it (harmless in eval mode, but be explicit)
    return m, embed_dim


def _load_efficientnet_b0() -> tuple[nn.Module, int]:
    m = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    # classifier: Sequential(Dropout, Linear(1280,1000))
    assert isinstance(m.classifier[-1], nn.Linear), (
        f"unexpected classifier tail: {m.classifier}"
    )
    embed_dim = m.classifier[-1].in_features
    m.classifier[-1] = nn.Identity()
    return m, embed_dim


def _load_dinov2(name: str) -> tuple[nn.Module, int]:
    hub_model = torch.hub.load("facebookresearch/dinov2", name)
    embed_dim = {"dinov2_vits14": 384, "dinov2_vitb14": 768}[name]
    return DinoV2Embedder(hub_model), embed_dim


LOADERS = {
    "resnet18": (_load_resnet18, 224, IMAGENET_MEAN, IMAGENET_STD),
    "mobilenet_v3_large": (_load_mobilenet_v3_large, 224, IMAGENET_MEAN, IMAGENET_STD),
    "efficientnet_b0": (_load_efficientnet_b0, 224, IMAGENET_MEAN, IMAGENET_STD),
    "dinov2_vits14": (lambda: _load_dinov2("dinov2_vits14"), 224, DINO_MEAN, DINO_STD),
    "dinov2_vitb14": (lambda: _load_dinov2("dinov2_vitb14"), 224, DINO_MEAN, DINO_STD),
}


def load_ensemble(names: list[str], device: str) -> dict[str, ModelWrapper]:
    out = {}
    for name in names:
        if name not in LOADERS:
            raise ValueError(f"unknown model {name!r}, available: {list(LOADERS)}")
        loader_fn, input_size, mean, std = LOADERS[name]
        backbone, embed_dim = loader_fn()
        wrapper = ModelWrapper(backbone, input_size, mean, std, embed_dim).to(device)
        out[name] = wrapper
    return out
