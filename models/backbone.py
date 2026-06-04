"""Backbones used by TaskShift training and analysis."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    feature_dim: int


class FrozenPrototypeBackbone(nn.Module):
    """Small frozen image featurizer for the local prototype pipeline.

    This is not the final research backbone. It exists so the training scripts,
    checkpoint format, and task heads can be exercised before DINOv2 is added.
    """

    def __init__(self, grid_size: int = 8) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.spec = BackboneSpec(
            name=f"frozen_prototype_grid{grid_size}",
            feature_dim=3 * grid_size * grid_size + 6,
        )

    @property
    def feature_dim(self) -> int:
        return self.spec.feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(images, (self.grid_size, self.grid_size))
        pooled = pooled.flatten(start_dim=1)
        means = images.mean(dim=(2, 3))
        stds = images.std(dim=(2, 3), unbiased=False)
        return torch.cat([pooled, means, stds], dim=1)


def build_backbone(name: str = "prototype") -> nn.Module:
    if name == "prototype":
        backbone = FrozenPrototypeBackbone()
    else:
        raise ValueError(f"unknown backbone: {name}")

    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.eval()
    return backbone

