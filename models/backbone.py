"""Backbones used by TaskShift training and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
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


class DINOv2Backbone(nn.Module):
    """Frozen DINOv2 backbone loaded from the official PyTorch Hub entrypoint."""

    def __init__(self, model_name: str = "dinov2_vits14") -> None:
        super().__init__()
        self.model_name = model_name
        self.model = load_dinov2_model(model_name)
        self.model.eval()
        self.spec = BackboneSpec(name=model_name, feature_dim=self._infer_feature_dim())

    @property
    def feature_dim(self) -> int:
        return self.spec.feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)
        if isinstance(output, dict):
            if "x_norm_clstoken" in output:
                output = output["x_norm_clstoken"]
            else:
                first_tensor = next(value for value in output.values() if isinstance(value, torch.Tensor))
                output = first_tensor
        return output

    def intermediate_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return selected DINOv2 block CLS-token activations."""

        if not hasattr(self.model, "get_intermediate_layers"):
            return {}

        num_blocks = len(getattr(self.model, "blocks", []))
        if num_blocks == 0:
            return {}

        indices = sorted(set([0, num_blocks // 4, num_blocks // 2, (3 * num_blocks) // 4, num_blocks - 1]))
        layers = self.model.get_intermediate_layers(
            images,
            n=indices,
            reshape=False,
            return_class_token=True,
            norm=True,
        )

        features: dict[str, torch.Tensor] = {}
        for block_index, layer_output in zip(indices, layers):
            if isinstance(layer_output, tuple):
                cls_token = layer_output[1]
            else:
                cls_token = layer_output[:, 0]
            features[f"block_{block_index}"] = cls_token
        return features

    def _infer_feature_dim(self) -> int:
        embed_dim = getattr(self.model, "embed_dim", None)
        if isinstance(embed_dim, int):
            return embed_dim

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            output = self.forward(dummy)
        return int(output.shape[-1])


def load_dinov2_model(model_name: str) -> nn.Module:
    local_repo = Path(__file__).resolve().parents[1] / ".external" / "dinov2"
    local_weight = local_repo / "weights" / f"{model_name}_pretrain.pth"
    kwargs: dict[str, object] = {}
    if local_weight.exists():
        kwargs["weights"] = str(local_weight)

    if local_repo.exists():
        return torch.hub.load(str(local_repo), model_name, source="local", **kwargs)
    return torch.hub.load("facebookresearch/dinov2", model_name, **kwargs)


def build_backbone(name: str = "prototype") -> nn.Module:
    if name == "prototype":
        backbone = FrozenPrototypeBackbone()
    elif name in {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"}:
        backbone = DINOv2Backbone(name)
    else:
        raise ValueError(f"unknown backbone: {name}")

    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.eval()
    return backbone


def image_transform_for_backbone(name: str) -> Callable[[Image.Image], torch.Tensor]:
    if name == "prototype":
        return pil_to_tensor
    if name.startswith("dinov2_"):
        return dinov2_transform
    raise ValueError(f"unknown backbone: {name}")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    width, height = image.size
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(height, width, 3)
    return data.permute(2, 0, 1).to(dtype=torch.float32).div(255.0)


def dinov2_transform(image: Image.Image) -> torch.Tensor:
    resized = image.resize((224, 224), Image.Resampling.BICUBIC)
    tensor = pil_to_tensor(resized)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std
