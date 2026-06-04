"""Task-specific heads for TaskShift."""

from __future__ import annotations

import torch
from torch import nn


class PassiveHead(nn.Module):
    """Predict passive recognition labels from frozen image features."""

    def __init__(self, input_dim: int, num_objects: int, num_rooms: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        self.object_classifier = nn.Linear(128, num_objects)
        self.room_classifier = nn.Linear(128, num_rooms)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(features)
        return {
            "object_logits": self.object_classifier(hidden),
            "room_logits": self.room_classifier(hidden),
        }


class NavigationHead(nn.Module):
    """Predict navigation and affordance labels from frozen image features."""

    def __init__(self, input_dim: int, num_binary: int, num_actions: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        self.binary_classifier = nn.Linear(128, num_binary)
        self.action_classifier = nn.Linear(128, num_actions)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(features)
        return {
            "binary_logits": self.binary_classifier(hidden),
            "action_logits": self.action_classifier(hidden),
        }

