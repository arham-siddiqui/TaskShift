"""PyTorch dataset loader for TaskShift metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image
import torch
from torch.utils.data import Dataset

from data.dataset_schema import (
    BEST_ACTIONS,
    CONCEPT_LABEL_KEYS,
    load_jsonl,
    validate_metadata_file,
)


OBJECT_VOCAB = (
    "door",
    "chair",
    "table",
    "sofa",
    "cabinet",
    "bed",
    "sink",
    "fridge",
    "wall",
    "floor",
)
ROOM_VOCAB = ("Kitchen", "LivingRoom", "Bedroom", "Bathroom")
NAVIGATION_BINARY_KEYS = (
    "door_visible",
    "path_blocked",
    "obstacle_visible",
    "reachable_goal_visible",
)


@dataclass(frozen=True)
class TaskShiftVocab:
    """Class and label ordering used to encode dataset targets."""

    objects: tuple[str, ...] = OBJECT_VOCAB
    rooms: tuple[str, ...] = ROOM_VOCAB
    navigation_binary: tuple[str, ...] = NAVIGATION_BINARY_KEYS
    navigation_actions: tuple[str, ...] = BEST_ACTIONS
    concepts: tuple[str, ...] = CONCEPT_LABEL_KEYS


class TaskShiftDataset(Dataset):
    """Load TaskShift frames with passive, navigation, and concept targets.

    Each sample has this structure:

    ```
    {
        "image": FloatTensor[C, H, W],
        "passive_targets": {
            "objects": FloatTensor[num_objects],
            "room": LongTensor[],
        },
        "navigation_targets": {
            "binary": FloatTensor[num_navigation_binary],
            "action": LongTensor[],
        },
        "concept_targets": FloatTensor[num_concepts],
        "metadata": dict,
    }
    ```
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
        validate: bool = True,
        vocab: TaskShiftVocab | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.metadata_path = self.dataset_dir / "metadata.jsonl"
        self.vocab = vocab or TaskShiftVocab()
        self.image_transform = image_transform or pil_to_tensor

        if validate:
            result = validate_metadata_file(self.metadata_path, self.dataset_dir)
            if not result.ok:
                errors = "\n".join(result.errors[:20])
                raise ValueError(f"invalid TaskShift dataset:\n{errors}")

        self.rows = load_jsonl(self.metadata_path)
        self.object_to_index = {name: index for index, name in enumerate(self.vocab.objects)}
        self.room_to_index = {name: index for index, name in enumerate(self.vocab.rooms)}
        self.action_to_index = {
            name: index for index, name in enumerate(self.vocab.navigation_actions)
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self.dataset_dir / row["image_path"]

        with Image.open(image_path) as image:
            image_tensor = self.image_transform(image.convert("RGB"))

        return {
            "image": image_tensor,
            "passive_targets": self._encode_passive_targets(row),
            "navigation_targets": self._encode_navigation_targets(row),
            "concept_targets": self._encode_concept_targets(row),
            "metadata": row,
        }

    def _encode_passive_targets(self, row: dict[str, Any]) -> dict[str, torch.Tensor]:
        objects = torch.zeros(len(self.vocab.objects), dtype=torch.float32)
        for raw_name in row["visible_objects"]:
            name = raw_name.lower()
            if name in self.object_to_index:
                objects[self.object_to_index[name]] = 1.0

        room_name = row["room_type"]
        if room_name not in self.room_to_index:
            raise ValueError(f"unknown room_type: {room_name}")

        return {
            "objects": objects,
            "room": torch.tensor(self.room_to_index[room_name], dtype=torch.long),
        }

    def _encode_navigation_targets(self, row: dict[str, Any]) -> dict[str, torch.Tensor]:
        labels = row["navigation_labels"]
        binary = torch.tensor(
            [float(labels[key]) for key in self.vocab.navigation_binary],
            dtype=torch.float32,
        )

        action_name = labels["best_action"]
        if action_name not in self.action_to_index:
            raise ValueError(f"unknown best_action: {action_name}")

        return {
            "binary": binary,
            "action": torch.tensor(self.action_to_index[action_name], dtype=torch.long),
        }

    def _encode_concept_targets(self, row: dict[str, Any]) -> torch.Tensor:
        labels = row["concept_labels"]
        return torch.tensor([float(labels[key]) for key in self.vocab.concepts], dtype=torch.float32)

    def concept_positive_counts(self) -> dict[str, int]:
        """Count positives per concept, useful for skipping all-negative probes."""

        return {
            concept: sum(1 for row in self.rows if row["concept_labels"][concept])
            for concept in self.vocab.concepts
        }

    def active_concepts(self, min_positive: int = 1) -> tuple[str, ...]:
        """Return concepts with enough positive examples for probe training."""

        counts = self.concept_positive_counts()
        return tuple(
            concept for concept in self.vocab.concepts if counts[concept] >= min_positive
        )


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a float tensor in [0, 1]."""

    width, height = image.size
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(height, width, 3)
    return data.permute(2, 0, 1).to(dtype=torch.float32).div(255.0)


def taskshift_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch samples while keeping variable-length metadata as a list."""

    return {
        "image": torch.stack([sample["image"] for sample in batch]),
        "passive_targets": {
            "objects": torch.stack(
                [sample["passive_targets"]["objects"] for sample in batch]
            ),
            "room": torch.stack([sample["passive_targets"]["room"] for sample in batch]),
        },
        "navigation_targets": {
            "binary": torch.stack(
                [sample["navigation_targets"]["binary"] for sample in batch]
            ),
            "action": torch.stack(
                [sample["navigation_targets"]["action"] for sample in batch]
            ),
        },
        "concept_targets": torch.stack([sample["concept_targets"] for sample in batch]),
        "metadata": [sample["metadata"] for sample in batch],
    }
