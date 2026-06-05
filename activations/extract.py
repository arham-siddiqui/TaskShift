"""Extract TaskShift activations from trained task heads."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from activations.hooks import ActivationRecorder
from data.taskshift_dataset import (
    NAVIGATION_BINARY_KEYS,
    OBJECT_VOCAB,
    ROOM_VOCAB,
    TaskShiftDataset,
    taskshift_collate,
)
from models.backbone import build_backbone
from models.backbone import image_transform_for_backbone
from models.heads import NavigationHead, PassiveHead
from models.train_heads import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TaskShift activations.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/prototype_dataset"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/activations"))
    parser.add_argument("--task", choices=("passive", "navigation", "both"), default="both")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    extract_requested_activations(
        dataset_dir=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        task=args.task,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
    )


def extract_requested_activations(
    dataset_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    task: str = "both",
    checkpoint: Path | None = None,
    batch_size: int = 64,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = ("passive", "navigation") if task == "both" else (task,)
    outputs: dict[str, Path] = {}

    for task_name in requested:
        checkpoint_path = checkpoint or checkpoint_dir / f"{task_name}_head.pt"
        output_path = output_dir / f"{task_name}_activations.pt"
        extract_activations(
            dataset_dir=dataset_dir,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            batch_size=batch_size,
        )
        outputs[task_name] = output_path

    return outputs


def extract_activations(
    dataset_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int = 64,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    task = checkpoint["task"]
    backbone_name = resolve_backbone_name(checkpoint)
    dataset = TaskShiftDataset(
        dataset_dir,
        image_transform=image_transform_for_backbone(backbone_name),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=taskshift_collate,
    )

    backbone = build_backbone(backbone_name).to(device)
    head = build_head(task, backbone.feature_dim, dataset, checkpoint).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    collected: dict[str, list[torch.Tensor]] = {
        "backbone_features": [],
        "head_hidden": [],
    }
    logits: dict[str, list[torch.Tensor]] = {}
    targets: dict[str, list[torch.Tensor]] = {
        "passive_objects": [],
        "passive_room": [],
        "navigation_binary": [],
        "navigation_action": [],
        "concepts": [],
    }
    metadata: list[dict[str, Any]] = []

    with ActivationRecorder() as recorder:
        recorder.watch("head_hidden", head.shared)

        with torch.no_grad():
            for batch in loader:
                recorder.clear()
                images = batch["image"].to(device)
                features = backbone(images)
                outputs = head(features)

                collected["backbone_features"].append(features.detach().cpu())
                if hasattr(backbone, "intermediate_features"):
                    for name, value in backbone.intermediate_features(images).items():
                        collected.setdefault(name, []).append(value.detach().cpu())
                collected["head_hidden"].append(recorder.activations["head_hidden"])
                for name, value in outputs.items():
                    logits.setdefault(name, []).append(value.detach().cpu())

                targets["passive_objects"].append(batch["passive_targets"]["objects"])
                targets["passive_room"].append(batch["passive_targets"]["room"])
                targets["navigation_binary"].append(batch["navigation_targets"]["binary"])
                targets["navigation_action"].append(batch["navigation_targets"]["action"])
                targets["concepts"].append(batch["concept_targets"])
                metadata.extend(batch["metadata"])

    artifact = {
        "task": task,
        "checkpoint_path": str(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "backbone": checkpoint.get("backbone", {}),
        "activations": stack_tensor_lists(collected),
        "logits": stack_tensor_lists(logits),
        "targets": stack_tensor_lists(targets),
        "metadata": metadata,
        "vocab": {
            "objects": list(OBJECT_VOCAB),
            "rooms": list(ROOM_VOCAB),
            "navigation_binary": list(NAVIGATION_BINARY_KEYS),
            "navigation_actions": list(dataset.vocab.navigation_actions),
            "concepts": list(dataset.vocab.concepts),
            "active_concepts": list(dataset.active_concepts()),
        },
        "source_metrics": checkpoint.get("metrics", {}),
    }
    torch.save(artifact, output_path)
    print_activation_summary(output_path, artifact)
    return output_path


def build_head(
    task: str,
    feature_dim: int,
    dataset: TaskShiftDataset,
    checkpoint: dict[str, Any],
) -> nn.Module:
    vocab = checkpoint.get("vocab", {})
    if task == "passive":
        return PassiveHead(
            feature_dim,
            len(vocab.get("objects", OBJECT_VOCAB)),
            len(vocab.get("rooms", ROOM_VOCAB)),
        )
    if task == "navigation":
        return NavigationHead(
            feature_dim,
            len(vocab.get("navigation_binary", NAVIGATION_BINARY_KEYS)),
            len(vocab.get("navigation_actions", dataset.vocab.navigation_actions)),
        )
    raise ValueError(f"unknown checkpoint task: {task}")


def resolve_backbone_name(checkpoint: dict[str, Any]) -> str:
    backbone = checkpoint.get("backbone", {})
    name = backbone.get("build_name") or backbone.get("name", "prototype")
    if name.startswith("frozen_prototype"):
        return "prototype"
    return name


def stack_tensor_lists(values: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.cat(tensors, dim=0) for name, tensors in values.items()}


def print_activation_summary(path: Path, artifact: dict[str, Any]) -> None:
    print(f"saved {artifact['task']} activations: {path}")
    for name, tensor in artifact["activations"].items():
        print(f"- {name}: {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
