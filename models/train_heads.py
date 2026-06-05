"""Train passive and navigation heads on a frozen TaskShift backbone."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

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


TaskName = Literal["passive", "navigation"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TaskShift task heads.")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/prototype_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--task", choices=("passive", "navigation", "both"), default="both")
    parser.add_argument(
        "--backbone",
        choices=("prototype", "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"),
        default="prototype",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    train_requested_heads(
        dataset_dir=args.dataset,
        output_dir=args.output_dir,
        task=args.task,
        backbone_name=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


def train_requested_heads(
    dataset_dir: Path,
    output_dir: Path,
    task: str = "both",
    backbone_name: str = "prototype",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 17,
) -> dict[str, dict[str, float]]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    dataset = TaskShiftDataset(
        dataset_dir,
        image_transform=image_transform_for_backbone(backbone_name),
    )
    train_loader, val_loader = build_loaders(dataset, batch_size, seed)
    backbone = build_backbone(backbone_name).to(device)

    results: dict[str, dict[str, float]] = {}
    if task in ("passive", "both"):
        head = PassiveHead(backbone.feature_dim, len(OBJECT_VOCAB), len(ROOM_VOCAB)).to(device)
        metrics = train_one_head("passive", backbone, head, train_loader, val_loader, device, epochs, lr)
        save_checkpoint(
            output_dir / "passive_head.pt",
            "passive",
            backbone,
            head,
            metrics,
            backbone_name,
            dataset.vocab.navigation_actions,
        )
        results["passive"] = metrics

    if task in ("navigation", "both"):
        head = NavigationHead(
            backbone.feature_dim,
            len(NAVIGATION_BINARY_KEYS),
            len(dataset.vocab.navigation_actions),
        ).to(device)
        metrics = train_one_head(
            "navigation", backbone, head, train_loader, val_loader, device, epochs, lr
        )
        save_checkpoint(
            output_dir / "navigation_head.pt",
            "navigation",
            backbone,
            head,
            metrics,
            backbone_name,
            dataset.vocab.navigation_actions,
        )
        results["navigation"] = metrics

    return results


def build_loaders(
    dataset: TaskShiftDataset,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=taskshift_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=taskshift_collate,
    )
    return train_loader, val_loader


def train_one_head(
    task: TaskName,
    backbone: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(task, backbone, head, train_loader, device, optimizer)
        val_metrics = run_epoch(task, backbone, head, val_loader, device)
        print(
            f"{task} epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"{format_metric_summary(task, val_metrics)}"
        )

    return run_epoch(task, backbone, head, val_loader, device)


def run_epoch(
    task: TaskName,
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    head.train(training)
    backbone.eval()

    totals: dict[str, float] = {"loss": 0.0}
    examples = 0

    for batch in loader:
        images = batch["image"].to(device)
        with torch.no_grad():
            features = backbone(images)

        if task == "passive":
            outputs = head(features)
            targets = {
                "objects": batch["passive_targets"]["objects"].to(device),
                "room": batch["passive_targets"]["room"].to(device),
            }
            loss, metrics = passive_loss_and_metrics(outputs, targets)
        else:
            outputs = head(features)
            targets = {
                "binary": batch["navigation_targets"]["binary"].to(device),
                "action": batch["navigation_targets"]["action"].to(device),
            }
            loss, metrics = navigation_loss_and_metrics(outputs, targets)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = images.shape[0]
        examples += batch_size
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size

    return {key: value / examples for key, value in totals.items()}


def passive_loss_and_metrics(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    object_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["object_logits"],
        targets["objects"],
    )
    room_loss = nn.functional.cross_entropy(outputs["room_logits"], targets["room"])
    loss = object_loss + room_loss

    object_predictions = (torch.sigmoid(outputs["object_logits"]) >= 0.5).float()
    object_accuracy = (object_predictions == targets["objects"]).float().mean()
    room_accuracy = (outputs["room_logits"].argmax(dim=1) == targets["room"]).float().mean()

    return loss, {
        "object_label_accuracy": float(object_accuracy.detach().cpu()),
        "room_accuracy": float(room_accuracy.detach().cpu()),
    }


def navigation_loss_and_metrics(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    binary_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["binary_logits"],
        targets["binary"],
    )
    action_loss = nn.functional.cross_entropy(outputs["action_logits"], targets["action"])
    loss = binary_loss + action_loss

    binary_predictions = (torch.sigmoid(outputs["binary_logits"]) >= 0.5).float()
    binary_accuracy = (binary_predictions == targets["binary"]).float().mean()
    action_accuracy = (
        outputs["action_logits"].argmax(dim=1) == targets["action"]
    ).float().mean()

    return loss, {
        "binary_label_accuracy": float(binary_accuracy.detach().cpu()),
        "action_accuracy": float(action_accuracy.detach().cpu()),
    }


def save_checkpoint(
    path: Path,
    task: TaskName,
    backbone: nn.Module,
    head: nn.Module,
    metrics: dict[str, float],
    backbone_name: str,
    navigation_actions: tuple[str, ...],
) -> None:
    checkpoint: dict[str, Any] = {
        "task": task,
        "backbone": {
            "name": getattr(getattr(backbone, "spec", None), "name", "unknown"),
            "build_name": backbone_name,
            "feature_dim": getattr(backbone, "feature_dim", None),
        },
        "head_state_dict": head.state_dict(),
        "metrics": metrics,
        "vocab": {
            "objects": OBJECT_VOCAB,
            "rooms": ROOM_VOCAB,
            "navigation_binary": NAVIGATION_BINARY_KEYS,
            "navigation_actions": navigation_actions,
        },
    }
    torch.save(checkpoint, path)
    print(f"saved {task} checkpoint: {path}")


def format_metric_summary(task: TaskName, metrics: dict[str, float]) -> str:
    if task == "passive":
        return (
            f"object_acc={metrics['object_label_accuracy']:.3f} "
            f"room_acc={metrics['room_accuracy']:.3f}"
        )
    return (
        f"binary_acc={metrics['binary_label_accuracy']:.3f} "
        f"action_acc={metrics['action_accuracy']:.3f}"
    )


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
