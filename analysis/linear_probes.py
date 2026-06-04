"""Train linear concept probes on saved TaskShift activations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train concept probes on TaskShift activations.")
    parser.add_argument("--activation-dir", type=Path, default=Path("artifacts/activations"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/probes"))
    parser.add_argument("--task", choices=("passive", "navigation", "both"), default="both")
    parser.add_argument("--activation", default="all", help="Activation name or 'all'.")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--min-positive", type=int, default=2)
    args = parser.parse_args()

    train_requested_probes(
        activation_dir=args.activation_dir,
        output_dir=args.output_dir,
        task=args.task,
        activation=args.activation,
        test_size=args.test_size,
        seed=args.seed,
        min_positive=args.min_positive,
    )


def train_requested_probes(
    activation_dir: Path,
    output_dir: Path,
    task: str = "both",
    activation: str = "all",
    test_size: float = 0.25,
    seed: int = 23,
    min_positive: int = 2,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = ("passive", "navigation") if task == "both" else (task,)
    outputs: dict[str, Path] = {}

    for task_name in requested:
        activation_path = activation_dir / f"{task_name}_activations.pt"
        output_path = output_dir / f"{task_name}_concept_probes.pt"
        train_probe_artifact(
            activation_path=activation_path,
            output_path=output_path,
            activation=activation,
            test_size=test_size,
            seed=seed,
            min_positive=min_positive,
        )
        outputs[task_name] = output_path

    return outputs


def train_probe_artifact(
    activation_path: Path,
    output_path: Path,
    activation: str = "all",
    test_size: float = 0.25,
    seed: int = 23,
    min_positive: int = 2,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    activation_artifact = torch.load(activation_path, map_location="cpu")
    concept_names = list(activation_artifact["vocab"]["concepts"])
    concept_targets = activation_artifact["targets"]["concepts"].numpy().astype(np.int64)
    activation_names = select_activation_names(activation_artifact, activation)

    probe_results = {}
    for activation_name in activation_names:
        features = activation_artifact["activations"][activation_name].numpy().astype(np.float64)
        probe_results[activation_name] = train_activation_probes(
            features=features,
            concept_targets=concept_targets,
            concept_names=concept_names,
            test_size=test_size,
            seed=seed,
            min_positive=min_positive,
        )

    artifact = {
        "task": activation_artifact["task"],
        "activation_path": str(activation_path),
        "probes": probe_results,
        "vocab": activation_artifact["vocab"],
        "source_metrics": activation_artifact.get("source_metrics", {}),
    }
    torch.save(artifact, output_path)
    print_probe_summary(output_path, artifact)
    return output_path


def train_activation_probes(
    features: np.ndarray,
    concept_targets: np.ndarray,
    concept_names: list[str],
    test_size: float,
    seed: int,
    min_positive: int,
) -> dict[str, Any]:
    mean = features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    normalized_features = (features - mean) / scale

    trained_concepts: list[str] = []
    weights: list[np.ndarray] = []
    intercepts: list[float] = []
    metrics: dict[str, dict[str, float | int | None]] = {}
    skipped: dict[str, str] = {}

    for concept_index, concept_name in enumerate(concept_names):
        labels = concept_targets[:, concept_index]
        positives = int(labels.sum())
        negatives = int(labels.shape[0] - positives)

        if positives < min_positive:
            skipped[concept_name] = f"only {positives} positive examples"
            continue
        if negatives < min_positive:
            skipped[concept_name] = f"only {negatives} negative examples"
            continue

        train_idx, val_idx = make_probe_split(labels, test_size, seed + concept_index)
        weight, intercept = fit_ridge_probe(
            normalized_features[train_idx],
            labels[train_idx],
            alpha=1.0,
        )

        scores = score_ridge_probe(normalized_features[val_idx], weight, intercept)
        predictions = (scores >= 0.0).astype(np.int64)
        val_labels = labels[val_idx]

        trained_concepts.append(concept_name)
        weights.append(weight.astype(np.float32))
        intercepts.append(float(intercept))
        metrics[concept_name] = {
            "positive_examples": positives,
            "negative_examples": negatives,
            "accuracy": float(accuracy_score(val_labels, predictions)),
            "balanced_accuracy": safe_balanced_accuracy(val_labels, predictions),
            "roc_auc": safe_roc_auc(val_labels, scores),
        }

    feature_dim = features.shape[1]
    weight_tensor = (
        torch.tensor(np.stack(weights), dtype=torch.float32)
        if weights
        else torch.empty((0, feature_dim), dtype=torch.float32)
    )
    intercept_tensor = torch.tensor(intercepts, dtype=torch.float32)

    return {
        "concepts": trained_concepts,
        "weights": weight_tensor,
        "intercepts": intercept_tensor,
        "metrics": metrics,
        "skipped": skipped,
        "standardization": {
            "mean": torch.tensor(mean.squeeze(0), dtype=torch.float32),
            "scale": torch.tensor(scale.squeeze(0), dtype=torch.float32),
        },
    }


def make_probe_split(
    labels: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(labels.shape[0])
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
        )

    if np.unique(labels[train_idx]).shape[0] < 2:
        raise ValueError("probe train split has only one class")
    return train_idx, val_idx


def fit_ridge_probe(features: np.ndarray, labels: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Fit a balanced binary ridge classifier and return a linear weight vector."""

    x = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(labels * 2 - 1, dtype=torch.float64).unsqueeze(1)
    design = torch.cat([x, torch.ones((x.shape[0], 1), dtype=torch.float64)], dim=1)

    positive_count = max(float(labels.sum()), 1.0)
    negative_count = max(float(labels.shape[0] - labels.sum()), 1.0)
    sample_weights = np.where(labels == 1, labels.shape[0] / positive_count, labels.shape[0] / negative_count)
    weight_diag = torch.tensor(sample_weights, dtype=torch.float64).unsqueeze(1)

    weighted_design = design * weight_diag
    regularizer = torch.eye(design.shape[1], dtype=torch.float64) * alpha
    regularizer[-1, -1] = 0.0
    solution = torch.linalg.solve(design.T @ weighted_design + regularizer, design.T @ (weight_diag * y))
    solution = solution.squeeze(1).numpy()
    return solution[:-1], float(solution[-1])


def score_ridge_probe(features: np.ndarray, weight: np.ndarray, intercept: float) -> np.ndarray:
    x = torch.tensor(features, dtype=torch.float64)
    w = torch.tensor(weight, dtype=torch.float64)
    return (x @ w + intercept).numpy()


def safe_balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    if np.unique(labels).shape[0] < 2:
        return None
    return float(balanced_accuracy_score(labels, predictions))


def safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(labels).shape[0] < 2:
        return None
    return float(roc_auc_score(labels, scores))


def select_activation_names(artifact: dict[str, Any], requested: str) -> list[str]:
    available = list(artifact["activations"].keys())
    if requested == "all":
        return available
    if requested not in available:
        raise ValueError(f"unknown activation '{requested}'; available: {available}")
    return [requested]


def print_probe_summary(path: Path, artifact: dict[str, Any]) -> None:
    print(f"saved {artifact['task']} concept probes: {path}")
    for activation_name, result in artifact["probes"].items():
        print(f"- {activation_name}: {len(result['concepts'])} trained, {len(result['skipped'])} skipped")
        for concept_name, metrics in result["metrics"].items():
            print(
                f"  {concept_name}: "
                f"balanced_acc={format_optional_metric(metrics['balanced_accuracy'])} "
                f"roc_auc={format_optional_metric(metrics['roc_auc'])}"
            )


def format_optional_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
