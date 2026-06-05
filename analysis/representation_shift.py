"""Compare passive and navigation representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from analysis.cka import linear_cka


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TaskShift representations.")
    parser.add_argument("--probe-dir", type=Path, default=Path("artifacts/probes"))
    parser.add_argument("--activation-dir", type=Path, default=Path("artifacts/activations"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/shift_metrics"))
    args = parser.parse_args()

    compare_representations(
        probe_dir=args.probe_dir,
        activation_dir=args.activation_dir,
        output_dir=args.output_dir,
    )


def compare_representations(
    probe_dir: Path,
    activation_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    passive_probes = torch.load(probe_dir / "passive_concept_probes.pt", map_location="cpu")
    navigation_probes = torch.load(probe_dir / "navigation_concept_probes.pt", map_location="cpu")
    passive_activations = torch.load(
        activation_dir / "passive_activations.pt",
        map_location="cpu",
    )
    navigation_activations = torch.load(
        activation_dir / "navigation_activations.pt",
        map_location="cpu",
    )

    artifact = {
        "probe_shift": compare_probe_artifacts(passive_probes, navigation_probes),
        "cka": compare_activation_artifacts(passive_activations, navigation_activations),
        "source_paths": {
            "passive_probes": str(probe_dir / "passive_concept_probes.pt"),
            "navigation_probes": str(probe_dir / "navigation_concept_probes.pt"),
            "passive_activations": str(activation_dir / "passive_activations.pt"),
            "navigation_activations": str(activation_dir / "navigation_activations.pt"),
        },
    }

    pt_path = output_dir / "representation_shift.pt"
    json_path = output_dir / "representation_shift_summary.json"
    torch.save(artifact, pt_path)
    json_path.write_text(json.dumps(to_jsonable(artifact), indent=2), encoding="utf-8")
    print_shift_summary(pt_path, json_path, artifact)
    return {"pt": pt_path, "json": json_path}


def compare_probe_artifacts(
    passive: dict[str, Any],
    navigation: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    activation_names = sorted(set(passive["probes"]) & set(navigation["probes"]))

    for activation_name in activation_names:
        passive_probe = passive["probes"][activation_name]
        navigation_probe = navigation["probes"][activation_name]
        common_concepts = [
            concept
            for concept in passive_probe["concepts"]
            if concept in set(navigation_probe["concepts"])
        ]

        rows = []
        passive_weights = weights_by_concept(passive_probe)
        navigation_weights = weights_by_concept(navigation_probe)
        for concept in common_concepts:
            passive_weight = passive_weights[concept]
            navigation_weight = navigation_weights[concept]
            cosine = cosine_similarity(passive_weight, navigation_weight)
            correlation = pearson_correlation(passive_weight, navigation_weight)
            shift = navigation_weight - passive_weight
            rows.append(
                {
                    "concept": concept,
                    "cosine_similarity": cosine,
                    "tuning_correlation": correlation,
                    "shift_magnitude": float(torch.linalg.vector_norm(shift)),
                    "passive_norm": float(torch.linalg.vector_norm(passive_weight)),
                    "navigation_norm": float(torch.linalg.vector_norm(navigation_weight)),
                    "navigation_minus_passive_norm": float(
                        torch.linalg.vector_norm(navigation_weight)
                        - torch.linalg.vector_norm(passive_weight)
                    ),
                    "passive_probe_metrics": passive_probe["metrics"].get(concept, {}),
                    "navigation_probe_metrics": navigation_probe["metrics"].get(concept, {}),
                }
            )

        rows.sort(key=lambda row: row["shift_magnitude"], reverse=True)
        results[activation_name] = {
            "concepts_compared": common_concepts,
            "concept_shift": rows,
            "mean_shift_magnitude": mean_value(row["shift_magnitude"] for row in rows),
            "mean_tuning_correlation": mean_value(row["tuning_correlation"] for row in rows),
        }

    return results


def compare_activation_artifacts(
    passive: dict[str, Any],
    navigation: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    activation_names = sorted(set(passive["activations"]) & set(navigation["activations"]))
    matrix = torch.zeros((len(activation_names), len(activation_names)), dtype=torch.float32)

    for row_index, passive_name in enumerate(activation_names):
        for column_index, navigation_name in enumerate(activation_names):
            value = linear_cka(
                passive["activations"][passive_name],
                navigation["activations"][navigation_name],
            )
            matrix[row_index, column_index] = value

    for index, activation_name in enumerate(activation_names):
        results[activation_name] = {
            "matched_layer_cka": float(matrix[index, index]),
        }

    return {
        "activation_names": activation_names,
        "matrix": matrix,
        "matched": results,
    }


def weights_by_concept(probe: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        concept: probe["weights"][index].to(dtype=torch.float64)
        for index, concept in enumerate(probe["concepts"])
    }


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) == 0.0:
        return 0.0
    return float((x @ y) / denominator)


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    return cosine_similarity(x_centered, y_centered)


def mean_value(values: Any) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(sum(values) / len(values))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def print_shift_summary(pt_path: Path, json_path: Path, artifact: dict[str, Any]) -> None:
    print(f"saved representation shift artifact: {pt_path}")
    print(f"saved JSON summary: {json_path}")

    for activation_name, result in artifact["probe_shift"].items():
        print(f"\n{activation_name} concept shifts")
        for row in result["concept_shift"]:
            print(
                f"- {row['concept']}: "
                f"shift={row['shift_magnitude']:.3f} "
                f"corr={row['tuning_correlation']:.3f} "
                f"cos={row['cosine_similarity']:.3f}"
            )

    print("\nmatched CKA")
    for activation_name, result in artifact["cka"]["matched"].items():
        print(f"- {activation_name}: {result['matched_layer_cka']:.3f}")


if __name__ == "__main__":
    main()

