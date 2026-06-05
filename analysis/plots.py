"""Plot TaskShift representation-shift results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TaskShift shift metrics.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/shift_metrics/representation_shift_summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/plots"))
    args = parser.parse_args()

    plot_shift_summary(args.summary, args.output_dir)


def plot_shift_summary(summary_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outputs: dict[str, Path] = {}

    for activation_name, result in summary["probe_shift"].items():
        shift_path = output_dir / f"{activation_name}_concept_shift.png"
        correlation_path = output_dir / f"{activation_name}_tuning_correlation.png"
        plot_concept_shift_bars(result, activation_name, shift_path)
        plot_tuning_correlation_bars(result, activation_name, correlation_path)
        outputs[f"{activation_name}_concept_shift"] = shift_path
        outputs[f"{activation_name}_tuning_correlation"] = correlation_path

    cka_path = output_dir / "cka_heatmap.png"
    plot_cka_heatmap(summary["cka"], cka_path)
    outputs["cka_heatmap"] = cka_path

    print("saved plots")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    return outputs


def plot_concept_shift_bars(result: dict[str, Any], activation_name: str, output_path: Path) -> None:
    rows = list(reversed(result["concept_shift"]))
    concepts = [row["concept"] for row in rows]
    shifts = [row["shift_magnitude"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = sns.color_palette("viridis", n_colors=max(len(rows), 3))
    ax.barh(concepts, shifts, color=colors[: len(rows)])
    ax.set_xlabel("Shift magnitude")
    ax.set_ylabel("Concept")
    ax.set_title(f"Concept Tuning Shift: {format_activation_name(activation_name)}")
    ax.grid(axis="x", color="#d0d7de", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    annotate_horizontal_bars(ax, shifts)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tuning_correlation_bars(result: dict[str, Any], activation_name: str, output_path: Path) -> None:
    rows = list(reversed(result["concept_shift"]))
    concepts = [row["concept"] for row in rows]
    correlations = [row["tuning_correlation"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#2f6f73" if value >= 0 else "#a44a3f" for value in correlations]
    ax.barh(concepts, correlations, color=colors)
    ax.axvline(0, color="#24292f", linewidth=1)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Tuning-vector correlation")
    ax.set_ylabel("Concept")
    ax.set_title(f"Passive vs Navigation Tuning Correlation: {format_activation_name(activation_name)}")
    ax.grid(axis="x", color="#d0d7de", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    annotate_horizontal_bars(ax, correlations)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_cka_heatmap(cka: dict[str, Any], output_path: Path) -> None:
    activation_names = cka["activation_names"]
    matrix = cka["matrix"]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="mako",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=[format_activation_name(name) for name in activation_names],
        yticklabels=[format_activation_name(name) for name in activation_names],
        cbar_kws={"label": "Linear CKA"},
    )
    ax.set_xlabel("Navigation activation")
    ax.set_ylabel("Passive activation")
    ax.set_title("Passive vs Navigation Representation Similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def annotate_horizontal_bars(ax: plt.Axes, values: list[float]) -> None:
    xmin, xmax = ax.get_xlim()
    width = xmax - xmin
    for index, value in enumerate(values):
        offset = width * 0.015
        if value >= 0:
            x = min(value + offset, xmax)
            ha = "left"
        else:
            x = max(value - offset, xmin)
            ha = "right"
        ax.text(x, index, f"{value:.3f}", va="center", ha=ha, fontsize=9)


def format_activation_name(name: str) -> str:
    return name.replace("_", " ").title()


if __name__ == "__main__":
    main()

