"""Run reproducible TaskShift experiment sweeps."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from activations.extract import extract_requested_activations
from analysis.linear_probes import train_requested_probes
from analysis.plots import plot_shift_summary
from analysis.representation_shift import compare_representations
from dashboard.build_static import build_dashboard
from data.build_prototype_dataset import build_dataset
from experiments.summarize_runs import build_comparison_report
from models.backbone import BACKBONE_TRAINING_MODES
from models.train_heads import train_requested_heads


@dataclass(frozen=True)
class RunConfig:
    experiment: str
    seed: int
    backbone: str
    train_backbone: str
    frames: int
    epochs: int
    batch_size: int
    lr: float
    backbone_lr: float

    @property
    def run_id(self) -> str:
        return f"{self.backbone}_{self.train_backbone}_seed{self.seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TaskShift experiment sweep.")
    parser.add_argument("--experiment", default="prototype_seed_sweep")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 31])
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=("prototype", "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"),
        default=["prototype"],
    )
    parser.add_argument(
        "--train-backbone-modes",
        nargs="+",
        choices=BACKBONE_TRAINING_MODES,
        default=["none"],
    )
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a run if its representation summary already exists.",
    )
    args = parser.parse_args()

    configs = [
        RunConfig(
            experiment=args.experiment,
            seed=seed,
            backbone=backbone,
            train_backbone=mode,
            frames=args.frames,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
        )
        for seed in args.seeds
        for backbone in args.backbones
        for mode in args.train_backbone_modes
    ]
    run_sweep(args.output_root, configs, skip_existing=args.skip_existing)


def run_sweep(
    output_root: Path,
    configs: list[RunConfig],
    skip_existing: bool = False,
) -> dict[str, Path]:
    if not configs:
        raise ValueError("at least one run config is required")

    experiment_dir = output_root / configs[0].experiment
    run_summaries: list[Path] = []
    for config in configs:
        run_paths = paths_for_run(experiment_dir, config)
        summary_path = run_paths["shift"] / "representation_shift_summary.json"
        if skip_existing and summary_path.exists():
            print(f"reusing existing run: {config.run_id}")
            run_summaries.append(summary_path)
            continue

        validate_config(config)
        run_single_experiment(config, run_paths)
        run_summaries.append(summary_path)

    comparison_paths = build_comparison_report(
        summary_paths=run_summaries,
        output_dir=experiment_dir / "comparison",
    )
    print(f"saved comparison summary: {comparison_paths['json']}")
    print(f"saved comparison dashboard: {comparison_paths['html']}")
    return comparison_paths


def run_single_experiment(config: RunConfig, paths: dict[str, Path]) -> None:
    print(f"\n=== {config.run_id} ===")
    build_dataset(
        output_dir=paths["dataset"],
        frame_count=config.frames,
        seed=config.seed,
        overwrite=True,
    )
    train_requested_heads(
        dataset_dir=paths["dataset"],
        output_dir=paths["checkpoints"],
        task="both",
        backbone_name=config.backbone,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        backbone_lr=config.backbone_lr,
        train_backbone=config.train_backbone,
        seed=config.seed,
    )
    extract_requested_activations(
        dataset_dir=paths["dataset"],
        checkpoint_dir=paths["checkpoints"],
        output_dir=paths["activations"],
        task="both",
        batch_size=config.batch_size,
    )
    train_requested_probes(
        activation_dir=paths["activations"],
        output_dir=paths["probes"],
        task="both",
        seed=config.seed + 1000,
    )
    compare_representations(
        probe_dir=paths["probes"],
        activation_dir=paths["activations"],
        output_dir=paths["shift"],
    )
    plot_shift_summary(
        summary_path=paths["shift"] / "representation_shift_summary.json",
        output_dir=paths["plots"],
    )
    build_dashboard(
        summary_path=paths["shift"] / "representation_shift_summary.json",
        output_path=paths["dashboard"] / "index.html",
        plot_dir=paths["plots"],
    )
    write_manifest(config, paths)


def paths_for_run(experiment_dir: Path, config: RunConfig) -> dict[str, Path]:
    run_dir = experiment_dir / "runs" / config.run_id
    return {
        "run": run_dir,
        "dataset": run_dir / "dataset",
        "checkpoints": run_dir / "checkpoints",
        "activations": run_dir / "activations",
        "probes": run_dir / "probes",
        "shift": run_dir / "shift_metrics",
        "plots": run_dir / "plots",
        "dashboard": run_dir / "dashboard",
    }


def write_manifest(config: RunConfig, paths: dict[str, Path]) -> Path:
    manifest: dict[str, Any] = {
        "config": asdict(config),
        "run_id": config.run_id,
        "paths": {name: str(path) for name, path in paths.items()},
    }
    output_path = paths["run"] / "run_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def validate_config(config: RunConfig) -> None:
    if config.train_backbone != "none" and not config.backbone.startswith("dinov2_"):
        raise ValueError(f"{config.train_backbone} training requires a DINOv2 backbone")
    if config.frames <= 0:
        raise ValueError("frames must be greater than 0")
    if config.epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")


if __name__ == "__main__":
    main()
