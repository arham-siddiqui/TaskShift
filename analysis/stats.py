"""Statistical checks for TaskShift experiment comparisons."""

from __future__ import annotations

import argparse
import html
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_CKA_ACTIVATIONS = ("backbone_features", "block_9", "block_11", "head_hidden")
DEFAULT_SHIFT_ACTIVATIONS = ("backbone_features", "block_11", "head_hidden")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap and permutation tests for TaskShift sweeps.")
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("artifacts/experiments/thor_dinov2_seed_sweep/comparison/comparison_summary.json"),
    )
    parser.add_argument("--baseline-condition", default="dinov2_vits14:none")
    parser.add_argument("--treatment-condition", default="dinov2_vits14:final_block")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stats"))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = build_stats_report(
        args.comparison,
        args.output_dir,
        baseline_condition=args.baseline_condition,
        treatment_condition=args.treatment_condition,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(f"saved stats summary: {paths['json']}")
    print(f"saved stats dashboard: {paths['html']}")
    for name, path in paths["plots"].items():
        print(f"saved {name}: {path}")


def build_stats_report(
    comparison_path: Path,
    output_dir: Path,
    *,
    baseline_condition: str,
    treatment_condition: str,
    bootstrap_samples: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(comparison_path.read_text(encoding="utf-8"))
    stats = compute_stats(
        report,
        baseline_condition=baseline_condition,
        treatment_condition=treatment_condition,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plots = {
        "metric_effects": plot_metric_effects(stats, plot_dir / "metric_effects.png"),
        "concept_effects": plot_concept_effects(stats, plot_dir / "concept_effects.png"),
    }

    json_path = output_dir / "stats_summary.json"
    html_path = output_dir / "index.html"
    json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    html_path.write_text(render_stats_dashboard(stats, plots, output_dir), encoding="utf-8")
    return {"json": json_path, "html": html_path, "plots": plots}


def compute_stats(
    report: dict[str, Any],
    *,
    baseline_condition: str,
    treatment_condition: str,
    bootstrap_samples: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    runs_by_condition = group_runs_by_condition_and_seed(report["runs"])
    if baseline_condition not in runs_by_condition:
        raise ValueError(f"baseline condition not found: {baseline_condition}")
    if treatment_condition not in runs_by_condition:
        raise ValueError(f"treatment condition not found: {treatment_condition}")

    baseline_runs = runs_by_condition[baseline_condition]
    treatment_runs = runs_by_condition[treatment_condition]
    paired_seeds = sorted(set(baseline_runs) & set(treatment_runs))
    if not paired_seeds:
        raise ValueError("no shared seeds found between baseline and treatment conditions")

    metric_tests = []
    for activation in DEFAULT_CKA_ACTIVATIONS:
        metric_tests.append(
            paired_metric_test(
                baseline_runs,
                treatment_runs,
                paired_seeds,
                family="matched_cka",
                activation=activation,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )
    for activation in DEFAULT_SHIFT_ACTIVATIONS:
        metric_tests.append(
            paired_metric_test(
                baseline_runs,
                treatment_runs,
                paired_seeds,
                family="mean_shift_magnitude",
                activation=activation,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1,
            )
        )

    concept_tests = []
    for activation in DEFAULT_SHIFT_ACTIVATIONS:
        concepts = sorted(
            set(concept_map_for_runs(baseline_runs.values(), activation))
            | set(concept_map_for_runs(treatment_runs.values(), activation))
        )
        for concept in concepts:
            concept_tests.append(
                paired_concept_test(
                    baseline_runs,
                    treatment_runs,
                    paired_seeds,
                    activation=activation,
                    concept=concept,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 2,
                )
            )

    return {
        "source": {
            "baseline_condition": baseline_condition,
            "treatment_condition": treatment_condition,
            "paired_seeds": paired_seeds,
            "run_count": len(paired_seeds) * 2,
            "bootstrap_samples": bootstrap_samples,
        },
        "interpretation": {
            "mean_difference": "treatment minus baseline",
            "confidence_interval": "bootstrap 95% CI over paired seed differences",
            "p_value": "exact paired sign-flip permutation p-value",
        },
        "metric_tests": [test for test in metric_tests if test is not None],
        "concept_tests": [test for test in concept_tests if test is not None],
    }


def group_runs_by_condition_and_seed(runs: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for run in runs:
        seed = run.get("seed")
        if seed is None:
            continue
        grouped.setdefault(run["condition"], {})[int(seed)] = run
    return grouped


def paired_metric_test(
    baseline_runs: dict[int, dict[str, Any]],
    treatment_runs: dict[int, dict[str, Any]],
    seeds: list[int],
    *,
    family: str,
    activation: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any] | None:
    baseline_values = []
    treatment_values = []
    used_seeds = []
    for run_seed in seeds:
        baseline = baseline_runs[run_seed].get(family, {}).get(activation)
        treatment = treatment_runs[run_seed].get(family, {}).get(activation)
        if baseline is None or treatment is None:
            continue
        baseline_values.append(float(baseline))
        treatment_values.append(float(treatment))
        used_seeds.append(run_seed)
    if not used_seeds:
        return None
    return summarize_paired_values(
        baseline_values,
        treatment_values,
        used_seeds,
        metric=f"{family}:{activation}",
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        extra={"family": family, "activation": activation},
    )


def paired_concept_test(
    baseline_runs: dict[int, dict[str, Any]],
    treatment_runs: dict[int, dict[str, Any]],
    seeds: list[int],
    *,
    activation: str,
    concept: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any] | None:
    baseline_values = []
    treatment_values = []
    used_seeds = []
    for run_seed in seeds:
        baseline = concept_shift_value(baseline_runs[run_seed], activation, concept)
        treatment = concept_shift_value(treatment_runs[run_seed], activation, concept)
        if baseline is None or treatment is None:
            continue
        baseline_values.append(baseline)
        treatment_values.append(treatment)
        used_seeds.append(run_seed)
    if not used_seeds:
        return None
    return summarize_paired_values(
        baseline_values,
        treatment_values,
        used_seeds,
        metric=f"concept_shift:{activation}:{concept}",
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        extra={"family": "concept_shift", "activation": activation, "concept": concept},
    )


def summarize_paired_values(
    baseline_values: list[float],
    treatment_values: list[float],
    seeds: list[int],
    *,
    metric: str,
    bootstrap_samples: int,
    seed: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    differences = [treatment - baseline for baseline, treatment in zip(baseline_values, treatment_values)]
    ci_low, ci_high = bootstrap_ci(differences, samples=bootstrap_samples, seed=seed)
    return {
        "metric": metric,
        **extra,
        "seeds": seeds,
        "baseline_mean": mean(baseline_values),
        "treatment_mean": mean(treatment_values),
        "mean_difference": mean(differences),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": paired_sign_flip_p_value(differences),
        "baseline_values": baseline_values,
        "treatment_values": treatment_values,
        "differences": differences,
    }


def concept_map_for_runs(runs: Any, activation: str) -> set[str]:
    concepts = set()
    for run in runs:
        for row in run.get("concept_shift", {}).get(activation, []):
            concepts.add(row["concept"])
    return concepts


def concept_shift_value(run: dict[str, Any], activation: str, concept: str) -> float | None:
    for row in run.get("concept_shift", {}).get(activation, []):
        if row["concept"] == concept:
            return float(row["shift_magnitude"])
    return None


def bootstrap_ci(values: list[float], *, samples: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if len(values) == 1:
        return values[0], values[0]

    import random

    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(draw))
    means.sort()
    low_index = max(0, math.floor((alpha / 2) * len(means)))
    high_index = min(len(means) - 1, math.ceil((1 - alpha / 2) * len(means)) - 1)
    return means[low_index], means[high_index]


def paired_sign_flip_p_value(differences: list[float]) -> float:
    if not differences:
        raise ValueError("permutation test requires at least one value")
    observed = abs(mean(differences))
    count = 0
    total = 0
    for signs in itertools.product((-1, 1), repeat=len(differences)):
        permuted = [value * sign for value, sign in zip(differences, signs)]
        if abs(mean(permuted)) >= observed - 1e-12:
            count += 1
        total += 1
    return count / total


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def plot_metric_effects(stats: dict[str, Any], output_path: Path) -> Path:
    rows = [
        row
        for row in stats["metric_tests"]
        if row["family"] in {"matched_cka", "mean_shift_magnitude"}
    ]
    labels = [format_metric_label(row) for row in rows]
    values = [row["mean_difference"] for row in rows]
    lows = [row["mean_difference"] - row["ci_low"] for row in rows]
    highs = [row["ci_high"] - row["mean_difference"] for row in rows]

    height = max(4.8, 0.42 * len(rows))
    fig, ax = plt.subplots(figsize=(9, height))
    y_positions = list(range(len(rows)))
    ax.barh(y_positions, values, color="#2f6f73")
    ax.errorbar(values, y_positions, xerr=[lows, highs], fmt="none", ecolor="#20242a", capsize=3)
    ax.axvline(0, color="#20242a", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean difference, treatment minus baseline")
    ax.set_title("Paired Metric Effects with Bootstrap 95% CIs")
    ax.grid(axis="x", color="#d0d7de", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_concept_effects(stats: dict[str, Any], output_path: Path) -> Path:
    rows = [
        row
        for row in stats["concept_tests"]
        if row["activation"] == "block_11"
    ]
    rows = sorted(rows, key=lambda row: row["mean_difference"], reverse=True)
    labels = [row["concept"] for row in rows]
    values = [row["mean_difference"] for row in rows]
    lows = [row["mean_difference"] - row["ci_low"] for row in rows]
    highs = [row["ci_high"] - row["mean_difference"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, max(4.2, 0.45 * len(rows))))
    y_positions = list(range(len(rows)))
    ax.barh(y_positions, values, color="#5f6f52")
    ax.errorbar(values, y_positions, xerr=[lows, highs], fmt="none", ecolor="#20242a", capsize=3)
    ax.axvline(0, color="#20242a", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean concept-shift difference")
    ax.set_title("Final Block Concept Effects with Bootstrap 95% CIs")
    ax.grid(axis="x", color="#d0d7de", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_stats_dashboard(stats: dict[str, Any], plots: dict[str, Path], output_dir: Path) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TaskShift Statistical Validation</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #20242a;
      --muted: #68707c;
      --line: #d9dee6;
      --accent: #2f6f73;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    .wrap {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
    }}
    .topbar {{
      min-height: 76px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
    }}
    h1 {{ margin: 0; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 4px 0 0; color: var(--muted); }}
    main {{ padding: 28px 0 42px; }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .plots {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    img {{
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: right;
      vertical-align: top;
      white-space: nowrap;
    }}
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2) {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: #f9fafb;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border-radius: 999px;
      background: #e8f1f0;
      color: #245c60;
      padding: 2px 9px;
      font-size: 12px;
      font-weight: 700;
    }}
    @media (max-width: 860px) {{
      .topbar {{ align-items: flex-start; flex-direction: column; justify-content: center; padding: 16px 0; }}
      th, td {{ white-space: normal; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <div>
        <h1>TaskShift Statistical Validation</h1>
        <p>{html.escape(stats["source"]["treatment_condition"])} compared with {html.escape(stats["source"]["baseline_condition"])} across seeds {html.escape(", ".join(map(str, stats["source"]["paired_seeds"])))}.</p>
      </div>
      <span class="badge">Bootstrap + Permutation</span>
    </div>
  </header>
  <main class="wrap">
    <section class="section">
      <h2>Effect Plots</h2>
      <div class="plots">
        <img src="{html.escape(relative_link(plots["metric_effects"], output_dir))}" alt="Metric effects">
        <img src="{html.escape(relative_link(plots["concept_effects"], output_dir))}" alt="Concept effects">
      </div>
    </section>
    <section class="section">
      <h2>Metric Tests</h2>
      {render_test_table(stats["metric_tests"])}
    </section>
    <section class="section">
      <h2>Final Block Concept Tests</h2>
      {render_test_table([row for row in stats["concept_tests"] if row["activation"] == "block_11"])}
    </section>
  </main>
</body>
</html>
"""


def render_test_table(rows: list[dict[str, Any]]) -> str:
    body = "\n".join(
        f"""
        <tr>
          <td>{html.escape(format_metric_label(row))}</td>
          <td>{html.escape(row.get("concept", row["activation"]))}</td>
          <td>{format_number(row["baseline_mean"])}</td>
          <td>{format_number(row["treatment_mean"])}</td>
          <td>{format_number(row["mean_difference"])}</td>
          <td>[{format_number(row["ci_low"])}, {format_number(row["ci_high"])}]</td>
          <td>{format_number(row["p_value"])}</td>
        </tr>
        """
        for row in rows
    )
    return f"""
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Target</th>
            <th>Baseline Mean</th>
            <th>Treatment Mean</th>
            <th>Difference</th>
            <th>95% CI</th>
            <th>Permutation p</th>
          </tr>
        </thead>
        <tbody>{body}</tbody>
      </table>
    """


def format_metric_label(row: dict[str, Any]) -> str:
    family = row["family"].replace("_", " ")
    activation = row["activation"].replace("_", " ")
    if row["family"] == "concept_shift":
        return f"{activation} concept shift"
    return f"{activation} {family}"


def format_number(value: Any) -> str:
    return f"{float(value):.3f}"


def relative_link(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path.resolve(), start=output_dir.resolve())


if __name__ == "__main__":
    main()
