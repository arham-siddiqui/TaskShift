"""Summarize multiple TaskShift runs into a comparison dashboard."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TaskShift experiment runs.")
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/comparison"))
    args = parser.parse_args()

    paths = build_comparison_report(args.summaries, args.output_dir)
    print(f"saved comparison summary: {paths['json']}")
    print(f"saved comparison dashboard: {paths['html']}")


def build_comparison_report(summary_paths: list[Path], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = summarize_runs(summary_paths)

    json_path = output_dir / "comparison_summary.json"
    html_path = output_dir / "index.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(render_comparison_dashboard(report, output_dir), encoding="utf-8")
    return {"json": json_path, "html": html_path}


def summarize_runs(summary_paths: list[Path]) -> dict[str, Any]:
    if not summary_paths:
        raise ValueError("at least one summary path is required")

    runs = [load_run_summary(path) for path in summary_paths]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["condition"]].append(run)

    conditions = []
    for condition, condition_runs in sorted(grouped.items()):
        conditions.append(summarize_condition(condition, condition_runs))

    return {
        "runs": runs,
        "conditions": conditions,
    }


def load_run_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    manifest = load_manifest(path)
    config = manifest.get("config", {})
    run_id = manifest.get("run_id") or path.parents[1].name
    backbone = config.get("backbone") or infer_backbone(summary)
    train_backbone = config.get("train_backbone") or infer_train_mode(summary)
    seed = config.get("seed")

    matched_cka = {
        name: values.get("matched_layer_cka")
        for name, values in summary["cka"].get("matched", {}).items()
    }
    mean_shift = {
        activation: result.get("mean_shift_magnitude")
        for activation, result in summary["probe_shift"].items()
    }
    mean_correlation = {
        activation: result.get("mean_tuning_correlation")
        for activation, result in summary["probe_shift"].items()
    }
    concept_shift = {
        activation: result.get("concept_shift", [])
        for activation, result in summary["probe_shift"].items()
    }

    return {
        "run_id": run_id,
        "summary_path": str(path),
        "dashboard_path": str(path.parents[1] / "dashboard" / "index.html"),
        "seed": seed,
        "backbone": backbone,
        "train_backbone": train_backbone,
        "condition": f"{backbone}:{train_backbone}",
        "matched_cka": matched_cka,
        "mean_shift_magnitude": mean_shift,
        "mean_tuning_correlation": mean_correlation,
        "concept_shift": concept_shift,
    }


def summarize_condition(condition: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    activation_names = sorted(
        {
            activation
            for run in runs
            for activation in run["matched_cka"]
        }
    )
    cka = {
        activation: describe_values(
            run["matched_cka"].get(activation)
            for run in runs
        )
        for activation in activation_names
    }
    mean_shift = {
        activation: describe_values(
            run["mean_shift_magnitude"].get(activation)
            for run in runs
        )
        for activation in activation_names
    }
    mean_correlation = {
        activation: describe_values(
            run["mean_tuning_correlation"].get(activation)
            for run in runs
        )
        for activation in activation_names
    }

    concept_shift_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for activation, rows in run["concept_shift"].items():
            for row in rows:
                concept_shift_values[activation][row["concept"]].append(row["shift_magnitude"])

    top_concepts = {
        activation: sorted(
            (
                {
                    "concept": concept,
                    **describe_values(values),
                }
                for concept, values in concepts.items()
            ),
            key=lambda row: row["mean"],
            reverse=True,
        )[:5]
        for activation, concepts in concept_shift_values.items()
    }

    return {
        "condition": condition,
        "run_count": len(runs),
        "seeds": sorted(seed for seed in (run.get("seed") for run in runs) if seed is not None),
        "matched_cka": cka,
        "mean_shift_magnitude": mean_shift,
        "mean_tuning_correlation": mean_correlation,
        "top_concepts": top_concepts,
    }


def describe_values(values: Any) -> dict[str, float | int | None]:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    mean = sum(cleaned) / len(cleaned)
    variance = sum((value - mean) ** 2 for value in cleaned) / len(cleaned)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(cleaned),
        "max": max(cleaned),
        "n": len(cleaned),
    }


def load_manifest(summary_path: Path) -> dict[str, Any]:
    manifest_path = summary_path.parents[1] / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def infer_backbone(summary: dict[str, Any]) -> str:
    source_paths = summary.get("source_paths", {})
    passive_activations = source_paths.get("passive_activations", "")
    if "dinov2" in passive_activations:
        return "dinov2_vits14"
    return "unknown"


def infer_train_mode(summary: dict[str, Any]) -> str:
    source_text = " ".join(str(value) for value in summary.get("source_paths", {}).values())
    if "finalblock" in source_text or "final_block" in source_text:
        return "final_block"
    matched = summary.get("cka", {}).get("matched", {})
    backbone_cka = matched.get("backbone_features", {}).get("matched_layer_cka")
    if backbone_cka is not None and backbone_cka < 0.999:
        return "trained_backbone"
    return "none"


def render_comparison_dashboard(report: dict[str, Any], output_dir: Path) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TaskShift Experiment Comparison</title>
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
    a {{ color: var(--accent); text-decoration: none; font-weight: 700; }}
    a:hover {{ text-decoration: underline; }}
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
        <h1>TaskShift Experiment Comparison</h1>
        <p>{len(report["runs"])} runs across {len(report["conditions"])} conditions.</p>
      </div>
      <span class="badge">Cross-run Report</span>
    </div>
  </header>
  <main class="wrap">
    <section class="section">
      <h2>Condition Summary</h2>
      {render_condition_table(report["conditions"])}
    </section>
    <section class="section">
      <h2>Run Index</h2>
      {render_run_table(report["runs"], output_dir)}
    </section>
  </main>
</body>
</html>
"""


def render_condition_table(conditions: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        f"""
        <tr>
          <td>{html.escape(condition["condition"])}</td>
          <td>{condition["run_count"]}</td>
          <td>{html.escape(', '.join(map(str, condition["seeds"])))}</td>
          <td>{format_stat(condition["matched_cka"].get("backbone_features"))}</td>
          <td>{format_stat(condition["matched_cka"].get("head_hidden"))}</td>
          <td>{format_stat(condition["mean_shift_magnitude"].get("backbone_features"))}</td>
          <td>{format_stat(condition["mean_shift_magnitude"].get("head_hidden"))}</td>
          <td>{html.escape(format_top_concepts(condition, "head_hidden"))}</td>
        </tr>
        """
        for condition in conditions
    )
    return f"""
      <table>
        <thead>
          <tr>
            <th>Condition</th>
            <th>Runs</th>
            <th>Seeds</th>
            <th>Backbone CKA</th>
            <th>Head CKA</th>
            <th>Backbone Shift</th>
            <th>Head Shift</th>
            <th>Top Head Concepts</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    """


def render_run_table(runs: list[dict[str, Any]], output_dir: Path) -> str:
    rows = "\n".join(
        f"""
        <tr>
          <td>{html.escape(run["run_id"])}</td>
          <td>{html.escape(run["condition"])}</td>
          <td>{html.escape(str(run.get("seed", "n/a")))}</td>
          <td>{format_number(run["matched_cka"].get("backbone_features"))}</td>
          <td>{format_number(run["matched_cka"].get("head_hidden"))}</td>
          <td>{format_number(run["mean_shift_magnitude"].get("backbone_features"))}</td>
          <td>{format_number(run["mean_shift_magnitude"].get("head_hidden"))}</td>
          <td><a href="{html.escape(relative_link(run["dashboard_path"], output_dir))}">dashboard</a></td>
        </tr>
        """
        for run in sorted(runs, key=lambda item: item["run_id"])
    )
    return f"""
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Condition</th>
            <th>Seed</th>
            <th>Backbone CKA</th>
            <th>Head CKA</th>
            <th>Backbone Shift</th>
            <th>Head Shift</th>
            <th>Report</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    """


def format_top_concepts(condition: dict[str, Any], activation: str) -> str:
    concepts = condition.get("top_concepts", {}).get(activation, [])
    return ", ".join(f"{row['concept']} {format_number(row['mean'])}" for row in concepts[:3]) or "n/a"


def format_stat(stat: dict[str, Any] | None) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    return f"{stat['mean']:.3f} +/- {stat['std']:.3f}"


def format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def relative_link(path: str, output_dir: Path) -> str:
    return os.path.relpath(Path(path).resolve(), start=output_dir.resolve())


if __name__ == "__main__":
    main()
