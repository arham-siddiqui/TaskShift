"""Build a static HTML dashboard for TaskShift results."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


PLOT_FILES = {
    "backbone_features": {
        "Concept Shift": "../plots/backbone_features_concept_shift.png",
        "Tuning Correlation": "../plots/backbone_features_tuning_correlation.png",
    },
    "head_hidden": {
        "Concept Shift": "../plots/head_hidden_concept_shift.png",
        "Tuning Correlation": "../plots/head_hidden_tuning_correlation.png",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static TaskShift dashboard.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/shift_metrics/representation_shift_summary.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/dashboard/index.html"))
    args = parser.parse_args()

    build_dashboard(args.summary, args.output)


def build_dashboard(summary_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_path.write_text(render_dashboard(summary), encoding="utf-8")
    print(f"saved dashboard: {output_path}")
    return output_path


def render_dashboard(summary: dict[str, Any]) -> str:
    activation_names = list(summary["probe_shift"].keys())
    first_activation = activation_names[0] if activation_names else ""
    head_result = summary["probe_shift"].get("head_hidden", {})
    top_shift = first_row(head_result.get("concept_shift", []))
    matched_cka = summary["cka"]["matched"]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TaskShift Results</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #20242a;
      --muted: #68707c;
      --line: #d9dee6;
      --accent: #2f6f73;
      --accent-2: #8a5a44;
      --warn: #a44a3f;
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
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    .wrap {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
    }}
    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      min-height: 76px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    main {{
      padding: 28px 0 40px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      min-height: 94px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric-value {{
      display: block;
      margin-top: 8px;
      font-size: 24px;
      font-weight: 700;
    }}
    .metric-note {{
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      margin: 22px 0 14px;
      border-bottom: 1px solid var(--line);
    }}
    .tab {{
      appearance: none;
      border: 1px solid var(--line);
      border-bottom: 0;
      border-radius: 8px 8px 0 0;
      background: #eef1f4;
      color: var(--text);
      cursor: pointer;
      font: inherit;
      padding: 10px 14px;
    }}
    .tab[aria-selected="true"] {{
      background: var(--panel);
      color: var(--accent);
      font-weight: 700;
    }}
    .panel {{
      display: none;
    }}
    .panel.active {{
      display: block;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      align-items: start;
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 15px;
      letter-spacing: 0;
    }}
    img {{
      display: block;
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
      vertical-align: middle;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: #f9fafb;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
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
    .caption {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 860px) {{
      .metrics, .grid {{
        grid-template-columns: 1fr;
      }}
      .topbar {{
        align-items: flex-start;
        flex-direction: column;
        justify-content: center;
        padding: 16px 0;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <div>
        <h1>TaskShift Results</h1>
        <p class="subtitle">Prototype representation-shift analysis across passive and navigation heads.</p>
      </div>
      <span class="badge">Static Report</span>
    </div>
  </header>
  <main class="wrap">
    <section class="metrics" aria-label="Summary metrics">
      {render_metric("Top Head Shift", format_metric(top_shift.get("shift_magnitude")), top_shift.get("concept", "n/a"))}
      {render_metric("Head Hidden CKA", format_metric(matched_cka.get("head_hidden", {}).get("matched_layer_cka")), "passive vs navigation")}
      {render_metric("Backbone CKA", format_metric(matched_cka.get("backbone_features", {}).get("matched_layer_cka")), "frozen shared features")}
      {render_metric("Compared Concepts", str(len(head_result.get("concepts_compared", []))), "agent skipped if all-negative")}
    </section>

    <section class="section">
      <h2>CKA Similarity</h2>
      <img src="../plots/cka_heatmap.png" alt="CKA heatmap comparing passive and navigation activations">
      <p class="caption">CKA compares the overall geometry of activation spaces. Higher values mean the same images are organized more similarly.</p>
    </section>

    <nav class="tabs" aria-label="Activation views">
      {render_tabs(activation_names, first_activation)}
    </nav>

    {render_activation_panels(summary, activation_names, first_activation)}
  </main>
  <script>
    const tabs = Array.from(document.querySelectorAll(".tab"));
    const panels = Array.from(document.querySelectorAll(".panel"));
    tabs.forEach((tab) => {{
      tab.addEventListener("click", () => {{
        const target = tab.dataset.target;
        tabs.forEach((item) => item.setAttribute("aria-selected", String(item === tab)));
        panels.forEach((panel) => panel.classList.toggle("active", panel.id === target));
      }});
    }});
  </script>
</body>
</html>
"""


def render_metric(label: str, value: str, note: str) -> str:
    return f"""
      <article class="metric">
        <span class="metric-label">{html.escape(label)}</span>
        <span class="metric-value">{html.escape(value)}</span>
        <span class="metric-note">{html.escape(note)}</span>
      </article>
    """


def render_tabs(activation_names: list[str], selected: str) -> str:
    return "\n".join(
        f'<button class="tab" type="button" data-target="panel-{html.escape(name)}" aria-selected="{str(name == selected).lower()}">{html.escape(format_activation_name(name))}</button>'
        for name in activation_names
    )


def render_activation_panels(
    summary: dict[str, Any],
    activation_names: list[str],
    selected: str,
) -> str:
    panels = []
    for activation_name in activation_names:
        result = summary["probe_shift"][activation_name]
        active = " active" if activation_name == selected else ""
        panels.append(
            f"""
    <section id="panel-{html.escape(activation_name)}" class="panel{active}">
      <div class="grid">
        {render_plot_card(activation_name, "Concept Shift")}
        {render_plot_card(activation_name, "Tuning Correlation")}
      </div>
      <section class="section">
        <h2>{html.escape(format_activation_name(activation_name))} Metrics</h2>
        {render_shift_table(result["concept_shift"])}
      </section>
    </section>
            """
        )
    return "\n".join(panels)


def render_plot_card(activation_name: str, title: str) -> str:
    src = PLOT_FILES[activation_name][title]
    return f"""
      <section class="section">
        <h2>{html.escape(title)}</h2>
        <img src="{html.escape(src)}" alt="{html.escape(format_activation_name(activation_name))} {html.escape(title)} plot">
      </section>
    """


def render_shift_table(rows: list[dict[str, Any]]) -> str:
    body = "\n".join(
        f"""
        <tr>
          <td>{html.escape(row["concept"])}</td>
          <td>{format_metric(row["shift_magnitude"])}</td>
          <td>{format_metric(row["tuning_correlation"])}</td>
          <td>{format_metric(row["cosine_similarity"])}</td>
          <td>{format_metric(row["navigation_minus_passive_norm"])}</td>
        </tr>
        """
        for row in rows
    )
    return f"""
      <table>
        <thead>
          <tr>
            <th>Concept</th>
            <th>Shift</th>
            <th>Correlation</th>
            <th>Cosine</th>
            <th>Norm Delta</th>
          </tr>
        </thead>
        <tbody>
          {body}
        </tbody>
      </table>
    """


def first_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return rows[0]


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_activation_name(name: str) -> str:
    return name.replace("_", " ").title()


if __name__ == "__main__":
    main()

