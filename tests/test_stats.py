from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analysis.stats import build_stats_report, compute_stats, paired_sign_flip_p_value


class StatsTests(unittest.TestCase):
    def test_compute_stats_uses_paired_seed_differences(self) -> None:
        report = make_report()

        stats = compute_stats(
            report,
            baseline_condition="model:none",
            treatment_condition="model:final_block",
            bootstrap_samples=50,
            seed=1,
        )

        self.assertEqual(stats["source"]["paired_seeds"], [1, 2, 3])
        block_11 = find_metric(stats["metric_tests"], "matched_cka", "block_11")
        self.assertAlmostEqual(block_11["baseline_mean"], 1.0)
        self.assertAlmostEqual(block_11["treatment_mean"], 0.9)
        self.assertAlmostEqual(block_11["mean_difference"], -0.1)
        self.assertEqual(block_11["differences"], [-0.09999999999999998] * 3)

        container = find_concept(stats["concept_tests"], "block_11", "container")
        self.assertAlmostEqual(container["baseline_mean"], 0.0)
        self.assertAlmostEqual(container["treatment_mean"], 1.5)
        self.assertAlmostEqual(container["mean_difference"], 1.5)

    def test_sign_flip_p_value_is_exact_for_three_same_direction_differences(self) -> None:
        self.assertAlmostEqual(paired_sign_flip_p_value([1.0, 1.0, 1.0]), 0.25)

    def test_build_stats_report_writes_json_html_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison_path = root / "comparison_summary.json"
            comparison_path.write_text(__import__("json").dumps(make_report()), encoding="utf-8")

            paths = build_stats_report(
                comparison_path,
                root / "stats",
                baseline_condition="model:none",
                treatment_condition="model:final_block",
                bootstrap_samples=50,
                seed=1,
            )

            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["html"].exists())
            self.assertTrue(paths["plots"]["metric_effects"].exists())
            self.assertTrue(paths["plots"]["concept_effects"].exists())


def find_metric(rows: list[dict], family: str, activation: str) -> dict:
    for row in rows:
        if row["family"] == family and row["activation"] == activation:
            return row
    raise AssertionError(f"missing metric {family}:{activation}")


def find_concept(rows: list[dict], activation: str, concept: str) -> dict:
    for row in rows:
        if row["activation"] == activation and row.get("concept") == concept:
            return row
    raise AssertionError(f"missing concept {activation}:{concept}")


def make_report() -> dict:
    runs = []
    for seed in [1, 2, 3]:
        runs.append(make_run(seed, "model:none", 1.0, 0.0))
        runs.append(make_run(seed, "model:final_block", 0.9, 1.5))
    return {"runs": runs, "conditions": []}


def make_run(seed: int, condition: str, block_11_cka: float, block_11_shift: float) -> dict:
    return {
        "run_id": f"{condition}-{seed}",
        "condition": condition,
        "seed": seed,
        "matched_cka": {
            "backbone_features": block_11_cka,
            "block_9": 1.0,
            "block_11": block_11_cka,
            "head_hidden": 0.5,
        },
        "mean_shift_magnitude": {
            "backbone_features": block_11_shift,
            "block_11": block_11_shift,
            "head_hidden": 2.0,
        },
        "concept_shift": {
            "backbone_features": [
                {"concept": "container", "shift_magnitude": block_11_shift},
            ],
            "block_11": [
                {"concept": "container", "shift_magnitude": block_11_shift},
            ],
            "head_hidden": [
                {"concept": "container", "shift_magnitude": 2.0},
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
