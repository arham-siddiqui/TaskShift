import json
from pathlib import Path
import tempfile
import unittest

from experiments.run_sweep import RunConfig, validate_config
from experiments.summarize_runs import build_comparison_report, summarize_runs


class ExperimentSummaryTest(unittest.TestCase):
    def test_summarizes_runs_by_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = write_fake_run(root, "prototype_none_seed1", seed=1, head_cka=0.8, head_shift=2.0)
            second = write_fake_run(root, "prototype_none_seed2", seed=2, head_cka=0.6, head_shift=4.0)

            report = summarize_runs([first, second])

            self.assertEqual(len(report["runs"]), 2)
            self.assertEqual(len(report["conditions"]), 1)
            condition = report["conditions"][0]
            self.assertEqual(condition["condition"], "prototype:none")
            self.assertEqual(condition["run_count"], 2)
            self.assertAlmostEqual(condition["matched_cka"]["head_hidden"]["mean"], 0.7)
            self.assertAlmostEqual(condition["mean_shift_magnitude"]["head_hidden"]["mean"], 3.0)
            self.assertEqual(condition["top_concepts"]["head_hidden"][0]["concept"], "container")

    def test_builds_comparison_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = write_fake_run(root, "prototype_none_seed1", seed=1, head_cka=0.8, head_shift=2.0)
            output_dir = root / "comparison"

            paths = build_comparison_report([summary], output_dir)

            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["html"].exists())
            html = paths["html"].read_text(encoding="utf-8")
            self.assertIn("TaskShift Experiment Comparison", html)
            self.assertIn("prototype:none", html)
            self.assertIn("dashboard", html)

    def test_final_block_requires_dinov2(self) -> None:
        config = RunConfig(
            experiment="test",
            seed=1,
            backbone="prototype",
            train_backbone="final_block",
            frames=12,
            epochs=1,
            batch_size=4,
            lr=1e-3,
            backbone_lr=1e-5,
        )

        with self.assertRaises(ValueError):
            validate_config(config)


def write_fake_run(root: Path, run_id: str, seed: int, head_cka: float, head_shift: float) -> Path:
    run_dir = root / "runs" / run_id
    shift_dir = run_dir / "shift_metrics"
    shift_dir.mkdir(parents=True)
    (run_dir / "dashboard").mkdir()
    (run_dir / "dashboard" / "index.html").write_text("<html></html>", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "config": {
            "seed": seed,
            "backbone": "prototype",
            "train_backbone": "none",
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary = {
        "probe_shift": {
            "backbone_features": {
                "mean_shift_magnitude": 0.0,
                "mean_tuning_correlation": 1.0,
                "concept_shift": [
                    {
                        "concept": "path",
                        "shift_magnitude": 0.0,
                        "tuning_correlation": 1.0,
                        "cosine_similarity": 1.0,
                    }
                ],
            },
            "head_hidden": {
                "mean_shift_magnitude": head_shift,
                "mean_tuning_correlation": 0.1,
                "concept_shift": [
                    {
                        "concept": "container",
                        "shift_magnitude": head_shift,
                        "tuning_correlation": 0.1,
                        "cosine_similarity": 0.1,
                    }
                ],
            },
        },
        "cka": {
            "matched": {
                "backbone_features": {"matched_layer_cka": 1.0},
                "head_hidden": {"matched_layer_cka": head_cka},
            }
        },
        "source_paths": {},
    }
    summary_path = shift_dir / "representation_shift_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


if __name__ == "__main__":
    unittest.main()
