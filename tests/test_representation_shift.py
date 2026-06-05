from pathlib import Path
import tempfile
import unittest

import torch

from activations.extract import extract_requested_activations
from analysis.cka import linear_cka
from analysis.linear_probes import train_requested_probes
from analysis.representation_shift import compare_representations
from data.build_prototype_dataset import build_dataset
from models.train_heads import train_requested_heads


class RepresentationShiftTest(unittest.TestCase):
    def test_linear_cka_bounds(self) -> None:
        x = torch.randn(10, 4)
        self.assertAlmostEqual(linear_cka(x, x), 1.0, places=5)
        self.assertGreaterEqual(linear_cka(x, torch.randn(10, 3)), 0.0)

    def test_compares_probe_and_activation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "prototype"
            checkpoint_dir = root / "checkpoints"
            activation_dir = root / "activations"
            probe_dir = root / "probes"
            output_dir = root / "shift"

            build_dataset(dataset_dir, frame_count=48, seed=9)
            train_requested_heads(
                dataset_dir=dataset_dir,
                output_dir=checkpoint_dir,
                task="both",
                epochs=1,
                batch_size=12,
            )
            extract_requested_activations(
                dataset_dir=dataset_dir,
                checkpoint_dir=checkpoint_dir,
                output_dir=activation_dir,
                task="both",
                batch_size=12,
            )
            train_requested_probes(
                activation_dir=activation_dir,
                output_dir=probe_dir,
                task="both",
                activation="head_hidden",
                min_positive=2,
            )

            outputs = compare_representations(probe_dir, activation_dir, output_dir)
            artifact = torch.load(outputs["pt"], map_location="cpu")

            self.assertTrue(outputs["json"].exists())
            self.assertIn("head_hidden", artifact["probe_shift"])
            self.assertIn("head_hidden", artifact["cka"]["matched"])
            self.assertGreater(len(artifact["probe_shift"]["head_hidden"]["concept_shift"]), 0)


if __name__ == "__main__":
    unittest.main()

