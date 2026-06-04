from pathlib import Path
import tempfile
import unittest

import torch

from activations.extract import extract_activations
from analysis.linear_probes import train_probe_artifact
from data.build_prototype_dataset import build_dataset
from models.train_heads import train_requested_heads


class LinearProbeTest(unittest.TestCase):
    def test_trains_probe_artifact_from_activations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "prototype"
            checkpoint_dir = root / "checkpoints"
            activation_path = root / "activations" / "passive_activations.pt"
            probe_path = root / "probes" / "passive_concept_probes.pt"

            build_dataset(dataset_dir, frame_count=40, seed=5)
            train_requested_heads(
                dataset_dir=dataset_dir,
                output_dir=checkpoint_dir,
                task="passive",
                epochs=1,
                batch_size=8,
            )
            extract_activations(
                dataset_dir=dataset_dir,
                checkpoint_path=checkpoint_dir / "passive_head.pt",
                output_path=activation_path,
                batch_size=8,
            )

            train_probe_artifact(
                activation_path=activation_path,
                output_path=probe_path,
                activation="head_hidden",
                min_positive=2,
            )

            artifact = torch.load(probe_path, map_location="cpu")
            head_probe = artifact["probes"]["head_hidden"]
            self.assertEqual(artifact["task"], "passive")
            self.assertIn("path", head_probe["concepts"])
            self.assertNotIn("agent", head_probe["concepts"])
            self.assertEqual(head_probe["weights"].shape[1], 128)
            self.assertIn("path", head_probe["metrics"])


if __name__ == "__main__":
    unittest.main()

