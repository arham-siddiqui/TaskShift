from pathlib import Path
import tempfile
import unittest

import torch

from activations.extract import extract_activations
from data.build_prototype_dataset import build_dataset
from models.train_heads import train_requested_heads


class ActivationExtractionTest(unittest.TestCase):
    def test_extracts_activation_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "prototype"
            checkpoint_dir = root / "checkpoints"
            output_path = root / "activations" / "passive_activations.pt"

            build_dataset(dataset_dir, frame_count=20, seed=3)
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
                output_path=output_path,
                batch_size=8,
            )

            artifact = torch.load(output_path, map_location="cpu")
            self.assertEqual(artifact["task"], "passive")
            self.assertEqual(tuple(artifact["activations"]["backbone_features"].shape), (20, 198))
            self.assertEqual(tuple(artifact["activations"]["head_hidden"].shape), (20, 128))
            self.assertEqual(tuple(artifact["targets"]["concepts"].shape), (20, 6))
            self.assertEqual(len(artifact["metadata"]), 20)
            self.assertIn("active_concepts", artifact["vocab"])


if __name__ == "__main__":
    unittest.main()

