from pathlib import Path
import tempfile
import unittest

import torch

from data.build_prototype_dataset import build_dataset
from data.taskshift_dataset import (
    NAVIGATION_BINARY_KEYS,
    OBJECT_VOCAB,
    ROOM_VOCAB,
    TaskShiftDataset,
)


class TaskShiftDatasetTest(unittest.TestCase):
    def test_dataset_returns_training_and_probe_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "prototype"
            build_dataset(dataset_dir, frame_count=16, seed=7)

            dataset = TaskShiftDataset(dataset_dir)
            sample = dataset[0]

            self.assertEqual(len(dataset), 16)
            self.assertEqual(tuple(sample["image"].shape), (3, 120, 160))
            self.assertEqual(sample["image"].dtype, torch.float32)
            self.assertGreaterEqual(float(sample["image"].min()), 0.0)
            self.assertLessEqual(float(sample["image"].max()), 1.0)

            passive = sample["passive_targets"]
            self.assertEqual(tuple(passive["objects"].shape), (len(OBJECT_VOCAB),))
            self.assertEqual(passive["objects"].dtype, torch.float32)
            self.assertEqual(passive["room"].dtype, torch.long)
            self.assertIn(int(passive["room"]), range(len(ROOM_VOCAB)))

            navigation = sample["navigation_targets"]
            self.assertEqual(tuple(navigation["binary"].shape), (len(NAVIGATION_BINARY_KEYS),))
            self.assertEqual(navigation["binary"].dtype, torch.float32)
            self.assertEqual(navigation["action"].dtype, torch.long)

            self.assertEqual(tuple(sample["concept_targets"].shape), (6,))
            self.assertEqual(sample["concept_targets"].dtype, torch.float32)
            self.assertIn("metadata", sample)

    def test_active_concepts_skip_all_negative_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "prototype"
            build_dataset(dataset_dir, frame_count=24, seed=11)

            dataset = TaskShiftDataset(dataset_dir)

            self.assertEqual(dataset.concept_positive_counts()["agent"], 0)
            self.assertNotIn("agent", dataset.active_concepts())
            self.assertIn("path", dataset.active_concepts())


if __name__ == "__main__":
    unittest.main()

