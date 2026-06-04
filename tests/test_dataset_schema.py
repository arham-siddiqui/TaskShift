from pathlib import Path
import tempfile
import unittest

from data.build_prototype_dataset import build_dataset
from data.dataset_schema import load_jsonl, validate_metadata_file


class DatasetSchemaTest(unittest.TestCase):
    def test_prototype_dataset_matches_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "prototype"

            build_dataset(dataset_dir, frame_count=12, seed=7)

            metadata_path = dataset_dir / "metadata.jsonl"
            rows = load_jsonl(metadata_path)
            result = validate_metadata_file(metadata_path, dataset_dir)

            self.assertTrue(result.ok)
            self.assertEqual(result.rows_checked, 12)
            self.assertTrue(rows[0]["image_path"].startswith("frames/"))
            self.assertTrue((dataset_dir / rows[0]["image_path"]).exists())
            self.assertIn("visible_objects", rows[0])
            self.assertIn("navigation_labels", rows[0])
            self.assertIn("concept_labels", rows[0])


if __name__ == "__main__":
    unittest.main()
