from pathlib import Path
import tempfile
import unittest

from analysis.plots import plot_shift_summary


class PlotTest(unittest.TestCase):
    def test_writes_plot_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "plots"
            outputs = plot_shift_summary(
                Path("artifacts/shift_metrics/representation_shift_summary.json"),
                output_dir,
            )

            self.assertIn("cka_heatmap", outputs)
            for path in outputs.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

