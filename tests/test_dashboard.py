from pathlib import Path
import tempfile
import unittest

from dashboard.build_static import build_dashboard


class DashboardTest(unittest.TestCase):
    def test_builds_static_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "dashboard" / "index.html"
            build_dashboard(
                Path("artifacts/shift_metrics/representation_shift_summary.json"),
                output_path,
            )

            html = output_path.read_text(encoding="utf-8")
            self.assertIn("TaskShift Results", html)
            self.assertIn("Head Hidden", html)
            self.assertIn("../plots/cka_heatmap.png", html)
            self.assertGreater(output_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

