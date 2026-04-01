from __future__ import annotations

import unittest
from pathlib import Path

from src.modeling import run_pipeline


class TimeSeriesForecastingTensorFlowTestCase(unittest.TestCase):
    def test_pipeline_contract(self) -> None:
        project_dir = Path(__file__).resolve().parents[1]
        summary = run_pipeline(project_dir)
        self.assertEqual(summary["row_count"], 240)
        self.assertEqual(summary["window_size"], 14)
        self.assertGreater(summary["train_window_count"], 100)
        self.assertGreater(summary["test_window_count"], 20)
        self.assertLess(summary["mae"], 12.0)
        self.assertLess(summary["rmse"], 13.0)


if __name__ == "__main__":
    unittest.main()
