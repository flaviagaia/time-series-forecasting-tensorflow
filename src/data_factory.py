from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


def ensure_time_series_dataset(base_dir: str | Path) -> str:
    base_path = Path(base_dir)
    raw_dir = base_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = raw_dir / "daily_demand_series.csv"

    days = np.arange(240)
    trend = 80 + 0.12 * days
    seasonality = 12 * np.sin(2 * np.pi * days / 30)
    weekly = 3 * np.sin(2 * np.pi * days / 7)
    demand = trend + seasonality + weekly

    dataframe = pd.DataFrame(
        {
            "day_index": days,
            "demand": np.round(demand, 4),
        }
    )

    with NamedTemporaryFile("w", suffix=".csv", delete=False, dir=raw_dir, encoding="utf-8") as tmp_file:
        temp_path = Path(tmp_file.name)
    try:
        dataframe.to_csv(temp_path, index=False)
        temp_path.replace(dataset_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return str(dataset_path)
