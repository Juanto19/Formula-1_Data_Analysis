# data_adapter/kaggle.py
from pathlib import Path
from typing import Dict

import pandas as pd


# Path to …/data/kaggle/   (use __file__, not file)
DATA_DIR = (Path(__file__).resolve()            # kaggle.py
            .parents[1]                         # → repo root
            / "data" / "kaggle")                # → …/data/kaggle


class KaggleAdapter:
    """Loads raw Kaggle CSVs and exposes high-level getters used by the app."""

    # ---------- constructor ----------
    def __init__(self, root: Path = DATA_DIR) -> None:
        self.root = root
        self._cache: Dict[str, pd.DataFrame] = {}   # read-once cache

    # ---------- low-level ----------
    def _load(self, name: str) -> pd.DataFrame:
        """Read <root>/<name>.csv once per process and cache."""
        if name not in self._cache:
            self._cache[name] = pd.read_csv(self.root / f"{name}.csv")
        return self._cache[name]

    # ---------- high-level ----------
    def season_results(self, year: int) -> pd.DataFrame:
        """All results for one championship season (joined to race meta)."""
        races   = self._load("races")
        results = self._load("results")
        return (
            races.query("year == @year")
                 .merge(results, on="raceId", how="inner")
        )

    def race_results(self, year: int, gp_name: str) -> pd.DataFrame:
        """Results for a single Grand Prix (uses GP ‘official name’)."""
        races   = self._load("races")
        results = self._load("results")
        race_id = (
            races.query("year == @year and name == @gp_name")
                 .iloc[0]["raceId"]
        )
        return results.query("raceId == @race_id")
