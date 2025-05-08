from pathlib import Path	
import pandas as pd	

DATA_DIR = Path(file).resolve().parents[2] / "data" / "kaggle"

class KaggleAdapter:
    """Loads raw Kaggle CSVs and exposes high-level getters used by the app."""
def __init__(self, root: Path = DATA_DIR):  
    self.root = root  
    self._cache = {}  # type: dict[str, pd.DataFrame]

# ---------- low-level ----------  
def _load(self, name: str) -> pd.DataFrame:  
    if name not in self._cache:  
        self._cache[name] = pd.read_csv(self.root / f"{name}.csv")  
    return self._cache[name]  

# ---------- high-level ----------  
def season_results(self, year: int) -> pd.DataFrame:  
    races = self._load("races")  
    results = self._load("results")  
    return (  
        races.query("year == @year")  
              .merge(results, on="raceId")  
    )  

def race_results(self, year: int, gp_name: str) -> pd.DataFrame:  
    races = self._load("races")  
    results = self._load("results")  
    race_id = races.query("year == @year & name == @gp_name").iloc[0]["raceId"]  
    return results.query("raceId == @race_id")  