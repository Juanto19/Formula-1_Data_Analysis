from enum import Enum
from .kaggle import KaggleAdapter
# later we could import FastF1Adapter

class Source(Enum):
    KAGGLE = "kaggle"
    FASTF1 = "fastf1"

def get_adapter(source: Source = Source.KAGGLE):
    if source is Source.KAGGLE:
        return KaggleAdapter()
    # if source is Source.FASTF1: return FastF1Adapter()
    raise ValueError(source)