# data_adapter/__init__.py
from enum import Enum

from .kaggle import KaggleAdapter
# from .fastf1 import FastF1Adapter   # ← you can add this later


class Source(Enum):
    """Enumerates every back-end the app knows how to talk to."""
    KAGGLE = "kaggle"
    FASTF1 = "fastf1"        # placeholder for future live data


def get_adapter(source: "Source | str" = Source.KAGGLE):
    """
    Factory that returns an *instance* of the requested adapter.

    Example
    -------
    >>> from data_adapter import get_adapter, Source
    >>> ad = get_adapter(Source.KAGGLE)
    >>> ad.season_results(2020).head()
    """
    # allow passing a raw string so `source="kaggle"` also works
    if isinstance(source, str):
        source = Source(source.lower())

    if source is Source.KAGGLE:
        return KaggleAdapter()
    if source is Source.FASTF1:
        raise NotImplementedError("FastF1Adapter coming soon")

    raise ValueError(f"Unknown data source: {source}")
