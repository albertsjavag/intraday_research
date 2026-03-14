from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class MarketData:
    """OHLCV DataFrames with DatetimeIndex (tz-naive UTC) and symbol columns."""

    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    def slice(self, start: str | None, end: str | None) -> "MarketData":
        """Return a MarketData covering [start, end] inclusive."""

        def _sl(df: pd.DataFrame) -> pd.DataFrame:
            return df.loc[start:end]

        return MarketData(
            close=_sl(self.close),
            open=_sl(self.open),
            high=_sl(self.high),
            low=_sl(self.low),
            volume=_sl(self.volume),
        )


class DataHandler(ABC):
    """Abstract base class for all data handlers."""

    @abstractmethod
    def load(
        self,
        symbols: list[str],
        start: str,
        end: str,
    ) -> MarketData:
        """Load OHLCV data for the given symbols and date range."""
        ...
