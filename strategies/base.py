from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from data.base import MarketData


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    A strategy maps MarketData → a DataFrame of target weights
    (DatetimeIndex × symbol columns).  Weights are target portfolio fractions:
        +1.0  = 100% long
        -1.0  = 100% short
         0.0  = flat
    Weights do NOT need to sum to 1 — the engine normalises them.
    """

    @property
    def name(self) -> str:
        """Human-readable strategy name used in reports and dashboards."""
        return self.__class__.__name__

    def fit(self, data: MarketData, params: dict | None = None) -> None:
        """Optional fitting step for parametric / ML strategies.

        Non-parametric strategies may leave this as a no-op.
        """

    @abstractmethod
    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        """Generate target weights for each (timestamp, symbol) pair.

        Parameters
        ----------
        data:
            Full MarketData for the window (including warmup bars).

        Returns
        -------
        pd.DataFrame
            DatetimeIndex × symbol columns, float weights.
        """
        ...
