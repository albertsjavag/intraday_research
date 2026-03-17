"""Hour-of-day position sizing filter.

Crypto liquidity and momentum predictability peaks during the London/NY
overlap (14:00–20:00 UTC). In the Asian dead zone (02:00–06:00 UTC),
volume thins out, spreads widen, and signals are noisier. This wrapper
scales an inner strategy's weights by a time-of-day multiplier to reduce
exposure during low-quality trading windows.

Reference: "The crypto world trades at tea time" (Springer, 2024).
"""
from __future__ import annotations

import pandas as pd

from data.base import MarketData
from strategies.base import Strategy


class TodScalingFilter(Strategy):
    """Wraps a strategy and scales weights by UTC hour-of-day multipliers.

    Parameters
    ----------
    strategy:
        Inner strategy to scale.
    off_hours:
        UTC hours to reduce exposure (default 2–5 inclusive).
    off_multiplier:
        Weight multiplier during off-hours (default 0.5).
    """

    def __init__(
        self,
        strategy: Strategy,
        off_hours: list[int] | None = None,
        off_multiplier: float = 0.5,
    ) -> None:
        self._strategy = strategy
        self._off_hours = set(off_hours if off_hours is not None else [2, 3, 4, 5])
        self._off_multiplier = off_multiplier

    @property
    def name(self) -> str:
        return f"tod_{self._strategy.name}"

    def fit(self, data: MarketData, params: dict | None = None) -> None:
        self._strategy.fit(data, params)

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        signals = self._strategy.generate_signals(data)

        multiplier = pd.Series(1.0, index=signals.index)
        multiplier[signals.index.hour.isin(self._off_hours)] = self._off_multiplier

        return signals.multiply(multiplier, axis=0)
