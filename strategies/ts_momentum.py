from __future__ import annotations

import pandas as pd

from data.base import MarketData
from strategies.base import Strategy

# Default hourly MA lookbacks: 1d, 3d, 1w, 2w
DEFAULT_LOOKBACKS = [24, 72, 168, 336]
DEFAULT_LONG_THRESHOLD = 0.25  # long if above >= 25% of MAs (i.e. ≥1 of 4)


class TimeSeriesMomentumStrategy(Strategy):
    """Time-series momentum using multi-timeframe hourly moving averages.

    For each symbol independently, compute the fraction of MAs that price
    is currently above.  Go long when that fraction exceeds long_threshold.
    Returns equal weight (1/N_long) for qualifying symbols, 0 otherwise.
    """

    def __init__(
        self,
        lookbacks: list[int] | None = None,
        long_threshold: float = DEFAULT_LONG_THRESHOLD,
    ) -> None:
        self._lookbacks = lookbacks or DEFAULT_LOOKBACKS
        self._long_threshold = long_threshold

    @property
    def name(self) -> str:
        return "ts_momentum"

    def fit(self, data: MarketData, params: dict | None = None) -> None:
        if params:
            self._lookbacks = params.get("lookbacks", self._lookbacks)
            self._long_threshold = params.get("long_threshold", self._long_threshold)

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        close = data.close

        # Compute MA score: fraction of lookbacks that price is above its MA
        scores = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for lb in self._lookbacks:
            ma = close.rolling(window=lb, min_periods=lb).mean()
            above = (close > ma).astype(float)
            scores += above

        scores /= len(self._lookbacks)

        # Long signal: score >= long_threshold
        long_mask = scores >= self._long_threshold

        # Equal weight across qualifying symbols at each bar
        n_long = long_mask.sum(axis=1).replace(0, float("nan"))
        weights = long_mask.astype(float).div(n_long, axis=0).fillna(0.0)

        return weights
