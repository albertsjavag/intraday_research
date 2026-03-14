from __future__ import annotations

import numpy as np
import pandas as pd

from data.base import MarketData
from strategies.base import Strategy

DEFAULT_LOOKBACK = 48       # 2 days of hourly bars
DEFAULT_Z_THRESHOLD = 0.5
DEFAULT_LONG_ONLY = True


class MeanReversionStrategy(Strategy):
    """Z-score mean reversion.

    Computes the z-score of price relative to its rolling mean over `lookback`
    bars.  Goes long when the z-score is below -z_threshold (price is cheap
    relative to its recent average).

    With long_only=True, short signals are suppressed — appropriate for
    trending crypto markets.

    Weight is proportional to the absolute z-score, capped at 1.0 per symbol.
    """

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
        long_only: bool = DEFAULT_LONG_ONLY,
    ) -> None:
        self._lookback = lookback
        self._z_threshold = z_threshold
        self._long_only = long_only

    @property
    def name(self) -> str:
        return "mean_reversion"

    def fit(self, data: MarketData, params: dict | None = None) -> None:
        if params:
            self._lookback = params.get("lookback", self._lookback)
            self._z_threshold = params.get("z_threshold", self._z_threshold)
            self._long_only = params.get("long_only", self._long_only)

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        close = data.close

        roll_mean = close.rolling(window=self._lookback, min_periods=self._lookback).mean()
        roll_std = close.rolling(window=self._lookback, min_periods=self._lookback).std()

        # Avoid division by zero
        roll_std = roll_std.replace(0, np.nan)
        zscore = (close - roll_mean) / roll_std

        # Long signal: price significantly below its mean
        long_signal = zscore < -self._z_threshold
        # Short signal (optional): price significantly above its mean
        short_signal = zscore > self._z_threshold

        if self._long_only:
            short_signal = pd.DataFrame(False, index=short_signal.index, columns=short_signal.columns)

        # Weights proportional to abs(z), capped at 1.0
        long_weights = (-zscore).clip(lower=0).where(long_signal, 0.0)
        short_weights = (-zscore).clip(upper=0).where(short_signal, 0.0)

        weights = long_weights + short_weights

        # Cap each symbol's weight at 1.0 in absolute terms
        weights = weights.clip(lower=-1.0, upper=1.0)

        # Normalise across symbols at each bar (sum of abs weights ≤ 1)
        abs_sum = weights.abs().sum(axis=1).replace(0, np.nan)
        weights = weights.div(abs_sum, axis=0).fillna(0.0)

        return weights
