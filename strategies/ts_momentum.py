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

    Parameters
    ----------
    min_holding_bars:
        Minimum consecutive bars a new signal must be stable before the
        position changes.  Prevents reacting to brief MA crossovers that
        reverse within hours.  Default 0 = no filter.
        Recommended 12-24 for hourly bars to reduce turnover.
    """

    def __init__(
        self,
        lookbacks: list[int] | None = None,
        long_threshold: float = DEFAULT_LONG_THRESHOLD,
        min_holding_bars: int = 0,
    ) -> None:
        self._lookbacks = lookbacks or DEFAULT_LOOKBACKS
        self._long_threshold = long_threshold
        self._min_holding_bars = min_holding_bars

    @property
    def name(self) -> str:
        return "ts_momentum"

    def fit(self, data: MarketData, params: dict | None = None) -> None:
        if params:
            self._lookbacks = params.get("lookbacks", self._lookbacks)
            self._long_threshold = params.get("long_threshold", self._long_threshold)
            self._min_holding_bars = params.get("min_holding_bars", self._min_holding_bars)

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
        raw_weights = long_mask.astype(float).div(n_long, axis=0).fillna(0.0)

        if self._min_holding_bars <= 0:
            return raw_weights

        # Signal persistence filter: only act on a signal change after it has
        # been stable for min_holding_bars consecutive bars.
        # Per-symbol: track how long the new signal has been in the new state.
        weights = raw_weights.copy()
        n_bars = len(close.index)
        symbols = list(close.columns)

        for sym in symbols:
            raw_col = raw_weights[sym].values
            out_col = raw_col.copy()
            bars_in_new_state: int = 0
            current_signal: float = raw_col[0]
            pending_signal: float = raw_col[0]

            for i in range(1, n_bars):
                new_val = raw_col[i]
                if new_val != pending_signal:
                    # Signal changed — reset countdown
                    pending_signal = new_val
                    bars_in_new_state = 1
                else:
                    bars_in_new_state += 1

                if bars_in_new_state >= self._min_holding_bars:
                    current_signal = pending_signal

                out_col[i] = current_signal

            weights[sym] = out_col

        return weights
