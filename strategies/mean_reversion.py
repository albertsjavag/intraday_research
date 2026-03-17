from __future__ import annotations

import numpy as np
import pandas as pd

from data.base import MarketData
from strategies.base import Strategy

DEFAULT_LOOKBACK = 168      # 7 days of hourly bars (was 48 = 2 days)
DEFAULT_Z_THRESHOLD = 0.5
DEFAULT_EXIT_Z = 0.0        # exit when z-score reverts to mean
DEFAULT_LONG_ONLY = True


class MeanReversionStrategy(Strategy):
    """Z-score mean reversion with binary hold-till-revert logic.

    Enters a long position when z-score falls below -z_threshold (price
    is cheap vs recent average).  Holds until z-score reverts to exit_z
    (default 0 = back to mean), then exits.  Produces binary 0/1 signals
    rather than continuous z-score weights, which dramatically reduces
    turnover compared to continuous proportional weighting.

    Parameters
    ----------
    lookback:
        Rolling window for z-score computation (hourly bars).
        Default 168 = 7 days.
    z_threshold:
        Entry trigger: go long when z < -z_threshold.
    exit_z:
        Exit trigger: close long when z > exit_z.  Default 0 (return to mean).
    long_only:
        If True (default), short signals are suppressed.
    """

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
        exit_z: float = DEFAULT_EXIT_Z,
        long_only: bool = DEFAULT_LONG_ONLY,
    ) -> None:
        self._lookback = lookback
        self._z_threshold = z_threshold
        self._exit_z = exit_z
        self._long_only = long_only

    @property
    def name(self) -> str:
        return "mean_reversion"

    def fit(self, data: MarketData, params: dict | None = None) -> None:
        if params:
            self._lookback = params.get("lookback", self._lookback)
            self._z_threshold = params.get("z_threshold", self._z_threshold)
            self._exit_z = params.get("exit_z", self._exit_z)
            self._long_only = params.get("long_only", self._long_only)

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        close = data.close
        symbols = list(close.columns)
        n = len(close)

        roll_mean = close.rolling(window=self._lookback, min_periods=self._lookback).mean()
        roll_std = close.rolling(window=self._lookback, min_periods=self._lookback).std()
        roll_std = roll_std.replace(0, np.nan)
        zscore = (close - roll_mean) / roll_std

        # Binary hold-till-revert: track in-position state per symbol
        in_long = pd.DataFrame(False, index=close.index, columns=symbols)

        for sym in symbols:
            z = zscore[sym].values
            state = False
            for i in range(n):
                if np.isnan(z[i]):
                    state = False
                elif not state:
                    # Enter long when z crosses below -threshold
                    if z[i] < -self._z_threshold:
                        state = True
                else:
                    # Exit long when z reverts above exit_z
                    if z[i] > self._exit_z:
                        state = False
                in_long.at[close.index[i], sym] = state

        # Equal weight across qualifying long symbols
        n_long = in_long.sum(axis=1).replace(0, float("nan"))
        weights = in_long.astype(float).div(n_long, axis=0).fillna(0.0)

        return weights
