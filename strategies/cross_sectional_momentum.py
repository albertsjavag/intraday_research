"""Cross-sectional momentum: long the top-ranked symbol(s) by recent return.

Unlike time-series momentum (which asks "is this asset trending up?"), this
asks "which asset is trending up the most relative to the others?" and
concentrates capital there. Rebalances on a weekly schedule to limit turnover.

Academic basis: Drogen/Hoffstein (SSRN 4322637), Huang/Sangiorgi (SSRN 4825389).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.base import MarketData
from strategies.base import Strategy


class CrossSectionalMomentumStrategy(Strategy):
    """Rank symbols by N-bar return; go long the top K, flat the rest.

    Parameters
    ----------
    lookback:
        Return lookback window in bars (default 720 = 30 days hourly).
    top_n:
        Number of top-ranked symbols to hold. Capped at universe size.
    rebalance_freq:
        Bars between rebalance decisions (default 168 = 1 week hourly).
    """

    def __init__(
        self,
        lookback: int = 720,
        top_n: int = 1,
        rebalance_freq: int = 168,
    ) -> None:
        self._lookback = lookback
        self._top_n = top_n
        self._rebalance_freq = rebalance_freq

    @property
    def name(self) -> str:
        return "cross_sectional_momentum"

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        close = data.close
        n_symbols = len(data.symbols)
        top_n = min(self._top_n, n_symbols)

        # Rolling return for ranking — shift(1) avoids using current bar's close
        returns = close.pct_change(self._lookback)

        # Build a rebalance-only signal: NaN on non-rebalance bars, then ffill
        n = len(data.index)
        rebalance_signal = pd.DataFrame(np.nan, index=data.index, columns=data.symbols)

        # Initialise flat before enough warmup data
        rebalance_signal.iloc[: self._lookback] = 0.0

        for i in range(self._lookback, n, self._rebalance_freq):
            row = returns.iloc[i]
            valid = row.dropna()
            if valid.empty:
                rebalance_signal.iloc[i] = 0.0
                continue
            ranked = valid.rank(ascending=False, method="first")
            weights = pd.Series(0.0, index=data.symbols)
            top_syms = ranked[ranked <= top_n].index.tolist()
            if top_syms:
                for sym in top_syms:
                    weights[sym] = 1.0 / len(top_syms)
            rebalance_signal.iloc[i] = weights

        # Forward-fill: hold weights between rebalances
        return rebalance_signal.ffill().fillna(0.0)
