"""Pairs relative value: tilt BTC/ETH weights based on their log-spread z-score.

BTC and ETH are cointegrated over medium horizons — their log-price ratio
oscillates around a rolling mean. When BTC is unusually expensive relative to
ETH (spread z-score high), tilt toward ETH. When BTC is cheap, tilt toward BTC.

This is a long-only relative-value strategy — it doesn't short, it just
rotates between the pair. Other symbols in the universe (e.g. SOL) receive a
constant equal base weight, unaffected by the spread signal.

Works best as a complement to momentum (different alpha source, low correlation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.base import MarketData
from strategies.base import Strategy

# Canonical pair — only these two are used for the spread signal
_PAIR = ("BTC/USD", "ETH/USD")


class PairsArbStrategy(Strategy):
    """Relative value tilt between BTC and ETH based on log-spread mean reversion.

    Parameters
    ----------
    lookback:
        Rolling window (bars) for spread mean and std. Default 336 = 14 days.
    tilt_per_sigma:
        How much weight to shift per unit of z-score. E.g. 0.15 → a z of 2
        shifts 30% of the base weight from BTC to ETH (or vice versa).
    z_cap:
        Maximum z-score magnitude used for tilt (clips extreme values).
    """

    def __init__(
        self,
        lookback: int = 336,
        tilt_per_sigma: float = 0.15,
        z_cap: float = 2.0,
    ) -> None:
        self._lookback = lookback
        self._tilt_per_sigma = tilt_per_sigma
        self._z_cap = z_cap

    @property
    def name(self) -> str:
        return "pairs_arb"

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        close = data.close
        symbols = data.symbols
        n = len(symbols)
        base_weight = 1.0 / n

        signals = pd.DataFrame(base_weight, index=data.index, columns=symbols)

        btc, eth = _PAIR
        if btc not in close.columns or eth not in close.columns:
            # Pair not in universe — return equal weight
            return signals

        # Log spread: positive means BTC expensive relative to ETH
        spread = np.log(close[btc] / close[eth])
        roll_mean = spread.rolling(self._lookback, min_periods=self._lookback // 2).mean()
        roll_std = spread.rolling(self._lookback, min_periods=self._lookback // 2).std()

        z = ((spread - roll_mean) / roll_std).clip(-self._z_cap, self._z_cap)

        # Tilt: high z (BTC expensive) → reduce BTC, increase ETH; and vice versa
        tilt = z * self._tilt_per_sigma * base_weight
        signals[btc] = (base_weight - tilt).clip(0.0)
        signals[eth] = (base_weight + tilt).clip(0.0)

        # Renormalise so weights still sum to 1
        row_sums = signals.sum(axis=1).replace(0.0, 1.0)
        return signals.div(row_sums, axis=0).fillna(base_weight)
