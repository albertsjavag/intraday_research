from __future__ import annotations

import pandas as pd

from data.base import MarketData
from strategies.base import Strategy


class BuyAndHoldStrategy(Strategy):
    """Equal-weight passive benchmark.

    Assigns weight 1/N to every symbol at every bar, regardless of price.
    Serves as the baseline against which all active strategies are compared.
    """

    @property
    def name(self) -> str:
        return "buy_and_hold"

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        n = len(data.symbols)
        if n == 0:
            return pd.DataFrame(index=data.index)

        weight = 1.0 / n
        signals = pd.DataFrame(
            weight,
            index=data.index,
            columns=data.symbols,
        )
        return signals
