from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ExecutionModel(ABC):
    """Abstract base class for execution / cost models.

    An execution model transforms a raw returns series by applying
    transaction costs, slippage, and other market-impact effects.
    """

    @abstractmethod
    def apply_costs(
        self,
        returns: pd.Series,
        trades: pd.Series,
        prices: pd.Series,
    ) -> pd.Series:
        """Return a net-of-cost returns series.

        Parameters
        ----------
        returns:
            Raw bar returns for a single symbol (pct change).
        trades:
            Signed notional traded per bar (positive = bought, negative = sold).
        prices:
            Close prices used to compute percentage cost.
        """
        ...
