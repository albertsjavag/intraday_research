from __future__ import annotations

import numpy as np
import pandas as pd

from execution.base import ExecutionModel


class ZeroCostModel(ExecutionModel):
    """No transaction costs — useful for theoretical upper-bound analysis."""

    def apply_costs(
        self,
        returns: pd.Series,
        trades: pd.Series,
        prices: pd.Series,
    ) -> pd.Series:
        return returns


class ProportionalCostModel(ExecutionModel):
    """Deducts a proportional round-trip cost on each trade.

    Parameters
    ----------
    commission_pct:
        One-way commission as a fraction (e.g. 0.0025 = 25 bps).
    slippage_pct:
        One-way slippage as a fraction (e.g. 0.0005 = 5 bps).
    """

    def __init__(
        self,
        commission_pct: float = 0.0025,
        slippage_pct: float = 0.0,
    ) -> None:
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    @property
    def one_way_cost(self) -> float:
        return self.commission_pct + self.slippage_pct

    def apply_costs(
        self,
        returns: pd.Series,
        trades: pd.Series,
        prices: pd.Series,
    ) -> pd.Series:
        """Subtract cost on bars where a trade occurred.

        Cost = one_way_cost × |notional_traded| / portfolio_value_proxy.
        Since trades is already in notional terms and returns is fractional,
        we deduct one_way_cost as a fraction of the position size traded.
        """
        # Trade occurred on any bar where trades != 0
        traded_mask = trades.abs() > 0
        cost_series = pd.Series(0.0, index=returns.index)

        if prices.empty or (prices == 0).all():
            return returns

        # Cost as fraction of portfolio: one_way_cost × |trade_notional| / price
        # Simplified: deduct cost on the fraction traded
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_traded = trades.abs() / prices.replace(0, np.nan)
        pct_traded = pct_traded.fillna(0)
        cost_series[traded_mask] = self.one_way_cost * pct_traded[traded_mask]

        return returns - cost_series
