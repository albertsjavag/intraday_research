from __future__ import annotations

import numpy as np
import pandas as pd


class Portfolio:
    """Tracks cash, positions, and equity over time.

    Positions are stored as fractional quantities (not notional).
    All prices are passed in at each bar by the engine.
    """

    def __init__(self, initial_capital: float) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: dict[str, float] = {}  # symbol → quantity
        self._history: list[dict] = []
        self._num_trades: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, float]:
        """Return a copy of the current positions dict (symbol → qty)."""
        return dict(self._positions)

    @property
    def num_trades(self) -> int:
        return self._num_trades

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def total_value(self, prices: dict[str, float | None]) -> float:
        """Return total portfolio value at the given prices.

        Symbols with NaN or None prices are valued at 0 for this bar
        (i.e. the position is ignored) rather than crashing or returning NaN.
        """
        value = self._cash
        for sym, qty in self._positions.items():
            price = prices.get(sym)
            if price is None or (isinstance(price, float) and np.isnan(price)):
                continue
            value += qty * price
        return value

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def compute_rebalance(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float | None],
        commission_pct: float = 0.0,
        drift_threshold: float = 0.02,
    ) -> None:
        """Rebalance positions to match target_weights.

        Parameters
        ----------
        target_weights:
            {symbol: weight} where weights are fractions of total portfolio
            value (may not sum to 1; engine should normalise before calling).
        prices:
            Current prices for all symbols. Symbols with None/NaN prices are
            skipped (no new position opened, no existing position changed).
        commission_pct:
            One-way commission fraction deducted from cash on each trade.
        drift_threshold:
            Minimum absolute weight deviation required to trigger a trade.
            Prevents paying commission on micro-rebalances every bar caused
            by price drift. Default 0.02 = only trade when current weight
            has drifted more than 2% from target.
        """
        portfolio_value = self.total_value(prices)
        if portfolio_value <= 0:
            return

        for sym, weight in target_weights.items():
            price = prices.get(sym)
            if price is None or (isinstance(price, float) and np.isnan(price)) or price == 0:
                continue

            current_qty = self._positions.get(sym, 0.0)
            current_weight = (current_qty * price) / portfolio_value

            # Only trade if weight has drifted beyond threshold
            if abs(current_weight - weight) < drift_threshold:
                continue

            target_notional = portfolio_value * weight
            target_qty = target_notional / price
            delta_qty = target_qty - current_qty

            if abs(delta_qty) < 1e-10:
                continue

            trade_notional = abs(delta_qty * price)
            cost = trade_notional * commission_pct
            self._cash -= delta_qty * price + cost
            self._positions[sym] = target_qty
            self._num_trades += 1

        # Close positions for symbols not in target_weights
        symbols_to_close = [
            sym for sym in list(self._positions.keys())
            if sym not in target_weights and self._positions.get(sym, 0) != 0
        ]
        for sym in symbols_to_close:
            price = prices.get(sym)
            if price is None or (isinstance(price, float) and np.isnan(price)) or price == 0:
                continue
            qty = self._positions.pop(sym)
            trade_notional = abs(qty * price)
            cost = trade_notional * commission_pct
            self._cash += qty * price - cost
            self._num_trades += 1

    def force_close(
        self,
        symbol: str,
        price: float,
        commission_pct: float = 0.0,
    ) -> None:
        """Force-close a position at the given price (used on data gaps)."""
        qty = self._positions.pop(symbol, 0.0)
        if abs(qty) < 1e-10:
            return
        trade_notional = abs(qty * price)
        cost = trade_notional * commission_pct
        self._cash += qty * price - cost
        self._num_trades += 1

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def update(
        self,
        timestamp: pd.Timestamp,
        prices: dict[str, float | None],
    ) -> None:
        """Record the portfolio state for the current bar."""
        self._history.append(
            {
                "timestamp": timestamp,
                "value": self.total_value(prices),
                "cash": self._cash,
            }
        )

    def equity_curve(self) -> pd.Series:
        """Return the equity curve as a Series indexed by timestamp."""
        if not self._history:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self._history).set_index("timestamp")
        return df["value"]
