from __future__ import annotations

import os
from typing import Any

from utils.secrets import load_dotenv, require


class AlpacaTrader:
    """Live / paper trading interface to Alpaca Markets.

    Credentials are read from environment variables (populated from .env):
        ALPACA_API_KEY
        ALPACA_SECRET_KEY
        ALPACA_LIVE  (default: false → paper trading)
    """

    def __init__(self) -> None:
        load_dotenv()

        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest
        except ImportError as exc:
            raise ImportError(
                "alpaca-py is required: pip install -e '.[alpaca]'"
            ) from exc

        api_key = require("ALPACA_API_KEY")
        secret_key = require("ALPACA_SECRET_KEY")
        live = os.environ.get("ALPACA_LIVE", "false").lower() == "true"

        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=not live,
        )
        self._live = live
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce
        self._MarketOrderRequest = MarketOrderRequest

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> dict[str, float]:
        """Return current account equity, cash, and buying power."""
        acct = self._client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
        }

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_all_positions(self) -> list[dict[str, Any]]:
        """Return all open positions."""
        positions = self._client.get_all_positions()
        result = []
        for p in positions:
            result.append(
                {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price) if p.current_price else None,
                    "market_value": float(p.market_value) if p.market_value else None,
                    "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl else None,
                    "unrealized_plpc": float(p.unrealized_plpc) if p.unrealized_plpc else None,
                }
            )
        return result

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Return the position for a symbol, or None if flat."""
        try:
            p = self._client.get_open_position(symbol)
            return {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value) if p.market_value else None,
                "side": p.side.value,
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def get_orders(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent orders."""
        from alpaca.trading.requests import GetOrdersRequest

        request = GetOrdersRequest(limit=limit)
        orders = self._client.get_orders(filter=request)
        result = []
        for o in orders:
            result.append(
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "notional": float(o.notional) if o.notional else None,
                    "qty": float(o.qty) if o.qty else None,
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                    "status": o.status.value,
                    "created_at": str(o.created_at),
                }
            )
        return result

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        notional: float,
        side: str,
    ) -> dict[str, Any]:
        """Place a notional market order.

        Parameters
        ----------
        symbol:
            Alpaca symbol, e.g. 'BTC/USD'.
        notional:
            Dollar notional to trade.
        side:
            'buy' or 'sell'.
        """
        order_side = (
            self._OrderSide.BUY if side.lower() == "buy" else self._OrderSide.SELL
        )
        request = self._MarketOrderRequest(
            symbol=symbol,
            notional=round(notional, 2),
            side=order_side,
            time_in_force=self._TimeInForce.GTC,
        )
        order = self._client.submit_order(request)
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "side": order.side.value,
            "notional": float(order.notional) if order.notional else None,
            "status": order.status.value,
        }

    def close_position(self, symbol: str) -> dict[str, Any] | None:
        """Close the position for a symbol. Returns None if already flat."""
        position = self.get_position(symbol)
        if position is None:
            return None
        try:
            result = self._client.close_position(symbol)
            return {"symbol": symbol, "closed": True, "order_id": str(result.id)}
        except Exception as exc:
            return {"symbol": symbol, "closed": False, "error": str(exc)}

    def close_all_positions(self) -> None:
        """Close all open positions and cancel open orders."""
        self._client.close_all_positions(cancel_orders=True)

    def execute_signal(
        self,
        symbol: str,
        signal: float,
        notional: float,
    ) -> dict[str, Any] | None:
        """Execute a signal for a symbol.

        Parameters
        ----------
        signal:
            +1 → long, -1 → short, 0 → flat.
        notional:
            Dollar amount to allocate when opening a position.
        """
        position = self.get_position(symbol)
        current_side = position["side"] if position else None

        # Determine desired side
        if signal > 0:
            desired_side = "long"
        elif signal < 0:
            desired_side = "short"
        else:
            desired_side = "flat"

        # Already correctly positioned → no-op
        if desired_side == current_side:
            return None

        # Close before reversing or going flat
        if current_side is not None:
            self.close_position(symbol)

        if desired_side == "flat":
            return None

        side = "buy" if desired_side == "long" else "sell"
        return self.place_order(symbol=symbol, notional=notional, side=side)
