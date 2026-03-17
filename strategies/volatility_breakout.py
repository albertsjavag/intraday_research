"""Volatility breakout: go long when price breaks above its recent channel.

Uses a Donchian channel (rolling N-bar high) with an ATR buffer to filter
noise. A genuine breakout — price clearly exceeding recent range — tends to
have follow-through momentum on crypto hourly bars.

Entry : close > rolling_max(close, channel_window) + atr * entry_mult
Exit  : close < rolling_max(close, channel_window) - atr * exit_mult
         (i.e. price falls meaningfully back inside the channel)
"""
from __future__ import annotations

import pandas as pd

from data.base import MarketData
from strategies.base import Strategy


class VolatilityBreakoutStrategy(Strategy):
    """Donchian channel breakout with ATR-based entry and exit buffers.

    Parameters
    ----------
    channel_window:
        Bars for the rolling high (channel width). Default 48 = 2 days.
    atr_window:
        Bars for the Average True Range. Default 14.
    entry_mult:
        ATR multiple added above the channel to confirm entry. Default 0.5.
    exit_mult:
        ATR multiple subtracted from the channel to trigger exit. Default 0.0
        means exit as soon as price falls back inside the channel.
    """

    def __init__(
        self,
        channel_window: int = 48,
        atr_window: int = 14,
        entry_mult: float = 0.5,
        exit_mult: float = 0.0,
    ) -> None:
        self._channel_window = channel_window
        self._atr_window = atr_window
        self._entry_mult = entry_mult
        self._exit_mult = exit_mult

    @property
    def name(self) -> str:
        return "volatility_breakout"

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        close = data.close
        high = data.high
        low = data.low

        # Average True Range (proxy using high-low range)
        atr = (high - low).rolling(self._atr_window, min_periods=1).mean()

        # Rolling channel high — shift(1) so we don't use the current bar's
        # close to signal on the same bar (look-ahead)
        channel_high = close.shift(1).rolling(self._channel_window, min_periods=1).max()

        entry_level = channel_high + atr * self._entry_mult
        exit_level = channel_high - atr * self._exit_mult

        # Vectorised state machine: in position or not
        # in_position[t] = 1 if close[t] > entry_level[t]
        #                 = 0 if close[t] < exit_level[t]
        #                 = previous state otherwise
        entry_signal = (close > entry_level).astype(float)
        exit_signal = (close < exit_level).astype(float)

        # Build position column-by-column using ffill trick:
        # Set 1 at entry, 0 at exit, NaN elsewhere, then ffill.
        signals = pd.DataFrame(index=data.index, columns=data.symbols, dtype=float)

        for sym in data.symbols:
            pos = pd.Series(pd.NA, index=data.index, dtype="Float64")
            pos[entry_signal[sym] == 1] = 1.0
            pos[exit_signal[sym] == 1] = 0.0
            # Start flat before first valid bar
            pos.iloc[0] = 0.0
            signals[sym] = pos.ffill().fillna(0.0).astype(float)

        # Equal-weight across all symbols with an active long signal
        n_active = signals.sum(axis=1).replace(0, 1)
        return signals.div(n_active, axis=0)
