"""Macro regime filter: wraps a strategy and scales its weights based on
a daily 200-day MA classification of the broader crypto market.

Regime logic (applied to a single barometer symbol, typically BTC/USD):

    bull     — price > MA200 × (1 + band) AND MA200 slope > 0
    bear     — price < MA200 × (1 - band) AND MA200 slope < 0
    sideways — everything else (price near MA200, or slope contradicts position)

Multipliers applied to the inner strategy's weights:
    bull     → 1.0  (full exposure)
    sideways → 0.5  (half exposure, regime uncertain)
    bear     → 0.0  (flat, long-only system has no edge in downtrends)

Regime is shifted by 1 day before applying so today's trading uses
yesterday's daily close — no lookahead bias.
"""
from __future__ import annotations

import pandas as pd

from data.base import MarketData
from strategies.base import Strategy

REGIME_MULTIPLIERS: dict[str, float] = {
    "bull": 1.0,
    "sideways": 0.5,
    "bear": 0.0,
}


class MacroRegimeFilter(Strategy):
    """Wraps a strategy and gates its signals through a daily macro regime.

    Parameters
    ----------
    strategy:
        Inner strategy whose weights are scaled.
    daily_close:
        Daily close price Series for the barometer symbol (BTC/USD).
        DatetimeIndex tz-naive UTC at day resolution.
    ma_period:
        Days for the moving average. Default 200.
    slope_window:
        Days over which to measure MA slope. Default 20.
    sideways_band:
        Fractional distance from MA200 counted as sideways, e.g. 0.02 = ±2%.
    """

    def __init__(
        self,
        strategy: Strategy,
        daily_close: pd.Series,
        ma_period: int = 200,
        slope_window: int = 20,
        sideways_band: float = 0.02,
    ) -> None:
        self._strategy = strategy
        self._daily_close = daily_close.sort_index().dropna()
        self._ma_period = ma_period
        self._slope_window = slope_window
        self._sideways_band = sideways_band
        self._regime: pd.Series | None = None  # date-level index → regime str

    @property
    def name(self) -> str:
        return f"regime_{self._strategy.name}"

    @property
    def current_regime(self) -> str:
        """Most recent regime label, for logging and dashboard state."""
        if self._regime is None or self._regime.empty:
            return "unknown"
        return str(self._regime.iloc[-1])

    def fit(self, data: MarketData, params: dict | None = None) -> None:
        self._strategy.fit(data, params)
        self._regime = self._compute_regime()

    def _compute_regime(self) -> pd.Series:
        close = self._daily_close
        ma = close.rolling(self._ma_period, min_periods=self._ma_period).mean()

        # Fractional slope of MA over slope_window days
        ma_slope = ma.pct_change(self._slope_window)

        # Price position relative to MA200 as a fraction
        price_vs_ma = (close - ma) / ma

        regime = pd.Series("sideways", index=close.index, dtype=object)
        bull = (price_vs_ma > self._sideways_band) & (ma_slope > 0)
        bear = (price_vs_ma < -self._sideways_band) & (ma_slope < 0)
        regime[bull] = "bull"
        regime[bear] = "bear"

        # Shift 1 day: today's trading uses yesterday's close regime.
        regime = regime.shift(1).fillna("sideways")
        return regime

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        signals = self._strategy.generate_signals(data)

        if self._regime is None:
            return signals

        # Map each hourly bar's date → daily regime → multiplier
        daily_dates = signals.index.normalize()
        unique_dates = daily_dates.unique().sort_values()

        # Forward-fill regime for any dates not in the daily series
        # (e.g. weekends in the daily barometer data)
        regime_by_date = (
            self._regime.reindex(unique_dates, method="ffill").fillna("sideways")
        )
        date_to_multiplier = {
            d: REGIME_MULTIPLIERS.get(r, 0.5)
            for d, r in regime_by_date.items()
        }

        multipliers = pd.Series(
            [date_to_multiplier.get(d, 0.5) for d in daily_dates],
            index=signals.index,
            dtype=float,
        )

        return signals.multiply(multipliers, axis=0)
