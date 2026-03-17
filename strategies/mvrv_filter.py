"""MVRV Z-Score regime filter.

MVRV (Market Value to Realized Value) Z-Score measures whether Bitcoin is
overvalued or undervalued relative to its "fair value" (aggregate cost basis
of all BTC holders). Historically:

    MVRV Z-Score > 7   → euphoria / cycle top → near-certain drawdown ahead
    MVRV Z-Score > 3.5 → elevated risk → reduce exposure
    MVRV Z-Score < 0   → undervalued / accumulation zone → full exposure

Data source: Glassnode or CryptoQuant export.
Required CSV format: two columns — date (YYYY-MM-DD), mvrv_z (float).
Download from: https://studio.glassnode.com/metrics?a=BTC&m=market.Mvrv

If the data file is not found, this filter acts as a transparent passthrough.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.base import MarketData
from strategies.base import Strategy

_DEFAULT_PATH = Path("data/mvrv.csv")


class MVRVFilter(Strategy):
    """Gates strategy exposure based on Bitcoin MVRV Z-Score.

    Parameters
    ----------
    strategy:
        Inner strategy to filter.
    mvrv_path:
        Path to CSV with columns [date, mvrv_z]. If missing, passthrough.
    caution_level:
        MVRV Z above this → reduce to caution_multiplier. Default 3.5.
    overbought_level:
        MVRV Z above this → reduce to overbought_multiplier. Default 7.0.
    caution_multiplier:
        Weight scale when MVRV is elevated. Default 0.5.
    overbought_multiplier:
        Weight scale when MVRV is in extreme territory. Default 0.0.
    """

    def __init__(
        self,
        strategy: Strategy,
        mvrv_path: str | Path = _DEFAULT_PATH,
        caution_level: float = 3.5,
        overbought_level: float = 7.0,
        caution_multiplier: float = 0.5,
        overbought_multiplier: float = 0.0,
    ) -> None:
        self._strategy = strategy
        self._caution_level = caution_level
        self._overbought_level = overbought_level
        self._caution_mult = caution_multiplier
        self._overbought_mult = overbought_multiplier
        self._mvrv = self._load(mvrv_path)

    def _load(self, path: str | Path) -> pd.Series | None:
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.set_index("date").sort_index()
            series = df["mvrv_z"].dropna()
            # Strip timezone if present
            if series.index.tz is not None:
                series.index = series.index.tz_convert(None)
            return series
        except (FileNotFoundError, KeyError, Exception):
            return None

    @property
    def name(self) -> str:
        suffix = "" if self._mvrv is None else "_mvrv"
        return f"{self._strategy.name}{suffix}"

    @property
    def mvrv_available(self) -> bool:
        return self._mvrv is not None

    def fit(self, data: MarketData, params: dict | None = None) -> None:
        self._strategy.fit(data, params)

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        signals = self._strategy.generate_signals(data)

        if self._mvrv is None:
            return signals

        # Map daily MVRV to each hourly bar (shift 1 day: no lookahead)
        mvrv_shifted = self._mvrv.shift(1).ffill()

        daily_dates = signals.index.normalize()
        unique_dates = daily_dates.unique().sort_values()
        mvrv_by_date = mvrv_shifted.reindex(unique_dates, method="ffill").fillna(0.0)
        date_to_mvrv = mvrv_by_date.to_dict()

        mvrv_hourly = pd.Series(
            [date_to_mvrv.get(d, 0.0) for d in daily_dates],
            index=signals.index,
            dtype=float,
        )

        # Compute per-bar multiplier
        multiplier = pd.Series(1.0, index=signals.index)
        multiplier[mvrv_hourly >= self._caution_level] = self._caution_mult
        multiplier[mvrv_hourly >= self._overbought_level] = self._overbought_mult

        return signals.multiply(multiplier, axis=0)
