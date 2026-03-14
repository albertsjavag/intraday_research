from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.base import MarketData


def load_csv_ohlcv(
    path: str | Path,
    symbol: str,
    date_col: str = "datetime",
    tz: str | None = None,
) -> MarketData:
    """Load a single-symbol OHLCV CSV into a MarketData object.

    The CSV must have columns: datetime, open, high, low, close, volume.
    The datetime column is parsed and set as index.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()

    if tz is not None:
        df.index = df.index.tz_localize(tz).tz_convert(None)
    elif df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df.columns = df.columns.str.lower()

    def _wrap(col: str) -> pd.DataFrame:
        return df[[col]].rename(columns={col: symbol})

    return MarketData(
        close=_wrap("close"),
        open=_wrap("open"),
        high=_wrap("high"),
        low=_wrap("low"),
        volume=_wrap("volume"),
    )


def normalize_index(data: MarketData, freq: str = "h") -> MarketData:
    """Reindex all frames to a complete DatetimeIndex at the given frequency.

    Gaps are left as NaN for prices (engine will handle them); volume → 0.
    """
    full_idx = pd.date_range(data.index[0], data.index[-1], freq=freq)

    def _reindex(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(full_idx)

    close = _reindex(data.close).ffill(limit=3)
    open_ = _reindex(data.open).ffill(limit=3)
    high = _reindex(data.high).ffill(limit=3)
    low = _reindex(data.low).ffill(limit=3)
    volume = _reindex(data.volume).fillna(0)

    return MarketData(close=close, open=open_, high=high, low=low, volume=volume)


def merge_market_data(datasets: list[MarketData]) -> MarketData:
    """Concatenate multiple single-symbol MarketData objects along the column axis.

    Uses ffill(limit=3) to handle minor index misalignments.
    The shared index is the union of all indices.
    """
    fields = ["close", "open", "high", "low", "volume"]
    merged: dict[str, pd.DataFrame] = {}

    for field in fields:
        frames = [getattr(d, field) for d in datasets]
        combined = pd.concat(frames, axis=1).sort_index()
        if field == "volume":
            combined = combined.fillna(0)
        else:
            combined = combined.ffill(limit=3)
        merged[field] = combined

    return MarketData(
        close=merged["close"],
        open=merged["open"],
        high=merged["high"],
        low=merged["low"],
        volume=merged["volume"],
    )
