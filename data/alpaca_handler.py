from __future__ import annotations

import warnings
from datetime import datetime, timezone

import pandas as pd

from data.base import DataHandler, MarketData


class AlpacaDataHandler(DataHandler):
    """Loads hourly crypto OHLCV bars from Alpaca Markets."""

    def __init__(self) -> None:
        try:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError as exc:
            raise ImportError(
                "alpaca-py is required: pip install -e '.[alpaca]'"
            ) from exc

        self._CryptoHistoricalDataClient = CryptoHistoricalDataClient
        self._CryptoBarsRequest = CryptoBarsRequest
        self._TimeFrame = TimeFrame
        self._client = CryptoHistoricalDataClient()

    def load(
        self,
        symbols: list[str],
        start: str,
        end: str,
    ) -> MarketData:
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        request = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Hour,
            start=datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
            end=datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
        )
        bars = self._client.get_crypto_bars(request)
        df_raw = bars.df  # MultiIndex: (symbol, timestamp)

        if df_raw.empty:
            raise ValueError(f"No data returned for {symbols} from {start} to {end}")

        # Flatten MultiIndex → wide DataFrames per field
        df_raw.index = df_raw.index.set_levels(
            df_raw.index.levels[1].tz_convert(None), level=1
        )

        fields = ["open", "high", "low", "close", "volume"]
        wide: dict[str, pd.DataFrame] = {}

        for field in fields:
            pivot = df_raw[field].unstack(level=0)
            pivot.index = pd.DatetimeIndex(pivot.index)
            wide[field] = pivot

        # Build a shared complete hourly index
        full_idx = wide["close"].index
        missing_threshold = 0.01  # warn if >1% bars missing per symbol

        for field in fields:
            wide[field] = wide[field].reindex(full_idx)

        # Fill volume gaps with 0 (not ffill)
        wide["volume"] = wide["volume"].fillna(0)

        # ffill price fields with a short limit only (max 3 bars)
        for field in ["open", "high", "low", "close"]:
            wide[field] = wide[field].ffill(limit=3)

        # Warn about symbols with >1% missing bars (after ffill)
        n_bars = len(full_idx)
        for sym in wide["close"].columns:
            n_missing = wide["close"][sym].isna().sum()
            pct = n_missing / n_bars
            if pct > missing_threshold:
                warnings.warn(
                    f"{sym}: {n_missing}/{n_bars} bars missing ({pct:.1%}) after ffill. "
                    "Long gaps left as NaN intentionally.",
                    stacklevel=2,
                )

        return MarketData(
            close=wide["close"],
            open=wide["open"],
            high=wide["high"],
            low=wide["low"],
            volume=wide["volume"],
        )
