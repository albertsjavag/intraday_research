from __future__ import annotations

import pandas as pd

from data.base import DataHandler, MarketData


class MockHandler(DataHandler):
    """Returns a pre-built MarketData object unchanged.

    Used in backtests to inject in-memory data without re-fetching.
    """

    def __init__(self, data: MarketData) -> None:
        self._data = data

    def load(self, symbols: list[str], start: str, end: str) -> MarketData:
        available = [s for s in symbols if s in self._data.symbols]
        sliced = MarketData(
            close=self._data.close[available].loc[start:end],
            open=self._data.open[available].loc[start:end],
            high=self._data.high[available].loc[start:end],
            low=self._data.low[available].loc[start:end],
            volume=self._data.volume[available].loc[start:end],
        )
        return sliced


class YFinanceHandler(DataHandler):
    """Loads OHLCV data from Yahoo Finance via yfinance.

    Useful for equities or as a free crypto fallback when Alpaca is unavailable.
    Note: yfinance tickers use '-' notation (e.g. 'BTC-USD'), not '/'.
    """

    def __init__(self, interval: str = "1h") -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "yfinance is required: pip install -e '.[data]'"
            ) from exc
        self._interval = interval

    def load(self, symbols: list[str], start: str, end: str) -> MarketData:
        import yfinance as yf

        # Normalise 'BTC/USD' → 'BTC-USD' for yfinance
        yf_symbols = [s.replace("/", "-") for s in symbols]
        sym_map = dict(zip(yf_symbols, symbols))

        raw = yf.download(
            yf_symbols,
            start=start,
            end=end,
            interval=self._interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        if len(yf_symbols) == 1:
            # Single ticker → flat columns
            sym = yf_symbols[0]
            data = {
                "close": raw[["Close"]].rename(columns={"Close": sym_map[sym]}),
                "open": raw[["Open"]].rename(columns={"Open": sym_map[sym]}),
                "high": raw[["High"]].rename(columns={"High": sym_map[sym]}),
                "low": raw[["Low"]].rename(columns={"Low": sym_map[sym]}),
                "volume": raw[["Volume"]].rename(columns={"Volume": sym_map[sym]}),
            }
        else:
            data = {}
            for field, col in [
                ("close", "Close"),
                ("open", "Open"),
                ("high", "High"),
                ("low", "Low"),
                ("volume", "Volume"),
            ]:
                df = raw[col].copy()
                df.columns = [sym_map.get(c, c) for c in df.columns]
                data[field] = df

        # Strip timezone
        for field in data:
            idx = data[field].index
            if hasattr(idx, "tz") and idx.tz is not None:
                data[field].index = idx.tz_convert(None)

        # Volume gaps → 0, price gaps → ffill(limit=3)
        data["volume"] = data["volume"].fillna(0)
        for field in ["open", "high", "low", "close"]:
            data[field] = data[field].ffill(limit=3)

        return MarketData(
            close=data["close"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            volume=data["volume"],
        )
