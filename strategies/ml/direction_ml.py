from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from data.base import MarketData
from strategies.ml.base_ml import FeatureBuilder, MLStrategy


class DirectionFeatureBuilder(FeatureBuilder):
    """Build a pooled feature matrix across all symbols.

    Features are computed per symbol then stacked with a MultiIndex
    (symbol, timestamp).  One model is trained on the full universe,
    allowing transfer of learning across symbols.

    Intraday features
    -----------------
    Returns: ret_1h, ret_4h, ret_8h, ret_24h, ret_72h, ret_168h
    Volatility: vol_8h, vol_24h, vol_ratio_8_24
    RSI: rsi_14
    MACD: macd, macd_signal, macd_hist (normalised by price)
    Bollinger: bb_pct, bb_width (24-bar, 2σ)
    Stochastic: stoch_k, stoch_d (14/3)
    ATR: atr_norm (14-bar)
    Trend: price_vs_ma24, price_vs_ma72, price_vs_ma168, ma24_vs_ma72, ma72_vs_ma168
    Intraday: hour_of_day, vwap_deviation, volume_ratio, obv_norm
    """

    def __init__(self, forward_bars: int = 8) -> None:
        self.forward_bars = forward_bars

    def build(self, data: MarketData) -> pd.DataFrame:
        frames = []
        for sym in data.symbols:
            close = data.close[sym].dropna()
            high = data.high[sym].reindex(close.index).ffill(limit=3)
            low = data.low[sym].reindex(close.index).ffill(limit=3)
            volume = data.volume[sym].reindex(close.index).fillna(0)

            if len(close) < 200:
                continue

            feat = pd.DataFrame(index=close.index)

            # --- Returns ---
            for period in [1, 4, 8, 24, 72, 168]:
                feat[f"ret_{period}h"] = np.log(close / close.shift(period))

            # --- Volatility ---
            feat["vol_8h"] = feat["ret_1h"].rolling(8).std()
            feat["vol_24h"] = feat["ret_1h"].rolling(24).std()
            with np.errstate(divide="ignore", invalid="ignore"):
                feat["vol_ratio_8_24"] = (feat["vol_8h"] / feat["vol_24h"]).replace([np.inf, -np.inf], np.nan)

            # --- RSI (14-bar) ---
            feat["rsi_14"] = _rsi(close, 14)

            # --- MACD (12/26/9) ---
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - macd_signal
            # Normalise by price
            feat["macd"] = macd_line / close
            feat["macd_signal"] = macd_signal / close
            feat["macd_hist"] = macd_hist / close

            # --- Bollinger Bands (24-bar, 2σ) ---
            bb_mid = close.rolling(24).mean()
            bb_std = close.rolling(24).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            bb_width = (bb_upper - bb_lower) / bb_mid
            with np.errstate(divide="ignore", invalid="ignore"):
                bb_pct = ((close - bb_lower) / (bb_upper - bb_lower)).replace([np.inf, -np.inf], np.nan)
            feat["bb_pct"] = bb_pct
            feat["bb_width"] = bb_width

            # --- Stochastic (14/3) ---
            low14 = low.rolling(14).min()
            high14 = high.rolling(14).max()
            with np.errstate(divide="ignore", invalid="ignore"):
                stoch_k = (100 * (close - low14) / (high14 - low14)).replace([np.inf, -np.inf], np.nan)
            feat["stoch_k"] = stoch_k
            feat["stoch_d"] = stoch_k.rolling(3).mean()

            # --- ATR normalised (14-bar) ---
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            feat["atr_norm"] = atr / close

            # --- Trend regime ---
            ma24 = close.rolling(24).mean()
            ma72 = close.rolling(72).mean()
            ma168 = close.rolling(168).mean()
            feat["price_vs_ma24"] = close / ma24 - 1
            feat["price_vs_ma72"] = close / ma72 - 1
            feat["price_vs_ma168"] = close / ma168 - 1
            feat["ma24_vs_ma72"] = ma24 / ma72 - 1
            feat["ma72_vs_ma168"] = ma72 / ma168 - 1

            # --- Intraday features ---
            feat["hour_of_day"] = close.index.hour

            # VWAP: cumulative(price × volume) / cumulative(volume), reset every 24 bars
            block = np.arange(len(close)) // 24
            feat["_block"] = block
            feat["_pv"] = close * volume
            feat["_vol"] = volume
            cum_pv = feat.groupby("_block")["_pv"].cumsum()
            cum_vol = feat.groupby("_block")["_vol"].cumsum().replace(0, np.nan)
            vwap = cum_pv / cum_vol
            feat["vwap_deviation"] = (close - vwap) / vwap.replace(0, np.nan)
            feat.drop(columns=["_block", "_pv", "_vol"], inplace=True)

            # Volume ratio: volume / 24-bar average volume
            avg_vol = volume.rolling(24).mean().replace(0, np.nan)
            feat["volume_ratio"] = volume / avg_vol

            # OBV z-scored over 24 bars
            direction = np.sign(close.diff())
            obv = (volume * direction).cumsum()
            obv_mean = obv.rolling(24).mean()
            obv_std = obv.rolling(24).std().replace(0, np.nan)
            feat["obv_norm"] = (obv - obv_mean) / obv_std

            # Add symbol label for MultiIndex
            feat["symbol"] = sym
            feat.index.name = "timestamp"
            feat = feat.reset_index().set_index(["symbol", "timestamp"])
            frames.append(feat)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames)

    def build_targets(self, data: MarketData) -> pd.Series:
        """Build binary classification targets: 1 if price up after forward_bars."""
        frames = []
        for sym in data.symbols:
            close = data.close[sym].dropna()
            if len(close) < self.forward_bars + 1:
                continue
            future = close.shift(-self.forward_bars)
            target = (future > close).astype(int)
            target.name = "target"
            target.index.name = "timestamp"
            target = target.reset_index()
            target["symbol"] = sym
            target = target.set_index(["symbol", "timestamp"])["target"]
            frames.append(target)

        if not frames:
            return pd.Series(dtype=int)

        return pd.concat(frames)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class DirectionMLStrategy(MLStrategy):
    """GBM direction classifier for intraday crypto signals.

    Trains a GradientBoostingClassifier on pooled features across all symbols.
    At inference, uses predicted probability of up class, thresholded at
    proba_threshold to generate long signals (long/flat by default).

    Parameters
    ----------
    n_estimators, max_depth, learning_rate:
        GBM hyperparameters.
    forward_bars:
        Prediction horizon (bars ahead to classify as up/down).
    proba_threshold:
        Minimum predicted up-probability to go long (default 0.52).
    long_short:
        If False (default), only long and flat signals — no shorts.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        forward_bars: int = 8,
        proba_threshold: float = 0.52,
        long_short: bool = False,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._forward_bars = forward_bars
        self._proba_threshold = proba_threshold
        self._long_short = long_short
        feature_builder = DirectionFeatureBuilder(forward_bars=forward_bars)
        super().__init__(feature_builder=feature_builder)

    @property
    def name(self) -> str:
        return "ml"

    def fit(self, data: MarketData, params: dict | None = None) -> None:
        if params:
            self._n_estimators = params.get("n_estimators", self._n_estimators)
            self._max_depth = params.get("max_depth", self._max_depth)
            self._learning_rate = params.get("learning_rate", self._learning_rate)
            self._forward_bars = params.get("forward_bars", self._forward_bars)
            self._proba_threshold = params.get("proba_threshold", self._proba_threshold)
            self._feature_builder = DirectionFeatureBuilder(forward_bars=self._forward_bars)

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required: pip install -e '.[ml]'"
            ) from exc

        features_df = self._feature_builder.build(data)
        targets = self._feature_builder.build_targets(data)

        if features_df.empty or targets.empty:
            self._is_fitted = False
            return

        # Align features and targets
        common_idx = features_df.index.intersection(targets.index)
        X = features_df.loc[common_idx].drop(columns=["symbol"], errors="ignore")
        y = targets.loc[common_idx]

        # Drop NaN rows
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        if len(X) < 50:
            warnings.warn("Too few training samples for ML strategy.", stacklevel=2)
            self._is_fitted = False
            return

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._feature_columns = list(X.columns)

        self._model = GradientBoostingClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_state=42,
        )
        self._model.fit(X_scaled, y)
        self._is_fitted = True

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        if not self._is_fitted or self._model is None:
            # Return zero signals if model not trained
            return pd.DataFrame(0.0, index=data.index, columns=data.symbols)

        features_df = self._feature_builder.build(data)
        if features_df.empty:
            return pd.DataFrame(0.0, index=data.index, columns=data.symbols)

        X = features_df.drop(columns=["symbol"], errors="ignore")
        # Keep only training columns, fill missing with 0
        X = X.reindex(columns=self._feature_columns, fill_value=0.0)
        X = X.fillna(0.0)

        X_scaled = self._scaler.transform(X)
        proba = self._model.predict_proba(X_scaled)[:, 1]  # P(up)

        # Build signal: 1 if proba > threshold, -1 if long_short and proba < 1-threshold
        signal_vals = np.zeros(len(proba))
        signal_vals[proba >= self._proba_threshold] = 1.0
        if self._long_short:
            signal_vals[proba <= (1 - self._proba_threshold)] = -1.0

        # Re-index to (symbol, timestamp) MultiIndex
        features_df["signal"] = signal_vals
        signal_series = features_df["signal"]

        # Pivot to wide format: rows=timestamp, cols=symbol
        signal_wide = signal_series.unstack(level=0)
        signal_wide = signal_wide.reindex(data.index).fillna(0.0)

        # Normalise abs weights to sum to 1 per bar
        abs_sum = signal_wide.abs().sum(axis=1).replace(0, np.nan)
        signal_wide = signal_wide.div(abs_sum, axis=0).fillna(0.0)

        # Ensure all symbols are present
        signal_wide = signal_wide.reindex(columns=data.symbols, fill_value=0.0)

        return signal_wide
