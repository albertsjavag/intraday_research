"""Regime-aware direction ML strategy.

Key improvements over the baseline:
1. Regime features: price vs 30d/90d MA, MA slope — model knows macro context.
2. Cross-sectional features: return relative to universe mean — model knows
   whether a symbol is outperforming or underperforming its peers.
3. Probability smoothing (EWM): raw probability is smoothed over N bars before
   thresholding, eliminating bar-by-bar flip-flopping and excess turnover.
4. Higher default threshold (0.58 vs 0.52): only trade high-confidence calls.
5. LightGBM: faster, better regularised, no StandardScaler needed. Falls back
   to sklearn GradientBoostingClassifier if lightgbm is not installed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from data.base import MarketData
from strategies.ml.base_ml import FeatureBuilder, MLStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder
# ─────────────────────────────────────────────────────────────────────────────

class DirectionFeatureBuilder(FeatureBuilder):
    """Build a pooled feature matrix across all symbols.

    Features are computed per symbol then stacked with a MultiIndex
    (symbol, timestamp). One model is trained on the full universe,
    allowing transfer of learning across symbols.

    Short-term technical
    --------------------
    Returns      : ret_1h, ret_4h, ret_8h, ret_24h, ret_72h, ret_168h
    Volatility   : vol_8h, vol_24h, vol_ratio_8_24
    RSI          : rsi_14
    MACD         : macd, macd_signal, macd_hist (price-normalised)
    Bollinger    : bb_pct, bb_width (24-bar, 2σ)
    Stochastic   : stoch_k, stoch_d (14/3)
    ATR          : atr_norm (14-bar)
    Short MAs    : price_vs_ma24, price_vs_ma72, price_vs_ma168,
                   ma24_vs_ma72, ma72_vs_ma168
    Intraday     : hour_of_day, vwap_deviation, volume_ratio, obv_norm

    Macro / regime (new)
    --------------------
    price_vs_ma720   : distance from 30-day MA (medium-term trend)
    price_vs_ma2160  : distance from 90-day MA (regime proxy)
    ma720_slope      : slope of 30-day MA over past 5 days

    Cross-sectional (new)
    ---------------------
    rel_ret_24h   : 24h return minus cross-sectional universe mean
    rel_ret_168h  : 7d return minus cross-sectional universe mean
    """

    def __init__(self, forward_bars: int = 8) -> None:
        self.forward_bars = forward_bars

    def build(self, data: MarketData) -> pd.DataFrame:
        # ── Cross-sectional returns (computed once across all symbols) ────────
        all_ret_24h: dict[str, pd.Series] = {}
        all_ret_168h: dict[str, pd.Series] = {}

        for sym in data.symbols:
            close = data.close[sym].dropna()
            if len(close) >= 200:
                all_ret_24h[sym] = np.log(close / close.shift(24))
                all_ret_168h[sym] = np.log(close / close.shift(168))

        cs_mean_24h: pd.Series | None = None
        cs_mean_168h: pd.Series | None = None
        if len(all_ret_24h) > 1:
            cs_mean_24h = pd.DataFrame(all_ret_24h).mean(axis=1)
            cs_mean_168h = pd.DataFrame(all_ret_168h).mean(axis=1)

        # ── Per-symbol features ───────────────────────────────────────────────
        frames = []
        for sym in data.symbols:
            close = data.close[sym].dropna()
            high = data.high[sym].reindex(close.index).ffill(limit=3)
            low = data.low[sym].reindex(close.index).ffill(limit=3)
            volume = data.volume[sym].reindex(close.index).fillna(0)

            if len(close) < 200:
                continue

            feat = pd.DataFrame(index=close.index)

            # --- Short-term returns ---
            for period in [1, 4, 8, 24, 72, 168]:
                feat[f"ret_{period}h"] = np.log(close / close.shift(period))

            # --- Volatility ---
            feat["vol_8h"] = feat["ret_1h"].rolling(8).std()
            feat["vol_24h"] = feat["ret_1h"].rolling(24).std()
            with np.errstate(divide="ignore", invalid="ignore"):
                feat["vol_ratio_8_24"] = (
                    feat["vol_8h"] / feat["vol_24h"]
                ).replace([np.inf, -np.inf], np.nan)

            # --- RSI (14-bar) ---
            feat["rsi_14"] = _rsi(close, 14)

            # --- MACD (12/26/9) ---
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            feat["macd"] = macd_line / close
            feat["macd_signal"] = macd_signal / close
            feat["macd_hist"] = (macd_line - macd_signal) / close

            # --- Bollinger Bands (24-bar, 2σ) ---
            bb_mid = close.rolling(24).mean()
            bb_std = close.rolling(24).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            bb_width = (bb_upper - bb_lower) / bb_mid
            with np.errstate(divide="ignore", invalid="ignore"):
                bb_pct = (
                    (close - bb_lower) / (bb_upper - bb_lower)
                ).replace([np.inf, -np.inf], np.nan)
            feat["bb_pct"] = bb_pct
            feat["bb_width"] = bb_width

            # --- Stochastic (14/3) ---
            low14 = low.rolling(14).min()
            high14 = high.rolling(14).max()
            with np.errstate(divide="ignore", invalid="ignore"):
                stoch_k = (100 * (close - low14) / (high14 - low14)).replace(
                    [np.inf, -np.inf], np.nan
                )
            feat["stoch_k"] = stoch_k
            feat["stoch_d"] = stoch_k.rolling(3).mean()

            # --- ATR normalised ---
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            feat["atr_norm"] = atr / close

            # --- Short-term trend MAs ---
            ma24 = close.rolling(24).mean()
            ma72 = close.rolling(72).mean()
            ma168 = close.rolling(168).mean()
            feat["price_vs_ma24"] = close / ma24 - 1
            feat["price_vs_ma72"] = close / ma72 - 1
            feat["price_vs_ma168"] = close / ma168 - 1
            feat["ma24_vs_ma72"] = ma24 / ma72 - 1
            feat["ma72_vs_ma168"] = ma72 / ma168 - 1

            # --- Macro / regime features (NEW) ---
            ma720 = close.rolling(720, min_periods=168).mean()
            # 90-day MA; fall back to 720 if not enough history
            ma2160 = close.rolling(2160, min_periods=720).mean()
            feat["price_vs_ma720"] = close / ma720 - 1
            feat["price_vs_ma2160"] = close / ma2160 - 1
            # Slope of 30d MA over past 5 days (120 bars)
            feat["ma720_slope"] = ma720.pct_change(120)

            # --- Cross-sectional relative returns (NEW) ---
            if cs_mean_24h is not None and sym in all_ret_24h:
                feat["rel_ret_24h"] = (
                    all_ret_24h[sym].reindex(close.index)
                    - cs_mean_24h.reindex(close.index)
                )
                feat["rel_ret_168h"] = (
                    all_ret_168h[sym].reindex(close.index)
                    - cs_mean_168h.reindex(close.index)
                )
            else:
                feat["rel_ret_24h"] = 0.0
                feat["rel_ret_168h"] = 0.0

            # --- Intraday features ---
            feat["hour_of_day"] = close.index.hour

            block = np.arange(len(close)) // 24
            feat["_block"] = block
            feat["_pv"] = close * volume
            feat["_vol"] = volume
            cum_pv = feat.groupby("_block")["_pv"].cumsum()
            cum_vol = feat.groupby("_block")["_vol"].cumsum().replace(0, np.nan)
            vwap = cum_pv / cum_vol
            feat["vwap_deviation"] = (close - vwap) / vwap.replace(0, np.nan)
            feat.drop(columns=["_block", "_pv", "_vol"], inplace=True)

            avg_vol = volume.rolling(24).mean().replace(0, np.nan)
            feat["volume_ratio"] = volume / avg_vol

            direction = np.sign(close.diff())
            obv = (volume * direction).cumsum()
            obv_mean = obv.rolling(24).mean()
            obv_std = obv.rolling(24).std().replace(0, np.nan)
            feat["obv_norm"] = (obv - obv_mean) / obv_std

            feat["symbol"] = sym
            feat.index.name = "timestamp"
            feat = feat.reset_index().set_index(["symbol", "timestamp"])
            frames.append(feat)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames)

    def build_targets(self, data: MarketData) -> pd.Series:
        """Binary target: 1 if price is higher after forward_bars."""
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


# ─────────────────────────────────────────────────────────────────────────────
# RSI helper
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

class DirectionMLStrategy(MLStrategy):
    """Regime-aware GBM direction classifier for intraday crypto.

    Uses LightGBM when available (faster, better regularised), falls back
    to sklearn GradientBoostingClassifier.

    Signal pipeline
    ---------------
    1. Build features (including macro regime + cross-sectional features).
    2. Predict P(up) per bar per symbol.
    3. Smooth probabilities with an EWM over `proba_smooth_span` bars —
       this eliminates bar-by-bar flip-flopping that causes excess turnover.
    4. Generate long signal where smoothed P(up) >= proba_threshold.
    5. Normalise across symbols.

    Parameters
    ----------
    n_estimators, max_depth, learning_rate:
        Tree model hyperparameters.
    forward_bars:
        Prediction horizon in bars.
    proba_threshold:
        Minimum smoothed probability to go long. Default 0.58 (vs old 0.52) —
        only trade high-confidence calls to preserve edge after commission.
    proba_smooth_span:
        EWM half-life (bars) for probability smoothing. Default 4 hours.
        Higher = fewer trades but slower to react.
    use_lightgbm:
        Try LightGBM first; fall back to sklearn if not installed.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.03,
        forward_bars: int = 8,
        proba_threshold: float = 0.58,
        proba_smooth_span: int = 4,
        use_lightgbm: bool = True,
        long_short: bool = False,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._forward_bars = forward_bars
        self._proba_threshold = proba_threshold
        self._proba_smooth_span = proba_smooth_span
        self._use_lightgbm = use_lightgbm
        self._long_short = long_short
        self._scaler = None
        self._feature_columns: list[str] = []
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
            self._proba_smooth_span = params.get("proba_smooth_span", self._proba_smooth_span)
            self._feature_builder = DirectionFeatureBuilder(forward_bars=self._forward_bars)

        features_df = self._feature_builder.build(data)
        targets = self._feature_builder.build_targets(data)

        if features_df.empty or targets.empty:
            self._is_fitted = False
            return

        common_idx = features_df.index.intersection(targets.index)
        X = features_df.loc[common_idx].drop(columns=["symbol"], errors="ignore")
        y = targets.loc[common_idx]

        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            warnings.warn("Too few training samples for ML strategy.", stacklevel=2)
            self._is_fitted = False
            return

        self._feature_columns = list(X.columns)

        # Try LightGBM first
        if self._use_lightgbm:
            try:
                import lightgbm as lgb
                self._model = lgb.LGBMClassifier(
                    n_estimators=self._n_estimators,
                    max_depth=self._max_depth,
                    learning_rate=self._learning_rate,
                    num_leaves=max(15, 2 ** self._max_depth - 1),
                    min_child_samples=30,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                )
                self._model.fit(X, y)
                self._scaler = None  # LightGBM is scale-invariant
                self._is_fitted = True
                print(f"  ML: LightGBM fitted on {len(X):,} samples, "
                      f"{len(self._feature_columns)} features.")
                return
            except ImportError:
                print("  ML: lightgbm not installed, falling back to sklearn GBM.")
                print("      Install with: pip install lightgbm")

        # Fallback: sklearn GBM (needs scaling)
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = GradientBoostingClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            subsample=0.8,
            random_state=42,
        )
        self._model.fit(X_scaled, y)
        self._is_fitted = True
        print(f"  ML: sklearn GBM fitted on {len(X):,} samples, "
              f"{len(self._feature_columns)} features.")

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        if not self._is_fitted or self._model is None:
            return pd.DataFrame(0.0, index=data.index, columns=data.symbols)

        features_df = self._feature_builder.build(data)
        if features_df.empty:
            return pd.DataFrame(0.0, index=data.index, columns=data.symbols)

        X = features_df.drop(columns=["symbol"], errors="ignore")
        X = X.reindex(columns=self._feature_columns, fill_value=0.0).fillna(0.0)

        if self._scaler is not None:
            X_input = self._scaler.transform(X)
        else:
            X_input = X.values

        raw_proba = self._model.predict_proba(X_input)[:, 1]  # P(up)

        # ── Smooth probabilities per symbol to kill flip-flopping ─────────────
        # Rebuild as (symbol, timestamp) MultiIndex Series, pivot, smooth, unpivot
        proba_series = pd.Series(raw_proba, index=features_df.index, name="proba")
        proba_wide = proba_series.unstack(level=0)  # cols=symbol, rows=timestamp

        # EWM smoothing: each bar's effective probability is a weighted average
        # of recent bars — signal only fires when model is persistently confident
        proba_smoothed = proba_wide.ewm(span=self._proba_smooth_span, adjust=False).mean()

        # ── Threshold → binary signal ─────────────────────────────────────────
        signal_wide = (proba_smoothed >= self._proba_threshold).astype(float)
        if self._long_short:
            signal_wide[proba_smoothed <= (1.0 - self._proba_threshold)] = -1.0

        signal_wide = signal_wide.reindex(data.index).fillna(0.0)
        signal_wide = signal_wide.reindex(columns=data.symbols, fill_value=0.0)

        # Normalise so weights sum to at most 1
        abs_sum = signal_wide.abs().sum(axis=1).replace(0, np.nan)
        signal_wide = signal_wide.div(abs_sum, axis=0).fillna(0.0)

        return signal_wide
