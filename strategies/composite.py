from __future__ import annotations

import numpy as np
import pandas as pd

from data.base import MarketData
from strategies.base import Strategy


class CompositeStrategy(Strategy):
    """Sharpe-weighted blend of N sub-strategies.

    At each rebalance bar, computes the rolling Sharpe ratio of each
    sub-strategy's hypothetical returns (lagged_signal × asset_return).
    Weights are floored at 0 (negative-Sharpe strategies get zero allocation),
    capped via iterative redistribution, and normalised to sum to 1.

    Falls back to equal weight if all strategies have negative recent Sharpe.

    Parameters
    ----------
    strategies:
        List of Strategy instances to blend.
    lookback:
        Rolling window (bars) for Sharpe estimation.
    rebalance_freq:
        How often (bars) to recompute strategy weights.
    max_weight:
        Maximum fraction any single strategy can receive (default 0.60).
        Excess is redistributed to uncapped strategies iteratively.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        lookback: int = 168,
        rebalance_freq: int = 24,
        max_weight: float = 0.60,
    ) -> None:
        self._strategies = strategies
        self._lookback = lookback
        self._rebalance_freq = rebalance_freq
        self._max_weight = max_weight
        self.last_strategy_weights: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "composite"

    def fit(self, data: MarketData, params: dict | None = None) -> None:
        if params:
            self._lookback = params.get("lookback", self._lookback)
            self._rebalance_freq = params.get("rebalance_freq", self._rebalance_freq)
            self._max_weight = params.get("max_weight", self._max_weight)

        for strategy in self._strategies:
            strategy.fit(data, params)

    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        # Generate signals from each sub-strategy
        sub_signals: dict[str, pd.DataFrame] = {}
        for s in self._strategies:
            sub_signals[s.name] = s.generate_signals(data)

        close = data.close
        asset_returns = close.pct_change()

        index = data.index
        n_bars = len(index)
        strategy_names = list(sub_signals.keys())
        n_strats = len(strategy_names)

        # Compute hypothetical return per strategy per bar:
        # hyp_return[t] = mean(lagged_signal[t-1] × asset_return[t]) across symbols
        hyp_returns: dict[str, pd.Series] = {}
        for sname in strategy_names:
            sig = sub_signals[sname]
            lagged = sig.shift(1)
            # Dot product across symbols, normalised by number of symbols
            hyp = (lagged * asset_returns).mean(axis=1)
            hyp_returns[sname] = hyp

        hyp_df = pd.DataFrame(hyp_returns, index=index)

        # Strategy weights: recomputed every rebalance_freq bars
        strategy_weights = pd.DataFrame(1.0 / n_strats, index=index, columns=strategy_names)
        current_weights = {sname: 1.0 / n_strats for sname in strategy_names}

        for i in range(n_bars):
            if i > 0 and i % self._rebalance_freq == 0:
                window = hyp_df.iloc[max(0, i - self._lookback): i]
                if len(window) >= 2:
                    current_weights = self._compute_weights(window, strategy_names)
            strategy_weights.iloc[i] = [current_weights.get(s, 0.0) for s in strategy_names]

        # Store last weights for dashboard / run_live.py
        self.last_strategy_weights = dict(strategy_weights.iloc[-1])

        # Blend sub-strategy signals using rolling strategy weights
        blended = pd.DataFrame(0.0, index=index, columns=close.columns)
        for sname in strategy_names:
            sig = sub_signals[sname].reindex(index).fillna(0.0)
            w = strategy_weights[sname]
            blended += sig.multiply(w, axis=0)

        return blended

    def _compute_weights(
        self, window: pd.DataFrame, strategy_names: list[str]
    ) -> dict[str, float]:
        """Compute Sharpe-based strategy weights with floor=0 and max_weight cap."""
        sharpes: dict[str, float] = {}
        for sname in strategy_names:
            col = window[sname].dropna()
            if len(col) < 2 or col.std() == 0:
                sharpes[sname] = 0.0
            else:
                sharpes[sname] = float(col.mean() / col.std())

        # Floor at 0
        raw = {s: max(0.0, v) for s, v in sharpes.items()}
        total = sum(raw.values())

        if total == 0:
            # All negative Sharpe — equal weight fallback
            n = len(strategy_names)
            return {s: 1.0 / n for s in strategy_names}

        # Normalise
        weights = {s: v / total for s, v in raw.items()}

        # Iterative max_weight cap redistribution
        weights = self._apply_max_weight_cap(weights)

        return weights

    def _apply_max_weight_cap(self, weights: dict[str, float]) -> dict[str, float]:
        """Iteratively redistribute excess weight from capped strategies."""
        weights = dict(weights)
        max_w = self._max_weight

        for _ in range(len(weights)):
            capped = {s: min(w, max_w) for s, w in weights.items()}
            excess = sum(w - max_w for w in weights.values() if w > max_w)

            if excess <= 1e-9:
                return capped

            # Redistribute excess to uncapped strategies
            uncapped = [s for s, w in weights.items() if w < max_w]
            if not uncapped:
                return capped

            uncapped_total = sum(capped[s] for s in uncapped)
            if uncapped_total == 0:
                share = excess / len(uncapped)
                for s in uncapped:
                    capped[s] += share
            else:
                for s in uncapped:
                    capped[s] += excess * (capped[s] / uncapped_total)

            weights = capped

        return weights
