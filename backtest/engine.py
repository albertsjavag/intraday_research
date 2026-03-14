from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from backtest.portfolio import Portfolio

if TYPE_CHECKING:
    from data.base import DataHandler, MarketData
    from execution.base import ExecutionModel
    from strategies.base import Strategy


@dataclass
class BacktestConfig:
    symbols: list[str]
    start: str
    end: str
    initial_capital: float = 100_000.0
    strategy_params: dict = field(default_factory=dict)
    signal_start: str | None = None  # load from here for indicator warmup


@dataclass
class BacktestResult:
    name: str
    equity_curve: pd.Series
    signals: pd.DataFrame
    num_trades: int
    config: BacktestConfig


class BacktestEngine:
    """Vectorised hourly backtest engine with full dependency injection.

    The engine never imports concrete strategy or data classes — only ABCs.
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_handler: "DataHandler",
        strategy: "Strategy",
        execution_model: "ExecutionModel",
    ) -> None:
        self._config = config
        self._data_handler = data_handler
        self._strategy = strategy
        self._execution_model = execution_model

    def run(self) -> BacktestResult:
        cfg = self._config

        # 1. Load data from signal_start (or start) for feature warmup
        load_start = cfg.signal_start or cfg.start
        data: MarketData = self._data_handler.load(
            symbols=cfg.symbols,
            start=load_start,
            end=cfg.end,
        )

        # 2. Fit strategy on full warmup data and generate signals
        self._strategy.fit(data, cfg.strategy_params)
        all_signals: pd.DataFrame = self._strategy.generate_signals(data)

        # 3. Slice to [start, end] for actual trading loop
        trade_data = data.slice(cfg.start, cfg.end)
        trade_signals = all_signals.loc[cfg.start:cfg.end]

        # Align signals to trade_data index
        trade_signals = trade_signals.reindex(trade_data.index)

        portfolio = Portfolio(initial_capital=cfg.initial_capital)
        commission = cfg.strategy_params.get("commission_pct", 0.0025)
        drift_threshold = cfg.strategy_params.get("drift_threshold", 0.02)

        # Track last known prices for force-close logic
        last_known_prices: dict[str, float] = {}

        symbols = trade_data.symbols

        for timestamp in trade_data.index:
            # Build current prices dict
            prices: dict[str, float | None] = {}
            for sym in symbols:
                raw = trade_data.close.loc[timestamp, sym]
                prices[sym] = None if (raw is None or (isinstance(raw, float) and np.isnan(raw))) else float(raw)

            # 4. Force-close positions for symbols whose price just went NaN
            force_closed = False
            for sym in list(portfolio.positions.keys()):
                if prices.get(sym) is None:
                    last_price = last_known_prices.get(sym)
                    if last_price is not None and portfolio.positions.get(sym, 0) != 0:
                        portfolio.force_close(sym, last_price, commission)
                        force_closed = True

            # Update last known prices
            for sym in symbols:
                p = prices[sym]
                if p is not None:
                    last_known_prices[sym] = p

            # If we force-closed, record state and skip normal update
            # (avoids duplicate portfolio.update() calls)
            if force_closed:
                portfolio.update(timestamp, prices)
                continue

            # 5. Build target weights from signals, excluding NaN-price symbols
            if timestamp in trade_signals.index:
                raw_weights = trade_signals.loc[timestamp]
            else:
                raw_weights = pd.Series(0.0, index=symbols)

            # Only include symbols with valid prices
            valid_weights: dict[str, float] = {}
            for sym in symbols:
                if prices.get(sym) is not None:
                    w = raw_weights.get(sym, 0.0)
                    if isinstance(w, float) and np.isnan(w):
                        w = 0.0
                    valid_weights[sym] = float(w)

            # Normalise weights so they sum to at most 1
            total_weight = sum(abs(w) for w in valid_weights.values())
            if total_weight > 1.0:
                valid_weights = {s: w / total_weight for s, w in valid_weights.items()}

            # 6. Rebalance and record — exactly once per bar
            portfolio.compute_rebalance(
                target_weights=valid_weights,
                prices=prices,
                commission_pct=commission,
                drift_threshold=drift_threshold,
            )
            portfolio.update(timestamp, prices)

        return BacktestResult(
            name=self._strategy.name,
            equity_curve=portfolio.equity_curve(),
            signals=trade_signals,
            num_trades=portfolio.num_trades,
            config=cfg,
        )
