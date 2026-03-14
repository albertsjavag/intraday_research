from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import pandas as pd

from backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from data.base import MarketData
from data.handlers import MockHandler

if TYPE_CHECKING:
    from strategies.base import Strategy


@dataclass
class WalkForwardWindow:
    train_start: str
    train_end: str
    test_start: str
    test_end: str


class WalkForwardValidator:
    """Splits data into rolling (train, test) windows and validates a strategy.

    Parameters
    ----------
    train_bars:
        Number of bars in each training window.
    test_bars:
        Number of bars in each test window.
    context_bars:
        Extra bars prepended before each test window for indicator warmup.
    step_bars:
        How many bars to advance the window each fold (defaults to test_bars).
    """

    def __init__(
        self,
        train_bars: int,
        test_bars: int,
        context_bars: int = 500,
        step_bars: int | None = None,
    ) -> None:
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.context_bars = context_bars
        self.step_bars = step_bars or test_bars

    def split(self, index: pd.DatetimeIndex) -> list[WalkForwardWindow]:
        """Return a list of WalkForwardWindow objects for the given index."""
        n = len(index)
        windows: list[WalkForwardWindow] = []

        start = 0
        while start + self.train_bars + self.test_bars <= n:
            train_end_idx = start + self.train_bars - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = min(test_start_idx + self.test_bars - 1, n - 1)

            windows.append(
                WalkForwardWindow(
                    train_start=str(index[start].date()),
                    train_end=str(index[train_end_idx].date()),
                    test_start=str(index[test_start_idx].date()),
                    test_end=str(index[test_end_idx].date()),
                )
            )
            start += self.step_bars

        return windows

    def validate(
        self,
        data: MarketData,
        strategy: "Strategy",
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.0025,
    ) -> list[BacktestResult]:
        """Run walk-forward validation and return one BacktestResult per fold."""
        windows = self.split(data.index)
        results: list[BacktestResult] = []

        for i, w in enumerate(windows):
            # Add context bars before the test window for indicator warmup
            test_start_pos = data.index.searchsorted(w.test_start)
            context_start_pos = max(0, test_start_pos - self.context_bars)
            signal_start = str(data.index[context_start_pos].date())

            # Slice train window for fitting
            train_data = data.slice(w.train_start, w.train_end)
            strategy.fit(train_data, params=None)

            # Build engine over [signal_start, test_end] with trade from test_start
            cfg = BacktestConfig(
                symbols=data.symbols,
                start=w.test_start,
                end=w.test_end,
                initial_capital=initial_capital,
                strategy_params={"commission_pct": commission_pct},
                signal_start=signal_start,
            )
            handler = MockHandler(data)
            engine = BacktestEngine(
                config=cfg,
                data_handler=handler,
                strategy=strategy,
                execution_model=_NoopExecution(),
            )
            result = engine.run()
            result.name = f"{strategy.name}_fold_{i+1}"
            results.append(result)

        return results


class ParameterSweep:
    """Iterates a parameter grid and returns one BacktestResult per config.

    Parameters
    ----------
    engine_factory:
        Callable that takes a BacktestConfig and returns a BacktestEngine.
    param_grid:
        List of dicts; each dict is passed as strategy_params in a config.
    base_config:
        Template BacktestConfig whose strategy_params will be overridden.
    """

    def __init__(
        self,
        engine_factory: Callable[[BacktestConfig], BacktestEngine],
        param_grid: list[dict],
        base_config: BacktestConfig,
    ) -> None:
        self.engine_factory = engine_factory
        self.param_grid = param_grid
        self.base_config = base_config

    def run(self) -> list[BacktestResult]:
        results: list[BacktestResult] = []
        for i, params in enumerate(self.param_grid):
            cfg = BacktestConfig(
                symbols=self.base_config.symbols,
                start=self.base_config.start,
                end=self.base_config.end,
                initial_capital=self.base_config.initial_capital,
                strategy_params={**self.base_config.strategy_params, **params},
                signal_start=self.base_config.signal_start,
            )
            engine = self.engine_factory(cfg)
            result = engine.run()
            result.name = f"sweep_{i+1}_{params}"
            results.append(result)
        return results


# Internal stub used by WalkForwardValidator to avoid importing cost model
class _NoopExecution:
    def apply_costs(self, returns, trades, prices):
        return returns
