"""ML-only backtest with explicit train/test split.

Compares DirectionMLStrategy vs Buy & Hold over the test period.
Shows trade markers on equity chart and a drawdown subplot.

Usage:
    python backtests/run_ml_backtest.py
    python backtests/run_ml_backtest.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from analytics.performance import compute_metrics
from backtest.engine import BacktestConfig, BacktestEngine
from data.handlers import MockHandler
from execution.cost_model import ProportionalCostModel
from strategies.buy_and_hold import BuyAndHoldStrategy
from strategies.ml.direction_ml import DirectionMLStrategy

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str = "config/default.yaml") -> None:
    cfg = load_config(config_path)
    sp = cfg.get("strategy_params", {})
    ml_params = sp.get("ml", {})

    bt = cfg.get("backtest", {})
    INITIAL_CAPITAL = float(bt.get("initial_capital", 10_000.0))
    COMMISSION_PCT = float(bt.get("commission_pct", 0.0025))

    all_symbols = (
        cfg["symbols"]["long_history"] + cfg["symbols"]["short_history"]
    )

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading data for ML backtest: {all_symbols}")

    try:
        from data.alpaca_handler import AlpacaDataHandler
        from data.loaders import merge_market_data

        handler_alpaca = AlpacaDataHandler()
        long_syms = cfg["symbols"]["long_history"]
        short_syms = cfg["symbols"]["short_history"]
        long_start = cfg["symbols"]["long_start"]
        short_start = cfg["symbols"]["short_start"]
        end_date = cfg["backtest"]["test_end"]

        datasets = []
        if long_syms:
            datasets.append(handler_alpaca.load(long_syms, long_start, end_date))
        if short_syms:
            datasets.append(handler_alpaca.load(short_syms, short_start, end_date))

        full_data = merge_market_data(datasets) if len(datasets) > 1 else datasets[0]

    except Exception as exc:
        print(f"Alpaca unavailable ({exc}). Using yfinance fallback...")
        from data.handlers import YFinanceHandler
        train_start = cfg["backtest"]["train_start"]
        test_end = cfg["backtest"]["test_end"]
        yf = YFinanceHandler(interval="1h")
        full_data = yf.load(all_symbols, train_start, test_end)

    print(f"Total bars: {len(full_data.index)}  ({full_data.index[0]} → {full_data.index[-1]})")

    # ── 70/30 train/test split by position ───────────────────────────────────
    n = len(full_data.index)
    split_idx = int(n * TRAIN_SPLIT)
    train_cutoff = str(full_data.index[split_idx - 1].date())
    test_start = str(full_data.index[split_idx].date())
    test_end = str(full_data.index[-1].date())
    train_start = str(full_data.index[0].date())

    print(f"Train: {train_start} → {train_cutoff}  ({split_idx} bars)")
    print(f"Test:  {test_start} → {test_end}  ({n - split_idx} bars)")

    train_data = full_data.slice(train_start, train_cutoff)

    # ── Build and fit ML strategy ─────────────────────────────────────────────
    ml = DirectionMLStrategy(
        n_estimators=ml_params.get("n_estimators", 200),
        max_depth=ml_params.get("max_depth", 3),
        learning_rate=ml_params.get("learning_rate", 0.05),
        forward_bars=ml_params.get("forward_bars", 8),
        proba_threshold=ml_params.get("proba_threshold", 0.52),
    )

    print("Fitting ML strategy on training data...")
    ml.fit(train_data)
    print("Fitting complete.")

    bah = BuyAndHoldStrategy()
    execution = ProportionalCostModel(commission_pct=COMMISSION_PCT)

    # ── Run backtests ─────────────────────────────────────────────────────────
    results = {}
    for strat in [ml, bah]:
        handler = MockHandler(full_data)
        bt_cfg = BacktestConfig(
            symbols=full_data.symbols,
            start=test_start,
            end=test_end,
            initial_capital=INITIAL_CAPITAL,
            strategy_params={"commission_pct": COMMISSION_PCT},
            signal_start=train_start,
        )
        engine = BacktestEngine(
            config=bt_cfg,
            data_handler=handler,
            strategy=strat,
            execution_model=execution,
        )
        result = engine.run()
        results[strat.name] = result
        m = compute_metrics(result)
        print(
            f"{strat.name:20s}  "
            f"total={m['total_return']:+.1%}  "
            f"sharpe={m['sharpe_ratio']:.2f}  "
            f"maxdd={m['max_drawdown']:.1%}  "
            f"trades={m['num_trades']}"
        )

    # ── Chart ─────────────────────────────────────────────────────────────────
    _save_ml_chart(results, ml)


def _save_ml_chart(results: dict, ml_strategy: DirectionMLStrategy) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})

    colors = {"ml": "steelblue", "buy_and_hold": "orange"}

    for name, result in results.items():
        eq = result.equity_curve
        if eq.empty:
            continue
        norm = eq / eq.iloc[0]
        ax1.plot(norm.index, norm.values, label=name, color=colors.get(name, "gray"))

    # Trade markers for ML
    ml_result = results.get("ml")
    if ml_result is not None and not ml_result.signals.empty:
        signals = ml_result.signals
        # Detect transitions
        prev = signals.shift(1).fillna(0)
        buys = ((signals > 0.05) & (prev <= 0.05)).any(axis=1)
        sells = ((signals <= 0.05) & (prev > 0.05)).any(axis=1)

        eq_ml = ml_result.equity_curve
        norm_ml = eq_ml / eq_ml.iloc[0]

        buy_times = norm_ml.index[norm_ml.index.isin(signals.index[buys])]
        sell_times = norm_ml.index[norm_ml.index.isin(signals.index[sells])]

        if len(buy_times):
            ax1.scatter(buy_times, norm_ml.loc[buy_times], marker="^", color="green",
                        s=30, zorder=5, label="Buy", alpha=0.7)
        if len(sell_times):
            ax1.scatter(sell_times, norm_ml.loc[sell_times], marker="v", color="red",
                        s=30, zorder=5, label="Sell", alpha=0.7)

    ax1.set_title("ML vs Buy & Hold — Test Period")
    ax1.set_ylabel("Normalised Equity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown subplot for ML
    if ml_result is not None and not ml_result.equity_curve.empty:
        eq = ml_result.equity_curve
        rolling_max = eq.cummax()
        drawdown = (eq - rolling_max) / rolling_max
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4, color="red")
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

    out_path = Path(__file__).parent / "ml_backtest.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nML backtest chart saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    main(args.config)
