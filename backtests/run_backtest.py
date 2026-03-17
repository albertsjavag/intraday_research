"""Multi-strategy backtest over the same test period.

All strategies are tested on an identical window — no cherry-picking.
Saves an equity chart PNG to backtests/.

Usage:
    python backtests/run_backtest.py
    python backtests/run_backtest.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from analytics.performance import aggregate_sweep, compute_metrics
from backtest.engine import BacktestConfig, BacktestEngine
from data.handlers import MockHandler
from execution.cost_model import ProportionalCostModel
from strategies.buy_and_hold import BuyAndHoldStrategy
from strategies.composite import CompositeStrategy
from strategies.macro_regime import MacroRegimeFilter
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml.direction_ml import DirectionMLStrategy
from strategies.ts_momentum import TimeSeriesMomentumStrategy

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str = "config/default.yaml") -> None:
    cfg = load_config(config_path)
    sp = cfg.get("strategy_params", {})

    bt = cfg.get("backtest", {})
    TRAIN_START = bt.get("train_start", "2021-01-01")
    TRAIN_END = bt.get("train_end", "2023-06-30")
    TEST_START = bt.get("test_start", "2023-07-01")
    TEST_END = bt.get("test_end", "2025-12-31")
    INITIAL_CAPITAL = float(bt.get("initial_capital", 10_000.0))
    COMMISSION_PCT = float(bt.get("commission_pct", 0.0025))
    DRIFT_THRESHOLD = float(bt.get("drift_threshold", 0.02))

    all_symbols = (
        cfg["symbols"]["long_history"] + cfg["symbols"]["short_history"]
    )

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading data: {all_symbols}  {TRAIN_START} → {TEST_END}")

    try:
        from data.alpaca_handler import AlpacaDataHandler
        from data.loaders import merge_market_data

        handler_alpaca = AlpacaDataHandler()
        long_syms = cfg["symbols"]["long_history"]
        short_syms = cfg["symbols"]["short_history"]
        long_start = cfg["symbols"]["long_start"]
        short_start = cfg["symbols"]["short_start"]

        datasets = []
        if long_syms:
            datasets.append(handler_alpaca.load(long_syms, long_start, TEST_END))
        if short_syms:
            datasets.append(handler_alpaca.load(short_syms, short_start, TEST_END))

        from data.loaders import merge_market_data
        full_data = merge_market_data(datasets) if len(datasets) > 1 else datasets[0]

    except Exception as exc:
        print(f"Alpaca unavailable ({exc}). Trying yfinance fallback...")
        from data.handlers import YFinanceHandler
        yf_handler = YFinanceHandler(interval="1h")
        full_data = yf_handler.load(all_symbols, TRAIN_START, TEST_END)

    print(f"Data loaded: {len(full_data.index)} bars, symbols: {full_data.symbols}")

    # ── Build strategies ─────────────────────────────────────────────────────
    ts_params = sp.get("ts_momentum", {})
    mr_params = sp.get("mean_reversion", {})
    ml_params = sp.get("ml", {})
    comp_params = sp.get("composite", {})
    regime_params = sp.get("macro_regime", {})

    bah = BuyAndHoldStrategy()
    ts_mom = TimeSeriesMomentumStrategy(
        lookbacks=ts_params.get("lookbacks", [72, 168, 336, 720]),
        long_threshold=ts_params.get("long_threshold", 0.25),
        min_holding_bars=ts_params.get("min_holding_bars", 0),
    )
    mean_rev = MeanReversionStrategy(
        lookback=mr_params.get("lookback", 168),
        z_threshold=mr_params.get("z_threshold", 0.5),
        exit_z=mr_params.get("exit_z", 0.0),
        long_only=mr_params.get("long_only", True),
    )
    ml = DirectionMLStrategy(
        n_estimators=ml_params.get("n_estimators", 200),
        max_depth=ml_params.get("max_depth", 3),
        learning_rate=ml_params.get("learning_rate", 0.05),
        forward_bars=ml_params.get("forward_bars", 8),
        proba_threshold=ml_params.get("proba_threshold", 0.52),
    )
    # ML is excluded from composite: its bar-by-bar signal changes contaminate
    # the blended output with excessive turnover. It runs best as a standalone.
    composite = CompositeStrategy(
        strategies=[bah, ts_mom, mean_rev],
        lookback=comp_params.get("lookback", 168),
        rebalance_freq=comp_params.get("rebalance_freq", 24),
        max_weight=comp_params.get("max_weight", 0.60),
    )

    # Derive daily BTC close from hourly data (resample avoids a second API call)
    barometer_sym = regime_params.get("barometer_symbol", "BTC/USD")
    if barometer_sym not in full_data.close.columns:
        barometer_sym = full_data.symbols[0]
    daily_btc_close = full_data.close[barometer_sym].resample("D").last().dropna()

    regime_ts_mom = MacroRegimeFilter(
        ts_mom,
        daily_btc_close,
        ma_period=regime_params.get("ma_period", 200),
        slope_window=regime_params.get("slope_window", 20),
        sideways_band=regime_params.get("sideways_band", 0.02),
    )

    strategies = [bah, ts_mom, mean_rev, regime_ts_mom]

    # Pre-fit all strategies with appropriate data windows.
    # ML is trained on train period only — avoids look-ahead contamination.
    # Non-ML strategies have no meaningful fit() so full_data is fine.
    train_data = full_data.slice(TRAIN_START, TRAIN_END)
    print(f"Fitting ML on {TRAIN_START} → {TRAIN_END} ({len(train_data.index)} bars)...")
    ml.fit(train_data)
    print("ML fitting complete.")

    bah.fit(full_data)
    ts_mom.fit(full_data)
    mean_rev.fit(full_data)
    regime_ts_mom.fit(full_data)

    # ── Run backtests ─────────────────────────────────────────────────────────
    execution = ProportionalCostModel(commission_pct=COMMISSION_PCT)
    results = []

    for strat in strategies:
        print(f"Running backtest: {strat.name}...")
        handler = MockHandler(full_data)
        cfg_bt = BacktestConfig(
            symbols=full_data.symbols,
            start=TEST_START,
            end=TEST_END,
            initial_capital=INITIAL_CAPITAL,
            strategy_params={"commission_pct": COMMISSION_PCT, "drift_threshold": DRIFT_THRESHOLD},
            signal_start=TRAIN_START,
            skip_fit=True,
        )
        engine = BacktestEngine(
            config=cfg_bt,
            data_handler=handler,
            strategy=strat,
            execution_model=execution,
        )
        result = engine.run()
        results.append(result)
        m = compute_metrics(result)
        print(
            f"  {strat.name:20s}  "
            f"total={m['total_return']:+.1%}  "
            f"sharpe={m['sharpe_ratio']:.2f}  "
            f"maxdd={m['max_drawdown']:.1%}  "
            f"trades={m['num_trades']}"
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Performance Summary ──────────────────────────────────────────")
    summary = aggregate_sweep(results)
    print(summary.to_string())

    # ── Equity chart ──────────────────────────────────────────────────────────
    _save_equity_chart(results)


def _save_equity_chart(results: list) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart. pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    for r in results:
        eq = r.equity_curve
        if eq.empty:
            continue
        normalised = eq / eq.iloc[0]
        ax.plot(normalised.index, normalised.values, label=r.name)

    ax.set_title("Equity Curves (normalised) — Test Period")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalised Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = Path(__file__).parent.parent / "plots" / "equity_curves.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nEquity chart saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    main(args.config)
