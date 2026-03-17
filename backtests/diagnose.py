"""Diagnostic script to trace exactly why strategies bleed to zero.

Runs on real Alpaca data (or yfinance fallback) and prints:
  - Monthly portfolio values, cash, position notionals
  - Signal long-percent per symbol per month
  - Trade counts per month
  - Any bars where portfolio value drops by >5% in a single step

Usage:
    python backtests/diagnose.py
    python backtests/diagnose.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str = "config/default.yaml") -> None:
    cfg = load_config(config_path)
    bt = cfg.get("backtest", {})
    sp = cfg.get("strategy_params", {})
    ts_params = sp.get("ts_momentum", {})

    TRAIN_START = bt.get("train_start", "2023-01-01")
    TRAIN_END = bt.get("train_end", "2024-06-30")
    TEST_START = bt.get("test_start", "2024-07-01")
    TEST_END = bt.get("test_end", "2025-12-31")
    INITIAL_CAPITAL = float(bt.get("initial_capital", 10_000.0))
    COMMISSION_PCT = 0.0  # zero costs to isolate signal quality
    DRIFT_THRESHOLD = 0.0  # always rebalance to see exact signal effect

    all_symbols = cfg["symbols"]["long_history"] + cfg["symbols"]["short_history"]

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading data for {all_symbols}")
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
        full_data = merge_market_data(datasets) if len(datasets) > 1 else datasets[0]

    except Exception as exc:
        print(f"Alpaca unavailable ({exc}). Using yfinance...")
        from data.handlers import YFinanceHandler
        yf = YFinanceHandler(interval="1h")
        full_data = yf.load(all_symbols, TRAIN_START, TEST_END)

    print(f"Data: {len(full_data.index)} bars  {full_data.index[0]} → {full_data.index[-1]}")

    # ── Inspect data quality ──────────────────────────────────────────────────
    print("\n── Data Quality ──")
    test_close = full_data.close.loc[TEST_START:TEST_END]
    for sym in full_data.symbols:
        col = test_close[sym]
        nan_pct = col.isna().mean() * 100
        big_moves = (col.pct_change().abs() > 0.10).sum()
        print(f"  {sym}: {len(col)} bars, {nan_pct:.2f}% NaN, {big_moves} bars with >10% move")

    # ── Generate ts_momentum signals on test period ───────────────────────────
    from strategies.ts_momentum import TimeSeriesMomentumStrategy

    ts = TimeSeriesMomentumStrategy(
        lookbacks=ts_params.get("lookbacks", [24, 72, 168, 336]),
        long_threshold=ts_params.get("long_threshold", 0.25),
    )
    ts.fit(full_data)

    # Use same data slice as engine would (signal_start = TRAIN_START)
    engine_data = full_data.close.loc[TRAIN_START:TEST_END]

    # Build signals on the full window (engine generates on full slice)
    full_md = full_data
    # Manually replicate engine logic:
    all_signals = ts.generate_signals(full_md)  # full range
    all_signals = all_signals.shift(1)           # 1-bar lag

    # Slice to test window
    test_signals = all_signals.loc[TEST_START:TEST_END]
    test_close_df = full_data.close.loc[TEST_START:TEST_END]

    print("\n── Signal Statistics (test period) ──")
    for sym in full_data.symbols:
        sig = test_signals[sym]
        long_pct = (sig > 0.05).mean() * 100
        zero_pct = (sig < 0.001).mean() * 100
        print(f"  {sym}: long {long_pct:.1f}% of bars, zero {zero_pct:.1f}% of bars")

    # ── Manual portfolio simulation ───────────────────────────────────────────
    print("\n── Manual Simulation (ts_momentum, zero costs) ──")

    cash = INITIAL_CAPITAL
    positions: dict[str, float] = {}  # sym → qty
    equity_curve = []
    trade_count = 0
    symbols = full_data.symbols

    last_known_prices: dict[str, float] = {}

    index = test_close_df.index

    for i, ts_idx in enumerate(index):
        prices: dict[str, float | None] = {}
        for sym in symbols:
            raw = test_close_df.loc[ts_idx, sym]
            prices[sym] = None if (raw is None or (isinstance(raw, float) and np.isnan(raw))) else float(raw)

        # Force-close NaN-price positions
        force_closed = False
        for sym in list(positions.keys()):
            if prices.get(sym) is None:
                lp = last_known_prices.get(sym)
                if lp is not None and positions.get(sym, 0) != 0:
                    qty = positions.pop(sym)
                    cash += qty * lp
                    trade_count += 1
                    force_closed = True

        # Update last known prices
        for sym in symbols:
            if prices[sym] is not None:
                last_known_prices[sym] = prices[sym]

        # Portfolio value
        pv = cash
        for sym, qty in positions.items():
            p = prices.get(sym)
            if p is not None:
                pv += qty * p

        if force_closed:
            equity_curve.append((ts_idx, pv, cash))
            continue

        # Get signals
        if ts_idx in test_signals.index:
            raw_w = test_signals.loc[ts_idx]
        else:
            raw_w = pd.Series(0.0, index=symbols)

        valid_weights: dict[str, float] = {}
        for sym in symbols:
            if prices.get(sym) is not None:
                w = raw_w.get(sym, 0.0)
                if isinstance(w, float) and np.isnan(w):
                    w = 0.0
                valid_weights[sym] = float(w)

        total_w = sum(abs(v) for v in valid_weights.values())
        if total_w > 1.0:
            valid_weights = {s: v / total_w for s, v in valid_weights.items()}

        # Rebalance
        pv2 = cash
        for sym, qty in positions.items():
            p = prices.get(sym)
            if p is not None:
                pv2 += qty * p

        for sym, weight in valid_weights.items():
            price = prices.get(sym)
            if price is None or price == 0:
                continue
            current_qty = positions.get(sym, 0.0)
            current_w = (current_qty * price) / pv2 if pv2 > 0 else 0.0
            if abs(current_w - weight) < DRIFT_THRESHOLD:
                continue
            target_qty = pv2 * weight / price
            delta = target_qty - current_qty
            if abs(delta) < 1e-10:
                continue
            cash -= delta * price
            positions[sym] = target_qty
            trade_count += 1

        # Close symbols not in target
        for sym in list(positions.keys()):
            if sym not in valid_weights or valid_weights.get(sym, 0) == 0:
                if positions.get(sym, 0) != 0:
                    price = prices.get(sym)
                    if price is not None and price != 0:
                        qty = positions.pop(sym)
                        cash += qty * price
                        trade_count += 1
            elif valid_weights.get(sym, 0) == 0:
                price = prices.get(sym)
                if price is not None and price != 0:
                    qty = positions.pop(sym, 0)
                    if qty != 0:
                        cash += qty * price
                        trade_count += 1

        pv_after = cash
        for sym, qty in positions.items():
            p = prices.get(sym)
            if p is not None:
                pv_after += qty * p

        equity_curve.append((ts_idx, pv_after, cash))

    eq_series = pd.Series(
        [v for _, v, _ in equity_curve],
        index=[t for t, _, _ in equity_curve],
    )
    cash_series = pd.Series(
        [c for _, _, c in equity_curve],
        index=[t for t, _, _ in equity_curve],
    )

    # ── Monthly breakdown ──────────────────────────────────────────────────────
    print("\n── Monthly Portfolio Values ──")
    monthly = eq_series.resample("ME").last()
    monthly_cash = cash_series.resample("ME").last()
    monthly_trades = pd.Series(1, index=eq_series.index).resample("ME").count()

    prev_val = INITIAL_CAPITAL
    for dt in monthly.index:
        val = monthly[dt]
        c = monthly_cash[dt]
        ret = (val / prev_val - 1) * 100 if prev_val > 0 else 0
        print(f"  {dt.strftime('%Y-%m')}: equity={val:>10,.0f}  cash={c:>10,.0f}  monthly_ret={ret:+.1f}%")
        prev_val = val

    total_ret = (eq_series.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    print(f"\nTotal return: {total_ret:+.1f}%  (trades: {trade_count})")

    # ── Detect large single-bar drops ─────────────────────────────────────────
    drops = eq_series.pct_change()
    big_drops = drops[drops < -0.03].sort_values()
    print(f"\n── Largest single-bar drops (>3%) ── ({len(big_drops)} total)")
    for dt, d in big_drops.head(10).items():
        print(f"  {dt}  {d:+.2%}")

    # ── Compare to buy-and-hold ────────────────────────────────────────────────
    print("\n── Buy & Hold comparison ──")
    test_c = full_data.close.loc[TEST_START:TEST_END]
    n = len(full_data.symbols)
    bah_weights = {s: 1.0 / n for s in full_data.symbols}
    bah_eq = sum(
        (test_c[sym] / test_c[sym].iloc[0]) * bah_weights[sym]
        for sym in full_data.symbols
        if sym in test_c.columns
    )
    print(f"  B&H total return: {(bah_eq.iloc[-1] - 1) * 100:+.1f}%")

    # Check if signals are generating ANY long positions
    any_long = (test_signals > 0.05).any(axis=1)
    print(f"\n  Bars with >=1 long signal: {any_long.sum()} / {len(any_long)} ({any_long.mean()*100:.1f}%)")
    zero_bars = (~any_long).sum()
    print(f"  Bars with ALL signals = 0 (cash): {zero_bars} ({zero_bars/len(any_long)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    main(args.config)
