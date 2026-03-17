"""Live trading loop: fit → signal → execute on Alpaca.

Usage
-----
    python run_live.py                        # run once
    python run_live.py --loop                 # fire at top of every hour
    python run_live.py --loop --utc-hour 8   # fire at 08:00 UTC only
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from data.alpaca_handler import AlpacaDataHandler
from data.loaders import merge_market_data
from execution.alpaca_trader import AlpacaTrader
from backtests.run_full_backtest import build_strategies_from_config
from utils.secrets import load_dotenv

STATE_FILE = Path("state/last_signals.json")
CONFIG_FILE = Path("config/default.yaml")


def load_config(path: Path = CONFIG_FILE) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_data(cfg: dict) -> "MarketData":  # type: ignore[name-defined]
    """Load data with symbol-specific start dates, then merge."""
    handler = AlpacaDataHandler()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    long_syms = cfg["symbols"]["long_history"]
    short_syms = cfg["symbols"]["short_history"]
    long_start = cfg["symbols"]["long_start"]
    short_start = cfg["symbols"]["short_start"]

    datasets = []

    if long_syms:
        long_data = handler.load(long_syms, long_start, today)
        datasets.append(long_data)

    if short_syms:
        short_data = handler.load(short_syms, short_start, today)
        datasets.append(short_data)

    if len(datasets) == 1:
        return datasets[0]

    return merge_market_data(datasets)


def build_strategy(cfg: dict, data: "MarketData", daily_btc_close: pd.Series):  # type: ignore[name-defined]
    """Build the composite live strategy from config, fit it, and return it."""
    bt = cfg.get("backtest", {})
    _, _, composite = build_strategies_from_config(
        cfg, data, daily_btc_close,
        train_start=bt.get("train_start", "2023-01-01"),
        train_end=bt.get("train_end", "2024-06-30"),
    )
    return composite


def run_once(cfg: dict) -> None:
    """Run a single iteration: load → fit → signal → execute → save state."""
    print(f"[{datetime.now(timezone.utc).isoformat()}] Loading data...")
    data = build_data(cfg)

    print(f"Loaded {len(data.symbols)} symbols, {len(data.index)} bars "
          f"({data.index[0]} → {data.index[-1]})")

    barometer_sym = cfg.get("strategy_params", {}).get("macro_regime", {}).get(
        "barometer_symbol", "BTC/USD"
    )
    if barometer_sym not in data.close.columns:
        barometer_sym = data.symbols[0]
    daily_btc_close = data.close[barometer_sym].resample("D").last().dropna()

    print("Building and fitting strategy...")
    strategy = build_strategy(cfg, data, daily_btc_close)
    strategy.fit(data)

    # Regime label (walks filter chain to find MacroRegimeFilter)
    from strategies.macro_regime import MacroRegimeFilter
    regime_label = "unknown"
    _s = strategy
    while _s is not None:
        if isinstance(_s, MacroRegimeFilter):
            regime_label = _s.current_regime
            break
        _s = getattr(_s, "_strategy", None)
    print(f"Macro regime: {regime_label.upper()}")

    print("Generating signals...")
    signals = strategy.generate_signals(data)

    # Latest weights
    latest = signals.iloc[-1] if not signals.empty else pd.Series(dtype=float)
    as_of = str(data.index[-1]) if len(data.index) else "unknown"
    computed_at = datetime.now(timezone.utc).isoformat()

    # Collect per-sub-strategy signals — walk to innermost CompositeStrategy
    from strategies.composite import CompositeStrategy
    strategy_signals: dict[str, dict[str, float]] = {}
    _s = strategy
    while _s is not None:
        if isinstance(_s, CompositeStrategy):
            for sub in _s._strategies:
                sub_sig = sub.generate_signals(data)
                row = sub_sig.iloc[-1] if not sub_sig.empty else pd.Series(dtype=float)
                strategy_signals[sub.name] = {sym: float(row.get(sym, 0.0)) for sym in data.symbols}
            break
        _s = getattr(_s, "_strategy", None)

    # Strategy weights from innermost CompositeStrategy
    strategy_weights: dict[str, float] = {}
    _s = strategy
    while _s is not None:
        if isinstance(_s, CompositeStrategy):
            strategy_weights = {k: round(v, 4) for k, v in _s.last_strategy_weights.items()}
            break
        _s = getattr(_s, "_strategy", None)

    state = {
        "as_of": as_of,
        "computed_at": computed_at,
        "macro_regime": regime_label,
        "symbols": data.symbols,
        "strategy_signals": strategy_signals,
        "strategy_weights": strategy_weights,
        "composite": {sym: float(latest.get(sym, 0.0)) for sym in data.symbols},
    }

    STATE_FILE.parent.mkdir(exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Signals saved → {STATE_FILE}")

    # Execute on Alpaca
    live_cfg = cfg.get("live", {})
    notional = float(live_cfg.get("notional_per_trade", 300.0))

    print("Connecting to Alpaca...")
    trader = AlpacaTrader()
    acct = trader.get_account()
    print(f"Account equity: ${acct['equity']:,.2f}  cash: ${acct['cash']:,.2f}")

    for sym in data.symbols:
        weight = float(latest.get(sym, 0.0))
        signal = 1 if weight > 0.05 else (-1 if weight < -0.05 else 0)
        print(f"  {sym}: weight={weight:.3f} → signal={signal:+d}")
        try:
            result = trader.execute_signal(sym, signal, notional)
            if result:
                print(f"    Order placed: {result}")
        except Exception as exc:
            print(f"    ERROR executing {sym}: {exc}")

    print(f"Done at {computed_at}")


def _wait_until_next_hour(target_utc_hour: int | None) -> None:
    """Block until the next top-of-hour (optionally filtered by UTC hour)."""
    while True:
        now = datetime.now(timezone.utc)
        if target_utc_hour is None or now.hour == target_utc_hour:
            # Wait until the next :05 mark past the hour
            if now.minute < 5:
                wait = (5 - now.minute) * 60 - now.second
            else:
                wait = (65 - now.minute) * 60 - now.second
        else:
            # Wait until target hour
            target = now.replace(hour=target_utc_hour, minute=5, second=0, microsecond=0)
            if target < now:
                target = target.replace(day=target.day + 1)
            wait = (target - now).total_seconds()

        print(f"Sleeping {wait:.0f}s until next run...")
        time.sleep(max(wait, 1))
        return


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Intraday live trading loop")
    parser.add_argument("--config", default=str(CONFIG_FILE), help="Path to config YAML")
    parser.add_argument("--loop", action="store_true", help="Run on a recurring schedule")
    parser.add_argument("--utc-hour", type=int, default=None,
                        help="Only execute at this UTC hour (default: every hour)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    if not args.loop:
        run_once(cfg)
    else:
        print("Running in loop mode. Press Ctrl+C to stop.")
        while True:
            try:
                run_once(cfg)
            except Exception as exc:
                print(f"ERROR: {exc}")
            _wait_until_next_hour(args.utc_hour)


if __name__ == "__main__":
    main()
