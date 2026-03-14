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
from strategies.buy_and_hold import BuyAndHoldStrategy
from strategies.composite import CompositeStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml.direction_ml import DirectionMLStrategy
from strategies.ts_momentum import TimeSeriesMomentumStrategy
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


def build_strategy(cfg: dict) -> CompositeStrategy:
    sp = cfg.get("strategy_params", {})

    ts_params = sp.get("ts_momentum", {})
    mr_params = sp.get("mean_reversion", {})
    ml_params = sp.get("ml", {})
    comp_params = sp.get("composite", {})

    bah = BuyAndHoldStrategy()
    ts_mom = TimeSeriesMomentumStrategy(
        lookbacks=ts_params.get("lookbacks", [24, 72, 168, 336]),
        long_threshold=ts_params.get("long_threshold", 0.25),
    )
    mean_rev = MeanReversionStrategy(
        lookback=mr_params.get("lookback", 48),
        z_threshold=mr_params.get("z_threshold", 0.5),
        long_only=mr_params.get("long_only", True),
    )
    ml = DirectionMLStrategy(
        n_estimators=ml_params.get("n_estimators", 200),
        max_depth=ml_params.get("max_depth", 3),
        learning_rate=ml_params.get("learning_rate", 0.05),
        forward_bars=ml_params.get("forward_bars", 8),
        proba_threshold=ml_params.get("proba_threshold", 0.52),
    )

    return CompositeStrategy(
        strategies=[bah, ts_mom, mean_rev, ml],
        lookback=comp_params.get("lookback", 168),
        rebalance_freq=comp_params.get("rebalance_freq", 24),
        max_weight=comp_params.get("max_weight", 0.60),
    )


def run_once(cfg: dict) -> None:
    """Run a single iteration: load → fit → signal → execute → save state."""
    print(f"[{datetime.now(timezone.utc).isoformat()}] Loading data...")
    data = build_data(cfg)

    print(f"Loaded {len(data.symbols)} symbols, {len(data.index)} bars "
          f"({data.index[0]} → {data.index[-1]})")

    strategy = build_strategy(cfg)

    print("Fitting strategy...")
    strategy.fit(data)

    print("Generating signals...")
    signals = strategy.generate_signals(data)

    # Latest weights
    latest = signals.iloc[-1] if not signals.empty else pd.Series(dtype=float)
    as_of = str(data.index[-1]) if len(data.index) else "unknown"
    computed_at = datetime.now(timezone.utc).isoformat()

    # Collect per-sub-strategy signals at latest bar
    strategy_signals: dict[str, dict[str, float]] = {}
    for sub in strategy._strategies:
        sub_sig = sub.generate_signals(data)
        row = sub_sig.iloc[-1] if not sub_sig.empty else pd.Series(dtype=float)
        strategy_signals[sub.name] = {sym: float(row.get(sym, 0.0)) for sym in data.symbols}

    state = {
        "as_of": as_of,
        "computed_at": computed_at,
        "symbols": data.symbols,
        "strategy_signals": strategy_signals,
        "strategy_weights": {k: round(v, 4) for k, v in strategy.last_strategy_weights.items()},
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
