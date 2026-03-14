"""CLI entry point for intraday_research.

Usage
-----
    python main.py                         # backtest with default config
    python main.py --config config/my.yaml
    python main.py --live                  # run live once
    python main.py --live --loop           # run live on schedule
    python main.py --dashboard             # open terminal dashboard
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="intraday_research",
        description="Intraday algorithmic trading — backtest and live trading CLI",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live trading (requires Alpaca credentials in .env)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="(with --live) Run on a recurring hourly schedule",
    )
    parser.add_argument(
        "--utc-hour",
        type=int,
        default=None,
        help="(with --live --loop) Only fire at this UTC hour",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Open the Rich terminal dashboard",
    )
    args = parser.parse_args()

    if args.dashboard:
        from dashboard import main as dashboard_main
        dashboard_main()

    elif args.live:
        import sys
        argv = ["run_live.py", "--config", args.config]
        if args.loop:
            argv.append("--loop")
        if args.utc_hour is not None:
            argv += ["--utc-hour", str(args.utc_hour)]

        import sys as _sys
        _sys.argv = argv
        from run_live import main as live_main
        live_main()

    else:
        # Default: run the multi-strategy backtest
        import backtests.run_backtest as rb
        rb.main()


if __name__ == "__main__":
    main()
