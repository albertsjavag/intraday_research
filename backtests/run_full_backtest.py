"""Comprehensive multi-strategy backtest with regime visualisation.

Reads which strategies to run from config/default.yaml (strategies.enabled).
Produces a 3-panel chart:
  1. Equity curves (all strategies + composite) with regime shading
  2. Regime strip showing bull / sideways / bear over time
  3. Drawdown chart for composite vs buy-and-hold

Usage:
    python backtests/run_full_backtest.py
    python backtests/run_full_backtest.py --config config/default.yaml
    python backtests/run_full_backtest.py --walk-forward   # fold validation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from analytics.performance import aggregate_sweep, compute_metrics
from backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from data.handlers import MockHandler
from execution.cost_model import ProportionalCostModel
from strategies.base import Strategy
from strategies.buy_and_hold import BuyAndHoldStrategy
from strategies.composite import CompositeStrategy
from strategies.macro_regime import MacroRegimeFilter
from strategies.mvrv_filter import MVRVFilter
from strategies.tod_filter import TodScalingFilter

# ── colour palette ─────────────────────────────────────────────────────────
_STRATEGY_COLORS = {
    "buy_and_hold":              "#aaaaaa",
    "ts_momentum":               "#2196F3",   # blue
    "cross_sectional_momentum":  "#FF9800",   # orange
    "volatility_breakout":       "#4CAF50",   # green
    "pairs_arb":                 "#9C27B0",   # purple
    "ml":                        "#F44336",   # red
    "composite":                 "#000000",   # black (thicker)
}
_REGIME_COLORS = {"bull": "#4CAF50", "sideways": "#FFC107", "bear": "#F44336"}


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_data(cfg: dict, test_end: str) -> "MarketData":  # type: ignore[name-defined]
    all_symbols = cfg["symbols"]["long_history"] + cfg["symbols"]["short_history"]
    try:
        from data.alpaca_handler import AlpacaDataHandler
        from data.loaders import merge_market_data

        handler = AlpacaDataHandler()
        datasets = []
        if cfg["symbols"]["long_history"]:
            datasets.append(
                handler.load(cfg["symbols"]["long_history"],
                             cfg["symbols"]["long_start"], test_end)
            )
        if cfg["symbols"]["short_history"]:
            datasets.append(
                handler.load(cfg["symbols"]["short_history"],
                             cfg["symbols"]["short_start"], test_end)
            )
        return merge_market_data(datasets) if len(datasets) > 1 else datasets[0]

    except Exception as exc:
        print(f"Alpaca unavailable ({exc}). Using yfinance fallback...")
        from data.handlers import YFinanceHandler
        yf = YFinanceHandler(interval="1h")
        return yf.load(all_symbols, cfg["backtest"]["train_start"], test_end)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy builder
# ─────────────────────────────────────────────────────────────────────────────

def build_strategies_from_config(
    cfg: dict,
    full_data: "MarketData",  # type: ignore[name-defined]
    daily_btc_close: pd.Series,
    train_start: str,
    train_end: str,
) -> tuple[BuyAndHoldStrategy, list[Strategy], Strategy]:
    """Return (benchmark, individual_strategies, composite).

    composite is the strategy that would be traded live — it is the
    Sharpe-weighted blend of all enabled strategies, wrapped in the
    optional filters (regime, tod, mvrv) specified in config.
    """
    sp = cfg.get("strategy_params", {})
    strat_cfg = cfg.get("strategies", {})
    enabled = strat_cfg.get("enabled", ["ts_momentum"])

    bah = BuyAndHoldStrategy()
    individual: list[Strategy] = []

    for name in enabled:
        strat = _build_one(name, sp, full_data, train_start, train_end)
        if strat is not None:
            individual.append(strat)

    if not individual:
        individual = [BuyAndHoldStrategy()]

    # Composite: Sharpe-weighted blend
    comp_p = sp.get("composite", {})
    composite_inner = CompositeStrategy(
        strategies=individual,
        lookback=comp_p.get("lookback", 168),
        rebalance_freq=comp_p.get("rebalance_freq", 24),
        max_weight=comp_p.get("max_weight", 0.60),
    )

    # Wrap with optional filters (outermost filter applied last)
    composite: Strategy = composite_inner

    # 1. TOD scaling
    tod_cfg = strat_cfg.get("tod_scaling", {})
    if tod_cfg.get("enabled", False):
        from strategies.tod_filter import TodScalingFilter
        composite = TodScalingFilter(
            composite,
            off_hours=tod_cfg.get("off_hours", [2, 3, 4, 5]),
            off_multiplier=tod_cfg.get("off_multiplier", 0.5),
        )

    # 2. MVRV filter
    mvrv_cfg = strat_cfg.get("mvrv_filter", {})
    if mvrv_cfg.get("enabled", False):
        from strategies.mvrv_filter import MVRVFilter
        composite = MVRVFilter(
            composite,
            mvrv_path=mvrv_cfg.get("data_file", "data/mvrv.csv"),
            caution_level=mvrv_cfg.get("caution_level", 3.5),
            overbought_level=mvrv_cfg.get("overbought_level", 7.0),
            caution_multiplier=mvrv_cfg.get("caution_multiplier", 0.5),
            overbought_multiplier=mvrv_cfg.get("overbought_multiplier", 0.0),
        )

    # 3. Macro regime filter (outermost — highest-level gate)
    regime_cfg = strat_cfg.get("regime_filter", {})
    if regime_cfg.get("enabled", True):
        composite = MacroRegimeFilter(
            composite,
            daily_btc_close,
            ma_period=regime_cfg.get("ma_period", 200),
            slope_window=regime_cfg.get("slope_window", 20),
            sideways_band=regime_cfg.get("sideways_band", 0.02),
        )

    return bah, individual, composite


def _build_one(
    name: str,
    sp: dict,
    full_data: "MarketData",  # type: ignore[name-defined]
    train_start: str,
    train_end: str,
) -> Strategy | None:
    """Instantiate one strategy by config name. Returns None for unknowns."""
    if name == "buy_and_hold":
        return BuyAndHoldStrategy()

    if name == "ts_momentum":
        from strategies.ts_momentum import TimeSeriesMomentumStrategy
        p = sp.get("ts_momentum", {})
        return TimeSeriesMomentumStrategy(
            lookbacks=p.get("lookbacks", [72, 168, 336, 720]),
            long_threshold=p.get("long_threshold", 0.25),
            min_holding_bars=p.get("min_holding_bars", 48),
        )

    if name == "cross_sectional":
        from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
        p = sp.get("cross_sectional", {})
        return CrossSectionalMomentumStrategy(
            lookback=p.get("lookback", 720),
            top_n=p.get("top_n", 1),
            rebalance_freq=p.get("rebalance_freq", 168),
        )

    if name == "volatility_breakout":
        from strategies.volatility_breakout import VolatilityBreakoutStrategy
        p = sp.get("volatility_breakout", {})
        return VolatilityBreakoutStrategy(
            channel_window=p.get("channel_window", 48),
            atr_window=p.get("atr_window", 14),
            entry_mult=p.get("entry_mult", 0.5),
            exit_mult=p.get("exit_mult", 0.0),
        )

    if name == "pairs_arb":
        from strategies.pairs_arb import PairsArbStrategy
        p = sp.get("pairs_arb", {})
        return PairsArbStrategy(
            lookback=p.get("lookback", 336),
            tilt_per_sigma=p.get("tilt_per_sigma", 0.15),
            z_cap=p.get("z_cap", 2.0),
        )

    if name == "ml":
        from strategies.ml.direction_ml import DirectionMLStrategy
        p = sp.get("ml", {})
        strat = DirectionMLStrategy(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.03),
            forward_bars=p.get("forward_bars", 24),
            min_return_pct=p.get("min_return_pct", 0.01),
            proba_threshold=p.get("proba_threshold", 0.60),
            proba_smooth_span=p.get("proba_smooth_span", 8),
            use_lightgbm=p.get("use_lightgbm", True),
        )
        # ML must be trained on train window only
        train_data = full_data.slice(train_start, train_end)
        print(f"  Fitting ML on {train_start} → {train_end}...")
        strat.fit(train_data)
        strat._pre_fitted = True  # sentinel: skip re-fit in main loop
        return strat

    print(f"  Unknown strategy '{name}' — skipped.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Backtest runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_one(
    strategy: Strategy,
    full_data: "MarketData",  # type: ignore[name-defined]
    cfg: dict,
    execution: ProportionalCostModel,
) -> BacktestResult:
    bt = cfg["backtest"]
    sp = cfg.get("strategy_params", {})
    bt_cfg = BacktestConfig(
        symbols=full_data.symbols,
        start=bt["test_start"],
        end=bt["test_end"],
        initial_capital=float(bt.get("initial_capital", 10_000.0)),
        strategy_params={
            "commission_pct": float(bt.get("commission_pct", 0.002)),
            "drift_threshold": float(bt.get("drift_threshold", 0.02)),
        },
        signal_start=bt["train_start"],
        skip_fit=True,
    )
    handler = MockHandler(full_data)
    engine = BacktestEngine(
        config=bt_cfg,
        data_handler=handler,
        strategy=strategy,
        execution_model=execution,
    )
    return engine.run()


# ─────────────────────────────────────────────────────────────────────────────
# Regime series helper (for visualisation)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_regime_series(
    composite: Strategy,
    full_data: "MarketData",  # type: ignore[name-defined]
) -> pd.Series | None:
    """Walk the filter chain and return the hourly regime series if available."""
    strat = composite
    while strat is not None:
        if isinstance(strat, MacroRegimeFilter) and strat._regime is not None:
            # Upscale daily regime to hourly for plotting
            daily_regime = strat._regime
            hourly_dates = full_data.index.normalize()
            unique_dates = hourly_dates.unique().sort_values()
            regime_by_date = (
                daily_regime.reindex(unique_dates, method="ffill").fillna("sideways")
            )
            date_to_regime = regime_by_date.to_dict()
            return pd.Series(
                [date_to_regime.get(d, "sideways") for d in hourly_dates],
                index=full_data.index,
                dtype=object,
            )
        # Unwrap one layer of filter
        inner = getattr(strat, "_strategy", None)
        if inner is strat or inner is None:
            break
        strat = inner
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_metrics(results: list[BacktestResult]) -> None:
    print("\n── Performance (test period) " + "─" * 48)
    header = f"{'Strategy':<30}  {'Return':>8}  {'Ann.Ret':>8}  {'Sharpe':>7}  {'MaxDD':>7}  {'Calmar':>7}  {'Trades':>7}"
    print(header)
    print("─" * len(header))
    for r in results:
        m = compute_metrics(r)
        print(
            f"{r.name:<30}  "
            f"{m['total_return']:>+8.1%}  "
            f"{m['annualized_return']:>+8.1%}  "
            f"{m['sharpe_ratio']:>7.2f}  "
            f"{m['max_drawdown']:>7.1%}  "
            f"{m['calmar_ratio']:>7.2f}  "
            f"{m['num_trades']:>7d}"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _plot(
    results: list[BacktestResult],
    regime_series: pd.Series | None,
    cfg: dict,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed — skipping chart.")
        return

    has_regime = regime_series is not None

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    if has_regime:
        gs = gridspec.GridSpec(3, 1, height_ratios=[5, 1, 2], hspace=0.08)
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 2], hspace=0.08)

    ax_eq = fig.add_subplot(gs[0])        # equity curves
    ax_dd = fig.add_subplot(gs[-1])       # drawdowns
    ax_regime = fig.add_subplot(gs[1]) if has_regime else None

    bt = cfg["backtest"]
    test_start = pd.Timestamp(bt["test_start"])

    # ── Regime background shading on equity panel ─────────────────────────
    if has_regime:
        test_regime = regime_series[regime_series.index >= test_start]
        _shade_regimes(ax_eq, test_regime)
        _shade_regimes(ax_dd, test_regime)

    # ── Equity curves ──────────────────────────────────────────────────────
    composite_result = None
    for r in results:
        eq = r.equity_curve
        if eq.empty:
            continue
        # Only plot test period
        eq = eq[eq.index >= test_start]
        if eq.empty:
            continue
        norm = eq / eq.iloc[0]

        is_composite = r.name.startswith("regime_") or r.name == "composite"
        is_bah = r.name == "buy_and_hold"

        color = _STRATEGY_COLORS.get(r.name, "#888888")
        lw = 2.5 if is_composite else (1.2 if is_bah else 1.6)
        ls = "--" if is_bah else "-"
        alpha = 1.0 if (is_composite or is_bah) else 0.75
        zorder = 10 if is_composite else (5 if is_bah else 3)

        ax_eq.plot(norm.index, norm.values, label=r.name,
                   color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder)

        if is_composite:
            composite_result = r

    ax_eq.set_ylabel("Normalised Equity (start = 1.0)", fontsize=10)
    ax_eq.set_title(
        f"Strategy Comparison — Test Period: {bt['test_start']} → {bt['test_end']}",
        fontsize=13, fontweight="bold",
    )
    ax_eq.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax_eq.grid(True, alpha=0.25)
    ax_eq.tick_params(labelbottom=False)

    # Regime legend patches
    if has_regime:
        legend_patches = [
            mpatches.Patch(color=_REGIME_COLORS["bull"],     alpha=0.25, label="Bull regime"),
            mpatches.Patch(color=_REGIME_COLORS["sideways"], alpha=0.25, label="Sideways"),
            mpatches.Patch(color=_REGIME_COLORS["bear"],     alpha=0.25, label="Bear regime"),
        ]
        ax_eq.legend(
            handles=ax_eq.get_legend_handles_labels()[0] + legend_patches,
            labels=ax_eq.get_legend_handles_labels()[1] + ["Bull", "Sideways", "Bear"],
            loc="upper left", fontsize=9, framealpha=0.8,
        )

    # ── Regime strip ───────────────────────────────────────────────────────
    if has_regime and ax_regime is not None:
        test_regime = regime_series[regime_series.index >= test_start]
        regime_num = test_regime.map({"bull": 2, "sideways": 1, "bear": 0}).fillna(1)
        regime_color_series = test_regime.map(_REGIME_COLORS).fillna(_REGIME_COLORS["sideways"])

        # Plot as colored scatter strip
        ax_regime.bar(
            test_regime.index, 1,
            color=regime_color_series.values,
            width=pd.Timedelta(hours=1),
            align="edge",
            alpha=0.9,
        )
        ax_regime.set_yticks([0.5])
        ax_regime.set_yticklabels(["Macro\nRegime"], fontsize=8)
        ax_regime.set_ylim(0, 1)
        ax_regime.tick_params(labelbottom=False)
        ax_regime.set_xlim(ax_eq.get_xlim())
        ax_regime.grid(False)

        # Regime annotations (percentage time in each)
        counts = test_regime.value_counts(normalize=True)
        label = "  |  ".join(
            f"{r}: {counts.get(r, 0):.0%}" for r in ["bull", "sideways", "bear"]
        )
        ax_regime.set_xlabel(label, fontsize=8, labelpad=2)

    # ── Drawdowns ──────────────────────────────────────────────────────────
    for r in results:
        eq = r.equity_curve
        if eq.empty:
            continue
        eq = eq[eq.index >= test_start]
        if eq.empty:
            continue

        is_composite = r.name.startswith("regime_") or r.name == "composite"
        is_bah = r.name == "buy_and_hold"

        rolling_max = eq.cummax()
        dd = (eq - rolling_max) / rolling_max

        if is_composite:
            ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.35,
                               color=_STRATEGY_COLORS.get(r.name, "black"),
                               label=r.name)
        elif is_bah:
            ax_dd.plot(dd.index, dd.values, lw=1.2, ls="--",
                       color=_STRATEGY_COLORS["buy_and_hold"],
                       alpha=0.7, label="buy_and_hold")
        else:
            ax_dd.plot(dd.index, dd.values, lw=0.8,
                       color=_STRATEGY_COLORS.get(r.name, "#888888"),
                       alpha=0.45)

    ax_dd.set_ylabel("Drawdown", fontsize=10)
    ax_dd.set_xlabel("Date", fontsize=10)
    ax_dd.legend(fontsize=9, framealpha=0.8)
    ax_dd.grid(True, alpha=0.25)
    ax_dd.set_xlim(ax_eq.get_xlim())

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved → {out_path}")


def _shade_regimes(ax: "plt.Axes", regime_series: pd.Series) -> None:  # type: ignore[name-defined]
    """Add axvspan shading for regime periods on a given axis."""
    if regime_series.empty:
        return
    current_regime = regime_series.iloc[0]
    span_start = regime_series.index[0]

    for ts, regime in regime_series.items():
        if regime != current_regime:
            ax.axvspan(span_start, ts,
                       color=_REGIME_COLORS.get(current_regime, "gray"),
                       alpha=0.08, linewidth=0)
            current_regime = regime
            span_start = ts

    # Close the final span
    ax.axvspan(span_start, regime_series.index[-1],
               color=_REGIME_COLORS.get(current_regime, "gray"),
               alpha=0.08, linewidth=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str = "config/default.yaml", walk_forward: bool = False) -> None:
    cfg = load_config(config_path)
    bt = cfg["backtest"]
    strat_cfg = cfg.get("strategies", {})

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading data through {bt['test_end']}...")
    full_data = _load_data(cfg, bt["test_end"])
    print(f"Loaded {len(full_data.index)} bars × {full_data.symbols}")

    # ── Daily BTC close for regime filter ────────────────────────────────────
    barometer = strat_cfg.get("regime_filter", {}).get("barometer_symbol", "BTC/USD")
    if barometer not in full_data.close.columns:
        barometer = full_data.symbols[0]
    daily_btc_close = full_data.close[barometer].resample("D").last().dropna()

    # ── Build strategies ─────────────────────────────────────────────────────
    print(f"\nEnabled strategies: {strat_cfg.get('enabled', [])}")
    bah, individual, composite = build_strategies_from_config(
        cfg, full_data, daily_btc_close,
        train_start=bt["train_start"],
        train_end=bt["train_end"],
    )

    # ── Fit (non-ML strategies are stateless; ML is pre-fitted in _build_one) ─
    print("Fitting strategies...")
    for s in [bah] + individual + [composite]:
        if not getattr(s, "_pre_fitted", False):
            s.fit(full_data)
    print("Fitting complete.\n")

    # ── Backtest each individual strategy + composite + benchmark ────────────
    execution = ProportionalCostModel(
        commission_pct=float(bt.get("commission_pct", 0.002))
    )

    all_strategies = [bah] + individual + [composite]
    results: list[BacktestResult] = []

    for strat in all_strategies:
        print(f"  Backtesting: {strat.name}...")
        result = _run_one(strat, full_data, cfg, execution)
        results.append(result)

    # ── Print metrics ────────────────────────────────────────────────────────
    _print_metrics(results)

    # ── Walk-forward validation ───────────────────────────────────────────────
    # Always run walk-forward when ML is enabled (mandatory for learned models).
    # Also run if explicitly requested via --walk-forward flag.
    ml_enabled = "ml" in strat_cfg.get("enabled", [])
    if walk_forward or ml_enabled:
        reason = "ML is enabled" if ml_enabled else "--walk-forward flag"
        print(f"\nRunning walk-forward validation on composite ({reason})...")
        from backtest.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(
            train_bars=4032,    # ~6 months hourly
            test_bars=1344,     # ~2 months per fold
            context_bars=720,   # warmup bars for slow indicators
        )
        wf_results = validator.validate(
            full_data, composite,
            initial_capital=float(bt.get("initial_capital", 10_000.0)),
            commission_pct=float(bt.get("commission_pct", 0.002)),
        )
        print("\n── Walk-Forward Folds (out-of-sample only) ─────────────────────")
        _print_metrics(wf_results)
        if ml_enabled and not walk_forward:
            print("  Note: walk-forward ran automatically because ML is enabled.")
            print("  Use --walk-forward to also run it without ML.")

    # ── Extract regime series for visualisation ──────────────────────────────
    regime_series = _extract_regime_series(composite, full_data)
    if regime_series is not None:
        test_regime = regime_series[regime_series.index >= pd.Timestamp(bt["test_start"])]
        counts = test_regime.value_counts(normalize=True)
        print("Regime breakdown (test period):")
        for r in ["bull", "sideways", "bear"]:
            print(f"  {r:>10}: {counts.get(r, 0):.1%}")
        print()

    # ── Plot ─────────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent.parent / "plots" / "full_backtest.png"
    out_path.parent.mkdir(exist_ok=True)
    _plot(results, regime_series, cfg, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full multi-strategy backtest")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Also run walk-forward validation on composite")
    args = parser.parse_args()
    main(args.config, walk_forward=args.walk_forward)
