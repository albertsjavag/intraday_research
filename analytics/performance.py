from __future__ import annotations

import numpy as np
import pandas as pd


# Annualisation factor for hourly bars (24 × 365)
HOURS_PER_YEAR = 24 * 365


def compute_metrics(result: "BacktestResult") -> dict:  # type: ignore[name-defined]
    """Compute standard performance metrics from a BacktestResult.

    Returns a dict with:
        total_return, annualized_return, sharpe_ratio, max_drawdown,
        calmar_ratio, num_trades, volatility
    """
    equity: pd.Series = result.equity_curve

    if equity.empty or len(equity) < 2:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "num_trades": 0,
            "volatility": 0.0,
        }

    returns = equity.pct_change().dropna()

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0

    n_bars = len(returns)
    ann_factor = HOURS_PER_YEAR / n_bars
    annualized_return = (1 + total_return) ** ann_factor - 1

    volatility = float(returns.std() * np.sqrt(HOURS_PER_YEAR))

    if volatility > 0:
        sharpe_ratio = float((returns.mean() * HOURS_PER_YEAR) / (returns.std() * np.sqrt(HOURS_PER_YEAR)))
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    num_trades = getattr(result, "num_trades", 0)

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar_ratio),
        "num_trades": int(num_trades),
        "volatility": float(volatility),
    }


def aggregate_sweep(results: list) -> pd.DataFrame:
    """Aggregate a list of BacktestResult objects into a summary DataFrame.

    Each result must have a .name attribute (used as row label) and
    an .equity_curve Series.
    """
    rows = []
    for r in results:
        metrics = compute_metrics(r)
        metrics["name"] = getattr(r, "name", "unknown")
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("name")

    # Format percentages for display
    pct_cols = ["total_return", "annualized_return", "max_drawdown", "volatility"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.2%}")

    return df
