"""Rich live terminal dashboard.

Displays account, open positions, recent orders, and strategy signals.
Reads live state from state/last_signals.json (written by run_live.py).

Usage:
    python dashboard.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path("state/last_signals.json")


def _load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def _build_signals_table(state: dict | None, Table, Style, Text) -> "Table":
    table = Table(title="Strategy Signals", show_footer=False, expand=True)
    table.add_column("Strategy", style="bold")
    table.add_column("Alloc %", style="cyan")

    if state is None:
        table.add_column("Status")
        table.add_row(
            "—",
            "—",
            "[dim]No signal data yet — run python run_live.py first[/dim]",
        )
        return table

    symbols = state.get("symbols", [])
    for sym in symbols:
        table.add_column(sym, justify="center")

    as_of = state.get("as_of", "?")
    computed_at = state.get("computed_at", "?")
    table.caption = f"as of {as_of} · computed {computed_at}"

    strategy_signals = state.get("strategy_signals", {})
    strategy_weights = state.get("strategy_weights", {})

    for strat_name, sym_weights in strategy_signals.items():
        alloc = strategy_weights.get(strat_name, 0.0)
        alloc_str = f"{alloc * 100:.1f}%"
        alloc_style = "cyan bold" if alloc > 0.15 else "dim"

        cells = []
        for sym in symbols:
            w = sym_weights.get(sym, 0.0)
            if w > 0.05:
                cells.append("[green]LONG[/green]")
            elif w < -0.05:
                cells.append("[red]SHORT[/red]")
            else:
                cells.append("[dim]FLAT[/dim]")

        table.add_row(strat_name, f"[{alloc_style}]{alloc_str}[/{alloc_style}]", *cells)

    # Composite row
    composite = state.get("composite", {})
    comp_cells = []
    for sym in symbols:
        w = composite.get(sym, 0.0)
        if w > 0.05:
            comp_cells.append(f"[green bold]{w:.2f}[/green bold]")
        elif w < -0.05:
            comp_cells.append(f"[red bold]{w:.2f}[/red bold]")
        else:
            comp_cells.append("[dim]0.00[/dim]")

    table.add_row("[bold]Composite[/bold]", "100%", *comp_cells)
    return table


def _build_positions_table(positions: list[dict], Table) -> "Table":
    table = Table(title="Open Positions", expand=True)
    table.add_column("Symbol")
    table.add_column("Side")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Mkt Value", justify="right")
    table.add_column("Unreal. P&L", justify="right")

    if not positions:
        table.add_row("[dim]No open positions[/dim]", "", "", "", "", "", "")
        return table

    for p in positions:
        pnl = p.get("unrealized_pl") or 0.0
        pnl_pct = p.get("unrealized_plpc") or 0.0
        pnl_str = f"${pnl:+,.2f} ({pnl_pct*100:+.1f}%)" if pnl else "—"
        pnl_style = "green" if pnl > 0 else "red" if pnl < 0 else "dim"

        table.add_row(
            p.get("symbol", "?"),
            p.get("side", "?"),
            f"{p.get('qty', 0):.4f}",
            f"${p.get('avg_entry_price', 0):,.2f}",
            f"${p.get('current_price', 0):,.2f}" if p.get("current_price") else "—",
            f"${p.get('market_value', 0):,.2f}" if p.get("market_value") else "—",
            f"[{pnl_style}]{pnl_str}[/{pnl_style}]",
        )

    return table


def _build_orders_table(orders: list[dict], Table) -> "Table":
    table = Table(title="Recent Orders", expand=True)
    table.add_column("Time")
    table.add_column("Symbol")
    table.add_column("Side")
    table.add_column("Amount", justify="right")
    table.add_column("Fill px", justify="right")
    table.add_column("Status")

    status_styles = {
        "filled": "green",
        "new": "yellow",
        "pending_new": "yellow",
        "partially_filled": "yellow",
        "canceled": "dim",
        "expired": "dim",
    }

    for o in orders:
        status = o.get("status", "unknown")
        style = status_styles.get(status, "dim")
        notional = o.get("notional")
        amt_str = f"${notional:,.2f}" if notional else (f"{o.get('qty', '?')} units")
        fill_px = o.get("filled_avg_price")
        fill_str = f"${fill_px:,.2f}" if fill_px else "—"
        created = o.get("created_at", "?")[:19]  # trim microseconds

        table.add_row(
            created,
            o.get("symbol", "?"),
            o.get("side", "?"),
            amt_str,
            fill_str,
            f"[{style}]{status}[/{style}]",
        )

    return table


def main() -> None:
    try:
        from rich.console import Console
        from rich.layout import Layout
        from rich.live import Live
        from rich.panel import Panel
        from rich.style import Style
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        raise ImportError("rich is required: pip install -e '.[alpaca]'")

    from utils.secrets import load_dotenv
    load_dotenv()

    try:
        from execution.alpaca_trader import AlpacaTrader
        trader = AlpacaTrader()
        live_mode = os.environ.get("ALPACA_LIVE", "false").lower() == "true"
    except Exception:
        trader = None
        live_mode = False

    console = Console()

    def make_layout() -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="account", size=4),
            Layout(name="signals", size=12),
            Layout(name="main"),
            Layout(name="footer", size=1),
        )
        layout["main"].split_row(
            Layout(name="positions"),
            Layout(name="orders"),
        )
        return layout

    def update(layout: Layout) -> None:
        state = _load_state()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        mode_str = "[red bold]LIVE[/red bold]" if live_mode else "[yellow]PAPER[/yellow]"
        symbols_str = ", ".join(state["symbols"]) if state else "—"

        # Header
        layout["header"].update(
            Panel(
                f"[bold cyan]intraday_research dashboard[/bold cyan]  "
                f"│  Mode: {mode_str}  │  Symbols: {symbols_str}",
                style="bold",
            )
        )

        # Account
        if trader:
            try:
                acct = trader.get_account()
                acct_text = (
                    f"[cyan]Equity:[/cyan] ${acct['equity']:>12,.2f}  "
                    f"[cyan]Cash:[/cyan] ${acct['cash']:>12,.2f}  "
                    f"[cyan]Buying Power:[/cyan] ${acct['buying_power']:>12,.2f}"
                )
            except Exception as exc:
                acct_text = f"[red]Error fetching account: {exc}[/red]"
        else:
            acct_text = "[dim]Alpaca not connected[/dim]"

        layout["account"].update(Panel(acct_text, title="Account"))

        # Signals
        layout["signals"].update(
            Panel(_build_signals_table(state, Table, Style, Text), title="Strategy Signals")
        )

        # Positions & Orders
        if trader:
            try:
                positions = trader.get_all_positions()
            except Exception:
                positions = []
            try:
                orders = trader.get_orders(limit=10)
            except Exception:
                orders = []
        else:
            positions = []
            orders = []

        layout["positions"].update(Panel(_build_positions_table(positions, Table)))
        layout["orders"].update(Panel(_build_orders_table(orders, Table)))

        # Footer
        layout["footer"].update(f"[dim]Last updated: {now}[/dim]")

    layout = make_layout()

    with Live(layout, console=console, screen=True, refresh_per_second=1) as live:
        while True:
            try:
                update(layout)
                live.refresh()
                import time
                time.sleep(1)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
