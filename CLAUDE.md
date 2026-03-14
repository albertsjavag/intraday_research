# intraday_research — AI Assistant Instructions

## Project purpose
Vectorised intraday (hourly bar) crypto backtesting and live trading system.
Architecturally mirrors a daily trading system but adapted for hourly bars.

## Key design principles
- **Dependency injection**: The engine imports only ABCs. Adding a strategy/data
  source/execution model requires zero engine changes.
- **Data flow**: DataHandler → MarketData → Strategy → weights → Portfolio →
  ExecutionModel → BacktestResult → analytics

## Critical bugs to always avoid

### 1. Data gap ffill spike
Never ffill more than 3 bars. Long gaps must stay NaN. The engine detects NaN
prices and force-closes positions at the last known price. Without this, the
portfolio holds a stale price and shows a fake equity spike when real data resumes.

### 2. Duplicate portfolio updates
`portfolio.update()` must be called exactly ONCE per bar. After a force-close,
use `continue` to skip the normal update. Two updates with the same timestamp
break chart rendering and metrics.

### 3. ML feature warmup
Use `signal_start` in `BacktestConfig` to load data from TRAIN_START so
indicator warmup (up to 336 bars for slow MA) is complete before trading begins.

### 4. CompositeStrategy weight explosion
Always use `max_weight=0.60` cap with iterative redistribution. ML's rolling
Sharpe can spike during a trend and claim 80-90% without the cap.

### 5. pandas compatibility
- Use `searchsorted()` not `get_loc()` for integer positions.
- Use `tz_convert(None)` not `tz_localize(None)` for Alpaca timestamps.
- Use `iloc[index.get_indexer(target, method='nearest')]` not
  `reindex(method='nearest')`.

## Timeframe: hourly bars
- MA lookbacks: fast=24h, mid=72h, slow=168h, trend=336h
- Mean reversion window: 48 bars
- ML forward target: 8 bars

## Data source
Alpaca CryptoHistoricalDataClient with TimeFrame.Hour.
Credentials in `.env`: ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_LIVE.

## Running
```bash
pip install -e ".[data,ml,alpaca,dev]"
cp .env.example .env  # fill in keys
python backtests/run_backtest.py   # validate
python run_live.py                 # live once
python run_live.py --loop          # every hour
python dashboard.py                # terminal dashboard
```

## File structure
- `data/`       — MarketData, DataHandler ABC, AlpacaDataHandler, MockHandler
- `strategies/` — Strategy ABC + concrete strategies + ML
- `backtest/`   — BacktestEngine, Portfolio, WalkForwardValidator
- `execution/`  — ExecutionModel ABC, cost models, AlpacaTrader
- `analytics/`  — performance metrics
- `utils/`      — secrets loader
- `backtests/`  — runnable backtest scripts
- `state/`      — runtime JSON written by run_live.py, read by dashboard
- `config/`     — default.yaml
