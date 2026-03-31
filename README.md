# Intraday Market Making with Avellaneda-Stoikov & Kalman Filter

A complete quantitative trading pipeline implementing an **Avellaneda-Stoikov market maker** on cointegrated equity pairs, with dynamic hedge ratio estimation via **Kalman Filter**.

**Pair:** WFC / AXP &nbsp;|&nbsp; **Data:** 619K minute bars (2020–2026) &nbsp;|&nbsp; **Broker:** Alpaca

## Results (Out-of-Sample: 2024–2026)

| Metric | Value |
|---|---|
| Sharpe Ratio | 1.21 |
| Net PnL | +$4,870 (+4.87%) |
| Max Drawdown | -2.49% |
| Total Fills | 266 |
| Model Active | 100% of bars |
| Capital Deployed (peak) | $35,923 |
| Return on Capital Deployed | 13.56% (27 months) |

The strategy **performs better in high-volatility regimes** and **preserves capital** in low-volatility periods.

All results are **net of transaction costs** ($0.005/share Alpaca fees) with phantom fill prevention applied.

## Pipeline

```
NB01: Pair Selection & Kalman Filter
  └─ Cointegration testing across S&P 100 sectors (train data only)
  └─ EM calibration on daily data, forward filter on minute data
  └─ Synthetic spread: S(t) = Y(t) - β(t)·X(t)

NB02: Baseline Taker Strategy
  └─ Z-score mean reversion (market taker)
  └─ Proves that crossing the spread destroys profitability

NB03: AS Hyperparameter Optimization
  └─ Empirical κ calibration via log-linear regression
  └─ γ calibration + grid search (42 combinations)
  └─ In-sample Sharpe: 1.45

NB04: Out-of-Sample Market Maker
  └─ Broker-aware execution with fee floor
  └─ Phantom fill gap detection (>2min gaps)
  └─ EOD inventory liquidation
  └─ OOS Sharpe: 1.21

Paper Trader: Live validation script
  └─ Online Kalman filter (single-step update)
  └─ Real-time AS quoting via Alpaca Paper API
  └─ Full tick and fill logging
```

## Key Technical Decisions

**Synthetic spread, not Kalman residual.** The Kalman estimates α(t) and β(t). The residual `Y - α - β·X` collapses to ~$0.25 std (useless for trading). The synthetic spread `Y - β·X` preserves ~$8.67 std — the actual asset the market maker trades.

**EM on daily, filter on minute.** pykalman EM on 600K bars is impractical. EM runs on ~1000 daily bars (seconds), Q scales to per-minute (Q_daily/390), filter() runs forward-only on all minute data.

**Variance on level, not diff.** `rolling(120).var()` on the spread level (~0.06), not `diff().rolling().var()` (~1e-7). The AS model needs σ² of how much the mid-price moves, not how smooth the changes are.

**elif execution.** Only one side (bid or ask) can fill per bar. Two independent `if` statements allowed phantom round-trips.

**Phantom fill prevention.** Bars >2 minutes apart are flagged as gaps. No execution on gap transitions. Removed 2,130 phantom fills from OOS results.

**Paper trader uses reduced position sizing.** The backtest uses CONTRACT_MULTIPLIER=100 and MAX_INVENTORY=50. The paper trader uses CONTRACT_MULTIPLIER=10 and MAX_INVENTORY=10 to keep capital requirements manageable. PnL scales linearly — the Sharpe and fill pattern should be comparable.

## Project Structure

```
├── notebooks/
│   ├── 01_kalman_pairs.ipynb        # Pair selection + Kalman filter
│   ├── 02_strategy_baseline.ipynb   # Z-score taker baseline
│   ├── 03_hyperparameter_optimization.ipynb  # AS calibration
│   └── 04_market_maker_engine.ipynb # OOS backtest
├── live/
│   └── paper_trader.py              # Real-time paper trading
├── data/
│   ├── kalman_results.csv           # NB01 output
│   ├── strategy_results.csv         # NB02 output
│   └── metadata.json                # Kalman parameters
└── results/
    ├── optimal_params.json          # γ, κ from grid search
    └── oos_results.json             # Final OOS metrics
```

## Setup

```bash
# Create virtual environment
python -m venv quant_env
source quant_env/bin/activate  # Linux/Mac
# or: quant_env\Scripts\activate  # Windows

# Install dependencies
pip install pandas numpy scipy scikit-learn matplotlib pykalman alpaca-py yfinance statsmodels

## Set API keys:

# Linux/Mac
export ALPACA_API_KEY="your_key"
export ALPACA_API_SECRET="your_secret"

# Windows (CMD)
set ALPACA_API_KEY=your_key
set ALPACA_API_SECRET=your_secret

# Windows (PowerShell)
$env:ALPACA_API_KEY="your_key"
$env:ALPACA_API_SECRET="your_secret"
```

Run notebooks in order: NB01 → NB02 → NB03 → NB04.

## Paper Trading

```bash
cd live
python paper_trader.py
```

Requires Alpaca paper trading account. Uses IEX feed (free tier). Logs saved to `./paper_logs/`.

## Mathematical Framework

**Kalman Filter** estimates dynamic hedge ratio β(t) via 2D state-space model with EM-calibrated Q and R matrices.

**Avellaneda-Stoikov** computes optimal quotes:
- Reservation price: `r = s - q·γ·σ²`
- Optimal half-spread: `δ* = (1/γ)·ln(1 + γ/κ) + (1/2)·γ·σ²`

**Calibration:** κ via empirical fill probability regression; γ via heuristic + grid search.

## References

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.*
- Guéant, O., Lehalle, C.A. & Fernandez-Tapia, J. (2012). *Dealing with the inventory risk.*
- Chan, E. *Algorithmic Trading: Winning Strategies and Their Rationale.*
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.*
