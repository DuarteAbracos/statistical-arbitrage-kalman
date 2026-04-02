#!/usr/bin/env python3
"""
==============================================================================
PAPER TRADING ENGINE — Avellaneda-Stoikov Market Maker (WFC / AXP)
==============================================================================

This script runs the AS market maker strategy in real-time using Alpaca's
paper trading API. It mirrors the backtested logic from NB04 as closely as
possible, with full logging so you can compare live fills vs what the
backtest would have predicted.

USAGE:
    1. Set environment variables:
         export ALPACA_API_KEY="your_paper_key"
         export ALPACA_API_SECRET="your_paper_secret"
    
    2. Run during market hours (9:30-16:00 ET):
         python paper_trader.py
    
    3. Check logs in ./paper_logs/

The script uses Alpaca PAPER trading endpoint — no real money at risk.

ARCHITECTURE:
    - Main loop runs every 60 seconds during market hours
    - Pulls latest minute bar for WFC and AXP
    - Maintains rolling window of spread values for variance calculation
    - Computes AS quotes (reservation price + optimal spread)
    - Submits/cancels limit orders via Alpaca API
    - Logs everything to CSV for post-analysis
    - Flattens all positions 5 minutes before close
==============================================================================
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from collections import deque

# ============================================================
# ALPACA SDK IMPORTS
# ============================================================
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame

# ============================================================
# CONFIGURATION
# ============================================================
# Pair
TICKER_X = "WFC"
TICKER_Y = "AXP"

# Strategy parameters (from NB03 grid search)
GAMMA = 0.084026
KAPPA = 0.7284

# Risk limits
MAX_INVENTORY = 10
CONTRACT_MULTIPLIER = 10

# Costs
ALPACA_FEE_PER_SHARE = 0.005
MINIMUM_PURE_PROFIT = 0.005

# Rolling window for variance (must match backtest)
VARIANCE_WINDOW = 120

# Timing
LOOP_INTERVAL_SEC = 60          # Check every 60 seconds
EOD_FLATTEN_MINUTES_BEFORE = 5  # Flatten 5 min before close
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MIN = 0

# Kalman filter state (loaded from NB01 results)
# These will be updated each minute
INITIAL_BETA = None  # Loaded from latest data
INITIAL_ALPHA = None

# Logging
LOG_DIR = "./paper_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/paper_trader.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("PaperTrader")


# ============================================================
# KALMAN FILTER (online, single-step update)
# ============================================================
class OnlineKalmanFilter:
    """
    Minimal online Kalman filter for 2D state [alpha, beta].
    Performs one predict-update step per new observation.
    Uses Q and R calibrated from NB01 EM.
    """

    def __init__(self, initial_state, initial_cov, Q, R):
        self.x = np.array(initial_state, dtype=float)  # [alpha, beta]
        self.P = np.array(initial_cov, dtype=float)     # 2x2
        self.Q = np.array(Q, dtype=float)               # 2x2 (per-minute)
        self.R = np.array(R, dtype=float)                # 1x1
        self.F = np.eye(2)                               # Transition: random walk

    def update(self, y_obs, x_price):
        """
        One Kalman step.
        y_obs: observed Y price (AXP)
        x_price: observed X price (WFC)
        Returns: alpha, beta, spread_synthetic
        """
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Observation matrix H = [1, x_price]
        H = np.array([[1.0, x_price]])

        # Innovation
        y_pred = H @ x_pred
        innovation = y_obs - y_pred[0]

        # Innovation covariance
        S = H @ P_pred @ H.T + self.R

        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Update
        self.x = x_pred + K.flatten() * innovation
        self.P = (np.eye(2) - K @ H) @ P_pred

        alpha = self.x[0]
        beta = self.x[1]

        # Synthetic spread (no intercept — same as backtest)
        spread_synthetic = y_obs - beta * x_price

        return alpha, beta, spread_synthetic


# ============================================================
# AVELLANEDA-STOIKOV QUOTING ENGINE
# ============================================================
class ASQuotingEngine:
    """
    Computes reservation price and optimal spread.
    Mirrors the backtest logic exactly.
    """

    def __init__(self, gamma, kappa, max_inventory):
        self.gamma = gamma
        self.kappa = kappa
        self.max_inventory = max_inventory

    def get_quotes(self, mid_price, variance, inventory, min_profitable_delta):
        """
        Returns: bid, ask, reservation_price, as_delta
        """
        # Reservation price
        r = mid_price - (inventory * self.gamma * variance)

        # Optimal half-spread
        liquidity_premium = (1 / self.gamma) * np.log(1 + (self.gamma / self.kappa))
        risk_premium = 0.5 * self.gamma * variance
        as_delta = liquidity_premium + risk_premium

        # Fee floor
        delta = max(as_delta, min_profitable_delta)

        # Quotes
        bid = r - delta
        ask = r + delta

        # Inventory hard stops
        if inventory >= self.max_inventory:
            bid = None  # Don't buy more
        elif inventory <= -self.max_inventory:
            ask = None  # Don't sell more

        return bid, ask, r, as_delta


# ============================================================
# TRADE LOGGER
# ============================================================
class TradeLogger:
    """Logs every tick, quote, and fill to CSV for post-analysis."""

    def __init__(self, log_dir):
        today = datetime.now().strftime("%Y-%m-%d")
        self.tick_file = os.path.join(log_dir, f"ticks_{today}.csv")
        self.fill_file = os.path.join(log_dir, f"fills_{today}.csv")

        # Initialize tick log
        if not os.path.exists(self.tick_file):
            pd.DataFrame(columns=[
                "timestamp", "x_price", "y_price", "alpha", "beta",
                "spread_synthetic", "variance", "inventory",
                "reservation_price", "bid", "ask", "as_delta",
                "cash", "equity",
            ]).to_csv(self.tick_file, index=False)

        # Initialize fill log
        if not os.path.exists(self.fill_file):
            pd.DataFrame(columns=[
                "timestamp", "side", "spread_price_theoretical", "spread_price_actual",
                "slippage", "y_fill_price", "x_fill_price",
                "qty_y", "qty_x", "quantity",
                "commission", "cash", "inventory_after", "equity",
                "fill_type",
            ]).to_csv(self.fill_file, index=False)

    def log_tick(self, data):
        pd.DataFrame([data]).to_csv(self.tick_file, mode="a", header=False, index=False)

    def log_fill(self, data):
        pd.DataFrame([data]).to_csv(self.fill_file, mode="a", header=False, index=False)


# ============================================================
# MAIN PAPER TRADER
# ============================================================
class PaperTrader:
    def __init__(self):
        # API clients
        api_key = os.environ.get("ALPACA_API_KEY")
        api_secret = os.environ.get("ALPACA_API_SECRET")

        if not api_key or not api_secret:
            raise ValueError("Set ALPACA_API_KEY and ALPACA_API_SECRET env vars!")

        # paper=True for paper trading
        self.trading_client = TradingClient(api_key, api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)

        # Load Kalman parameters from NB01
        self._load_kalman_params()

        # AS engine
        self.as_engine = ASQuotingEngine(GAMMA, KAPPA, MAX_INVENTORY)

        # State
        self.inventory = 0  # Spread inventory (positive = long spread)
        self.cash = 0.0
        self.spread_history = deque(maxlen=VARIANCE_WINDOW + 10)
        self.current_bid_order_id = None
        self.current_ask_order_id = None

        # Logger
        self.logger = TradeLogger(LOG_DIR)

        log.info(f"PaperTrader initialized: {TICKER_X}/{TICKER_Y}")
        log.info(f"Params: gamma={GAMMA:.6f}, kappa={KAPPA:.4f}, max_inv={MAX_INVENTORY}")

    def _load_kalman_params(self):
        """Load Q, R, and last state from NB01 output."""
        metadata_path = os.path.join(os.path.dirname(__file__), "..", "data", "metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            Q_daily = np.array(meta["Q_daily"])
            R_daily = np.array(meta["R_daily"])
            Q_minute = Q_daily / 390
            self.kalman = OnlineKalmanFilter(
                initial_state=[meta.get("alpha_ols", 27.0), meta.get("beta_ols", 2.9)],
                initial_cov=np.eye(2) * 1.0,
                Q=Q_minute,
                R=R_daily,
            )
            log.info(f"Kalman loaded from metadata: alpha0={self.kalman.x[0]:.2f}, beta0={self.kalman.x[1]:.4f}")
        else:
            # Fallback defaults (from your NB01 run)
            log.warning(f"metadata.json not found at {metadata_path}, using defaults")
            Q_minute = np.array([[2.61e-06, 2.58e-07], [2.58e-07, 7.38e-06]])
            R_daily = np.array([[0.95]])
            self.kalman = OnlineKalmanFilter(
                initial_state=[27.27, 2.92],
                initial_cov=np.eye(2) * 1.0,
                Q=Q_minute,
                R=R_daily,
            )

    def _warmup(self):
        """Pull last VARIANCE_WINDOW minutes of data to seed the spread history."""
        log.info(f"Warming up: fetching last {VARIANCE_WINDOW + 30} minute bars...")

        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=8)  # Enough to cover a full trading day

        from alpaca.data.requests import StockBarsRequest
        request = StockBarsRequest(
            symbol_or_symbols=[TICKER_X, TICKER_Y],
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed="iex",
        )

        # 1. Obtaining brute response
        response = self.data_client.get_stock_bars(request)

        # 2. Verifiy if the DataFrame came back empty (Alpaca can be flaky sometimes)
        if response.df.empty:
            log.warning("A Alpaca não retornou dados de histórico (DataFrame vazio). Tentando novamente em 60 segundos...")
            time.sleep(60)
            return self._warmup()  # try to run the warmup again

        # 3. Reset the index safely, ensuring 'timestamp' is a column
        bars = response.df.reset_index()

        # 4. Extra prevension: Alpaca changed 'timestamp' to 'time' in some versions, so we check and rename if needed
        if "timestamp" not in bars.columns and "time" in bars.columns:
            bars = bars.rename(columns={"time": "timestamp"})

        # 5. Pivot safely
        pivot = bars.pivot(index="timestamp", columns="symbol")

        x_prices = pivot["close"][TICKER_X].dropna()
        y_prices = pivot["close"][TICKER_Y].dropna()

        # Align
        common_idx = x_prices.index.intersection(y_prices.index)
        x_prices = x_prices.loc[common_idx]
        y_prices = y_prices.loc[common_idx]

        # Run Kalman over warmup data to get state up to date
        for i in range(len(common_idx)):
            alpha, beta, spread = self.kalman.update(y_prices.iloc[i], x_prices.iloc[i])
            self.spread_history.append(spread)

        log.info(f"Warmup done: {len(self.spread_history)} bars, "
                 f"beta={self.kalman.x[1]:.4f}, last spread={self.spread_history[-1]:.4f}")

    def _get_latest_prices(self):
        """Fetch latest minute bar for both tickers."""
        try:
            request = StockLatestBarRequest(symbol_or_symbols=[TICKER_X, TICKER_Y], feed="iex")
            bars = self.data_client.get_stock_latest_bar(request)

            x_price = bars[TICKER_X].close
            y_price = bars[TICKER_Y].close

            return x_price, y_price
        except Exception as e:
            log.error(f"Failed to get prices: {e}")
            return None, None

    def _compute_variance(self):
        """Rolling variance of the spread (same as backtest)."""
        if len(self.spread_history) < VARIANCE_WINDOW:
            # Not enough data yet — use a safe default
            return max(np.var(list(self.spread_history)), 1e-8)

        recent = list(self.spread_history)[-VARIANCE_WINDOW:]
        var = np.var(recent, ddof=1)
        return max(var, 1e-8)

    def _compute_fee_floor(self, beta):
        """Minimum profitable delta given current beta and costs."""
        total_shares = CONTRACT_MULTIPLIER + (CONTRACT_MULTIPLIER * abs(beta))
        commission = total_shares * ALPACA_FEE_PER_SHARE
        commission_per_unit = commission / CONTRACT_MULTIPLIER
        return commission_per_unit + MINIMUM_PURE_PROFIT

    def _cancel_existing_orders(self):
        """Cancel any outstanding limit orders."""
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            open_orders = self.trading_client.get_orders(request)
            for order in open_orders:
                if order.symbol in [TICKER_X, TICKER_Y]:
                    self.trading_client.cancel_order_by_id(order.id)
                    log.info(f"Cancelled order {order.id} ({order.symbol} {order.side})")
        except Exception as e:
            log.error(f"Error cancelling orders: {e}")

    def _submit_spread_orders(self, bid_spread, ask_spread, beta, x_price):
        """
        Submit limit orders to execute the spread trade.
        
        Buying the spread = Buy Y, Sell beta*X
        Selling the spread = Sell Y, Buy beta*X
        
        We translate spread bid/ask into actual stock limit prices.
        """
        self._cancel_existing_orders()

        beta_abs = abs(beta)
        qty_y = CONTRACT_MULTIPLIER
        qty_x = int(round(CONTRACT_MULTIPLIER * beta_abs))

        if qty_x == 0:
            log.warning(f"qty_x is 0 (beta={beta:.4f}), skipping")
            return

        # The spread = Y - beta * X
        # Bid on spread: we want to BUY the spread
        #   -> Buy Y at current market, Sell X at current market
        #   -> Only submit if the spread price touches our bid
        # Ask on spread: we want to SELL the spread
        #   -> Sell Y, Buy X

        # For simplicity in paper trading, we use market orders when the
        # current spread crosses our bid/ask. This is more conservative
        # than limit orders (worse fills) but avoids queue position issues.

        # We don't submit orders proactively — instead, on each tick we
        # check if the current spread would fill our quote, and if so,
        # execute at market. This mirrors the backtest logic.
        
        # (Orders are executed in the main loop via _check_and_execute)
        pass

    def _check_and_execute(self, current_spread, beta, x_price, y_price):
        """
        Check if current spread crosses our bid/ask and execute if so.
        Uses market orders for simplicity in paper trading.
        """
        variance = self._compute_variance()
        fee_floor = self._compute_fee_floor(beta)

        bid, ask, r, as_delta = self.as_engine.get_quotes(
            current_spread, variance, self.inventory, fee_floor
        )

        # Log the tick
        equity = self.cash + self.inventory * current_spread * CONTRACT_MULTIPLIER
        self.logger.log_tick({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "x_price": x_price,
            "y_price": y_price,
            "alpha": self.kalman.x[0],
            "beta": beta,
            "spread_synthetic": current_spread,
            "variance": variance,
            "inventory": self.inventory,
            "reservation_price": r,
            "bid": bid,
            "ask": ask,
            "as_delta": as_delta,
            "cash": self.cash,
            "equity": equity,
        })

        beta_abs = abs(beta)
        qty_y = CONTRACT_MULTIPLIER
        qty_x = int(round(CONTRACT_MULTIPLIER * beta_abs))

        if qty_x == 0:
            return

        total_shares = qty_y + qty_x
        commission = total_shares * ALPACA_FEE_PER_SHARE

        # Check bid fill (buy the spread)
        if bid is not None and current_spread <= bid and self.inventory < MAX_INVENTORY:
            try:
                # Buy Y
                order_y = self.trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=TICKER_Y,
                        qty=qty_y,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                # Sell X
                order_x = self.trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=TICKER_X,
                        qty=qty_x,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                )

                # Wait briefly for fills, then fetch actual fill prices
                time.sleep(2)
                y_fill_price = y_price  # default
                x_fill_price = x_price  # default
                try:
                    filled_y = self.trading_client.get_order_by_id(order_y.id)
                    filled_x = self.trading_client.get_order_by_id(order_x.id)
                    if filled_y.filled_avg_price:
                        y_fill_price = float(filled_y.filled_avg_price)
                    if filled_x.filled_avg_price:
                        x_fill_price = float(filled_x.filled_avg_price)
                except Exception:
                    pass

                actual_spread_price = y_fill_price - beta_abs * x_fill_price
                slippage = actual_spread_price - bid

                self.inventory += 1
                self.cash -= (actual_spread_price * CONTRACT_MULTIPLIER) + commission
                equity = self.cash + self.inventory * current_spread * CONTRACT_MULTIPLIER

                log.info(f"BUY SPREAD @ theo={bid:.4f} actual={actual_spread_price:.4f} "
                         f"slip={slippage:.4f} | inv={self.inventory} | "
                         f"Y: buy {qty_y}@{y_fill_price:.2f} | X: sell {qty_x}@{x_fill_price:.2f}")

                self.logger.log_fill({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "side": "BUY",
                    "spread_price_theoretical": bid,
                    "spread_price_actual": actual_spread_price,
                    "slippage": slippage,
                    "y_fill_price": y_fill_price,
                    "x_fill_price": x_fill_price,
                    "qty_y": qty_y,
                    "qty_x": qty_x,
                    "quantity": 1,
                    "commission": commission,
                    "cash": self.cash,
                    "inventory_after": self.inventory,
                    "equity": equity,
                    "fill_type": "INTRADAY",
                })

            except Exception as e:
                log.error(f"BUY execution failed: {e}")

        # Check ask fill (sell the spread)
        elif ask is not None and current_spread >= ask and self.inventory > -MAX_INVENTORY:
            try:
                # Sell Y
                order_y = self.trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=TICKER_Y,
                        qty=qty_y,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                # Buy X
                order_x = self.trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=TICKER_X,
                        qty=qty_x,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                )

                # Wait briefly for fills, then fetch actual fill prices
                time.sleep(2)
                y_fill_price = y_price  # default
                x_fill_price = x_price  # default
                try:
                    filled_y = self.trading_client.get_order_by_id(order_y.id)
                    filled_x = self.trading_client.get_order_by_id(order_x.id)
                    if filled_y.filled_avg_price:
                        y_fill_price = float(filled_y.filled_avg_price)
                    if filled_x.filled_avg_price:
                        x_fill_price = float(filled_x.filled_avg_price)
                except Exception:
                    pass

                actual_spread_price = y_fill_price - beta_abs * x_fill_price
                slippage = ask - actual_spread_price

                self.inventory -= 1
                self.cash += (actual_spread_price * CONTRACT_MULTIPLIER) - commission
                equity = self.cash + self.inventory * current_spread * CONTRACT_MULTIPLIER

                log.info(f"SELL SPREAD @ theo={ask:.4f} actual={actual_spread_price:.4f} "
                         f"slip={slippage:.4f} | inv={self.inventory} | "
                         f"Y: sell {qty_y}@{y_fill_price:.2f} | X: buy {qty_x}@{x_fill_price:.2f}")

                self.logger.log_fill({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "side": "SELL",
                    "spread_price_theoretical": ask,
                    "spread_price_actual": actual_spread_price,
                    "slippage": slippage,
                    "y_fill_price": y_fill_price,
                    "x_fill_price": x_fill_price,
                    "qty_y": qty_y,
                    "qty_x": qty_x,
                    "quantity": 1,
                    "commission": commission,
                    "cash": self.cash,
                    "inventory_after": self.inventory,
                    "equity": equity,
                    "fill_type": "INTRADAY",
                })

            except Exception as e:
                log.error(f"SELL execution failed: {e}")

        else:
            log.debug(f"No fill | spread={current_spread:.4f} | "
                      f"bid={bid} | ask={ask} | inv={self.inventory}")

    def _flatten_positions(self):
        """EOD: close all positions at market."""
        if self.inventory == 0:
            log.info("EOD: No inventory to flatten.")
            return

        log.info(f"EOD FLATTEN: inventory={self.inventory}")

        beta = self.kalman.x[1]
        beta_abs = abs(beta)
        qty_y = CONTRACT_MULTIPLIER * abs(self.inventory)
        qty_x = int(round(CONTRACT_MULTIPLIER * beta_abs * abs(self.inventory)))

        try:
            if self.inventory > 0:
                # Long spread -> sell Y, buy X to flatten
                order_y = self.trading_client.submit_order(
                    MarketOrderRequest(symbol=TICKER_Y, qty=qty_y,
                                       side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
                order_x = self.trading_client.submit_order(
                    MarketOrderRequest(symbol=TICKER_X, qty=qty_x,
                                       side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
            else:
                # Short spread -> buy Y, sell X to flatten
                order_y = self.trading_client.submit_order(
                    MarketOrderRequest(symbol=TICKER_Y, qty=qty_y,
                                       side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                order_x = self.trading_client.submit_order(
                    MarketOrderRequest(symbol=TICKER_X, qty=qty_x,
                                       side=OrderSide.SELL, time_in_force=TimeInForce.DAY))

            # Fetch actual fill prices
            time.sleep(2)
            y_fill_price = 0.0
            x_fill_price = 0.0
            try:
                filled_y = self.trading_client.get_order_by_id(order_y.id)
                filled_x = self.trading_client.get_order_by_id(order_x.id)
                if filled_y.filled_avg_price:
                    y_fill_price = float(filled_y.filled_avg_price)
                if filled_x.filled_avg_price:
                    x_fill_price = float(filled_x.filled_avg_price)
            except Exception:
                pass

            actual_spread_price = y_fill_price - beta_abs * x_fill_price
            total_shares_flat = qty_y + qty_x
            flat_commission = total_shares_flat * ALPACA_FEE_PER_SHARE

            if self.inventory > 0:
                self.cash += (actual_spread_price * CONTRACT_MULTIPLIER * abs(self.inventory)) - flat_commission
            else:
                self.cash -= (actual_spread_price * CONTRACT_MULTIPLIER * abs(self.inventory)) + flat_commission

            log.info(f"EOD: Flattened {self.inventory} units @ spread={actual_spread_price:.4f}")

            self.logger.log_fill({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "side": "EOD_FLATTEN",
                "spread_price_theoretical": 0,
                "spread_price_actual": actual_spread_price,
                "slippage": 0,
                "y_fill_price": y_fill_price,
                "x_fill_price": x_fill_price,
                "qty_y": qty_y,
                "qty_x": qty_x,
                "quantity": abs(self.inventory),
                "commission": flat_commission,
                "cash": self.cash,
                "inventory_after": 0,
                "equity": self.cash,
                "fill_type": "EOD_FLATTEN",
            })

            self.inventory = 0

        except Exception as e:
            log.error(f"EOD flatten failed: {e}")

    def _is_market_open(self):
        """Check if we're within trading hours (ET)."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception:
            return False

    def _is_near_close(self):
        """Check if we're within EOD_FLATTEN_MINUTES_BEFORE of close."""
        try:
            clock = self.trading_client.get_clock()
            if clock.next_close:
                time_to_close = clock.next_close - clock.timestamp
                return time_to_close.total_seconds() < EOD_FLATTEN_MINUTES_BEFORE * 60
        except Exception:
            pass
        return False

    def run(self):
        """Main loop."""
        log.info("=" * 60)
        log.info("PAPER TRADER STARTING")
        log.info("=" * 60)

        # Wait for market to open
        while not self._is_market_open():
            log.info("Market closed. Waiting 60s...")
            time.sleep(60)

        # Warmup Kalman with recent data
        self._warmup()

        log.info("Entering main trading loop...")

        while True:
            try:
                # Check market status
                if not self._is_market_open():
                    log.info("Market closed. Stopping.")
                    break

                # Check if near close -> flatten
                if self._is_near_close():
                    self._flatten_positions()
                    log.info(f"Waiting for market close...")
                    time.sleep(300)  # Wait 5 min
                    continue

                # Get latest prices
                x_price, y_price = self._get_latest_prices()
                if x_price is None or y_price is None:
                    log.warning("Could not get prices, skipping tick")
                    time.sleep(LOOP_INTERVAL_SEC)
                    continue

                # Update Kalman
                alpha, beta, spread = self.kalman.update(y_price, x_price)
                self.spread_history.append(spread)

                log.info(f"TICK | X={x_price:.2f} Y={y_price:.2f} | "
                         f"beta={beta:.4f} spread={spread:.4f} | inv={self.inventory}")

                # Check for execution
                self._check_and_execute(spread, beta, x_price, y_price)

                # Sleep until next bar
                time.sleep(LOOP_INTERVAL_SEC)

            except KeyboardInterrupt:
                log.info("Interrupted by user. Flattening and exiting...")
                self._flatten_positions()
                break

            except Exception as e:
                log.error(f"Unexpected error: {e}", exc_info=True)
                time.sleep(LOOP_INTERVAL_SEC)

        # End of day summary
        log.info("=" * 60)
        log.info("SESSION SUMMARY")
        log.info(f"Final inventory: {self.inventory}")
        log.info(f"Cash PnL: ${self.cash:.2f}")
        log.info(f"Logs saved to {LOG_DIR}/")
        log.info("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    trader = PaperTrader()
    trader.run()
