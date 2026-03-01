# Statistical Arbitrage & Optimal Execution Engine

> **⚠️ Status Note:** This repository is currently being refactored from a local development environment for public open-source release. The full source code, including the C++ execution module, is scheduled to be pushed by **March 2nd, 2026**.

## 1. Project Overview
This project implements an end-to-end quantitative trading pipeline focused on **Statistical Arbitrage (Pairs Trading)** in crypto markets. Unlike traditional cointegration approaches that assume a static relationship between assets, this engine utilizes a **Kalman Filter** to estimate time-varying hedge ratios ($\beta$) in real-time, allowing for adaptation to structural market breaks.

Furthermore, it integrates an **Optimal Execution** module based on the Avellaneda-Stoikov framework to manage inventory risk and minimize transaction costs.

## 2. Mathematical Framework

### A. Dynamic Hedge Ratio (Kalman Filter)
We model the spread between two cointegrated assets, $Y$ and $X$, using a state-space representation where the hedge ratio $\beta$ is treated as a hidden state following a random walk.

**Observation Equation:**
$$Y_t = \alpha_t + \beta_t X_t + \epsilon_t$$

**State Equation:**
$$\beta_t = \beta_{t-1} + \omega_t$$

The Kalman Filter recursively updates the estimate of $\beta_t$ (the hedge ratio) based on new price observations, minimizing the mean squared error of the spread prediction.

### B. Optimal Execution (HJB Equation)
To mitigate inventory risk, the agent solves the Hamilton-Jacobi-Bellman (HJB) equation to determine optimal bid/ask quotes ($r_b, r_a$) around the mid-price. This ensures the algo skews quotes to neutralize inventory as the end of the trading session approaches, minimizing adverse selection.

## 3. System Architecture

The system is designed with a modular architecture to separate signal generation from execution logic:

```text
[Data Feed (Websocket)] 
       |
       v
[Kalman Filter Engine] --> (Calculates Z-Score Signal)
       |
       v
[Risk Controller] --> (Checks Position Limits)
       |
       v
[HJB Execution Algo] --> (Optimizes Bid/Ask Spread)
       |
       v
[Exchange API] <--> [Portfolio Manager]
```
### Module Breakdown (Pending Upload)
* **`src/signals/kalman.py`**: Implementation of the iterative predict-update cycle using `pykalman` and `numpy`.
* **`src/execution/hjb_agent.cpp`**: Low-latency C++ module for solving the inventory equations and placing orders.
* **`src/backtest/event_driven.py`**: Custom backtester simulating L2 order book latency and FIFO queue position.

## 4. Roadmap & Progress

- [x] Mathematical Formulation & Literature Review
- [x] Data Pipeline (Binance/Alpaca APIs)
- [x] Signal Generation (Kalman Implementation)
- [ ] **Code Cleanup & Documentation (In Progress)**
- [ ] Unit Tests Coverage
- [ ] Public Repo Push (ETA: 8 Days)

## 5. References
* *Avellaneda, M., & Stoikov, S. (2008).* High-frequency trading in a limit order book.
* *Chan, E.* Algorithmic Trading: Winning Strategies and Their Rationale.

---
*For inquiries regarding the implementation details prior to the code push, please contact me directly.*
