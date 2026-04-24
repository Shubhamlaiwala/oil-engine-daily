# Kalshi Oil Prediction Market Research Engine

A research engine for analyzing Kalshi prediction markets using Black-Scholes pricing.

This system polls Kalshi's oil (`KXWTI`) event contracts in real time, prices them against a terminal-probability model derived from live WTI crude data, identifies apparent mispricings, and manages a simulated portfolio end-to-end — from candidate ranking through order lifecycle, position monitoring, and structured ML-ready logging.

The project is a personal research build focused on the applied mechanics of short-horizon binary options markets. It is not a profitable strategy, and the findings section below discusses what I learned about why.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Components](#key-components)
- [Pricing Model](#pricing-model)
- [Portfolio and Risk Management](#portfolio-and-risk-management)
- [Logging and Data Collection](#logging-and-data-collection)
- [Findings](#findings)
- [Setup](#setup)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Overview

Kalshi lists binary ("yes/no") contracts on oil price outcomes — for example, *"Will WTI settle above $94.99 at close?"* Each contract is effectively a cash-or-nothing digital option. The engine does the following on every polling cycle (default: 60 seconds):

1. Fetches live WTI and USO prices from Yahoo Finance (with a Twelve Data fallback path).
2. Computes realized volatility across short (10-bar) and medium (30-bar) windows with configurable blending and floor/ceiling bounds.
3. Fetches the current Kalshi `KXWTI` event and its associated contract strikes.
4. For each contract, computes a terminal settlement probability using Black-Scholes closed form, and optionally a touch probability via Monte Carlo.
5. Compares the model probability against the market's implied probability (ask prices) to compute edge on both YES and NO sides.
6. Filters candidates through liquidity, spread, strike-distance, and trading-phase gates.
7. Ranks the surviving candidates and passes them to a portfolio planner that decides ENTER / HOLD / ROTATE / EXIT across current paper positions.
8. Generates order intents, reconciles them against a paper ledger, monitors open positions for exit triggers, and writes structured JSONL logs for downstream analysis.

The engine supports simulation (paper trading), with a live-trading code path gated behind config flags and Kalshi's RSA-signed API authentication.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (entry)                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
               ┌───────────────▼────────────────┐
               │     oil_engine_core.py         │
               │  - Market data ingestion       │
               │  - Volatility & drift          │
               │  - Black-Scholes pricing       │
               │  - Edge computation & ranking  │
               └───────────────┬────────────────┘
                               │
               ┌───────────────▼────────────────┐
               │    oil_engine_runner.py        │
               │  - Cycle orchestration         │
               │  - Paper ledger                │
               │  - Order lifecycle             │
               │  - State persistence           │
               │  - Alerts (Telegram)           │
               └──┬─────────────┬──────────────┬┘
                  │             │              │
        ┌─────────▼──┐  ┌───────▼─────┐  ┌────▼──────────┐
        │ portfolio_ │  │  position_  │  │    ml/        │
        │ manager.py │  │  manager.py │  │ (schema &     │
        │            │  │             │  │  data writer) │
        │ ENTER/HOLD │  │ Exit rules  │  │               │
        │ ROTATE     │  │ Stop/profit │  │ JSONL logs    │
        │ decisions  │  │ thresholds  │  │               │
        └────────────┘  └─────────────┘  └───────────────┘
```

## Key Components

| File | Purpose |
|---|---|
| `main.py` | Thin entry point; loads config and runs the engine loop. |
| `oil_engine_core.py` | Market data, volatility estimation, Black-Scholes probability, candidate ranking. |
| `oil_engine_runner.py` | Cycle orchestration, paper trading ledger, state persistence, alerting. |
| `portfolio_manager.py` | Capital allocation, position limits, rotation rules, re-entry cooldowns. |
| `position_manager.py` | Open position monitoring; exit rule evaluation (stop-loss, profit-lock, edge-collapse). |
| `kalshi_account_client.py` | RSA-signed Kalshi API authentication, market/order/position endpoints. |
| `state_manager.py` | Atomic JSON persistence for runtime state and paper positions. |
| `ml/ml_schema.py` | Schema-versioned record builders for candidates, decisions, and trade outcomes. |
| `ml/ml_data_writer.py` | Append-only JSONL writers for ML-ready datasets. |
| `persistent_log_uploader.py` | Log bundling and remote archival (Supabase / presigned S3 / webhook). |
| `settings.daily.yaml` | All runtime configuration — pricing, filters, portfolio, exits. |

---

## Pricing Model

For a binary "Will settle above strike K?" contract with spot price S, annualized vol σ, drift μ, and time-to-expiry T (in years):

```
d₂ = [ln(S/K) + (μ − ½σ²)T] / (σ√T)
P(YES) = Φ(d₂)
```

Where Φ is the standard normal CDF. This is the closed-form solution of the Black-Scholes PDE for a cash-or-nothing digital option. An alternative touch-probability path is available for barrier-style reasoning via a Monte Carlo simulation (2,000 paths × 40 steps by default).

Volatility is estimated from logarithmic returns on 1-minute WTI bars, blended across short and medium windows:

```
σ_blended = w_short · σ_short + w_medium · σ_medium
σ_final = clip(σ_blended, σ_min, σ_max)
```

Drift is similarly estimated from recent mean log-returns and clipped to configurable bounds.

**Edge** for each side:

```
edge_yes = P(YES) − ask_yes
edge_no  = P(NO)  − ask_no    where P(NO) = 1 − P(YES)
```

---

## Portfolio and Risk Management

The portfolio planner enforces the following rules (all configurable):

- **Capital limits**: max total deployment fraction, reserve cash fraction, per-trade dollar and fraction caps.
- **Position limits**: maximum open trades, per-event position caps, per-event per-side caps.
- **Edge threshold**: minimum edge required to add a position; separate (lower) threshold to hold an existing position.
- **Trading-phase gates**: entries restricted to configured phases (e.g. `ACTIVE_TRADING`, `CLOSE_ONLY`).
- **Anti-clustering**: minimum strike gap between same-event positions; re-entry cooldown after exits.
- **Rotation**: a held position is rotated out only if a new candidate's edge exceeds it by a configurable improvement margin.

Exits are evaluated every cycle by `position_manager.py` with a layered set of rules: hard stop loss, early stop loss, profit lock, soft profit defense, edge collapse with confirmation cycles, stale-position exits, and emergency guards.

---

## Logging and Data Collection

The engine produces three append-only JSONL datasets suitable for downstream analysis or ML training:

- **`ml_candidate_dataset.jsonl`** — every ranked candidate at every cycle, with model features (edge, prob, distance, confidence, vol), market features (bid/ask/spread), and execution flags (was top-ranked, was executed).
- **`portfolio_decisions.jsonl`** — per-cycle portfolio plan: recommendation, capital state, enter/hold/exit counts, reasoning.
- **`ml_trade_outcomes.jsonl`** — realized P&L per closed paper trade: entry/exit prices, hold duration, exit reason, settlement.

All records are schema-versioned and config-hashed so runs with different parameters can be compared cleanly. The optional `persistent_log_uploader.py` can bundle and ship logs to Supabase Storage, a presigned S3 URL, or a generic webhook endpoint.

---

## Findings

This is the honest section. The engine is architecturally sound, but it is not a profitable strategy. A few weeks of paper trading and the resulting log analysis made clear why.

**1. Black-Scholes is miscalibrated for hourly-horizon binaries.**
For a contract with ~1 hour to settlement at realized vol around 0.02–0.05 annualized, σ·√T compresses to roughly 0.0002–0.0006. This makes a price move of even $0.50 from spot to strike appear as a many-sigma event in the model, producing "edges" of 0.30+ that are not real — the market is pricing in end-of-day noise, volatility jumps, and pin behavior that a Gaussian terminal assumption cannot capture. In the logs, the top candidate per cycle almost always showed `p ≈ 0.001` or `p ≈ 0.999`, which is a model artifact rather than a probability.

**2. Spread cost dominates at small contract notionals.**
Kalshi spreads on the relevant strikes were typically 2–4 cents on contracts priced 0.60–0.80, or roughly 3–6% per round trip. With exit thresholds set at ±5%, expected value per trade was negative even before any question of signal quality.

**3. Trading phase selection matters a lot.**
Entries in `CLOSE_ONLY` phase (hours before settlement) were where the model error was largest and the spread was most expensive. Restricting entries to longer-horizon phases would be a meaningful structural change.

**4. Realized intraday volatility from 1-minute bars systematically understates the volatility that actually matters at settlement.**
Jumps, news, and settlement mechanics are not in the return series the model is trained on.

**What this means for the project**: the engine is a research artifact, not a production strategy. The portion of the system with real ongoing value is the data pipeline — the schema-versioned candidate/outcome logs that can be used to study market behavior and test alternative pricing models. A credible next research step would be to replace the Gaussian terminal assumption with either an empirical distribution bootstrapped from historical EOD residuals, or an implied-vol surface fit back out from near-the-money contract prices.

---

## Setup

```bash
git clone https://github.com/Shubhamlaiwala/oil-engine-daily.git
cd oil-engine-daily
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Review `settings.daily.yaml` and edit as needed. Then:

```bash
python main.py
```

For live trading (not recommended without a tested edge), set Kalshi credentials as environment variables:

```
KALSHI_KEY_ID=...
KALSHI_PRIVATE_KEY_PATH=...
```

Optional remote archive backends (Supabase, presigned S3, webhook) are documented in `persistent_log_uploader.py`.

---

## Configuration

All behavior is controlled by `settings.daily.yaml`. Key sections:

- `runtime` — poll intervals, retries, execution mode (`simulation` / `live`).
- `kalshi` — API base, series ticker (`KXWTI`), contract resolution mode.
- `market_data` — Yahoo tickers and fallback sources.
- `model` — drift, vol windows, vol floor/ceiling, probability clipping, blend weights, Monte Carlo parameters.
- `filters` — bid/ask minimums, spread limits, strike distance rules (static or dynamic).
- `portfolio` — capital caps, position limits, edge thresholds, cooldowns.
- `exit` — stop loss, profit lock, edge-collapse thresholds, stagnation and time-based exits.

---

## Tech Stack

- **Python 3.11+**
- **pandas**, **numpy**, **scipy** — data handling and statistics
- **yfinance** — WTI and USO price ingestion
- **requests** — Kalshi REST API
- **cryptography** — RSA signing for Kalshi authentication
- **boto3** — Cloudflare R2 log upload (optional)
- **pyyaml** — configuration

No ML libraries are a runtime dependency; the system is designed to *produce* ML-ready datasets rather than train models inline.

---

## Author

**Shubham Laiwala**
Personal research project exploring the applied mechanics of prediction market pricing, portfolio management, and algorithmic execution in a live market context.

GitHub: [@Shubhamlaiwala](https://github.com/Shubhamlaiwala)

---

## Disclaimer

This code is shared for educational and research purposes. It is not investment advice, and it is not a profitable trading strategy. Trading prediction markets or any financial instrument involves real risk of loss. Do not use this code or any derivative of it with capital you are not prepared to lose.
