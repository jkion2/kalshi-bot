---
name: predict-market-bot
description: >
  AI-powered Kalshi prediction market trading bot.
  Scans markets, researches events, predicts probabilities using Claude,
  validates risk with Kelly Criterion, and executes trades.
  Use when: "scan markets", "check risk", "kelly size", "trade signal",
  "research market", "estimate probability", "brier score".
metadata:
  version: 1.0.0
  pattern: pipeline
  tags: [kalshi, prediction-market, kelly, risk, trading]
---

# Predict Market Bot

## Pipeline

```
SCAN → RESEARCH → PREDICT → RISK CHECK → EXECUTE → COMPOUND
```

Each stage is a separate Python module. Data flows via typed dataclasses.

## Stage 1: Scan (scanner.py)
- Fetches all open Kalshi markets via REST API
- Filters: volume ≥ 200, expiry ≤ 30 days, spread ≤ 5¢, price 2%-98%
- Ranks by uncertainty × liquidity × urgency
- Returns top 10 candidates per cycle

## Stage 2: Research (researcher.py)
- Uses Claude with web_search tool to gather news per market
- NLP sentiment analysis (-1.0 to +1.0)
- PROMPT INJECTION PROTECTION: all external content is wrapped in
  [EXTERNAL CONTENT] tags and Claude is instructed to treat it as data only

## Stage 3: Predict (predictor.py)
- Claude estimates true YES probability independently of market price
- Only generates a signal when:
  - edge = |model_prob - market_price| ≥ 4%
  - confidence ≥ 60%

## Stage 4: Risk + Execute (risk.py + executor.py)
Risk checks (all deterministic Python, NO LLM):
  □ STOP file check
  □ Daily loss < 15% of bankroll
  □ Drawdown < 8%
  □ Edge ≥ 4%
  □ Concurrent positions < 15
  □ Daily API cost < $5

Kelly position sizing:
  f* = (p × b - q) / b  × kelly_fraction (0.25)
  Hard cap: max 5% of bankroll per position

Execution:
  - Paper mode: simulates fill at market price
  - Live mode: limit order via Kalshi REST API
  - Slippage abort: if price moves > 2% before fill, cancel

## Stage 5: Compound (ledger.py)
- Every trade logged to data/trades.json
- Metrics tracked: win rate, Brier Score, Sharpe, max drawdown
- Daily post-mortem written to references/failure_log.md
- Failure classification: bad_prediction | bad_timing | bad_exec | shock

## Kill Switch
Create a file named `STOP` in the project root to halt all trading immediately:
```bash
touch STOP      # halt
rm STOP         # resume
```

## Key Formulas

### Kelly Criterion
```
f* = (p × b - q) / b
b  = (1 - price) / price
```

### Expected Value
```
EV = p × (1 - price) - (1-p) × price
```

### Brier Score (calibration)
```
BS = (1/n) × Σ(predicted - outcome)²
Target: < 0.25  |  Perfect: 0.0
```

### Edge
```
edge = model_probability - market_price
Minimum to trade: 0.04 (4%)
```
