# kalshi-bot

An AI-powered prediction market trading bot that runs on [Kalshi](https://kalshi.com). It scans open markets, researches them using Claude, estimates probabilities, and places trades when it finds a real edge. Currently running in paper trading mode while I validate the strategy.

Still a work in progress, built this as a personal project to learn more about AI agents, financial APIs, and algorithmic trading.

---

## How it works

The bot runs a 5-step pipeline every hour:

```
Scan → Research → Predict → Risk Check → Execute
```

1. **Scan** — pulls all open Kalshi markets and filters down to the most tradeable ones
2. **Research** — uses Claude Haiku to assess each market, with web search triggered only when real-time data is needed (prices, weather, scores)
3. **Predict** — Claude estimates the true probability of the outcome and compares it to the market price
4. **Risk Check** — validates the trade against Kelly Criterion sizing, daily loss limits, drawdown limits, and more
5. **Execute** — places a paper trade (or a real one if you switch it to live mode)

---

## Stack

- Python 3.11+
- [Anthropic API](https://console.anthropic.com) (Claude Haiku)
- [Kalshi REST API](https://kalshi.com)
- `httpx` for async HTTP
- `python-dotenv` for config

---

## Status

Currently in **paper trading mode** on Kalshi's demo environment. Tracking win rate, Brier Score, and drawdown before considering going live. No real money involved yet.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/jkion2/kalshi-bot.git
cd kalshi-bot
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your .env file

```bash
cp .env.example .env
```

Then fill in your credentials:

```
KALSHI_API_KEY_ID=your-key-id
KALSHI_API_KEY=your-private-key
KALSHI_DEMO=true
ANTHROPIC_API_KEY=sk-ant-xxxx
STARTING_BANKROLL=100
```

Get your Kalshi API keys at [kalshi.com](https://kalshi.com) under Account → API Keys. Start with `KALSHI_DEMO=true` to use their demo environment with fake money.

Get your Anthropic API key at [console.anthropic.com](https://console.anthropic.com).

### 5. Run it

```bash
# Paper trading (no real money)
python main.py --mode paper

# Live trading (real money — only after validating in paper mode)
python main.py --mode live
```

---

## Kill switch

If you ever need to stop the bot immediately, just create a file called `STOP` in the project root:

```bash
touch STOP   # halt
rm STOP      # resume
```

---

## Project structure

```
kalshi-bot/
├── main.py                  # entry point
├── config.py                # all settings in one place
├── bot/
│   ├── scanner.py           # step 1: find markets
│   ├── researcher.py        # step 2: research with Claude
│   ├── predictor.py         # step 3: estimate probability
│   ├── risk.py              # step 4a: Kelly sizing + safety checks
│   ├── executor.py          # step 4b: place or simulate orders
│   └── ledger.py            # step 5: track trades and metrics
└── skills/predict-market-bot/
    └── SKILL.md             # bot architecture reference
```

---

## Disclaimer

This is a personal project for learning purposes. Not financial advice. Trading prediction markets involves real financial risk. Always start with paper trading and never trade money you can't afford to lose.
