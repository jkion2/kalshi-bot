# Failure Log
Auto-updated by ledger.daily_review(). Read by scanner and researcher
before processing new markets to avoid repeating mistakes.

## Known Failure Patterns

| Pattern | Description | Mitigation |
|---------|-------------|------------|
| bad_prediction | Model probability was far from actual outcome | Check Brier Score; reduce position size if score > 0.25 |
| bad_timing | Correct direction but market moved against us before resolution | Widen edge threshold for slow-moving markets |
| bad_exec | Slippage or fill issues | Increase slippage tolerance or reduce size |
| shock | Unpredictable external event (news, black swan) | Reduce max_concurrent_positions |

---
```

**.gitignore** — create the file at the root of `kalshi-bot/` (just named `.gitignore`, no extension) and paste this in:
```
.env
__pycache__/
*.pyc
*.pyo
.venv/
venv/
logs/
data/
STOP