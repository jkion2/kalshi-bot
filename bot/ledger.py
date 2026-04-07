"""
bot/ledger.py — Step 5: Compound
Persists every trade to disk. Tracks bankroll, daily P&L, drawdown,
win rate, Sharpe ratio, Brier Score, and API costs.
Generates post-mortem reports after losses.
"""
 
import json
import logging
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional
 
from bot import TradeSignal, TradeResult
from config import BotConfig
 
log = logging.getLogger(__name__)
 
 
class TradeLedger:
    def __init__(self, cfg: BotConfig):
        self.cfg           = cfg
        self.bankroll      = cfg.starting_bankroll
        self.peak_bankroll = cfg.starting_bankroll
        self._trades: list[dict] = []
        self._load()
 
    # ── Record ─────────────────────────────────────────────────────────────────
    def record(
        self,
        signal: TradeSignal,
        position_size: float,
        result: TradeResult,
    ) -> None:
        """Record a trade immediately after execution."""
        entry = {
            "trade_id":          result.order_id,
            "market_id":         signal.market_id,
            "title":             signal.title,
            "side":              signal.side,
            "model_probability": signal.model_probability,
            "market_price":      signal.market_price,
            "edge":              signal.edge,
            "confidence":        signal.confidence,
            "expected_value":    signal.expected_value,
            "reasoning":         signal.reasoning,
            "position_usd":      position_size,
            "fill_price":        result.fill_price,
            "contracts":         result.contracts,
            "simulated":         result.simulated,
            "outcome":           None,      # filled in when market resolves
            "pnl":               None,      # filled in when market resolves
            "status":            "open",    # open | won | lost | error
            "opened_at":         datetime.utcnow().isoformat(),
            "closed_at":         None,
            "failure_class":     None,      # bad_prediction | bad_timing | bad_exec | shock
        }
        self._trades.append(entry)
        log.debug(f"Ledger recorded: {entry['trade_id']}")
 
    # ── Metrics ────────────────────────────────────────────────────────────────
    def today_pnl(self) -> float:
        """Sum of P&L for all trades opened today."""
        today = date.today().isoformat()
        return sum(
            t["pnl"] or 0
            for t in self._trades
            if t["opened_at"][:10] == today and t["pnl"] is not None
        )
 
    def current_drawdown(self) -> float:
        """Current drawdown as a fraction from peak bankroll."""
        if self.peak_bankroll == 0:
            return 0.0
        return max(0.0, (self.peak_bankroll - self.bankroll) / self.peak_bankroll)
 
    def open_position_count(self) -> int:
        return sum(1 for t in self._trades if t["status"] == "open")
 
    def today_api_cost(self) -> float:
        """
        Rough estimate: each research + prediction cycle costs ~$0.02
        (2 Claude Haiku calls × ~500 tokens each).
        Track externally if you need precise costs.
        """
        today = date.today().isoformat()
        today_cycles = sum(
            1 for t in self._trades
            if t["opened_at"][:10] == today
        )
        return today_cycles * 0.02
 
    def win_rate(self) -> Optional[float]:
        closed = [t for t in self._trades if t["status"] in ("won", "lost")]
        if not closed:
            return None
        wins = sum(1 for t in closed if t["status"] == "won")
        return wins / len(closed)
 
    def brier_score(self) -> Optional[float]:
        """
        Brier Score measures calibration quality.
        BS = (1/n) * Σ (predicted_prob - actual_outcome)²
        Lower is better. Perfect calibration = 0. Random = 0.25.
        """
        resolved = [t for t in self._trades if t["outcome"] is not None]
        if not resolved:
            return None
        total = sum(
            (t["model_probability"] - float(t["outcome"])) ** 2
            for t in resolved
        )
        return total / len(resolved)
 
    def summary(self) -> dict:
        closed = [t for t in self._trades if t["status"] in ("won", "lost")]
        pnls   = [t["pnl"] for t in closed if t["pnl"] is not None]
        return {
            "bankroll":        self.bankroll,
            "total_trades":    len(self._trades),
            "open_positions":  self.open_position_count(),
            "closed_trades":   len(closed),
            "win_rate":        self.win_rate(),
            "brier_score":     self.brier_score(),
            "today_pnl":       self.today_pnl(),
            "drawdown":        self.current_drawdown(),
            "total_pnl":       sum(pnls),
            "today_api_cost":  self.today_api_cost(),
        }
 
    # ── Daily review (post-mortem) ─────────────────────────────────────────────
    def daily_review(self) -> None:
        """
        Run at end of day. Writes a plain-English summary to the failure log
        so the scanner and researcher can check what went wrong.
        """
        today = date.today().isoformat()
        summary = self.summary()
 
        entry = f"""
## Daily Review — {today}
- Bankroll: ${summary['bankroll']:.2f}
- Today P&L: ${summary['today_pnl']:.2f}
- Win Rate (all time): {f"{summary['win_rate']:.0%}" if summary['win_rate'] else 'N/A'}
- Brier Score: {f"{summary['brier_score']:.3f}" if summary['brier_score'] else 'N/A'}
- Drawdown: {summary['drawdown']:.1%}
- Open positions: {summary['open_positions']}
 
### Losses Today
"""
        today_losses = [
            t for t in self._trades
            if t["opened_at"][:10] == today and t["status"] == "lost"
        ]
        for loss in today_losses:
            entry += (
                f"- {loss['market_id']}: ${loss['pnl']:.2f} | "
                f"Predicted {loss['model_probability']:.0%}, "
                f"actual {loss['outcome']} | "
                f"Class: {loss.get('failure_class', 'unknown')}\n"
            )
 
        if not today_losses:
            entry += "- No losses today\n"
 
        failure_log = self.cfg.failure_log_path
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_log, "a") as f:
            f.write(entry)
 
        log.info(f"Daily review written to {failure_log}")
 
    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self) -> None:
        self.cfg.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.trade_log_path, "w") as f:
            json.dump(
                {
                    "bankroll":      self.bankroll,
                    "peak_bankroll": self.peak_bankroll,
                    "trades":        self._trades,
                },
                f,
                indent=2,
            )
        log.debug(f"Ledger saved to {self.cfg.trade_log_path}")
 
    def _load(self) -> None:
        if self.cfg.trade_log_path.exists():
            with open(self.cfg.trade_log_path) as f:
                data = json.load(f)
            self.bankroll      = data.get("bankroll",      self.cfg.starting_bankroll)
            self.peak_bankroll = data.get("peak_bankroll", self.cfg.starting_bankroll)
            self._trades       = data.get("trades",        [])
            log.info(
                f"Ledger loaded: {len(self._trades)} trades, "
                f"bankroll=${self.bankroll:.2f}"
            )
        else:
            log.info("No existing ledger found. Starting fresh.")
 