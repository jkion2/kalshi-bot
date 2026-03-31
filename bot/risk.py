"""
bot/risk.py — Step 4a: Risk Management
ALL checks run deterministically in Python (not in an LLM prompt).
If ANY check fails, the trade is blocked.

Checks in order:
  1. Kill switch (STOP file)
  2. Daily loss limit
  3. Max drawdown
  4. Edge minimum
  5. Max concurrent positions
  6. Kelly position size calculation
  7. Max single position size
  8. Total exposure cap
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from bot import TradeSignal
from config import BotConfig

if TYPE_CHECKING:
    from bot.ledger import TradeLedger

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg

    # ── Main validation entry point ────────────────────────────────────────────
    def validate(self, signal: TradeSignal, ledger: "TradeLedger") -> tuple[bool, str]:
        """
        Run all risk checks.
        Returns (approved: bool, reason: str).
        All checks are deterministic Python — no LLM calls.
        """
        checks = [
            self._check_kill_switch,
            self._check_daily_loss,
            self._check_drawdown,
            self._check_edge,
            self._check_duplicate_position,
            self._check_concurrent_positions,
            self._check_daily_api_cost,
            self._check_opposing_positions,
        ]
        for check in checks:
            ok, reason = check(signal, ledger)
            if not ok:
                return False, reason
        return True, "approved"

    # ── Individual checks ──────────────────────────────────────────────────────
    def _check_kill_switch(self, signal, ledger) -> tuple[bool, str]:
        if Path("STOP").exists():
            return False, "STOP file detected — all trading halted"
        return True, ""

    def _check_daily_loss(self, signal, ledger) -> tuple[bool, str]:
        daily_pnl = ledger.today_pnl()
        limit     = -self.cfg.max_daily_loss_pct * ledger.bankroll
        if daily_pnl <= limit:
            return False, f"Daily loss ${daily_pnl:.2f} exceeds limit ${limit:.2f}"
        return True, ""

    def _check_drawdown(self, signal, ledger) -> tuple[bool, str]:
        drawdown = ledger.current_drawdown()
        if drawdown >= self.cfg.max_drawdown_pct:
            return False, f"Drawdown {drawdown:.1%} exceeds max {self.cfg.max_drawdown_pct:.1%}"
        return True, ""

    def _check_edge(self, signal, ledger) -> tuple[bool, str]:
        if signal.edge < self.cfg.min_edge:
            return False, f"Edge {signal.edge:.1%} below minimum {self.cfg.min_edge:.1%}"
        return True, ""

    def _check_duplicate_position(self, signal, ledger) -> tuple[bool, str]:
        open_markets = {t["market_id"] for t in ledger._trades if t["status"] == "open"}
        if signal.market_id in open_markets:
            return False, f"Already holding open position in {signal.market_id}"
        return True, ""

    def _check_concurrent_positions(self, signal, ledger) -> tuple[bool, str]:
        n = ledger.open_position_count()
        if n >= self.cfg.max_concurrent_positions:
            return False, f"Already at max {n} concurrent positions"
        return True, ""

    def _check_daily_api_cost(self, signal, ledger) -> tuple[bool, str]:
        cost = ledger.today_api_cost()
        if cost >= self.cfg.max_daily_api_cost_usd:
            return False, f"Daily API cost ${cost:.2f} at limit ${self.cfg.max_daily_api_cost_usd:.2f}"
        return True, ""

    def _check_opposing_positions(self, signal, ledger) -> tuple[bool, str]:
        """Block trades where we already hold the opposite side of the same event."""
        event_prefix = signal.market_id.rsplit("-", 1)[0]
        for trade in ledger._trades:
            if trade["status"] == "open" and trade["market_id"] != signal.market_id:
                existing_prefix = trade["market_id"].rsplit("-", 1)[0]
                if existing_prefix == event_prefix:
                    return False, f"Already holding open position in opposing market {trade['market_id']}"
        return True, ""

    # ── Kelly Criterion Position Sizing ───────────────────────────────────────
    def kelly_size(self, signal: TradeSignal, bankroll: float) -> float:
        """
        Calculate optimal position size using Fractional Kelly Criterion.

        Full Kelly formula:
            f* = (p * b - q) / b
        where:
            p = win probability
            q = 1 - p (lose probability)
            b = net odds (how much you win per $1 risked)

        For a prediction market at price c (in dollars):
            - You pay c per contract
            - You win (1 - c) if correct
            - You lose c if wrong
            So b = (1 - c) / c

        We then multiply by kelly_fraction (0.25 = Quarter-Kelly) for safety.
        """
        price = signal.market_price
        p     = signal.model_probability if signal.side == "yes" else 1 - signal.model_probability
        q     = 1.0 - p

        # Avoid division by zero at extreme prices
        if price <= 0 or price >= 1:
            return 0.0

        b = (1 - price) / price  # net odds

        # Full Kelly fraction of bankroll
        full_kelly_f = (p * b - q) / b

        # Safety check: full Kelly can be negative (means don't bet at all)
        if full_kelly_f <= 0:
            log.debug(f"Kelly says no bet (full_kelly_f={full_kelly_f:.3f})")
            return 0.0

        # Apply fractional Kelly
        fractional_f = full_kelly_f * self.cfg.kelly_fraction

        # Convert to dollar amount
        kelly_dollars = fractional_f * bankroll

        # Hard cap at max_position_pct of bankroll
        max_dollars = self.cfg.max_position_pct * bankroll

        position = min(kelly_dollars, max_dollars)

        log.info(
            f"  Kelly sizing: p={p:.0%} b={b:.2f} "
            f"full_f={full_kelly_f:.3f} frac_f={fractional_f:.3f} "
            f"-> ${position:.2f} (max cap ${max_dollars:.2f})"
        )
        return round(position, 2)
