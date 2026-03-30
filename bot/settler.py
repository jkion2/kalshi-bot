"""
bot/settler.py — Auto-Settlement
Checks all open paper trades against the Kalshi API to see if they've
resolved. If resolved, calculates P&L, updates trade status, and
adjusts the bankroll automatically.

Runs as part of the main cycle — no manual trades.json editing needed.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from bot.kalshi_client import KalshiClient
from bot.ledger import TradeLedger

log = logging.getLogger(__name__)


class TradeSettler:
    def __init__(self, client: KalshiClient, ledger: TradeLedger):
        self.client = client
        self.ledger = ledger

    async def settle_open_trades(self) -> int:
        """
        Check all open trades against Kalshi API.
        Returns the number of trades settled this run.
        """
        open_trades = [t for t in self.ledger._trades if t["status"] == "open"]
        if not open_trades:
            log.info("Settler: no open trades to check")
            return 0

        log.info(f"Settler: checking {len(open_trades)} open trades")
        settled = 0

        for trade in open_trades:
            result = await self._check_market(trade)
            if result is not None:
                self._apply_settlement(trade, result)
                settled += 1

        if settled > 0:
            self.ledger.save()
            log.info(f"Settler: settled {settled} trades | Bankroll: ${self.ledger.bankroll:.2f}")

        return settled

    # ── Check a single market ─────────────────────────────────────────────────
    async def _check_market(self, trade: dict) -> Optional[dict]:
        """
        Fetch market status from Kalshi.
        Returns settlement dict if resolved, None if still open.
        """
        try:
            data = await self.client.get_market(trade["market_id"])
            market = data.get("market", data)  # handle both response formats

            status = market.get("status", "")
            result = market.get("result", "")

            # Kalshi statuses: active, closed, settled, finalized
            if status not in ("settled", "finalized"):
                return None

            # result is "yes" or "no"
            if not result:
                return None

            return {
                "result":    result.lower(),
                "status":    status,
                "close_time": market.get("close_time", ""),
            }

        except Exception as exc:
            log.debug(f"Settler: could not fetch {trade['market_id']}: {exc}")
            return None

    # ── Apply settlement to a trade ───────────────────────────────────────────
    def _apply_settlement(self, trade: dict, settlement: dict) -> None:
        """
        Update the trade record and adjust the bankroll.

        For a YES contract bought at price p with n contracts:
          - WIN  (result=yes): profit = n * (1 - p)
          - LOSS (result=no):  loss   = n * p (what we paid)

        For a NO contract bought at price p with n contracts:
          - WIN  (result=no):  profit = n * (1 - p)
          - LOSS (result=yes): loss   = n * p
        """
        side       = trade["side"]          # "yes" or "no"
        result     = settlement["result"]   # "yes" or "no"
        contracts  = trade.get("contracts", 0) or 0
        fill_price = trade.get("fill_price", 0) or 0

        # Did we win?
        won = (side == result)

        if won:
            pnl = contracts * (1.0 - fill_price)
            status = "won"
            outcome = 1
        else:
            pnl = -(contracts * fill_price)
            status = "lost"
            outcome = 0

        # Round to cents
        pnl = round(pnl, 2)

        # Update trade record
        trade["outcome"]   = outcome
        trade["pnl"]       = pnl
        trade["status"]    = status
        trade["closed_at"] = settlement.get("close_time") or datetime.now(timezone.utc).isoformat()

        # Update bankroll and peak
        self.ledger.bankroll = round(self.ledger.bankroll + pnl, 2)
        if self.ledger.bankroll > self.ledger.peak_bankroll:
            self.ledger.peak_bankroll = self.ledger.bankroll

        log.info(
            f"Settler: {trade['market_id']} -> {status.upper()} | "
            f"side={side} result={result} | "
            f"P&L=${pnl:+.2f} | "
            f"Bankroll=${self.ledger.bankroll:.2f}"
        )