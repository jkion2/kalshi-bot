"""
bot/executor.py — Step 4b: Execute
Places orders on Kalshi (or simulates them in paper mode).
Handles slippage detection and order confirmation.
"""

import logging
import math
from datetime import datetime

from bot import TradeSignal, TradeResult
from bot.kalshi_client import KalshiClient
from config import BotConfig

log = logging.getLogger(__name__)


class TradeExecutor:
    def __init__(self, client: KalshiClient, cfg: BotConfig):
        self.client = client
        self.cfg    = cfg

    async def execute(
        self,
        signal: TradeSignal,
        position_size_usd: float,
    ) -> TradeResult:
        """
        Execute a trade. In paper mode, simulates without placing real orders.
        In live mode, submits a limit order to Kalshi.
        """
        if position_size_usd <= 0:
            return TradeResult(
                success=False,
                order_id=None,
                fill_price=None,
                contracts=None,
                error="Position size is zero",
            )

        price      = signal.market_price
        contracts  = self._contracts_from_usd(position_size_usd, price)

        if contracts < 1:
            return TradeResult(
                success=False,
                order_id=None,
                fill_price=None,
                contracts=None,
                error=f"Position ${position_size_usd:.2f} too small for even 1 contract at {price:.0%}",
            )

        if self.cfg.paper_mode:
            return await self._simulate(signal, contracts, price)
        else:
            return await self._place_live_order(signal, contracts, price)

    # ── Paper trading ──────────────────────────────────────────────────────────
    async def _simulate(
        self,
        signal: TradeSignal,
        contracts: int,
        price: float,
    ) -> TradeResult:
        """Simulate an order fill at the current market price."""
        fake_order_id = f"PAPER-{signal.market_id}-{int(datetime.utcnow().timestamp())}"
        cost = contracts * price
        log.info(
            f"  [PAPER] Simulated {contracts} {signal.side.upper()} contracts "
            f"at {price:.0%} = ${cost:.2f}"
        )
        return TradeResult(
            success    = True,
            order_id   = fake_order_id,
            fill_price = price,
            contracts  = contracts,
            simulated  = True,
        )

    # ── Live trading ───────────────────────────────────────────────────────────
    async def _place_live_order(
        self,
        signal: TradeSignal,
        contracts: int,
        price: float,
    ) -> TradeResult:
        """
        Submit a limit order to Kalshi.
        Uses a limit order (not market order) to control slippage.
        """
        try:
            # Get current orderbook to check for slippage before committing
            orderbook = await self.client.get_orderbook(signal.market_id)
            live_price = self._extract_best_price(orderbook, signal.side)

            if live_price is None:
                return TradeResult(
                    success=False, order_id=None, fill_price=None, contracts=None,
                    error="Could not read live orderbook price",
                )

            slippage = abs(live_price - price)
            if slippage > self.cfg.max_slippage_pct:
                return TradeResult(
                    success=False, order_id=None, fill_price=None, contracts=None,
                    error=f"Slippage {slippage:.1%} exceeds max {self.cfg.max_slippage_pct:.1%}. Aborting.",
                )

            # Convert to cents (Kalshi uses integer cents 1-99)
            price_cents = round(live_price * 100)

            resp = await self.client.create_order(
                ticker     = signal.market_id,
                side       = signal.side,
                action     = "buy",
                count      = contracts,
                price      = price_cents,
            )

            order_id = resp.get("order", {}).get("order_id", "unknown")
            log.info(
                f"  [LIVE] Order placed: {order_id} | "
                f"{contracts} {signal.side.upper()} @ {price_cents}¢"
            )
            return TradeResult(
                success    = True,
                order_id   = order_id,
                fill_price = live_price,
                contracts  = contracts,
                simulated  = False,
            )

        except Exception as exc:
            log.error(f"Order placement failed for {signal.market_id}: {exc}")
            return TradeResult(
                success=False, order_id=None, fill_price=None, contracts=None,
                error=str(exc),
            )

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _contracts_from_usd(self, usd: float, price: float) -> int:
        """How many contracts can we buy for this dollar amount at this price?"""
        if price <= 0:
            return 0
        return math.floor(usd / price)

    def _extract_best_price(self, orderbook: dict, side: str) -> float | None:
        """Extract best ask price for the side we want to buy."""
        try:
            if side == "yes":
                asks = orderbook.get("orderbook", {}).get("yes", [])
            else:
                asks = orderbook.get("orderbook", {}).get("no", [])
            # asks are sorted lowest price first (best ask for buyer)
            if asks:
                return asks[0][0] / 100   # cents to fraction
        except (IndexError, KeyError, TypeError):
            pass
        return None
