"""
bot/scanner.py — Step 1: Scan
Pulls all open Kalshi markets, applies liquidity/timing filters,
and returns a ranked shortlist of candidates worth researching.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from bot import Market
from bot.kalshi_client import KalshiClient
from config import BotConfig

log = logging.getLogger(__name__)


class MarketScanner:
    def __init__(self, client: KalshiClient, cfg: BotConfig):
        self.client = client
        self.cfg    = cfg

    async def scan(self) -> list[Market]:
        """
        Fetch all open markets → filter → rank → return top candidates.
        """
        raw_markets = await self._fetch_all_markets()
        log.info(f"Fetched {len(raw_markets)} raw open markets from Kalshi")

        candidates = []
        for raw in raw_markets:
            market = self._parse(raw)
            if market and self._passes_filters(market):
                candidates.append(market)

        log.info(f"{len(candidates)} markets passed filters")
        return self._rank(candidates)

    # ── Fetching ───────────────────────────────────────────────────────────────
    async def _fetch_all_markets(self) -> list[dict]:
        """Paginate through all open markets."""
        all_markets = []
        cursor = ""
        while True:
            resp = await self.client.get_markets(limit=100, cursor=cursor)
            markets = resp.get("markets", [])
            all_markets.extend(markets)
            cursor = resp.get("cursor", "")
            if not cursor or not markets:
                break
        return all_markets

    # ── Parsing ────────────────────────────────────────────────────────────────
    def _parse(self, raw: dict) -> Optional[Market]:
        try:
            ticker    = raw["ticker"]
            yes_bid   = float(raw.get("yes_bid_dollars", 0))
            yes_ask   = float(raw.get("yes_ask_dollars", 1))
            yes_price = (yes_bid + yes_ask) / 2
            no_price  = 1.0 - yes_price
            spread    = yes_ask - yes_bid
            volume    = float(raw.get("volume_fp", 0))

            close_time = raw.get("close_time", "")
            if close_time:
                expiry_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                days_left = (expiry_dt - datetime.now(timezone.utc)).total_seconds() / 86400
            else:
                days_left = 999

            return Market(
                market_id     = ticker,
                title         = raw.get("title", ticker),
                yes_price     = yes_price,
                no_price      = no_price,
                volume        = int(volume),
                open_interest = int(float(raw.get("open_interest_fp", 0))),
                days_to_expiry= days_left,
                spread        = spread,
                category      = raw.get("event_ticker", "unknown"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            log.debug(f"Failed to parse market {raw.get('ticker','?')}: {exc}")
            return None

    # ── Filters ────────────────────────────────────────────────────────────────
    def _passes_filters(self, m: Market) -> bool:
        """
        All conditions must be True for a market to proceed to research.
        """
        # Must have real liquidity
        if m.volume < self.cfg.min_volume:
            return False

        # Must resolve soon enough to be actionable
        if m.days_to_expiry > self.cfg.max_days_to_expiry:
            return False

        # Dead markets (price at extremes) have no edge
        if m.yes_price <= 0.02 or m.yes_price >= 0.98:
            return False

        # Wide spread = hard to make money after slippage
        if m.spread > self.cfg.max_spread_cents:
            return False

        return True

    # ── Ranking ────────────────────────────────────────────────────────────────
    def _rank(self, markets: list[Market]) -> list[Market]:
        """
        Sort by a composite opportunity score:
        - Prefer markets near 50% (most uncertain → most potential edge)
        - Prefer markets with higher volume (easier to fill)
        - Prefer markets closer to expiry (faster resolution)

        Returns top 10 to keep research costs bounded.
        """
        def score(m: Market) -> float:
            uncertainty = 1.0 - abs(m.yes_price - 0.5) * 2   # peaks at 0.5 price
            liquidity   = min(m.volume / 1000, 1.0)            # normalize to 0-1
            urgency     = max(0, 1.0 - m.days_to_expiry / 30)  # closer = higher
            return uncertainty * 0.5 + liquidity * 0.3 + urgency * 0.2

        ranked = sorted(markets, key=score, reverse=True)
        top    = ranked[:3]
        for m in top:
            log.debug(f"  Candidate: {m.market_id} | YES={m.yes_price:.0%} | vol={m.volume}")
        return top
