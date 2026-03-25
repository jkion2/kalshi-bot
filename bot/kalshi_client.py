"""
bot/kalshi_client.py
Thin wrapper around the Kalshi REST API.

Demo base URL : https://demo-api.kalshi.co/trade-api/v2
Live base URL : https://trading-api.kalshi.co/trade-api/v2

Docs: https://trading-api.readme.io
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Optional
import httpx

log = logging.getLogger(__name__)

DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
LIVE_BASE = "https://trading-api.kalshi.co/trade-api/v2"


class KalshiClient:
    def __init__(self, api_key: str, api_key_id: str, demo: bool = True):
        self.api_key    = api_key
        self.api_key_id = api_key_id
        self.base_url   = DEMO_BASE if demo else LIVE_BASE
        self.demo       = demo
        self._client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=10.0,
        )
        log.info(f"KalshiClient connected to {'DEMO' if self.demo else 'LIVE'} API")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    # ── Authentication ─────────────────────────────────────────────────────────
    def _sign(self, method: str, path: str, body: str = "") -> dict[str, str]:
        """
        Kalshi uses HMAC-SHA256 request signing.
        Header format: Kalshi-Access-Key, Kalshi-Access-Signature, Kalshi-Access-Timestamp
        """
        ts = str(int(time.time() * 1000))  # milliseconds
        msg = ts + method.upper() + path + body
        sig = hmac.new(
            self.api_key.encode(),
            msg.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "Kalshi-Access-Key":       self.api_key_id,
            "Kalshi-Access-Signature": sig,
            "Kalshi-Access-Timestamp": ts,
            "Content-Type":            "application/json",
        }

    # ── Generic request ────────────────────────────────────────────────────────
    async def _request(
        self, method: str, path: str, body: Optional[dict] = None
    ) -> dict[str, Any]:
        body_str = json.dumps(body) if body else ""
        headers  = self._sign(method, path, body_str)

        resp = await self._client.request(
            method  = method,
            url     = path,
            headers = headers,
            content = body_str or None,
        )

        if resp.status_code == 429:
            log.warning("Rate limited. Sleeping 2s.")
            await asyncio.sleep(2)
            return await self._request(method, path, body)

        resp.raise_for_status()
        return resp.json()

    # ── Market endpoints ───────────────────────────────────────────────────────
    async def get_markets(
        self,
        limit: int = 100,
        cursor: str = "",
        status: str = "open",
    ) -> dict:
        """
        GET /markets — paginated list of all open markets.
        Returns: { markets: [...], cursor: "..." }
        """
        params = f"?limit={limit}&status={status}"
        if cursor:
            params += f"&cursor={cursor}"
        return await self._request("GET", f"/markets{params}")

    async def get_market(self, ticker: str) -> dict:
        """GET /markets/{ticker} — single market details."""
        return await self._request("GET", f"/markets/{ticker}")

    async def get_orderbook(self, ticker: str, depth: int = 5) -> dict:
        """GET /markets/{ticker}/orderbook — best bids/asks."""
        return await self._request("GET", f"/markets/{ticker}/orderbook?depth={depth}")

    # ── Order endpoints ────────────────────────────────────────────────────────
    async def create_order(
        self,
        ticker: str,
        side: str,         # "yes" or "no"
        action: str,       # "buy" or "sell"
        count: int,        # number of contracts
        price: int,        # price in cents (1-99)
        order_type: str = "limit",
    ) -> dict:
        """
        POST /portfolio/orders — place a limit order.
        price is in cents: e.g. 0.65 probability = price 65
        """
        body = {
            "ticker":     ticker,
            "side":       side,
            "action":     action,
            "count":      count,
            "type":       order_type,
            "yes_price":  price if side == "yes" else 100 - price,
            "no_price":   price if side == "no"  else 100 - price,
        }
        return await self._request("POST", "/portfolio/orders", body)

    async def cancel_order(self, order_id: str) -> dict:
        """DELETE /portfolio/orders/{order_id}"""
        return await self._request("DELETE", f"/portfolio/orders/{order_id}")

    async def get_positions(self) -> dict:
        """GET /portfolio/positions — all open positions."""
        return await self._request("GET", "/portfolio/positions")

    async def get_balance(self) -> float:
        """GET /portfolio/balance — available cash in USD."""
        data = await self._request("GET", "/portfolio/balance")
        # Kalshi returns balance in cents
        return data.get("balance", 0) / 100
