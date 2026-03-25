import asyncio
from bot.kalshi_client import KalshiClient
from config import BotConfig

async def check():
    cfg = BotConfig.load()
    client = KalshiClient(cfg.kalshi_api_key, cfg.kalshi_api_key_id, demo=True)
    await client.connect()
    resp = await client.get_markets(limit=5)
    markets = resp.get("markets", [])
    print(f"Got {len(markets)} markets")
    print("---")
    for m in markets:
        print(m)
        print("---")
    await client.close()

asyncio.run(check())
