import asyncio
from bot.kalshi_client import KalshiClient
from config import BotConfig

async def check():
    cfg = BotConfig.load(paper_mode=True)
    client = KalshiClient(
        api_key=cfg.kalshi_api_key,
        api_key_id=cfg.kalshi_api_key_id,
        demo=cfg.demo_mode,
    )
    await client.connect()
    ids = [
        "KXTRUMPMEET-26MAR-XJIN",
        "KXWTI-26APR02-T105.99",
        "KXRT-SUP-45",
    ]
    for mid in ids:
        data = await client.get_market(mid)
        market = data.get("market", data)
        print(mid)
        print("  status:", market.get("status"))
        print("  result:", market.get("result"))

asyncio.run(check())