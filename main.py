"""
Kalshi AI Trading Bot - Main Entry Point
Run: python main.py --mode paper   (paper trading, no real money)
Run: python main.py --mode live    (live trading, REAL MONEY)
"""
 
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
 
from bot.scanner import MarketScanner
from bot.researcher import MarketResearcher
from bot.predictor import MarketPredictor
from bot.risk import RiskManager
from bot.executor import TradeExecutor
from bot.ledger import TradeLedger
from bot.kalshi_client import KalshiClient
from config import BotConfig
from bot.settler import TradeSettler
 
# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/bot_{datetime.now().strftime('%Y%m%d')}.log"),
    ],
)
log = logging.getLogger("main")
 
# How long to cache research before re-fetching (2 hours)
CACHE_TTL_SECONDS = 7200
 
# How many markets to research per cycle (after filtering blocked events)
RESEARCH_SLOTS = 3
 
 
def _event_key(market_id: str) -> str:
    """
    Extract the date+teams portion of a market ID to match across market types.
 
    Examples:
      KXMLBGAME-26MAR311835TEXBAL-TEX  ->  26MAR311835TEXBAL
      KXMLBTOTAL-26MAR311835TEXBAL-10  ->  26MAR311835TEXBAL
      KXNBAGAME-26MAR31NYKHOU-HOU      ->  26MAR31NYKHOU
 
    This ensures that holding a position in any market type for a given
    real-world event blocks all other market types for that same event.
    """
    parts = market_id.split("-")
    if len(parts) >= 2:
        return parts[1]
    return market_id
 
 
async def run_cycle(
    scanner: MarketScanner,
    researcher: MarketResearcher,
    predictor: MarketPredictor,
    risk: RiskManager,
    executor: TradeExecutor,
    ledger: TradeLedger,
    settler: TradeSettler,
    cfg: BotConfig,
    research_cache: dict,
) -> None:
    """One full scan → research → predict → risk-check → execute cycle."""
 
    log.info("═" * 60)
    log.info("Starting new trading cycle")
 
    # 0. SETTLE - check if any open trades have been resolved
    await settler.settle_open_trades()
 
    # 1. SCAN — find top 10 markets worth looking at
    candidates = await scanner.scan()
    log.info(f"Scanner found {len(candidates)} candidate markets")
    if not candidates:
        log.info("No candidates this cycle. Sleeping.")
        return
 
    # 2. PRE-FILTER — build set of event keys we already hold positions in.
    # Uses the date+teams portion (parts[1]) so that KXMLBGAME and KXMLBTOTAL
    # for the same real-world game are both blocked when either is held.
    open_event_keys = {
        _event_key(t["market_id"])
        for t in ledger._trades
        if t["status"] == "open"
    }
 
    # 3. RESEARCH — skip blocked events and cached markets.
    # Always research exactly RESEARCH_SLOTS fresh markets per cycle.
    now = datetime.now(timezone.utc)
 
    # Expire stale cache entries older than CACHE_TTL_SECONDS
    expired = [
        mid for mid, entry in research_cache.items()
        if (now - entry["cached_at"]).total_seconds() > CACHE_TTL_SECONDS
    ]
    for mid in expired:
        del research_cache[mid]
        log.info(f"Cache expired: {mid}")
 
    briefs = []
    researched_candidates = []
 
    for m in candidates:
        # Stop once we have filled all research slots
        if len(researched_candidates) >= RESEARCH_SLOTS:
            break
 
        event_key = _event_key(m.market_id)
 
        # Skip if we already hold any position in this real-world event
        if event_key in open_event_keys:
            log.info(f"Skipping research for {m.market_id}: open position exists in this event")
            continue
 
        # Use cached research if still fresh
        if m.market_id in research_cache:
            log.info(f"Cache hit: {m.market_id} — skipping Claude research call")
            briefs.append(research_cache[m.market_id]["brief"])
            researched_candidates.append(m)
        else:
            brief = await researcher.research(m)
            await asyncio.sleep(6) # prevent the token rate limit bursts
            if brief is not None:
                research_cache[m.market_id] = {
                    "brief":     brief,
                    "cached_at": datetime.now(timezone.utc),
                }
                briefs.append(brief)
                researched_candidates.append(m)
 
    candidates = researched_candidates
 
    if not candidates:
        log.info("All candidates blocked by open positions or served from cache. No API calls needed.")
        return
 
    # 4. PREDICT — estimate true probability
    signals = []
    for market, brief in zip(candidates, briefs):
        signal = await predictor.predict(market, brief)
        if signal:
            signals.append(signal)
    log.info(f"Predictor generated {len(signals)} trade signals")
 
    # 5. RISK + EXECUTE — validate and place (or simulate) orders
    traded_events = set()
    for signal in signals:
        # Block opposing positions within the same cycle
        event_key = _event_key(signal.market_id)
        if event_key in traded_events:
            log.warning(f"Skipping {signal.market_id}: already traded this event this cycle")
            continue
 
        approved, reason = risk.validate(signal, ledger)
        if not approved:
            log.warning(f"Risk rejected {signal.market_id}: {reason}")
            continue
 
        position_size = risk.kelly_size(signal, ledger.bankroll)
        result = await executor.execute(signal, position_size)
        ledger.record(signal, position_size, result)
 
        # Bust cache after trading so next cycle re-researches fresh
        research_cache.pop(signal.market_id, None)
 
        traded_events.add(event_key)
        log.info(
            f"{'[PAPER]' if cfg.paper_mode else '[LIVE]'} "
            f"Traded {signal.market_id} | "
            f"Side={signal.side} | Size=${position_size:.2f} | "
            f"Edge={signal.edge:.1%}"
        )
 
    # 6. COMPOUND — nightly post-mortem is handled by ledger.daily_review()
    ledger.save()
 
 
async def main(cfg: BotConfig) -> None:
    log.info(f"Bot starting in {'PAPER' if cfg.paper_mode else '*** LIVE ***'} mode")
    log.info(f"Bankroll: ${cfg.starting_bankroll:.2f} | Max position: {cfg.max_position_pct:.0%}")
 
    # Check for kill switch before doing anything
    if Path("STOP").exists():
        log.critical("STOP file detected. Bot will not start. Delete STOP to proceed.")
        return
 
    client = KalshiClient(
        api_key=cfg.kalshi_api_key,
        api_key_id=cfg.kalshi_api_key_id,
        demo=cfg.demo_mode,
    )
    await client.connect()
 
    scanner    = MarketScanner(client, cfg)
    researcher = MarketResearcher(cfg)
    predictor  = MarketPredictor(cfg)
    risk       = RiskManager(cfg)
    executor   = TradeExecutor(client, cfg)
    ledger     = TradeLedger(cfg)
    settler    = TradeSettler(client, ledger)
    await settler.sync_positions_on_startup()
 
    # Shared research cache — persists across cycles, expires after CACHE_TTL_SECONDS
    research_cache: dict = {}
 
    cycle = 0
    while True:
        if Path("STOP").exists():
            log.critical("STOP file detected mid-run. Halting immediately.")
            break
 
        cycle += 1
        log.info(f"Cycle #{cycle}")
        try:
            await run_cycle(
                scanner, researcher, predictor,
                risk, executor, ledger, settler,
                cfg, research_cache,
            )
        except Exception as exc:
            log.exception(f"Unhandled error in cycle #{cycle}: {exc}")
 
        log.info(f"Sleeping {cfg.cycle_interval_seconds}s until next cycle...")
        await asyncio.sleep(cfg.cycle_interval_seconds)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi AI Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="paper = no real money, live = real money",
    )
    args = parser.parse_args()
 
    config = BotConfig.load(paper_mode=(args.mode == "paper"))
    asyncio.run(main(config))