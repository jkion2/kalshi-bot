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
from datetime import datetime

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


async def run_cycle(
    scanner: MarketScanner,
    researcher: MarketResearcher,
    predictor: MarketPredictor,
    risk: RiskManager,
    executor: TradeExecutor,
    ledger: TradeLedger,
    settler: TradeSettler,
    cfg: BotConfig,
) -> None:
    """One full scan → research → predict → risk-check → execute cycle."""

    log.info("═" * 60)
    log.info("Starting new trading cycle")

    # 0. SETTLE - check if any open trades have been resolved
    await settler.settle_open_trades()

    # 1. SCAN — find markets worth looking at
    candidates = await scanner.scan()
    log.info(f"Scanner found {len(candidates)} candidate markets")
    if not candidates:
        log.info("No candidates this cycle. Sleeping.")
        return

    # 2. RESEARCH — gather intel on each candidate (runs in parallel)
    briefs = []
    for m in candidates:
        brief = await researcher.research(m)
        briefs.append(brief)

    # 3. PREDICT — estimate true probability
    signals = []
    for market, brief in zip(candidates, briefs):
        signal = await predictor.predict(market, brief)
        if signal:
            signals.append(signal)
    log.info(f"Predictor generated {len(signals)} trade signals")

    # 4. RISK + EXECUTE — validate and place (or simulate) orders
    for signal in signals:
        approved, reason = risk.validate(signal, ledger)
        if not approved:
            log.warning(f"Risk rejected {signal.market_id}: {reason}")
            continue

        position_size = risk.kelly_size(signal, ledger.bankroll)
        result = await executor.execute(signal, position_size)
        ledger.record(signal, position_size, result)
        log.info(
            f"{'[PAPER]' if cfg.paper_mode else '[LIVE]'} "
            f"Traded {signal.market_id} | "
            f"Side={signal.side} | Size=${position_size:.2f} | "
            f"Edge={signal.edge:.1%}"
        )

    # 5. COMPOUND — nightly post-mortem is handled by ledger.daily_review()
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

    scanner   = MarketScanner(client, cfg)
    researcher = MarketResearcher(cfg)
    predictor  = MarketPredictor(cfg)
    risk       = RiskManager(cfg)
    executor   = TradeExecutor(client, cfg)
    ledger     = TradeLedger(cfg)
    settler    = TradeSettler(client, ledger)

    cycle = 0
    while True:
        if Path("STOP").exists():
            log.critical("STOP file detected mid-run. Halting immediately.")
            break

        cycle += 1
        log.info(f"Cycle #{cycle}")
        try:
            await run_cycle(scanner, researcher, predictor, risk, executor, ledger, settler, cfg)
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
