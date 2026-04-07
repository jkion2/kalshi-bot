"""
config.py — All tunable parameters in one place.
Edit this file to change strategy settings.
Never hardcode secrets here — use .env instead.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env file automatically


@dataclass
class BotConfig:
    # ── API Credentials ────────────────────────────────────────────────────────
    kalshi_api_key: str = ""
    kalshi_api_key_id: str = ""
    anthropic_api_key: str = ""

    # ── Mode Flags ─────────────────────────────────────────────────────────────
    paper_mode: bool = True        # True = simulate trades only, no real orders
    demo_mode: bool = True         # True = use Kalshi demo environment

    # ── Bankroll & Position Sizing ─────────────────────────────────────────────
    starting_bankroll: float = 300.0   # Your starting capital in USD
    max_position_pct: float = 0.05     # Max 5% of bankroll per trade
    max_concurrent_positions: int = 25
    kelly_fraction: float = 0.25       # Quarter-Kelly (safer than full Kelly)

    # ── Edge Thresholds ────────────────────────────────────────────────────────
    min_edge: float = 0.06             # Only trade if model_prob - market_price > 6%
    min_confidence: float = 0.68       # Model must be at least 68% confident

    # ── Market Filters ─────────────────────────────────────────────────────────
    min_volume: int = 10              # Minimum contracts traded (liquidity filter)
    max_days_to_expiry: int = 30       # Only markets resolving within 30 days
    max_spread_cents: float = 0.50     # Skip markets with spread > 5 cents
    volume_spike_multiplier: float = 2.0  # Flag if volume > 2x 7-day average

    # ── Risk Controls ──────────────────────────────────────────────────────────
    max_daily_loss_pct: float = 0.15   # Stop trading if daily loss > 15%
    max_drawdown_pct: float = 0.15     # Stop if drawdown from peak > 15%
    max_daily_api_cost_usd: float = 5.0  # Hard cap on Claude API spend per day
    max_slippage_pct: float = 0.02     # Abort if price moves > 2% before fill

    # ── Scheduling ─────────────────────────────────────────────────────────────
    cycle_interval_seconds: int = 1800  # Run full cycle every 30 minutes

    # ── File Paths ─────────────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: Path("data"))
    log_dir: Path  = field(default_factory=lambda: Path("logs"))
    trade_log_path: Path = field(default_factory=lambda: Path("data/trades.json"))
    failure_log_path: Path = field(default_factory=lambda: Path("skills/predict-market-bot/references/failure_log.md"))

    @classmethod
    def load(cls, paper_mode: bool = True) -> "BotConfig":
        """Load config from environment variables with safe defaults."""
        cfg = cls(
            kalshi_api_key    = os.getenv("KALSHI_API_KEY", ""),
            kalshi_api_key_id = os.getenv("KALSHI_API_KEY_ID", ""),
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", ""),
            paper_mode        = paper_mode,
            demo_mode         = os.getenv("KALSHI_DEMO", "true").lower() == "true",
            starting_bankroll = float(os.getenv("STARTING_BANKROLL", "300")),
        )

        # Enforce safety: if demo creds are absent, force demo mode
        if not cfg.kalshi_api_key:
            cfg.demo_mode = True
            cfg.paper_mode = True

        # Create directories
        cfg.data_dir.mkdir(exist_ok=True)
        cfg.log_dir.mkdir(exist_ok=True)

        return cfg
