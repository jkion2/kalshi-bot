"""
bot/__init__.py
Shared dataclasses used across all bot modules.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Market:
    """A Kalshi market returned by the scanner."""
    market_id: str          # Kalshi ticker e.g. "INXD-23DEC31-B4000"
    title: str              # Human-readable title
    yes_price: float        # Current YES price (0-1)
    no_price: float         # Current NO price (0-1)
    volume: int             # Total contracts traded
    open_interest: int      # Outstanding contracts
    days_to_expiry: float   # Days until market closes
    spread: float           # Best ask - best bid
    category: str           # e.g. "politics", "economics", "weather"


@dataclass
class ResearchBrief:
    """Output of the research step for one market."""
    market_id: str
    sentiment_score: float      # -1.0 (bearish) to +1.0 (bullish)
    sentiment_sources: int      # How many sources analyzed
    headline_summary: str       # 1-2 sentence summary of key news
    key_facts: list[str]        # Bullet points of important facts
    raw_text: str               # Full aggregated text for LLM


@dataclass
class TradeSignal:
    """Output of the prediction step — a potential trade."""
    market_id: str
    title: str
    side: str                   # "yes" or "no"
    model_probability: float    # Our estimated true probability (0-1)
    market_price: float         # What the market is pricing (0-1)
    edge: float                 # model_probability - market_price
    confidence: float           # Model's confidence in the estimate
    expected_value: float       # EV of the trade
    reasoning: str              # Claude's explanation
    days_to_expiry: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradeResult:
    """Result after attempting to place an order."""
    success: bool
    order_id: Optional[str]
    fill_price: Optional[float]
    contracts: Optional[int]
    error: Optional[str] = None
    simulated: bool = False     # True when paper trading
