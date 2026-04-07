"""
bot/predictor.py — Step 3: Predict
Uses Claude to estimate the true probability of a market outcome,
then calculates edge vs. the current market price.
Only emits a TradeSignal if edge > min_edge and confidence > min_confidence.
"""
 
import json
import logging
from anthropic import AsyncAnthropic
from typing import Optional
 
from bot import Market, ResearchBrief, TradeSignal
from config import BotConfig
 
log = logging.getLogger(__name__)
 
PREDICTOR_SYSTEM_PROMPT = """You are an expert prediction market forecaster.
Your job is to estimate the true probability of a binary outcome.
 
Rules:
1. Be calibrated: a 70% prediction should be right about 70% of the time.
2. Do not anchor to the current market price — form your own independent estimate first.
3. Consider base rates, recent evidence, and time remaining.
4. Be especially skeptical of extreme probabilities (<10% or >90%).
5. Express uncertainty honestly in the confidence field.
 
CRITICAL — MARKET STRUCTURE AWARENESS:
Before estimating probability, read the market title and ID carefully to understand
what YES and NO mean. Common traps:
 
- Markets with TIE, DRAW, or similar in the ID or title:
  YES = the tie/draw happens, NO = it does not happen
  e.g. "Mexico vs Belgium Winner? (TIE)" — YES means there IS a tie, not that someone wins
 
- Markets asking "Will X win by over N points?":
  YES = they win by more than N, NO = they win by less or lose
 
- Markets asking "Will X be above/below threshold?":
  YES = the condition is met, NO = it is not
 
- "Winner?" markets with a specific team suffix (e.g. -TEX, -BAL):
  YES = that team wins, NO = the other team wins
 
Always state your interpretation of YES/NO in your reasoning before giving a probability.
 
CRITICAL — COMPLETED EVENT SKEPTICISM:
If your research suggests an event has already occurred, be highly skeptical before
treating it as certain. Web search results frequently return:
- Results from different games with the same teams
- AI-generated or hallucinated match results
- Results from previous rounds/seasons
- Misidentified tournaments or leagues
 
If you believe an event is already resolved, still cap your confidence at 0.85 maximum
and your probability at 0.92 maximum, because web search results are often wrong about
completed events. The market price being far from your estimate is a warning sign,
not a confirmation — it may mean your information is wrong, not that the market is mispriced.
 
Always respond with a JSON object in this EXACT format:
{
  "probability": <float 0.0 to 1.0, your YES probability estimate>,
  "confidence":  <float 0.5 to 1.0, how confident you are in this estimate>,
  "reasoning":   "<2-3 sentence explanation including your YES/NO interpretation>"
}
 
CRITICAL: Output the JSON object FIRST, before any other text. No preamble, no analysis before the JSON. Start your response with { and end with }.
 
probability: your estimate of the true YES probability
confidence: 0.5 = very uncertain, 1.0 = highly certain
"""
 
# Live game dampening settings
LIVE_GAME_THRESHOLD_DAYS = 0.1
LIVE_DAMPEN_FACTOR       = 0.75
LIVE_MAX_CONFIDENCE      = 0.80
 
# Completed-event dampening — phrases that indicate Claude thinks the event already happened
# These are high-loss patterns: confident bets on "confirmed" results that turn out to be wrong
COMPLETED_EVENT_PHRASES = [
    "already occurred",
    "already happened",
    "already resolved",
    "already been played",
    "already took place",
    "completed game",
    "completed match",
    "historical outcome",
    "historical result",
    "match has been played",
    "game has been played",
    "outcome is essentially determined",
    "outcome has already",
    "factual outcome",
    "definitive outcome",
    "outcome is known",
    "result is known",
    "this is a past event",
    "this is a completed",
    "resolution date has passed",
    "event has already occurred",
    "event has occurred",
]
 
# How much to dampen when a completed-event claim is detected
COMPLETED_EVENT_DAMPEN_FACTOR = 0.80   # pull probability toward 50%
COMPLETED_EVENT_MAX_CONFIDENCE = 0.75  # cap confidence
 
 
def _is_tie_market(market: Market) -> bool:
    """Detect markets where YES = a tie/draw occurring."""
    keywords = ["TIE", "DRAW", "TIED", "DRAWN"]
    market_upper = market.market_id.upper()
    title_upper  = market.title.upper()
    return any(kw in market_upper or kw in title_upper for kw in keywords)
 
 
def _has_completed_event_claim(reasoning: str) -> bool:
    """Check if the reasoning claims the event has already happened."""
    reasoning_lower = reasoning.lower()
    return any(phrase in reasoning_lower for phrase in COMPLETED_EVENT_PHRASES)
 
 
class MarketPredictor:
    def __init__(self, cfg: BotConfig):
        self.cfg    = cfg
        self.client = AsyncAnthropic(api_key=cfg.anthropic_api_key)
 
    async def predict(
        self,
        market: Market,
        brief: ResearchBrief,
    ) -> Optional[TradeSignal]:
        """
        Estimate true probability for a market.
        Returns a TradeSignal if edge and confidence thresholds are met,
        or None if we shouldn't trade this market.
        """
        log.info(f"Predicting: {market.market_id}")
        log.info(f"  days_to_expiry: {market.days_to_expiry:.4f}")
 
        if _is_tie_market(market):
            log.info(f"  TIE/DRAW market detected — YES = tie occurs")
 
        prediction = await self._call_claude(market, brief)
        if not prediction:
            return None
 
        model_prob = prediction["probability"]
        confidence = prediction["confidence"]
        reasoning  = prediction["reasoning"]
 
        # ── Completed-event dampener ───────────────────────────────────────────
        # The bot frequently finds web search results claiming a game/match is
        # already completed, then bets with extreme confidence. These claims are
        # often wrong — wrong game, wrong date, hallucinated results.
        # We pull the probability toward 50% and cap confidence when this pattern
        # is detected, reducing position size and filtering weak signals out.
        if _has_completed_event_claim(reasoning):
            raw_prob   = model_prob
            raw_conf   = confidence
            model_prob = 0.5 + (model_prob - 0.5) * COMPLETED_EVENT_DAMPEN_FACTOR
            confidence = min(confidence, COMPLETED_EVENT_MAX_CONFIDENCE)
            log.info(
                f"  Completed-event dampener: prob {raw_prob:.0%} -> {model_prob:.0%} | "
                f"conf {raw_conf:.0%} -> {confidence:.0%}"
            )
 
        # ── Live game dampener ─────────────────────────────────────────────────
        # Markets expiring very soon are likely live. Claude reads current scores
        # and gets overconfident. Pull toward 50% and cap confidence.
        if market.days_to_expiry < LIVE_GAME_THRESHOLD_DAYS:
            raw_prob   = model_prob
            raw_conf   = confidence
            model_prob = 0.5 + (model_prob - 0.5) * LIVE_DAMPEN_FACTOR
            confidence = min(confidence, LIVE_MAX_CONFIDENCE)
            log.info(
                f"  Live game dampener: prob {raw_prob:.0%} -> {model_prob:.0%} | "
                f"conf {raw_conf:.0%} -> {confidence:.0%}"
            )
 
        # ── Edge calculation ───────────────────────────────────────────────────
        yes_edge = model_prob - market.yes_price
 
        if yes_edge >= self.cfg.min_edge:
            side      = "yes"
            edge      = yes_edge
            mkt_price = market.yes_price
        elif -yes_edge >= self.cfg.min_edge:
            side      = "no"
            edge      = -yes_edge
            mkt_price = market.no_price
        else:
            log.info(
                f"  No signal: edge {yes_edge:.1%} < min {self.cfg.min_edge:.1%} "
                f"(model={model_prob:.0%}, market={market.yes_price:.0%})"
            )
            return None
 
        if confidence < self.cfg.min_confidence:
            log.info(f"  No signal: confidence {confidence:.0%} < min {self.cfg.min_confidence:.0%}")
            return None
 
        # Expected Value: EV = p * win_amount - (1-p) * lose_amount
        win_prob = model_prob if side == "yes" else 1 - model_prob
        ev       = win_prob * (1 - mkt_price) - (1 - win_prob) * mkt_price
 
        signal = TradeSignal(
            market_id         = market.market_id,
            title             = market.title,
            side              = side,
            model_probability = model_prob,
            market_price      = mkt_price,
            edge              = edge,
            confidence        = confidence,
            expected_value    = ev,
            reasoning         = reasoning,
            days_to_expiry    = market.days_to_expiry,
        )
        log.info(
            f"  SIGNAL: {side.upper()} | edge={edge:.1%} | "
            f"EV={ev:.3f} | conf={confidence:.0%} | "
            f"model={model_prob:.0%} vs market={market.yes_price:.0%}"
        )
        return signal
 
    async def _call_claude(self, market: Market, brief: ResearchBrief) -> Optional[dict]:
        """Call Claude to get probability estimate. Returns parsed JSON or None."""
        raw_text = ""
        try:
            prompt = self._build_prompt(market, brief)
            response = await self.client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 800,
                system     = PREDICTOR_SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text.strip()
            if not raw_text:
                log.warning(f"Predictor: Claude returned empty response for {market.market_id}")
                return None
            text = raw_text
            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1]
                    if text.startswith("json"):
                        text = text[4:]
            return json.loads(text)
        except json.JSONDecodeError as exc:
            import re
            match = re.search(r'\{[^{}]*"probability"[^{}]*\}', raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            log.warning(
                f"Predictor JSON parse failed for {market.market_id}: {exc} | "
                f"Raw response: {raw_text[:300]!r}"
            )
            return None
        except Exception as exc:
            log.warning(f"Predictor Claude call failed for {market.market_id}: {exc}")
            return None
 
    def _build_prompt(self, market: Market, brief: ResearchBrief) -> str:
        facts = "\n".join(f"  • {f}" for f in brief.key_facts) or "  (none found)"
 
        structure_hint = ""
        if _is_tie_market(market):
            structure_hint = (
                "\nMARKET STRUCTURE NOTE: This market contains TIE or DRAW in its ID/title. "
                "YES = a tie/draw occurs. NO = there is a decisive winner. "
                "Make sure your probability reflects the chance of a TIE, not a win."
            )
 
        live_note = ""
        if market.days_to_expiry < LIVE_GAME_THRESHOLD_DAYS:
            live_note = (
                "\nNOTE: This market expires very soon and may be live/in-progress. "
                "Account for remaining time and variance — do not treat current score "
                "as a certain outcome."
            )
 
        return f"""
Prediction market question: {market.title}
Market ID: {market.market_id}
 
Market data:
  Current YES price: {market.yes_price:.0%}
  Current NO price:  {market.no_price:.0%}
  Days to resolution: {market.days_to_expiry:.1f}
  Category: {market.category}
 
Research summary:
  Sentiment score: {brief.sentiment_score:+.2f} (range -1 to +1)
  Headline: {brief.headline_summary}
 
Key facts:
{facts}
{structure_hint}{live_note}
Task: First state what YES and NO mean for this specific market.
Then estimate the true probability that this market resolves YES.
Ignore the current market price when forming your initial estimate.
Then consider whether the current price is too high or too low.
""".strip()