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

Always respond with a JSON object in this EXACT format:
{
  "probability": <float 0.0 to 1.0, your YES probability estimate>,
  "confidence":  <float 0.5 to 1.0, how confident you are in this estimate>,
  "reasoning":   "<2-3 sentence explanation of your reasoning>"
}

probability: your estimate of the true YES probability
confidence: 0.5 = very uncertain, 1.0 = highly certain
"""


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

        prediction = await self._call_claude(market, brief)
        if not prediction:
            return None

        model_prob = prediction["probability"]
        confidence = prediction["confidence"]
        reasoning  = prediction["reasoning"]

        # Determine which side to trade (YES or NO)
        # If we think YES is more likely than priced: buy YES
        # If we think NO is more likely (YES less likely than priced): buy NO
        yes_edge = model_prob - market.yes_price
        no_edge  = (1 - model_prob) - market.no_price  # same as -(yes_edge)

        if abs(yes_edge) >= abs(no_edge):
            side      = "yes"
            edge      = yes_edge
            mkt_price = market.yes_price
        else:
            side      = "no"
            edge      = no_edge
            mkt_price = market.no_price

        # Gate: only signal if edge and confidence are high enough
        if edge < self.cfg.min_edge:
            log.info(
                f"  No signal: edge {edge:.1%} < min {self.cfg.min_edge:.1%} "
                f"(model={model_prob:.0%}, market={market.yes_price:.0%})"
            )
            return None

        if confidence < self.cfg.min_confidence:
            log.info(f"  No signal: confidence {confidence:.0%} < min {self.cfg.min_confidence:.0%}")
            return None

        # Expected Value: EV = p * win_amount - (1-p) * lose_amount
        # For a $1 contract: win = (1 - price), lose = price
        win_prob  = model_prob if side == "yes" else 1 - model_prob
        ev        = win_prob * (1 - mkt_price) - (1 - win_prob) * mkt_price

        signal = TradeSignal(
            market_id        = market.market_id,
            title            = market.title,
            side             = side,
            model_probability= model_prob,
            market_price     = mkt_price,
            edge             = edge,
            confidence       = confidence,
            expected_value   = ev,
            reasoning        = reasoning,
        )
        log.info(
            f"  SIGNAL: {side.upper()} | edge={edge:.1%} | "
            f"EV={ev:.3f} | conf={confidence:.0%} | "
            f"model={model_prob:.0%} vs market={market.yes_price:.0%}"
        )
        return signal

    async def _call_claude(self, market: Market, brief: ResearchBrief) -> Optional[dict]:
        """Call Claude to get probability estimate. Returns parsed JSON or None."""
        try:
            prompt = self._build_prompt(market, brief)
            response = await self.client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 400,
                system     = PREDICTOR_SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception as exc:
            log.warning(f"Predictor Claude call failed: {exc}")
            return None

    def _build_prompt(self, market: Market, brief: ResearchBrief) -> str:
        facts = "\n".join(f"  • {f}" for f in brief.key_facts) or "  (none found)"
        return f"""
Prediction market question: {market.title}

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

Task: Estimate the true probability that this market resolves YES.
Ignore the current market price when forming your initial estimate.
Then consider whether the current price is too high or too low.
""".strip()
