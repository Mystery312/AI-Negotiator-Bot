from __future__ import annotations
import os, logging, json, re, argparse, collections, textwrap, random
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

import torch                          # NEW – for PreferenceEstimator
from neo4j import GraphDatabase
from openai import OpenAI             # keep low‑level SDK

# ─────────────────────────────────────────────────────────────────────────────
#  Project‑local imports
# ─────────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
import os
import sys

# Add parent directory to sys.path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try package imports first, then fall back to local imports
try:
    from app.graph import fetch_last_n
    from app.rag import retrieve_rag_context
    from app.casino_rag import get_casino_context
    from app.pareto import best_offer, utility  # NEW – Pareto helper
    from app.preference import load_pref_model, estimate_preferences  # NEW
except ImportError:
    # Fall back to local imports when running as script
    from graph import fetch_last_n
    from rag import retrieve_rag_context
    from casino_rag import get_casino_context
    from pareto import best_offer, utility  # NEW – Pareto helper
    from preference import load_pref_model, estimate_preferences  # NEW


# Load environment variables
load_dotenv()

# Verbosity flag for production vs debug
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Conciseness helper

MAX_WORDS = 40
def _concise(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")


# Preference Estimator (once at import) – re‑uses the training ckpt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREF_CKPT = os.getenv("PREF_MODEL_CKPT", "checkpoints/pref_estimator.pt")
try:
    pref_model, pref_tok = load_pref_model(PREF_CKPT, device=DEVICE)
    logger.info("PreferenceEstimator loaded from %s", PREF_CKPT)
except FileNotFoundError:
    pref_model = pref_tok = None
    logger.warning("PreferenceEstimator checkpoint not found – Pareto suggestions disabled.")

def estimate_preferences(turn_texts: List[str]) -> tuple[Optional[List[float]], Optional[List[float]]]:
    """Infer (my_weights, opp_weights) from recent utterances.
    Returns (None, None) if model is missing."""
    if pref_model is None:
        return None, None
    text = " ".join(turn_texts[-6:])
    enc = pref_tok(text, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        w_me, w_opp = pref_model(enc["input_ids"], enc["attention_mask"])
    return w_me.squeeze().tolist(), w_opp.squeeze().tolist()


# (all existing taxonomy / scoring / util sections remain *unchanged*)
#   … large block omitted for brevity – everything up to get_advice() stays


# Rich Negotiation Move Taxonomy
NEGOTIATION_MOVES = {
    # Information moves
    "INFO_GATHER": "information_gathering",
    "INFO_SHARE": "information_sharing", 
    "INFO_REQUEST": "information_request",
    "INFO_DISCLOSE": "information_disclosure",
    
    # Value creation moves
    "EXPAND_PIE": "expand_pie",
    "INTEGRATIVE": "integrative_bargaining",
    "CREATIVE_SOLUTION": "creative_solution",
    "MUTUAL_GAIN": "mutual_gain_focus",
    
    # Value claiming moves
    "DISTRIBUTIVE": "distributive_bargaining",
    "HARD_BALL": "hard_ball_tactics",
    "POSITIONAL": "positional_bargaining",
    "COMPETITIVE": "competitive_approach",
    
    # Concession moves
    "CONCESSION": "concession",
    "CONDITIONAL_CONCESSION": "conditional_concession",
    "GRADUAL_CONCESSION": "gradual_concession",
    "RECIPROCAL_CONCESSION": "reciprocal_concession",
    
    # Pressure moves
    "DEADLINE": "deadline_pressure",
    "ULTIMATUM": "ultimatum",
    "WALK_AWAY": "walk_away_threat",
    "ESCALATION": "escalation",
    
    # Relationship moves
    "BUILD_TRUST": "build_trust",
    "APPEAL_EMOTION": "emotional_appeal",
    "RELATIONSHIP_FOCUS": "relationship_focus",
    "COLLABORATIVE": "collaborative_approach",
    
    # Communication moves
    "SUMMARIZE": "summarize",
    "CLARIFY": "clarify",
    "REFORMULATE": "reformulate",
    "ACTIVE_LISTEN": "active_listening",
    
    # Strategic moves
    "ANCHOR": "anchoring",
    "FRAMING": "framing",
    "NORM_APPEAL": "normative_appeal",
    "PRECEDENT": "precedent_reference",
    
    # Defensive moves
    "DEFEND_POSITION": "defend_position",
    "COUNTER_OFFER": "counter_offer",
    "REJECT": "reject_offer",
    "STALL": "stalling",
    
    # Closure moves
    "ACCEPT": "accept_offer",
    "FINAL_OFFER": "final_offer",
    "AGREEMENT": "agreement_reached",
    "CLOSE_DEAL": "close_deal"
}

# Move categories for analysis
MOVE_CATEGORIES = {
    "information": ["INFO_GATHER", "INFO_SHARE", "INFO_REQUEST", "INFO_DISCLOSE"],
    "value_creation": ["EXPAND_PIE", "INTEGRATIVE", "CREATIVE_SOLUTION", "MUTUAL_GAIN"],
    "value_claiming": ["DISTRIBUTIVE", "HARD_BALL", "POSITIONAL", "COMPETITIVE"],
    "concession": ["CONCESSION", "CONDITIONAL_CONCESSION", "GRADUAL_CONCESSION", "RECIPROCAL_CONCESSION"],
    "pressure": ["DEADLINE", "ULTIMATUM", "WALK_AWAY", "ESCALATION"],
    "relationship": ["BUILD_TRUST", "APPEAL_EMOTION", "RELATIONSHIP_FOCUS", "COLLABORATIVE"],
    "communication": ["SUMMARIZE", "CLARIFY", "REFORMULATE", "ACTIVE_LISTEN"],
    "strategic": ["ANCHOR", "FRAMING", "NORM_APPEAL", "PRECEDENT"],
    "defensive": ["DEFEND_POSITION", "COUNTER_OFFER", "REJECT", "STALL"],
    "closure": ["ACCEPT", "FINAL_OFFER", "AGREEMENT", "CLOSE_DEAL"]
}

# Move intensity levels
MOVE_INTENSITY = {
    "low": ["INFO_GATHER", "INFO_SHARE", "CLARIFY", "ACTIVE_LISTEN", "BUILD_TRUST"],
    "medium": ["CONCESSION", "CONDITIONAL_CONCESSION", "SUMMARIZE", "REFORMULATE", "ANCHOR"],
    "high": ["HARD_BALL", "ULTIMATUM", "ESCALATION", "WALK_AWAY", "FINAL_OFFER"]
}

# Strategy dataclass for pluggable advice
@dataclass
class Strategy:
    name: str
    trigger: Callable[[Dict[str, Any]], bool]
    advice: str

# Numeric offer tracking
@dataclass
class Offer:
    """Track numeric offers and concessions."""
    value: float
    currency: str = "USD"
    offer_type: str = "price"  # price, quantity, discount, etc.
    conditions: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Concession:
    """Track concession amounts and types."""
    amount: float
    concession_type: str  # monetary, percentage, quantity, etc.
    from_value: float
    to_value: float
    conditions: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

def extract_numeric_offers(text: str) -> List[Offer]:
    """
    Extract numeric offers from text using regex patterns.
    
    Args:
        text: Text to analyze for offers
        
    Returns:
        List of Offer objects
    """
    import re
    
    offers = []
    
    # Patterns for different types of offers
    patterns = {
        'price': r'\$?(\d+(?:\.\d{2})?)\s*(?:dollars?|USD)?',
        'percentage': r'(\d+(?:\.\d{1,2})?)%\s*(?:discount|off|reduction)?',
        'quantity': r'(\d+)\s*(?:units?|items?|pieces?)',
        'time': r'(\d+)\s*(?:days?|weeks?|months?)'
    }
    
    for offer_type, pattern in patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            value = float(match.group(1))
            offers.append(Offer(
                value=value,
                offer_type=offer_type,
                timestamp=datetime.now()
            ))
    
    return offers

def track_concession_history(conv_id: str, speaker: str, current_offers: List[Offer], 
                           previous_offers: List[Offer]) -> List[Concession]:
    """
    Track concession history between offer rounds.
    
    Args:
        conv_id: Conversation identifier
        speaker: Speaker identifier
        current_offers: Current round of offers
        previous_offers: Previous round of offers
        
    Returns:
        List of Concession objects
    """
    concessions = []
    
    # Group offers by type for comparison
    current_by_type = {}
    previous_by_type = {}
    
    for offer in current_offers:
        if offer.offer_type not in current_by_type:
            current_by_type[offer.offer_type] = []
        current_by_type[offer.offer_type].append(offer)
    
    for offer in previous_offers:
        if offer.offer_type not in previous_by_type:
            previous_by_type[offer.offer_type] = []
        previous_by_type[offer.offer_type].append(offer)
    
    # Compare offers by type
    for offer_type in current_by_type:
        if offer_type in previous_by_type:
            current_values = [o.value for o in current_by_type[offer_type]]
            previous_values = [o.value for o in previous_by_type[offer_type]]
            
            # Find the most significant change
            if current_values and previous_values:
                current_avg = sum(current_values) / len(current_values)
                previous_avg = sum(previous_values) / len(previous_values)
                
                if current_avg != previous_avg:
                    concession_amount = abs(current_avg - previous_avg)
                    concession_type = "increase" if current_avg > previous_avg else "decrease"
                    
                    concessions.append(Concession(
                        amount=concession_amount,
                        concession_type=concession_type,
                        from_value=previous_avg,
                        to_value=current_avg,
                        timestamp=datetime.now()
                    ))
    
    return concessions

def calculate_concession_rate(concessions: List[Concession], time_window_hours: int = 24) -> float:
    """
    Calculate the rate of concessions over time.
    
    Args:
        concessions: List of Concession objects
        time_window_hours: Time window for calculation
        
    Returns:
        Concession rate (concessions per hour)
    """
    if not concessions:
        return 0.0
    
    cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
    recent_concessions = [c for c in concessions if c.timestamp > cutoff_time]
    
    if not recent_concessions:
        return 0.0
    
    time_span = (datetime.now() - cutoff_time).total_seconds() / 3600  # hours
    return len(recent_concessions) / time_span

def analyze_offer_progression(offers: List[Offer]) -> Dict[str, Any]:
    """
    Analyze the progression of offers over time.
    
    Args:
        offers: List of Offer objects
        
    Returns:
        Dictionary with analysis results
    """
    if not offers:
        return {"error": "No offers to analyze"}
    
    # Group by offer type
    by_type = {}
    for offer in offers:
        if offer.offer_type not in by_type:
            by_type[offer.offer_type] = []
        by_type[offer.offer_type].append(offer)
    
    analysis = {}
    
    for offer_type, type_offers in by_type.items():
        # Sort by timestamp
        sorted_offers = sorted(type_offers, key=lambda x: x.timestamp)
        values = [o.value for o in sorted_offers]
        
        if len(values) >= 2:
            # Calculate trends
            trend = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
            
            # Calculate volatility (standard deviation)
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            volatility = variance ** 0.5
            
            # Calculate concession frequency
            concessions = 0
            for i in range(1, len(values)):
                if values[i] != values[i-1]:
                    concessions += 1
            
            analysis[offer_type] = {
                "trend": trend,
                "volatility": volatility,
                "concession_frequency": concessions,
                "initial_value": values[0],
                "final_value": values[-1],
                "total_change": values[-1] - values[0],
                "offer_count": len(values)
            }
    
    return analysis

# Strategy definitions using the new dataclass approach
STRATEGIES = [
    Strategy(
        name="TitForTat",
        trigger=lambda scores: scores.get("reciprocity", 0) > 0.6 and scores.get("comp_opp", 0) > 0.4 and scores.get("volatility", 0) < 0.3,
        advice="Maintain tit-for-tat reciprocity while gradually building trust through small concessions."
    ),
    Strategy(
        name="EscalateDeadlock",
        trigger=lambda scores: scores.get("comp_opp", 0) > 0.7 and scores.get("volatility", 0) < 0.2 and scores.get("reciprocity", 0) < 0.3,
        advice="Escalate—signal consequences if the deadlock persists."
    ),
    Strategy(
        name="PackageOffer", 
        trigger=lambda scores: scores.get("coop_opp", 0) > 0.5 and scores.get("phase", "") == "bargaining",
        advice="Propose a package deal that addresses multiple interests simultaneously."
    ),
    Strategy(
        name="MirrorCoop",
        trigger=lambda scores: scores.get("coop_opp", 0) > 0.7 and scores.get("coop_me", 0) < 0.5,
        advice="Mirror their cooperation with a matching concession."
    ),
    Strategy(
        name="StabilizeVolatile",
        trigger=lambda scores: scores.get("volatility", 0) > 0.6,
        advice="Stabilise talks by summarising points of agreement."
    ),
    Strategy(
        name="BuildMomentum",
        trigger=lambda scores: 0.4 <= scores.get("coop_me", 0) <= 0.6 and 0.4 <= scores.get("coop_opp", 0) <= 0.6,
        advice="Propose a small concession to build momentum toward agreement."
    ),
    Strategy(
        name="GatherInfo",
        trigger=lambda scores: scores.get("move_count", 0) <= 2,
        advice="Start by gathering information about their priorities and constraints."
    ),
    Strategy(
        name="TestFlexibility",
        trigger=lambda scores: scores.get("coop_opp", 0) < 0.4 and scores.get("volatility", 0) > 0.4,
        advice="Test their flexibility with a small concession to gauge response."
    ),
    Strategy(
        name="CreateValue",
        trigger=lambda scores: scores.get("coop_me", 0) > 0.6 and scores.get("coop_opp", 0) > 0.6,
        advice="Build on the positive momentum by exploring deeper interests."
    ),
    Strategy(
        name="BreakPattern",
        trigger=lambda scores: scores.get("reciprocity", 0) < 0.3,
        advice="Break the pattern by making a unilateral gesture of goodwill."
    ),
    Strategy(
        name="ManageIntensity",
        trigger=lambda scores: scores.get("avg_opp_intensity", 2) > 2.5,
        advice="Respond with measured intensity to avoid escalation while maintaining position."
    ),
    # ――― NEW: Nice-Guy recovery when we're the blocker ―――
    Strategy(
        name="NiceGuyRecovery",
        trigger=lambda s: (
            s.get("comp_me", 0) > .5
            and s.get("coop_opp", 0) > .6
            and "no deal" in " ".join(s.get("opp_move_sequence", [])).lower()
            and s.get("no_deal_count", 0) >= 2
        ),
        advice=(
            "You are the one stalling the agreement.  Pivot to a **Nice-Guy counter-offer**: "
            "concede one low-value item (or a small % discount) and explicitly invite a reciprocal move. "
            "Phrase it as 'I can give ground here if we both take a step forward.'"
        )
    )
]

def rule_based_advice_enhanced(scores: Dict[str, Any], priorities: Dict[str, Dict[str, Any]] = None, current_offers: Dict[str, Dict[str, Any]] = None) -> str:
    """
    Enhanced rule-based advice using pluggable strategies with item-specific context.
    
    Args:
        scores: Dictionary with rich scoring metrics
        priorities: Dictionary of item priorities for each speaker
        current_offers: Dictionary of current offers for each speaker
        
    Returns:
        String advice based on selected strategy
    """
    logger.info(f"rule_based_advice_enhanced called with scores: {scores}")
    
    # Inject "nice-guy" (tit-for-tat) bias
    if scores.get("coop_opp", 0) > .6 and scores.get("coop_me", 0) > .4:
        mirror_strategy = next((s for s in STRATEGIES if s.name == "MirrorCoop"), None)
        if mirror_strategy:
            logger.info("Using MirrorCoop strategy due to cooperative bias")
            return mirror_strategy.advice
    
    # Check for specific strategies
    for strat in STRATEGIES:
        if strat.trigger(scores):
            logger.info(f"Selected strategy: {strat.name}")
            return strat.advice
    
    # If no strategy matches, provide context-aware fallback advice
    logger.info("No strategy matched, providing context-aware fallback")
    
    # Analyze the scores to provide better fallback advice
    coop_opp = scores.get("coop_opp", 0)
    coop_me = scores.get("coop_me", 0)
    comp_opp = scores.get("comp_opp", 0)
    comp_me = scores.get("comp_me", 0)
    reciprocity = scores.get("reciprocity", 0)
    volatility = scores.get("volatility", 0)
    
    # Include item-specific context if available
    item_context = ""
    if priorities and current_offers:
        # Extract key items and their priorities using actual item names
        you_high_priority = [data.get("item_name", item) for item, data in priorities.get("You", {}).items() 
                           if data.get("priority") == "high"]
        them_high_priority = [data.get("item_name", item) for item, data in priorities.get("Them", {}).items() 
                            if data.get("priority") == "high"]
        
        if you_high_priority and them_high_priority:
            item_context = f" Focus on {', '.join(you_high_priority)} vs {', '.join(them_high_priority)}."
        elif you_high_priority:
            item_context = f" Prioritize securing {', '.join(you_high_priority)}."
        elif them_high_priority:
            item_context = f" Address their need for {', '.join(them_high_priority)}."
    
    # Context-aware fallback advice with item-specific suggestions
    if coop_opp > 0.6 and coop_me < 0.4:
        return f"Mirror their cooperative approach with a matching concession to build trust.{item_context}"
    elif comp_opp > 0.7 and comp_me < 0.3:
        return f"Stand firm but offer a small concession to show flexibility and avoid deadlock.{item_context}"
    elif reciprocity < 0.3:
        return f"Break the negative pattern by making a unilateral gesture of goodwill.{item_context}"
    elif volatility > 0.6:
        return f"Stabilize the conversation by summarizing points of agreement and shared interests.{item_context}"
    elif coop_opp < 0.4 and comp_opp > 0.6:
        return f"Respond to their competitive stance with measured firmness while keeping options open.{item_context}"
    elif coop_me > 0.6 and coop_opp > 0.6:
        return f"Build on the positive momentum by exploring deeper interests and creative solutions.{item_context}"
    else:
        # If we have specific items, provide more targeted advice
        if item_context:
            return f"Probe their underlying interests with open-ended questions to find common ground.{item_context}"
        else:
            return f"Probe their underlying interests with open-ended questions to find common ground."

@dataclass
class StructuredAdvice:
    """Structured advice with multiple components."""
    strategy: str
    reasoning: str
    example_offer: str
    example_dialog: str
    implementation: str
    risks: List[str]
    benefits: List[str]
    confidence: float  # 0.0 to 1.0
    priority: str  # "high", "medium", "low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy": self.strategy,
            "reasoning": self.reasoning,
            "example_offer": self.example_offer,
            "example_dialog": self.example_dialog,
            "implementation": self.implementation,
            "risks": self.risks,
            "benefits": self.benefits,
            "confidence": self.confidence,
            "priority": self.priority
        }

def generate_structured_advice(scores: Dict[str, Any], strategy: Strategy) -> StructuredAdvice:
    """
    Generate structured advice with example offers and implementation guidance.
    
    Args:
        scores: Dictionary with scoring metrics
        strategy: Selected negotiation strategy
        
    Returns:
        StructuredAdvice object
    """
    # Extract key metrics for reasoning
    coop_opp = scores.get("coop_opp", 0)
    coop_me = scores.get("coop_me", 0)
    reciprocity = scores.get("reciprocity", 0)
    volatility = scores.get("volatility", 0)
    phase = scores.get("phase", "exploration")
    risk_level = scores.get("risk_level", 0)
    
    # Generate reasoning based on strategy and metrics
    reasoning = _generate_reasoning(strategy, scores)
    
    # Generate example offer based on strategy and context
    example_offer = _generate_example_offer(strategy, scores)
    
    # Generate example dialog that corresponds to the advice
    example_dialog = _generate_example_dialog(strategy, scores)
    
    # Generate implementation guidance
    implementation = _generate_implementation(strategy, scores)
    
    # Assess risks and benefits
    risks = _assess_risks(strategy, scores)
    benefits = _assess_benefits(strategy, scores)
    
    # Calculate confidence based on data quality and strategy fit
    confidence = _calculate_confidence(strategy, scores)
    
    # Determine priority based on urgency and impact
    priority = _determine_priority(strategy, scores)
    
    return StructuredAdvice(
        strategy=strategy.name,
        reasoning=reasoning,
        example_offer=example_offer,
        example_dialog=example_dialog,
        implementation=implementation,
        risks=risks,
        benefits=benefits,
        confidence=confidence,
        priority=priority
    )

def _generate_reasoning(strategy: Strategy, scores: Dict[str, Any]) -> str:
    """Generate reasoning for the selected strategy."""
    
    if strategy.name == "GatherInfo":
        return (
            "Early in the negotiation, we need to understand the other party's priorities, "
            "constraints, and underlying interests. This information will inform our strategy "
            "and help identify potential areas for value creation."
        )
    
    elif strategy.name == "TitForTat":
        return (
            f"High reciprocity ({scores.get('reciprocity', 0):.1%}) with competitive behavior "
            f"({scores.get('comp_opp', 0):.1%}) and low volatility ({scores.get('volatility', 0):.1%}) "
            "indicates a tit-for-tat pattern. The opponent is matching your moves systematically. "
            "Maintain this pattern while gradually building trust through small concessions."
        )
    
    elif strategy.name == "EscalateDeadlock":
        return (
            f"The opponent is showing high competitive behavior ({scores.get('comp_opp', 0):.1%}) "
            f"with low volatility ({scores.get('volatility', 0):.1%}) and low reciprocity "
            f"({scores.get('reciprocity', 0):.1%}), indicating a stable competitive stance. "
            "Escalation may be necessary to break the deadlock."
        )
    
    elif strategy.name == "MirrorCoop":
        return (
            f"High reciprocity ({scores.get('reciprocity', 0):.1%}) and cooperation "
            f"({scores.get('coop_opp', 0):.1%}) suggest the opponent is matching our moves. "
            "Building on this pattern can create positive momentum."
        )
    
    elif strategy.name == "StabilizeVolatile":
        return (
            f"High volatility ({scores.get('volatility', 0):.1%}) indicates the opponent "
            "is changing behavior frequently. Stabilizing the conversation can create "
            "a more predictable environment for productive negotiation."
        )
    
    elif strategy.name == "NiceGuyRecovery":
        return (
            "You have been perceived as blocking progress while the opponent remains cooperative. "
            "Switching to a measured concession (nice-guy move) rebuilds goodwill and prevents permanent breakdown."
        )
    
    else:
        return f"Applying {strategy.name} strategy based on current negotiation dynamics."

def _generate_example_offer(strategy: Strategy, scores: Dict[str, Any]) -> str:
    """Generate example offer based on strategy and context."""
    
    if strategy.name == "GatherInfo":
        return "Could you help me understand what's most important to you in this arrangement?"
    
    elif strategy.name == "TitForTat":
        return "I appreciate your approach. Let me match your gesture with a similar concession."
    
    elif strategy.name == "EscalateDeadlock":
        return "If we can't reach agreement today, I'll need to explore other options."
    
    elif strategy.name == "MirrorCoop":
        return "I'm willing to offer a 15% discount if you can commit to the full order today."
    
    elif strategy.name == "StabilizeVolatile":
        return "Let's review what we've agreed on so far and build from there."
    
    elif strategy.name == "BuildMomentum":
        return "I appreciate your cooperative approach - let me match that with a similar gesture."
    
    elif strategy.name == "CreateValue":
        return "What if we explore combining our resources to create additional value for both parties?"
    
    elif strategy.name == "BreakPattern":
        return "I'd like to make a goodwill gesture to show my commitment to finding a solution."
    
    elif strategy.name == "TestFlexibility":
        return "What are the key factors driving your position on this issue?"
    
    elif strategy.name == "PackageOffer":
        return "I'd like to propose a comprehensive package that addresses multiple aspects of our discussion."
    
    elif strategy.name == "ManageIntensity":
        return "Let me respond thoughtfully to your concerns while maintaining our core interests."
    
    elif strategy.name == "NiceGuyRecovery":
        # choose either the lowest-weighted item or a 5 % concession
        return (
            "I'm willing to include one of the lower-priority items in your share, "
            "provided we can wrap this up today."
        )

    else:
        return "I'd like to explore how we can work together on this."

def _generate_example_dialog(strategy: Strategy, scores: Dict[str, Any]) -> str:
    """Generate example dialog that corresponds to the advice given."""
    
    if strategy.name == "GatherInfo":
        return ("You: \"I'd like to better understand your perspective. What are the most important factors for you in this deal?\"\n"
                "Other Party: \"Well, delivery timeline is critical for us.\"\n"
                "You: \"I see. What makes the timeline so important for your business?\"\n"
                "Other Party: \"We have a major product launch scheduled, and we need the materials by then.\"\n"
                "You: \"That's helpful context. Are there other priorities I should know about?\"")
    
    elif strategy.name == "TitForTat":
        return ("You: \"I understand your position on the delivery timeline.\"\n"
                "Other Party: \"And I understand your concerns about the payment terms.\"\n"
                "You: \"Let me offer a small concession on the timeline if you can be flexible on payment.\"\n"
                "Other Party: \"That's reasonable. I can offer a similar concession on payment terms.\"\n"
                "You: \"Good. We're making progress by matching each other's cooperative moves.\"")
    
    elif strategy.name == "EscalateDeadlock":
        return ("You: \"I understand your position, but we seem to be at an impasse.\"\n"
                "Other Party: \"We can't move on the price.\"\n"
                "You: \"If we can't find common ground today, I'll need to explore alternative suppliers.\"\n"
                "Other Party: \"You're bluffing.\"\n"
                "You: \"I'm not bluffing. I have other options, but I'd prefer to work with you if we can find a solution.\"")
    
    elif strategy.name == "MirrorCoop":
        return ("You: \"I appreciate that you've been flexible on the payment terms.\"\n"
                "Other Party: \"We want this to work for both of us.\"\n"
                "You: \"In that spirit, I'm willing to offer a 15% discount if you can commit to the full order today.\"\n"
                "Other Party: \"That's a generous offer. Let me think about it.\"\n"
                "You: \"Take your time. I want to show that I value our partnership.\"")
    
    elif strategy.name == "StabilizeVolatile":
        return ("You: \"Let's take a step back and review what we've accomplished so far.\"\n"
                "Other Party: \"We've agreed on the basic terms.\"\n"
                "You: \"Exactly. We've made good progress on the core issues. Now let's build on that foundation.\"\n"
                "Other Party: \"What do you suggest?\"\n"
                "You: \"Let's focus on the remaining details one at a time, starting with the delivery schedule.\"")
    
    elif strategy.name == "BuildMomentum":
        return ("You: \"I see we're both interested in making this work.\"\n"
                "Other Party: \"Yes, we have common ground.\"\n"
                "You: \"To keep the momentum going, I'm willing to offer a small concession on the warranty terms.\"\n"
                "Other Party: \"That's helpful. What would that look like?\"\n"
                "You: \"I can extend the warranty from 1 year to 18 months if you can finalize the order this week.\"")
    
    elif strategy.name == "CreateValue":
        return ("You: \"What if we think beyond just price and delivery?\"\n"
                "Other Party: \"What do you have in mind?\"\n"
                "You: \"We could explore a longer-term partnership that benefits both of us.\"\n"
                "Other Party: \"How would that work?\"\n"
                "You: \"We could offer volume discounts and priority support if you commit to quarterly orders.\"")
    
    elif strategy.name == "BreakPattern":
        return ("You: \"I want to break the cycle we've been in.\"\n"
                "Other Party: \"What do you mean?\"\n"
                "You: \"We've been going back and forth on the same issues. Let me make a unilateral gesture.\"\n"
                "Other Party: \"What kind of gesture?\"\n"
                "You: \"I'm willing to accept your current offer on the main contract if you'll consider our proposal for the add-on services.\"")
    
    elif strategy.name == "TestFlexibility":
        return ("You: \"I'd like to understand your flexibility on the timeline.\"\n"
                "Other Party: \"We need it by the end of the month.\"\n"
                "You: \"What if we could deliver 80% by then and the rest a week later?\"\n"
                "Other Party: \"That might work, but we'd need a discount for the delay.\"\n"
                "You: \"I can offer a 5% discount on the delayed portion. How does that sound?\"")
    
    elif strategy.name == "PackageOffer":
        return ("You: \"Instead of negotiating each item separately, let me propose a comprehensive package.\"\n"
                "Other Party: \"What's included?\"\n"
                "You: \"The main contract, add-on services, extended warranty, and training - all at a bundled discount.\"\n"
                "Other Party: \"What's the total package price?\"\n"
                "You: \"I can offer the complete package at 20% less than if we negotiated each item individually.\"")
    
    elif strategy.name == "ManageIntensity":
        return ("You: \"I understand your concerns about the timeline.\"\n"
                "Other Party: \"It's a deal-breaker for us.\"\n"
                "You: \"I hear that. Let me respond thoughtfully while maintaining our core interests.\"\n"
                "Other Party: \"What does that mean?\"\n"
                "You: \"I can't change the timeline, but I can offer additional support to help you meet your goals within the current schedule.\"")
    
    else:
        return ("You: \"I'd like to explore how we can work together on this.\"\n"
                "Other Party: \"What do you have in mind?\"\n"
                "You: \"Let's discuss our respective needs and see where we can find common ground.\"\n"
                "Other Party: \"That sounds reasonable.\"\n"
                "You: \"Great. Let's start by understanding each other's priorities.\"")

def _generate_implementation(strategy: Strategy, scores: Dict[str, Any]) -> str:
    """Generate implementation guidance."""
    
    if strategy.name == "GatherInfo":
        return (
            "1. Ask open-ended questions about their priorities\n"
            "2. Listen actively and take notes\n"
            "3. Probe for underlying interests, not just positions\n"
            "4. Share some of your own priorities to encourage reciprocity"
        )
    
    elif strategy.name == "EscalateDeadlock":
        return (
            "1. Clearly state the consequences of continued impasse\n"
            "2. Set a firm deadline for resolution\n"
            "3. Be prepared to walk away if necessary\n"
            "4. Maintain professional tone while being firm"
        )
    
    elif strategy.name == "MirrorCoop":
        return (
            "1. Make a conditional concession\n"
            "2. Clearly state what you expect in return\n"
            "3. Monitor their response to reinforce the pattern\n"
            "4. Be prepared to escalate if they don't reciprocate"
        )
    
    elif strategy.name == "StabilizeVolatile":
        return (
            "1. Summarize points of agreement\n"
            "2. Acknowledge progress made\n"
            "3. Focus on common ground\n"
            "4. Propose structured next steps"
        )
    
    elif strategy.name == "BuildMomentum":
        return (
            "1. Offer a small, meaningful concession\n"
            "2. Frame it as building on positive momentum\n"
            "3. Request a corresponding gesture\n"
            "4. Monitor their response to maintain the pattern"
        )
    
    elif strategy.name == "CreateValue":
        return (
            "1. Explore deeper interests and needs\n"
            "2. Look for opportunities to expand the pie\n"
            "3. Propose integrative solutions\n"
            "4. Focus on mutual gains rather than concessions"
        )
    
    elif strategy.name == "BreakPattern":
        return (
            "1. Make a unilateral goodwill gesture\n"
            "2. Clearly explain your motivation\n"
            "3. Don't expect immediate reciprocation\n"
            "4. Monitor their response over time"
        )
    
    elif strategy.name == "TestFlexibility":
        return (
            "1. Propose a small concession to test their flexibility\n"
            "2. Observe their response carefully\n"
            "3. Adjust your approach based on their reaction\n"
            "4. Be prepared to escalate if they remain rigid"
        )
    
    elif strategy.name == "PackageOffer":
        return (
            "1. Bundle multiple issues into a comprehensive proposal\n"
            "2. Highlight the value of the package deal\n"
            "3. Offer a discount for the bundled approach\n"
            "4. Be prepared to negotiate individual components if needed"
        )
    
    elif strategy.name == "ManageIntensity":
        return (
            "1. Respond with measured intensity\n"
            "2. Acknowledge their concerns without conceding\n"
            "3. Maintain your core position\n"
            "4. Offer alternative solutions that address their needs"
        )
    
    elif strategy.name == "NiceGuyRecovery":
        return (
            "1. Identify a genuinely low-value item or a ≤ 5 % discount you can give up.\n"
            "2. Frame the concession as a gesture to move things forward, not as capitulation.\n"
            "3. Ask explicitly for the opponent's matching step (e.g., confirm timing, quantity, or add-on fees).\n"
            "4. If they reciprocate, lock in the tentative agreement quickly to maintain momentum."
        )

    else:
        return (
            "1. Deliver the message clearly and confidently\n"
            "2. Monitor their response\n"
            "3. Be prepared to adjust based on their reaction\n"
            "4. Follow up on any commitments made"
        )

def _assess_risks(strategy: Strategy, scores: Dict[str, Any]) -> List[str]:
    """Assess potential risks of the strategy."""
    risks = []
    
    if strategy.name == "EscalateDeadlock":
        risks.extend([
            "May damage the relationship permanently",
            "Could lead to complete breakdown of talks",
            "Might trigger defensive reactions"
        ])
    
    elif strategy.name == "MirrorCoop":
        risks.extend([
            "Opponent may not reciprocate",
            "Could be seen as weakness",
            "May set precedent for future concessions"
        ])
    
    elif strategy.name == "BreakPattern":
        risks.extend([
            "May be seen as desperation",
            "Could encourage more demands",
            "Might not be reciprocated"
        ])
    
    elif strategy.name == "BuildMomentum":
        risks.extend([
            "May be seen as too eager to make concessions",
            "Could set a pattern of one-sided concessions",
            "Might not be reciprocated"
        ])
    
    elif strategy.name == "TestFlexibility":
        risks.extend([
            "May reveal your flexibility too early",
            "Could be seen as a sign of weakness",
            "Might encourage more demands"
        ])
    
    elif strategy.name == "PackageOffer":
        risks.extend([
            "May be seen as trying to force a decision",
            "Could be rejected entirely",
            "Might complicate the negotiation unnecessarily"
        ])
    
    else:
        risks.append("Standard negotiation risks apply")
    
    # Add context-specific risks
    if scores.get("risk_level", 0) > 2:
        risks.append("High-risk negotiation environment")
    
    if scores.get("volatility", 0) > 0.5:
        risks.append("Unpredictable opponent behavior")
    
    return risks

def _assess_benefits(strategy: Strategy, scores: Dict[str, Any]) -> List[str]:
    """Assess potential benefits of the strategy."""
    benefits = []
    
    if strategy.name == "GatherInfo":
        benefits.extend([
            "Better understanding of opponent's interests",
            "Identification of potential value creation opportunities",
            "Improved strategic positioning"
        ])
    
    elif strategy.name == "CreateValue":
        benefits.extend([
            "Potential for mutual gains",
            "Strengthened relationship",
            "Creative solutions to complex problems"
        ])
    
    elif strategy.name == "MirrorCoop":
        benefits.extend([
            "Builds positive momentum",
            "Establishes cooperative pattern",
            "Increases likelihood of agreement"
        ])
    
    elif strategy.name == "StabilizeVolatile":
        benefits.extend([
            "Reduces uncertainty",
            "Creates more predictable environment",
            "Builds trust through consistency"
        ])
    
    elif strategy.name == "BuildMomentum":
        benefits.extend([
            "Creates positive momentum",
            "May lead to reciprocal concessions",
            "Builds trust and cooperation",
            "Accelerates progress toward agreement"
        ])
    
    elif strategy.name == "BreakPattern":
        benefits.extend([
            "May break negative cycles",
            "Demonstrates goodwill and commitment",
            "Could lead to reciprocal gestures",
            "May improve the overall atmosphere"
        ])
    
    elif strategy.name == "TestFlexibility":
        benefits.extend([
            "Gains valuable information about their flexibility",
            "May reveal hidden opportunities",
            "Helps calibrate your approach",
            "May lead to unexpected concessions"
        ])
    
    elif strategy.name == "PackageOffer":
        benefits.extend([
            "May accelerate agreement",
            "Creates value through bundling",
            "Simplifies complex negotiations",
            "May lead to better overall terms"
        ])
    
    elif strategy.name == "ManageIntensity":
        benefits.extend([
            "Prevents escalation",
            "Maintains professional relationship",
            "Demonstrates emotional control",
            "May lead to more productive discussions"
        ])
    
    elif strategy.name == "EscalateDeadlock":
        benefits.extend([
            "May break through resistance",
            "Could accelerate decision-making",
            "Demonstrates seriousness and commitment",
            "May reveal their true bottom line"
        ])
    
    else:
        benefits.append("Advances negotiation objectives")
    
    # Add context-specific benefits
    if scores.get("coop_opp", 0) > 0.6:
        benefits.append("Opponent appears cooperative")
    
    if scores.get("trust_level", 0) > 0.3:
        benefits.append("Some trust already established")
    
    return benefits

def _calculate_confidence(strategy: Strategy, scores: Dict[str, Any]) -> float:
    """Calculate confidence level in the strategy recommendation."""
    confidence = 0.5  # Base confidence
    
    # Adjust based on data quality
    move_count = scores.get("move_count", 0)
    if move_count >= 6:
        confidence += 0.2
    elif move_count >= 3:
        confidence += 0.1
    
    # Adjust based on strategy fit
    if strategy.trigger(scores):
        confidence += 0.2
    
    # Adjust based on risk level
    risk_level = scores.get("risk_level", 0)
    if risk_level == 0:
        confidence += 0.1
    elif risk_level > 3:
        confidence -= 0.2
    
    # Adjust based on volatility
    volatility = scores.get("volatility", 0)
    if volatility < 0.3:
        confidence += 0.1
    elif volatility > 0.7:
        confidence -= 0.1
    
    return min(max(confidence, 0.0), 1.0)

def _determine_priority(strategy: Strategy, scores: Dict[str, Any]) -> str:
    """Determine priority level of the advice."""
    
    # High priority conditions
    if strategy.name == "EscalateDeadlock":
        return "high"
    
    if scores.get("risk_level", 0) > 2:
        return "high"
    
    if scores.get("comp_opp", 0) > 0.7:
        return "high"
    
    # Medium priority conditions
    if scores.get("volatility", 0) > 0.5:
        return "medium"
    
    if scores.get("phase", "") == "closing":
        return "medium"
    
    # Default to low priority
    return "low"

def get_structured_advice(conv_id: str, speaker: str, model: str = "qwen3:latest") -> Dict[str, Any]:
    """
    Get structured negotiation advice with example offers and implementation guidance.
    
    Args:
        conv_id: Conversation identifier
        speaker: Speaker identifier
        model: Model to use for advice generation
        
    Returns:
        Dictionary with structured advice
    """
    try:
        # Check if deal has been reached
        deal_reached = check_deal_reached(conv_id)
        
        if deal_reached:
            # Return closure advice
            return {
                "strategy": "Closure",
                "reasoning": "Agreement has been reached, focus on proper closure",
                "example_offer": "Let's summarize our agreement and confirm next steps.",
                "implementation": "Document the agreement and establish clear next steps.",
                "risks": ["Misunderstanding of terms", "Lack of follow-through"],
                "benefits": ["Clear agreement", "Positive relationship maintained"],
                "confidence": 0.9,
                "priority": "high"
            }
        
        # Get conversation data
        turns = fetch_last_n(conv_id, 5)
        
        if not turns:
            # No conversation history - use information gathering strategy
            scores = {"move_count": 0, "phase": "opening"}
            # Find the first strategy that triggers
            strategy = None
            for strat in STRATEGIES:
                if strat.trigger(scores):
                    strategy = strat
                    break
            if not strategy:
                strategy = STRATEGIES[5]  # GatherInfo strategy as fallback
            structured_advice = generate_structured_advice(scores, strategy)
            return structured_advice.to_dict()
        
        # Extract moves and get scores
        my_moves = [t['pd'] for t in turns if t['speaker'] == speaker]
        opp_moves = [t['pd'] for t in turns if t['speaker'] != speaker]
        
        scores = score_turns(my_moves, opp_moves)
        
        # Find the first strategy that triggers
        strategy = None
        for strat in STRATEGIES:
            if strat.trigger(scores):
                strategy = strat
                break
        if not strategy:
            strategy = STRATEGIES[5]  # GatherInfo strategy as fallback
        
        # Generate structured advice
        structured_advice = generate_structured_advice(scores, strategy)
        
        return structured_advice.to_dict()
        
    except Exception as e:
        logger.error(f"Error getting structured advice: {e}")
        return {
            "strategy": "Fallback",
            "reasoning": "Unable to analyze negotiation context",
            "example_offer": "I'd like to continue our discussion.",
            "implementation": "Proceed with caution and gather more information.",
            "risks": ["Limited information available"],
            "benefits": ["Safe approach"],
            "confidence": 0.3,
            "priority": "low"
        }

# Initialize LLM client (robust import for package vs script)
try:
    from app.llm_client import create_llm_client, get_provider_from_model
except Exception:  # ModuleNotFoundError in some run modes
    from llm_client import create_llm_client, get_provider_from_model

# Default client (will be updated based on model selection)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3:latest")
provider = get_provider_from_model(DEFAULT_MODEL)
default_client = create_llm_client(provider, DEFAULT_MODEL)

# Initialize Neo4j driver directly
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "6xlBSIDu8Nc8gjXrpt3kNuwM7AZHGI3WJrfpN2fFDXE")

# Tightened prompt templates for focused advice
SYSTEM_PROMPT = (
    "You are a negotiation coach. Give ONE crisp, context-specific suggestion "
    "the user can say next (≈ 35 words max). "
    "Incorporate specific items, quantities, and concessions mentioned in the coach's analysis. "
    "No bullet lists, no section headers—just the sentence. "
    "Ground it in the last exchange; do not repeat earlier generic advice. "
    "IMPORTANT: Respond directly with actionable advice, not internal thoughts or explanations."
)

# USER_TEMPLATE is replaced with inline prompt in _llm_generate_reply

try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    logger.info("Connected to Neo4j database for coach module")
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {e}")
    neo4j_driver = None

def score_turns(my_moves, opp_moves):
    """
    Analyze negotiation moves using rich taxonomy instead of binary C/D.
    
    Args:
        my_moves: List of my recent moves (from NEGOTIATION_MOVES)
        opp_moves: List of opponent's recent moves (from NEGOTIATION_MOVES)
        
    Returns:
        Dictionary with rich scoring metrics
    """
    # window = last 6 turns (pad if shorter)
    m = (my_moves + ['INFO_GATHER']*6)[-6:]
    o = (opp_moves + ['INFO_GATHER']*6)[-6:]
    
    # Categorize moves by type
    def categorize_moves(moves):
        categories = {cat: 0 for cat in MOVE_CATEGORIES.keys()}
        for move in moves:
            for cat, move_list in MOVE_CATEGORIES.items():
                if move in move_list:
                    categories[cat] += 1
                    break
        return categories
    
    my_categories = categorize_moves(m)
    opp_categories = categorize_moves(o)
    
    # Calculate cooperation rates (value creation vs claiming)
    coop_rate_me = (my_categories.get('value_creation', 0) + my_categories.get('relationship', 0)) / 6
    coop_rate_opp = (opp_categories.get('value_creation', 0) + opp_categories.get('relationship', 0)) / 6
    
    # Calculate competitive rates
    comp_rate_me = (my_categories.get('value_claiming', 0) + my_categories.get('pressure', 0)) / 6
    comp_rate_opp = (opp_categories.get('value_claiming', 0) + opp_categories.get('pressure', 0)) / 6
    
    # Reciprocity - how often moves match in category
    reciprocity = 0
    for i in range(1, min(6, len(m), len(o))):
        my_cat = next((cat for cat, moves in MOVE_CATEGORIES.items() if m[i] in moves), None)
        opp_cat = next((cat for cat, moves in MOVE_CATEGORIES.items() if o[i-1] in moves), None)
        if my_cat == opp_cat:
            reciprocity += 1
    reciprocity = reciprocity / 5 if len(m) > 1 and len(o) > 1 else 0
    
    # Volatility - how often opponent changes move categories
    volatility = 0
    for i in range(1, len(o)):
        prev_cat = next((cat for cat, moves in MOVE_CATEGORIES.items() if o[i-1] in moves), None)
        curr_cat = next((cat for cat, moves in MOVE_CATEGORIES.items() if o[i] in moves), None)
        if prev_cat != curr_cat:
            volatility += 1
    volatility = volatility / 5 if len(o) > 1 else 0
    
    # Recent trend analysis
    recent_opp_moves = o[-3:]
    recent_trend = "stable"
    if len(recent_opp_moves) >= 2:
        recent_cats = [next((cat for cat, moves in MOVE_CATEGORIES.items() if move in moves), None) 
                      for move in recent_opp_moves]
        if recent_cats[-1] != recent_cats[-2]:
            recent_trend = "changing"
        elif all(cat in ['value_creation', 'relationship'] for cat in recent_cats):
            recent_trend = "cooperating"
        elif all(cat in ['value_claiming', 'pressure'] for cat in recent_cats):
            recent_trend = "competing"
    
    # Momentum - direction of negotiation
    momentum = 0
    if len(m) >= 2 and len(o) >= 2:
        my_momentum = 1 if m[-1] in MOVE_CATEGORIES['value_creation'] and m[-2] in MOVE_CATEGORIES['value_claiming'] else \
                     -1 if m[-1] in MOVE_CATEGORIES['value_claiming'] and m[-2] in MOVE_CATEGORIES['value_creation'] else 0
        opp_momentum = 1 if o[-1] in MOVE_CATEGORIES['value_creation'] and o[-2] in MOVE_CATEGORIES['value_claiming'] else \
                      -1 if o[-1] in MOVE_CATEGORIES['value_claiming'] and o[-2] in MOVE_CATEGORIES['value_creation'] else 0
        momentum = (my_momentum + opp_momentum) / 2
    
    # Count explicit "no deal" utterances in last 6 turns
    no_deal_count = sum(
        1 for t in (m + o)[-6:]
        if isinstance(t, str) and "no deal" in t.lower()
    )

    # Intensity analysis
    def get_intensity(move):
        for level, moves in MOVE_INTENSITY.items():
            if move in moves:
                return level
        return "medium"
    
    my_intensity = [get_intensity(move) for move in m]
    opp_intensity = [get_intensity(move) for move in o]
    
    # Calculate average intensity
    intensity_map = {"low": 1, "medium": 2, "high": 3}
    avg_my_intensity = sum(intensity_map.get(i, 2) for i in my_intensity) / len(my_intensity)
    avg_opp_intensity = sum(intensity_map.get(i, 2) for i in opp_intensity) / len(opp_intensity)
    
    # Enhanced metrics for strategy selection
    # Concession analysis
    concession_moves = ['CONCESSION', 'CONDITIONAL_CONCESSION', 'GRADUAL_CONCESSION', 'RECIPROCAL_CONCESSION']
    my_concessions = sum(1 for move in m if move in concession_moves)
    opp_concessions = sum(1 for move in o if move in concession_moves)
    concession_ratio = my_concessions / (my_concessions + opp_concessions + 1)  # Avoid division by zero
    
    # Pressure analysis
    pressure_moves = ['DEADLINE', 'ULTIMATUM', 'WALK_AWAY', 'ESCALATION']
    my_pressure = sum(1 for move in m if move in pressure_moves)
    opp_pressure = sum(1 for move in o if move in pressure_moves)
    pressure_ratio = opp_pressure / (my_pressure + opp_pressure + 1)
    
    # Information exchange analysis
    info_moves = ['INFO_GATHER', 'INFO_SHARE', 'INFO_REQUEST', 'INFO_DISCLOSE']
    my_info = sum(1 for move in m if move in info_moves)
    opp_info = sum(1 for move in o if move in info_moves)
    info_balance = (my_info - opp_info) / 6  # Normalized difference
    
    # Strategic positioning
    strategic_moves = ['ANCHOR', 'FRAMING', 'NORM_APPEAL', 'PRECEDENT']
    my_strategic = sum(1 for move in m if move in strategic_moves)
    opp_strategic = sum(1 for move in o if move in strategic_moves)
    strategic_advantage = (my_strategic - opp_strategic) / 6
    
    # Communication quality
    comm_moves = ['SUMMARIZE', 'CLARIFY', 'REFORMULATE', 'ACTIVE_LISTEN']
    my_comm = sum(1 for move in m if move in comm_moves)
    opp_comm = sum(1 for move in o if move in comm_moves)
    comm_quality = (my_comm + opp_comm) / 12  # Combined communication quality
    
    # Trust building
    trust_moves = ['BUILD_TRUST', 'RELATIONSHIP_FOCUS', 'COLLABORATIVE']
    my_trust = sum(1 for move in m if move in trust_moves)
    opp_trust = sum(1 for move in o if move in trust_moves)
    trust_level = (my_trust + opp_trust) / 12
    
    # Defensive posture
    defensive_moves = ['DEFEND_POSITION', 'COUNTER_OFFER', 'REJECT', 'STALL']
    my_defensive = sum(1 for move in m if move in defensive_moves)
    opp_defensive = sum(1 for move in o if move in defensive_moves)
    defensive_ratio = opp_defensive / (my_defensive + opp_defensive + 1)
    
    # Value creation vs claiming balance
    value_creation_ratio = (my_categories.get('value_creation', 0) + opp_categories.get('value_creation', 0)) / 12
    value_claiming_ratio = (my_categories.get('value_claiming', 0) + opp_categories.get('value_claiming', 0)) / 12
    value_balance = value_creation_ratio - value_claiming_ratio
    
    # Negotiation phase detection
    total_moves = len(my_moves) + len(opp_moves)
    if total_moves <= 4:
        phase = "opening"
    elif total_moves <= 12:
        phase = "exploration"
    elif total_moves <= 20:
        phase = "bargaining"
    else:
        phase = "closing"
    
    # Risk assessment
    risk_factors = []
    if comp_rate_opp > 0.7:
        risk_factors.append("high_competition")
    if volatility > 0.6:
        risk_factors.append("high_volatility")
    if pressure_ratio > 0.5:
        risk_factors.append("pressure_tactics")
    if defensive_ratio > 0.6:
        risk_factors.append("defensive_posture")
    
    risk_level = len(risk_factors)
    
    return {
        # Core metrics
        "coop_me": coop_rate_me,
        "coop_opp": coop_rate_opp,
        "comp_me": comp_rate_me,
        "comp_opp": comp_rate_opp,
        "reciprocity": reciprocity,
        "volatility": volatility,
        "last_pair": (m[-1], o[-1]),
        "recent_trend": recent_trend,
        "momentum": momentum,
        "move_count": len(my_moves) + len(opp_moves),
        "my_categories": my_categories,
        "opp_categories": opp_categories,
        "avg_my_intensity": avg_my_intensity,
        "avg_opp_intensity": avg_opp_intensity,
        "my_intensity": my_intensity,
        "opp_intensity": opp_intensity,
        
        # Enhanced metrics for strategy selection
        "concession_ratio": concession_ratio,
        "pressure_ratio": pressure_ratio,
        "info_balance": info_balance,
        "strategic_advantage": strategic_advantage,
        "comm_quality": comm_quality,
        "trust_level": trust_level,
        "defensive_ratio": defensive_ratio,
        "value_balance": value_balance,
        "phase": phase,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "no_deal_count": no_deal_count,
        
        # Move sequences for pattern analysis
        "my_move_sequence": m,
        "opp_move_sequence": o,
        
        # Category distributions
        "my_category_distribution": {k: v/6 for k, v in my_categories.items()},
        "opp_category_distribution": {k: v/6 for k, v in opp_categories.items()},
        
        # Intensity distributions
        "my_intensity_distribution": {
            "low": sum(1 for i in my_intensity if i == "low") / len(my_intensity),
            "medium": sum(1 for i in my_intensity if i == "medium") / len(my_intensity),
            "high": sum(1 for i in my_intensity if i == "high") / len(my_intensity)
        },
        "opp_intensity_distribution": {
            "low": sum(1 for i in opp_intensity if i == "low") / len(opp_intensity),
            "medium": sum(1 for i in opp_intensity if i == "medium") / len(opp_intensity),
            "high": sum(1 for i in opp_intensity if i == "high") / len(opp_intensity)
        },
        
        # New metrics for Step 5
        "concession_curve_me": 0,  # Default when data missing
        "concession_curve_opp": 0,  # Default when data missing
        "offer_variance_opp": 0,  # Default when data missing
        "aspiration_gap": 0  # Default when data missing
    }

#  DE-DUPLICATION & VARIETY -
# Keep a rolling deck of 5 phrasings per "idea" and pop at random.
ALT_CACHE = collections.defaultdict(list) # key = idea, val = list[str]

def get_alternative_advice(scores: Dict[str, Any], idea: str) -> str:
    """Return a non-repeated variant of *idea* (15-30 words)."""
    # 1) Pre-compute the variant list once per idea
    if not ALT_CACHE[idea]:
        base = [s.strip() for s in textwrap.wrap(idea, width=60)] # crude split
        # simple re-phraser – replace first verb with synonym set
        synonyms = {"Consider": ["Try", "Perhaps", "You could"],
                   "Escalate": ["Apply pressure", "Signal urgency"],
                   "Probe": ["Ask", "Clarify", "Explore"]}
        first = base[0].split()[0]
        for repl in synonyms.get(first, [first]):
            variant = base[0].replace(first, repl, 1) + (" " + " ".join(base[1:]) if len(base) > 1 else "")
            ALT_CACHE[idea].append(variant)
    # 2) Pick a variant we haven't used in the last 3 times
    history = ALT_CACHE[idea][:]
    random.shuffle(history)
    return history.pop()[:150] # trim


def summarize_turns_for_llm(turns):
    """
    Create a compressed dialogue summary for LLM context.
    
    Args:
        turns: List of conversation turns
        
    Returns:
        String summary of 1-3 sentences
    """
    if not turns:
        return "No conversation history available."
    
    if len(turns) <= 2:
        # For very short conversations, just concatenate the turns
        summary_parts = []
        for turn in turns[-2:]:
            speaker = turn.get('speaker', 'Unknown')
            move = turn.get('pd', 'Unknown')
            summary_parts.append(f"{speaker}: {move}")
        return " | ".join(summary_parts)
    
    # For longer conversations, create a more sophisticated summary
    recent_turns = turns[-4:]  # Last 4 turns
    summary_parts = []
    
    # Group by speaker and summarize patterns
    speaker_moves = {}
    for turn in recent_turns:
        speaker = turn.get('speaker', 'Unknown')
        move = turn.get('pd', 'Unknown')
        if speaker not in speaker_moves:
            speaker_moves[speaker] = []
        speaker_moves[speaker].append(move)
    
    # Create summary with more context
    for speaker, moves in speaker_moves.items():
        if len(moves) >= 2:
            pattern = "→".join(moves[-2:])  # Last 2 moves
            # Add context about the move type
            last_move = moves[-1]
            if last_move == 'C':
                context = "cooperative"
            elif last_move == 'D':
                context = "competitive"
            else:
                context = "neutral"
            summary_parts.append(f"{speaker}: {pattern} ({context})")
        else:
            move = moves[0]
            context = "cooperative" if move == 'C' else "competitive" if move == 'D' else "neutral"
            summary_parts.append(f"{speaker}: {move} ({context})")
    
    # Add overall tone assessment
    coop_count = sum(1 for turn in recent_turns if turn.get('pd') == 'C')
    total_moves = len(recent_turns)
    if total_moves > 0:
        coop_rate = coop_count / total_moves
        if coop_rate > 0.7:
            tone = "generally cooperative"
        elif coop_rate < 0.3:
            tone = "generally competitive"
        else:
            tone = "mixed"
        summary_parts.append(f"Overall tone: {tone}")
    
    return " | ".join(summary_parts)

def fetch_graph_patterns(conv_id, limit=3):
    """
    Fetch recent move patterns from Neo4j for graph-based analysis.
    
    Args:
        conv_id: Conversation identifier
        limit: Number of recent moves to fetch
        
    Returns:
        List of recent moves (C/D)
    """
    try:
        if not neo4j_driver:
            return []
            
        with neo4j_driver.session() as session:
            query = """
            MATCH (c:Conv {id: $conv_id})
            OPTIONAL MATCH (c)-[:HAS_TURN]->(t:Turn)
            WITH t ORDER BY t.turn_number DESC
            LIMIT $limit
            RETURN collect(t.pd) AS last_moves
            """
            
            result = session.run(query, {"conv_id": conv_id, "limit": limit})
            record = result.single()
            
            if record and record["last_moves"]:
                return list(reversed(record["last_moves"]))  # Reverse to get chronological order
            else:
                return []
                
    except Exception as e:
        logger.error(f"Error fetching graph patterns: {e}")
        return []

def check_recent_advice(conv_id, speaker, advice_text, limit=3):
    """
    Check if the same advice was given recently to avoid repetition.
    
    Args:
        conv_id: Conversation identifier
        speaker: Speaker identifier
        advice_text: The advice text to check
        limit: Number of recent advice to check
        
    Returns:
        Boolean indicating if advice was recently given
    """
    try:
        if not neo4j_driver:
            return False
            
        with neo4j_driver.session() as session:
            query = """
            MATCH (p:Person {name: $speaker})
            OPTIONAL MATCH (p)-[:GAVE_ADVICE]->(a:Advice)
            WHERE a.text = $advice_text
            WITH a ORDER BY a.ts DESC
            LIMIT $limit
            RETURN count(a) as count
            """
            
            result = session.run(query, {"speaker": speaker, "advice_text": advice_text, "limit": limit})
            record = result.single()
            
            return record and record["count"] > 0
                
    except Exception as e:
        logger.error(f"Error checking recent advice: {e}")
        return False

def store_advice(conv_id, speaker, advice_text):
    """
    Store advice in the graph database to track repetition.
    
    Args:
        conv_id: Conversation identifier
        speaker: Speaker identifier
        advice_text: The advice text to store
    """
    try:
        if not neo4j_driver:
            return
            
        with neo4j_driver.session() as session:
            query = """
            MATCH (p:Person {name: $speaker})
            CREATE (a:Advice {text: $advice_text, ts: datetime()})
            CREATE (p)-[:GAVE_ADVICE]->(a)
            """
            
            session.run(query, {"speaker": speaker, "advice_text": advice_text})
                
    except Exception as e:
        logger.error(f"Error storing advice: {e}")

def extract_strategy_from_hint(hint: str) -> str:
    """
    Extract strategy from advice hint for CaSiNo RAG.
    
    Args:
        hint: Advice hint string
        
    Returns:
        Strategy string for CaSiNo retrieval
    """
    hint_lower = hint.lower()
    
    if any(word in hint_lower for word in ['escalate', 'consequence', 'deadlock']):
        return 'escalation'
    elif any(word in hint_lower for word in ['concession', 'conditional']):
        return 'concession'
    elif any(word in hint_lower for word in ['stabilise', 'summarise', 'agreement']):
        return 'stabilization'
    elif any(word in hint_lower for word in ['mirror', 'cooperation', 'matching']):
        return 'mirroring'
    elif any(word in hint_lower for word in ['momentum', 'explore', 'interests']):
        return 'exploration'
    elif any(word in hint_lower for word in ['goodwill', 'unilateral', 'gesture']):
        return 'goodwill'
    elif any(word in hint_lower for word in ['question', 'reveal', 'interests']):
        return 'information_gathering'
    else:
        return 'general'

def get_casino_dialogue_examples(strategy: str, recent_moves: List[str]) -> str:
    """
    Get CaSiNo dialogue examples for few-shot learning.
    
    Args:
        strategy: Strategy string for CaSiNo retrieval
        recent_moves: Recent power dynamics moves
        
    Returns:
        String with few-shot examples or empty string if none found
    """
    try:
        # Get CaSiNo context
        casino_context = get_casino_context(strategy, recent_moves)
        
        if not casino_context or casino_context == "No relevant CaSiNo examples found.":
            return ""
        
        # Parse the context to extract dialogue examples
        # The context should contain dialogue examples in a structured format
        # For now, we'll use the context as-is, but in a real implementation
        # we'd parse it to extract specific utterance pairs
        
        # Extract dialogue examples from the context
        # This is a simplified version - in practice, you'd parse the CaSiNo data
        # to extract actual utterance pairs that led to successful deals
        
        # For demonstration, we'll create a few-shot example based on the strategy
        if strategy == 'concession':
            few_shot = f"""
Here is how humans phrased a similar move:
H1: "I'm willing to offer a 15% discount if you can commit to the full order today."
H2: "Let's split the difference - I'll reduce my price by 10% if you increase your order size."
"""
        elif strategy == 'escalation':
            few_shot = f"""
Here is how humans phrased a similar move:
H1: "If we can't reach agreement today, I'll need to explore other options."
H2: "The current impasse is costing both of us - we need to find a solution."
"""
        elif strategy == 'stabilization':
            few_shot = f"""
Here is how humans phrased a similar move:
H1: "Let's review what we've agreed on so far and build from there."
H2: "We've made good progress - let's focus on the areas where we align."
"""
        elif strategy == 'mirroring':
            few_shot = f"""
Here is how humans phrased a similar move:
H1: "I appreciate your cooperative approach - let me match that with a similar gesture."
H2: "Your willingness to work together is encouraging - I'll reciprocate that spirit."
"""
        elif strategy == 'exploration':
            few_shot = f"""
Here is how humans phrased a similar move:
H1: "What are the key factors driving your position on this issue?"
H2: "Help me understand your priorities so we can find common ground."
"""
        elif strategy == 'goodwill':
            few_shot = f"""
Here is how humans phrased a similar move:
H1: "I'd like to make a goodwill gesture to show my commitment to this deal."
H2: "Let me start by offering something that demonstrates my sincerity."
"""
        elif strategy == 'information_gathering':
            few_shot = f"""
Here is how humans phrased a similar move:
H1: "What would be most valuable to you in this arrangement?"
H2: "Can you help me understand your key concerns and priorities?"
"""
        else:
            few_shot = f"""
Here is how humans phrased a similar move:
H1: "I'd like to explore how we can work together on this."
H2: "Let's find a solution that works for both of us."
"""
        
        return few_shot
        
    except Exception as e:
        logger.error(f"Error getting CaSiNo dialogue examples: {e}")
        return ""

def clean_llm_response(response_text):
    """
    Clean up LLM response for descriptive advice.
    
    Args:
        response_text: Raw response from LLM
        
    Returns:
        Cleaned descriptive advice
    """
    logger.info(f"clean_llm_response called with: '{response_text}'")
    if not response_text:
        logger.info("Empty response_text, returning default")
        return "Consider starting with information sharing to build trust and understand their priorities."
    
    # Remove <think> tags and their content
    cleaned = response_text.strip()
    
    # Remove <think> tags and their content
    import re
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)  # Handle unclosed tags
    
    # Clean up any remaining whitespace
    cleaned = cleaned.strip()
    
    logger.info(f"After removing think tags: '{cleaned}'")
    
    # For descriptive advice, we want to keep the full explanation
    # Increased limit to handle longer, more detailed responses (5000 chars as requested)
    if len(cleaned) > 10:
        # Allow much longer responses for detailed advice
        if len(cleaned) > 5000:
            # Only truncate if extremely long, and do it more intelligently
            sentences = cleaned.split('.')
            if len(sentences) > 1:
                # Take first 8-10 sentences for very long responses
                truncated = '. '.join(sentences[:8]) + '.'
                if len(truncated) <= 5000:
                    result = truncated
                else:
                    # If still too long, take first 6 sentences
                    truncated = '. '.join(sentences[:6]) + '.'
                    result = truncated[:5000]
            else:
                result = cleaned[:5000]
        else:
            result = cleaned
        
        # Enforce conciseness for short replies only (not for long advice)
        result_short = _concise(result, MAX_WORDS)
        logger.info(f"Final result: '{result_short}'")
        return result_short
    
    # If the cleaned response is too short, return a varied default
    import random
    fallback_advice = [
        "Consider starting with information sharing to build trust and understand their priorities.",
        "Try asking open-ended questions to explore their underlying interests and constraints.",
        "Begin by establishing common ground and identifying shared objectives.",
        "Start with a collaborative approach to understand their perspective and needs.",
        "Focus on building rapport before diving into specific terms and conditions.",
        "Propose a small concession to demonstrate flexibility and build momentum.",
        "Ask about their priorities and constraints to find mutually beneficial solutions.",
        "Summarize what you understand so far and invite their perspective.",
        "Explore creative options that might satisfy both parties' core interests.",
        "Take a step back and identify the key issues that need to be resolved."
    ]
    result = random.choice(fallback_advice)
    
    # Enforce conciseness
    result = _concise(result, MAX_WORDS)
    logger.info(f"Using fallback advice: '{result}'")
    return result

def clean_llm_response_long(response_text: str) -> str:
    """Clean up LLM response for long, justified advice without word trimming.
    - Strips <think> tags and surrounding whitespace
    - Returns up to ~5000 characters to keep payload reasonable
    """
    logger.info(f"clean_llm_response_long called with: '{response_text[:120] + ('…' if len(response_text)>120 else '')}'")
    if not response_text:
        return "Provide a brief justification tied to the conversation, then a concrete recommendation that names items and exact quantities."
    import re as _re
    cleaned = response_text.strip()
    cleaned = _re.sub(r'<think>.*?</think>', '', cleaned, flags=_re.DOTALL)
    cleaned = _re.sub(r'<think>.*', '', cleaned, flags=_re.DOTALL)
    cleaned = cleaned.strip()
    if len(cleaned) <= 5000:
        return cleaned
    # Smart-ish truncation by sentences
    sentences = cleaned.split('.')
    if len(sentences) > 6:
        return '.'.join(sentences[:6]).strip() + '.'
    return cleaned[:5000]

def check_deal_reached(conv_id):
    """
    Check if a deal has been reached in the conversation.
    
    Args:
        conv_id: Conversation identifier
        
    Returns:
        Boolean indicating if deal was reached
    """
    try:
        if not neo4j_driver:
            return False
            
        with neo4j_driver.session() as session:
            # Check for outcome indicating deal reached - use OPTIONAL MATCH to avoid warnings
            query = """
            MATCH (c:Conv {id: $conv_id})
            OPTIONAL MATCH (c)-[:OF_CONV]->(o:Outcome)
            WHERE o.deal_reached = true OR o.status = 'accepted'
            RETURN count(o) as deal_count
            """
            
            result = session.run(query, {"conv_id": conv_id})
            record = result.single()
            
            if record and record["deal_count"] > 0:
                return True
            
            # Also check if any turn was annotated with accept - use OPTIONAL MATCH
            query2 = """
            MATCH (c:Conv {id: $conv_id})
            OPTIONAL MATCH (c)-[:HAS_TURN]->(t:Turn)
            WHERE t.annotation = 'accept' OR (t.status IS NOT NULL AND t.status = 'accepted')
            RETURN count(t) as accept_count
            """
            
            result2 = session.run(query2, {"conv_id": conv_id})
            record2 = result2.single()
            
            return record2 and record2["accept_count"] > 0
                
    except Exception as e:
        logger.error(f"Error checking deal status: {e}")
        return False

def get_closure_advice():
    """
    Get advice for when a deal has been reached.
    
    Returns:
        String with closure advice
    """
    import random
    
    closure_options = [
        "Summarise the agreement and confirm next steps.",
        "Review the key terms we've agreed upon and outline the implementation timeline.",
        "Let's document our agreement and establish clear next steps for both parties.",
        "Confirm the main points of our agreement and discuss the follow-up process.",
        "Recap the key decisions we've made and clarify the action items for each of us.",
        "Let's summarize what we've accomplished and confirm our next steps.",
        "Review the agreement we've reached and establish our next meeting or milestone.",
        "Document the key terms of our agreement and confirm the timeline for implementation."
    ]
    
    return random.choice(closure_options)


# ⟨⟨  Unchanged: Rich move taxonomy, dataclasses Offer / Concession, rule‑based
#      strategy definitions, scoring / helper utilities …  ⟩⟩

# (The huge middle section of the file is identical to the previous version
#  you pasted – it is omitted here to keep the canvas readable.)


# ░░░   Pareto‑powered hint injection inside get_advice()   ░░░

# We add three small tweaks near the top of get_advice():
#   1.  call `estimate_preferences()` (returns None, None if model missing)
#   2.  call `best_offer()` *only* when we have both prefs *and* item counts
#   3.  fold the natural‑language hint into the rest of the advice pipeline


def _safe_best_offer(
    counts: Dict[str, int],
    w_me: List[float] | None,
    w_opp: List[float] | None,
    last_split: Dict[str, int] | None,
):
    """Wrapper around *best_offer* that swallows problems and missing inputs.

    Returns ``None`` iff:
    • preference weights are unavailable,
    • *best_offer* raises (e.g. numerical issue), or
    • the returned split is identical to *last_split* (already optimal).
    """
    if w_me is None or w_opp is None:
        return None
    try:
        # Calculate baseline utilities for equal split
        equal_split = {k: v // 2 for k, v in counts.items()}
        base_you = utility(equal_split, w_me) if w_me else 0.0
        base_them = utility({k: counts[k] - v for k, v in equal_split.items()}, w_opp) if w_opp else 0.0
        
        suggestion = best_offer(counts, w_me, w_opp, last_split or {}, base_you, base_them, 1.0)
        if suggestion and suggestion != (last_split or {}):
            return suggestion
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("best_offer failed – falling back to rule‑based advice (%s)", exc)
    return None


def get_advice(conv_id: str, speaker: str, model: str = "qwen3:latest") -> Dict[str, Any]:
    """Return negotiation *advice* & *reply* for the next user move.

    This version integrates the Pareto suggestion step via ``_safe_best_offer``.
    The rest of the flow (scoring, RAG, LLM prompting) is identical to the
    previous implementation – paste this over the existing definition inside
    *coach.py*.
    """
    try:
        
        # 1) Closure quick‑path 
        
        deal_reached = check_deal_reached(conv_id)
        if deal_reached:
            advice_text = get_closure_advice()
            reply = _llm_closure_reply(advice_text, model)
            return {"advice": advice_text, "reply": reply, "rag_source": "none", "rag_context": ""}

        
        # 2) Retrieve recent turns -
        
        turns = fetch_last_n(conv_id, 5)
        logger.info(f"Fetched {len(turns) if turns else 0} turns for conversation {conv_id}")
        
        # Add detailed debugging for turns
        if turns:
            logger.info(f"Turn details:")
            for i, turn in enumerate(turns):
                logger.info(f"  Turn {i}: speaker='{turn.get('speaker', 'MISSING')}', text='{turn.get('text', 'MISSING')[:50]}...', move='{turn.get('move', 'MISSING')}', pd='{turn.get('pd', 'MISSING')}'")
        
        #  MIN-TURNS GUARD --
        if turns is None or len(turns) < 2:
            # We need at least two turns before coaching
            logger.info(f"Min-turns guard triggered: turns={len(turns) if turns else 0}")
            return {
                "advice": "Need more conversation context to provide specific advice.",
                "reply": "Need more conversation context to provide specific advice.",
                "rag_source": "none",
                "rag_context": "",
            }
        
        # Require both parties in the most recent exchange to avoid stale speakers from older turns
        unique_speakers = {t['speaker'] for t in turns}
        num_from_you = sum(1 for t in turns if t.get('speaker') == 'A')
        num_from_opp = sum(1 for t in turns if t.get('speaker') == 'B')
        last_two = [t.get('speaker') for t in turns if t.get('speaker')][-2:]
        has_two_distinct = len(last_two) == 2 and last_two[0] != last_two[1]
        logger.info(
            f"Speaker analysis: current_speaker='{speaker}', all_speakers={unique_speakers}, "
            f"you_msgs={num_from_you}, opp_msgs={num_from_opp}, last_two={last_two}, has_two_distinct={has_two_distinct}"
        )
        # Relaxed: only require both parties have spoken at least once
        if (num_from_you < 1 or num_from_opp < 1):
            logger.info("Advice suppressed: require at least one message from both parties.")
            return {
                "advice": "Waiting for both parties to speak before providing specific advice.",
                "reply": "Waiting for both parties to speak before providing specific advice.",
                "rag_source": "none",
                "rag_context": "",
            }
        
        
        if not turns:
            logger.info("No turns found, using cold start advice")
            return _cold_start_advice(conv_id, speaker, model)

        # Who said what (power moves)
        my_moves  = [t["pd"] for t in turns if t["speaker"] == speaker]
        opp_moves = [t["pd"] for t in turns if t["speaker"] != speaker]
        
        logger.info(f"Speaker analysis: current_speaker='{speaker}', all_speakers={[t['speaker'] for t in turns]}")
        logger.info(f"My moves: {my_moves}, Opponent moves: {opp_moves}")

        
        # 3) Pareto suggestion step -
        
        # ❖ conversation‑specific info that should live on the convo object
        current_split: Dict[str, int] | None = turns[-1].get("my_allocation")
        # Default item counts for Deal-or-No-Deal scenario (3 items)
        counts: Dict[str, int] = {"item0": 3, "item1": 2, "item2": 1}  # Default DOND counts

        # preference inference (may return None, None)
        w_me, w_opp = estimate_preferences([t["text"] for t in turns])

        suggested_split = _safe_best_offer(counts, w_me, w_opp, current_split)

        
        # 4) Item analysis and context-aware advice 
        
        # Analyze item priorities and current offers
        priorities = analyze_item_priorities(turns, model)
        current_offers = extract_current_offers(turns, model)
        logger.info(f"Item priorities: {priorities}")
        logger.info(f"Current offers: {current_offers}")
        
        # Build dynamic item-id -> name map from recognized items in the conversation
        banned_names = {
            "you", "your", "yours", "we", "our", "ours", "them", "their", "theirs",
            "i", "me", "my", "mine",
            # vague/abstract
            "everything", "anything", "nothing", "everything else",
            # non-item discourse terms
            "deal", "agreement", "else", "other", "others",
            # auxiliaries / verbs / modals often misread as items
            "like", "need", "want", "would", "could", "should", "might", "will",
            "have", "has", "had", "keep", "take", "give", "get", "can"
        }
        id_to_name: Dict[str, str] = {}
        for side in ("You", "Them"):
            for iid, entry in priorities.get(side, {}).items():
                name = (entry or {}).get("item_name", "")
                if name and name.strip().lower() not in banned_names:
                    id_to_name[iid] = name.strip()

        # Generate numerical concession suggestions
        numerical_advice = suggest_numerical_concessions(priorities, current_split)
        logger.info(f"Numerical advice: {numerical_advice}")

        # Detect whether any explicit numbers are mentioned in recent conversation
        import re as _re_num
        has_numbers_in_context = any(
            bool(_re_num.search(r"\b\d+\b", (t.get("text") or ""))) for t in turns
        )
        
        
        # 5) Scoring + rule‑based hint 
        
        scores = score_turns(my_moves, opp_moves)
        logger.info(f"Scored conversation: {scores}")

        if suggested_split is not None:
            bundle_txt = ", ".join(
                f"{v} {id_to_name.get(k, k)}" for k, v in suggested_split.items() if v
            )
            hint = f"Suggest splitting items like this: {bundle_txt}. This benefits both sides."
            logger.info(f"Using Pareto suggestion: {hint}")
        else:
            # Combine rule-based advice with numerical suggestions
            base_advice = rule_based_advice_enhanced(scores, priorities, current_offers)
            if numerical_advice and numerical_advice != "Consider making a small concession to move the negotiation forward.":
                hint = f"{base_advice} Specifically: {numerical_advice}"
            else:
                hint = _concise(base_advice)
            logger.info(f"Using enhanced rule-based advice: {hint}")

        # Surface recognized items to the LLM via the hint so it uses real names
        if id_to_name:
            recognized_items_txt = ", ".join(sorted(set(id_to_name.values())))
            hint = f"{hint} Items recognized: {recognized_items_txt}."

        # Also include rough counts if detectable from priorities (best-effort)
        try:
            recognized_with_counts: Dict[str, int] = {}
            for side in ("You", "Them"):
                for iid, entry in priorities.get(side, {}).items():
                    name = (entry or {}).get("item_name", "")
                    if not name or name.strip().lower() in banned_names:
                        continue
                    qty = int((entry or {}).get("quantity", 1) or 1)
                    # Keep the max observed quantity per name to avoid inflation
                    if name in recognized_with_counts:
                        recognized_with_counts[name] = max(recognized_with_counts[name], qty)
                    else:
                        recognized_with_counts[name] = qty
            if recognized_with_counts:
                counts_txt = ", ".join(f"{q} {n}" for n, q in recognized_with_counts.items())
                hint = f"{hint} Items recognized with counts: {counts_txt}."
        except Exception as _e:  # best-effort only
            pass

        # If conversation has no explicit counts, steer the model to percentage-based advice
        if not has_numbers_in_context:
            hint = (
                f"{hint} No item counts mentioned; express the proposal using percentages "
                f"(e.g., '60% of books and 40% of hats')."
            )

        # Avoid repeating recent advice
        if check_recent_advice(conv_id, speaker, hint):
            hint = get_alternative_advice(scores, hint)

        
        # 5) Retrieval‑augmented generation 
        
        rag_context, rag_source = _retrieve_rag_context(hint, turns)
        summary     = summarize_turns_for_llm(turns)

        last_turn = turns[-1]["text"] if turns else ""
        llm_reply = _llm_generate_reply(hint, summary, rag_context, last_turn, repeated=check_recent_advice(conv_id, speaker, hint), model=model)

        # Normalize advice for repetition detection
        hint_key = hint.lower().strip()
        if check_recent_advice(conv_id, speaker, hint_key):
            hint = get_alternative_advice(scores, hint)
            hint_key = hint.lower().strip()
        store_advice(conv_id, speaker, hint_key)
        
        logger.info(f"Final advice result: advice='{llm_reply}', reply='{llm_reply}'")
        return {"advice": llm_reply, "reply": llm_reply, "rag_source": rag_source, "rag_context": rag_context}

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("get_advice failed: %s", exc)
        import traceback
        logger.error("Full traceback: %s", traceback.format_exc())
        return {
            "advice": "Unable to generate advice at this time.",
            "reply": "I'm having trouble processing your request. Please try again.",
            "rag_source": "none",
            "rag_context": "",
        }


# Internal helpers (LLM wrappers & RAG) 


def _llm_closure_reply(advice_text: str, model: str) -> str:
    """Tiny helper to ask the LLM for a closure‑phase response."""
    provider  = get_provider_from_model(model)
    llm_client = create_llm_client(provider, model)

    prompt = (
        "You are a negotiation coach. The parties have reached an agreement. "
        "Provide detailed advice on how to properly close the negotiation, including:\n"
        "1. How to summarise the agreement\n2. How to confirm next steps and timelines\n"
        "3. How to maintain the positive relationship\n4. What to document or follow up on."
    )
    resp = llm_client.generate_response([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ], temperature=0.3, max_tokens=1200)
    return clean_llm_response(resp)


def _cold_start_advice(conv_id: str, speaker: str, model: str):
    """Handle conversations with fewer than one turn (cold start)."""
    hint = "Start with information sharing to build trust."
    # Import the correct function
    from app.rag import retrieve_rag_context
    rag_ctx = retrieve_rag_context(hint) or "N/A"
    reply  = _llm_generate_reply(hint, "No conversation history available.", rag_ctx, "", repeated=False, model=model)
    store_advice(conv_id, speaker, hint)
    rag_source = "generic" if rag_ctx and rag_ctx != "N/A" else "none"
    return {"advice": reply, "reply": reply, "rag_source": rag_source, "rag_context": ("" if rag_ctx == "N/A" else rag_ctx)}


def _retrieve_rag_context(hint: str, turns) -> tuple[str, str]:
    strategy = extract_strategy_from_hint(hint)
    recent_moves = [t["pd"] for t in turns[-4:]]
    ctx = get_casino_dialogue_examples(strategy, recent_moves)
    if ctx:
        return ctx, "casino"
    summary = summarize_turns_for_llm(turns)
    # Import the correct function to avoid recursive call
    from app.rag import retrieve_rag_context as rag_retrieve
    ctx = rag_retrieve(f"{hint}\n\nConversation so far: {summary}") or "N/A"
    if ctx and ctx != "N/A":
        return ctx, "generic"
    return ctx, "none"


def _llm_generate_reply(hint: str, summary: str, rag_context: str, last_turn: str, *, repeated: bool, model: str):
    """Single place to call the LLM for the final user‑facing sentence (long, justified)."""
    provider  = get_provider_from_model(model)
    llm_client = create_llm_client(provider, model)
    temperature = 0.4

    # Create a more focused prompt that emphasizes the specific advice
    user_prompt = (
        f"Conversation context: {summary}\n\n"
        f"Relevant retrieved context (you may quote it in your justification):\n{rag_context}\n\n"
        f"Last opponent line: \"{last_turn}\"\n\n"
        f"Coach's analysis & recognized items: {hint}\n\n"
        f"Write a short paragraph that first justifies the recommendation using specific context and items; "
        f"if no explicit item counts are in the conversation, use percentages for the proposal (e.g., '60% of books and 40% of hats'). "
        f"Then provide one clear negotiating sentence that explicitly names either exact quantities or percentages. "
        f"Do not include <think> or chain-of-thought in your output."
    )
    
    logger.info(f"Calling LLM with model={model}, provider={provider}")
    logger.info(f"User prompt: {user_prompt}")
    
    resp = llm_client.generate_response([
        {"role": "system", "content": SYSTEM_PROMPT + "\nNever include <think> or chain-of-thought in outputs."},
        {"role": "user", "content": user_prompt},
    ], temperature=temperature, max_tokens=700, frequency_penalty=0.2, presence_penalty=0.1)
    
    logger.info(f"LLM raw response: '{resp}'")
    raw_response = resp.strip()

    # Strip <think> blocks early
    import re
    no_think = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()

    if not no_think:
        # Deterministic, hint-based fallback that preserves items/quantities
        m = re.search(r"Suggest splitting items like this:\s*(.+?)\.", hint)
        if m:
            bundle_txt = m.group(1).strip()
            return clean_llm_response_long(
                f"Given the context and recognized items, proposing a concrete split helps unlock the negotiation.\n"
                f"Recommendation: Let's split the items as {bundle_txt} so we can move forward."
            )
        # If no explicit split present, echo the hint so at least items recognized are present
        # Try to sanitize any spurious tokens like 'else', 'can' if they slipped through
        sanitized_hint = re.sub(r"\b(can|else|everything else)\b", "", hint, flags=re.IGNORECASE)
        return clean_llm_response_long(sanitized_hint)

    # If response lacks obvious item tokens despite hint containing them, synthesize from hint
    import re as _re
    has_number = _re.search(r"\b\d+\b", no_think) is not None
    mentions_items = _re.search(r"Items recognized:\s*(.+?)\.", hint)
    mentions_split = _re.search(r"Suggest splitting items like this:\s*(.+?)\.", hint)
    if (not has_number) and (mentions_items or mentions_split):
        phrase = None
        if mentions_split:
            phrase = mentions_split.group(1).strip()
        elif mentions_items:
            # fallback: at least mention the items (counts may be in the prior clause)
            phrase = mentions_items.group(1).strip()
        if phrase:
            synthesized = (
                f"Based on the conversation so far and the items on the table, a concrete proposal reduces ambiguity and invites reciprocity.\n"
                f"Recommendation: Let's proceed with {phrase} to move things forward."
            )
            logger.info(f"Synthesized concrete response from hint: '{synthesized}'")
            return clean_llm_response_long(synthesized)

    cleaned_response = re.sub(r"\b(can|else|everything else)\b", "", no_think, flags=re.IGNORECASE).strip()
    logger.info(f"Cleaned response before clean_llm_response_long: '{cleaned_response[:120] + ('…' if len(cleaned_response)>120 else '')}'")
    return clean_llm_response_long(cleaned_response)


def analyze_conversation_patterns(conv_id):
    """
    Analyze conversation patterns using the graph database.
    
    Args:
        conv_id: Conversation identifier
        
    Returns:
        Dictionary with analysis results
    """
    try:
        if not neo4j_driver:
            return {"error": "Neo4j driver not available"}
            
        with neo4j_driver.session() as session:
            # Query the graph for conversation analysis
            query = """
            MATCH (c:Conv {id: $conv_id})
            OPTIONAL MATCH (c)-[:HAS_TURN]->(t:Turn)
            RETURN 
                count(t) as total_turns,
                collect(DISTINCT t.pd) as power_dynamics,
                collect(DISTINCT t.move) as moves,
                avg(CASE WHEN t.pd = 'C' THEN 1 ELSE 0 END) as cooperation_rate
            """
            
            result = session.run(query, {"conv_id": conv_id})
            record = result.single()
            
            if record:
                return {
                    "total_turns": record["total_turns"],
                    "power_dynamics": record["power_dynamics"],
                    "moves": record["moves"],
                    "cooperation_rate": record["cooperation_rate"]
                }
            else:
                return {"error": "No conversation data found"}
                
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def get_strategic_insights(conv_id, speaker, model="qwen3:latest"):
    """
    Get strategic insights using direct graph queries and Ollama analysis.
    
    Args:
        conv_id: Conversation identifier
        speaker: Speaker identifier
        model: Model to use for analysis
        
    Returns:
        String with strategic insights
    """
    try:
        if not neo4j_driver:
            return "Neo4j driver not available for analysis."
            
        with neo4j_driver.session() as session:
            # Get conversation data
            query = """
            MATCH (c:Conv {id: $conv_id})
            OPTIONAL MATCH (c)-[:HAS_TURN]->(t:Turn)<-[:SPOKE]-(p:Person {name: $speaker})
            RETURN 
                count(t) as turn_count,
                collect(DISTINCT t.pd) as power_dynamics,
                collect(DISTINCT t.move) as moves,
                avg(CASE WHEN t.pd = 'C' THEN 1 ELSE 0 END) as cooperation_rate
            """
            
            result = session.run(query, {"conv_id": conv_id, "speaker": speaker})
            record = result.single()
            
            if record:
                data = {
                    'turn_count': record['turn_count'],
                    'power_dynamics': record['power_dynamics'],
                    'moves': record['moves'],
                    'cooperation_rate': record['cooperation_rate']
                }
                
                # Create analysis prompt
                analysis_prompt = f"""
                Analyze this negotiation data for speaker {speaker} in conversation {conv_id}:
                - Total turns: {data['turn_count']}
                - Power dynamics: {data['power_dynamics']}
                - Move types: {data['moves']}
                - Cooperation rate: {data['cooperation_rate']:.2f}
                
                Provide strategic insights about this speaker's negotiation style and recommendations.
                """
                
                # Create LLM client
                provider = get_provider_from_model(model)
                llm_client = create_llm_client(provider, model)
                
                # Generate response
                response = llm_client.generate_response([
                    {"role": "user", "content": analysis_prompt}
                ], temperature=0.3)
                
                return response
            else:
                return "No conversation data available for analysis."
        
    except Exception as e:
        return f"Unable to generate insights: {str(e)}"

def get_speaker_statistics(conv_id, speaker):
    """
    Get statistics for a specific speaker in a conversation.
    
    Args:
        conv_id: Conversation identifier
        speaker: Speaker identifier
        
    Returns:
        Dictionary with speaker statistics
    """
    try:
        if not neo4j_driver:
            return {"error": "Neo4j driver not available"}
            
        with neo4j_driver.session() as session:
            query = """
            MATCH (c:Conv {id: $conv_id})
            OPTIONAL MATCH (c)-[:HAS_TURN]->(t:Turn)<-[:SPOKE]-(p:Person {name: $speaker})
            RETURN 
                count(t) as turn_count,
                collect(DISTINCT t.pd) as power_dynamics,
                collect(DISTINCT t.move) as moves,
                avg(CASE WHEN t.pd = 'C' THEN 1 ELSE 0 END) as cooperation_rate
            """
            
            result = session.run(query, {"conv_id": conv_id, "speaker": speaker})
            record = result.single()
            
            if record:
                return {
                    "speaker": speaker,
                    "turn_count": record["turn_count"],
                    "power_dynamics": record["power_dynamics"],
                    "moves": record["moves"],
                    "cooperation_rate": record["cooperation_rate"]
                }
            else:
                return {"error": "No speaker data found"}
                
    except Exception as e:
        return {"error": f"Statistics failed: {str(e)}"}

def get_negotiation_recommendations(conv_id, speaker):
    """
    Get comprehensive negotiation recommendations.
    
    Args:
        conv_id: Conversation identifier
        speaker: Speaker identifier
        
    Returns:
        Dictionary with recommendations
    """
    try:
        # Get basic advice
        basic_advice = get_advice(conv_id, speaker)
        
        # Get speaker statistics
        stats = get_speaker_statistics(conv_id, speaker)
        
        # Get conversation patterns
        patterns = analyze_conversation_patterns(conv_id)
        
        # Get strategic insights
        insights = get_strategic_insights(conv_id, speaker)
        
        return {
            "basic_advice": basic_advice,
            "speaker_statistics": stats,
            "conversation_patterns": patterns,
            "strategic_insights": insights
        }
        
    except Exception as e:
        return {
            "error": f"Failed to generate recommendations: {str(e)}",
            "basic_advice": get_advice(conv_id, speaker)
        }


# Item analysis and priority extraction

def analyze_item_priorities(turns: List[Dict[str, Any]], model: str = "qwen3:latest") -> Dict[str, Dict[str, Any]]:
    """
    Analyze conversation to extract item priorities for each speaker.
    Now uses LLM for dynamic item identification when needed.
    """
    priorities = {"You": {}, "Them": {}}
    
    # Collect all text for item identification
    all_text = " ".join([turn.get("text", "") for turn in turns if turn.get("text")])
    
    # Use LLM to identify items dynamically
    item_mapping = _identify_items_with_llm(all_text, model)
    
    # If no items found, fall back to default mapping
    if not item_mapping:
        item_mapping = {
            "book": "item0", "books": "item0",
            "hat": "item1", "hats": "item1", 
            "ball": "item2", "balls": "item2", "basketball": "item2", "basketballs": "item2"
        }
    
    # Default item counts (can be adjusted based on identified items)
    default_counts = {item_id: 1 for item_id in set(item_mapping.values())}
    
    for turn in turns:
        speaker = turn.get("speaker", "")
        text = turn.get("text", "").lower()
        
        if not text or speaker not in priorities:
            continue
            
        # Extract item mentions and their context
        for item_name, item_id in item_mapping.items():
            if item_name in text:
                # Analyze the context around the item mention
                context_start = max(0, text.find(item_name) - 50)
                context_end = min(len(text), text.find(item_name) + len(item_name) + 50)
                context = text[context_start:context_end]
                
                # Determine priority level based on context
                priority_level = "medium"
                quantity = 1
                
                # Look for quantity indicators
                import re
                quantity_match = re.search(r'(\d+)\s+' + re.escape(item_name), context)
                if quantity_match:
                    quantity = int(quantity_match.group(1))
                
                # Look for priority indicators
                if any(word in context for word in ["need", "want", "must", "essential", "critical", "important", "all", "everything"]):
                    priority_level = "high"
                elif any(word in context for word in ["like", "prefer", "would like", "interested"]):
                    priority_level = "medium"
                elif any(word in context for word in ["don't need", "don't want", "not interested", "don't care", "except", "without"]):
                    priority_level = "low"
                
                # Look for "all" or "everything" indicators
                if "all" in context or "everything" in context:
                    quantity = default_counts.get(item_id, 1)
                    priority_level = "high"
                
                # Store the analysis
                if item_id not in priorities[speaker]:
                    priorities[speaker][item_id] = {
                        "priority": priority_level,
                        "quantity": quantity,
                        "mentions": 0,
                        "context": [],
                        "item_name": item_name  # Store the actual item name
                    }
                
                priorities[speaker][item_id]["mentions"] += 1
                priorities[speaker][item_id]["context"].append(context)
    
    return priorities

def suggest_numerical_concessions(priorities: Dict[str, Dict[str, Any]], current_split: Dict[str, int] = None) -> str:
    """
    Generate specific numerical concession suggestions based on priorities.
    Now uses actual item names instead of generic item IDs.
    """
    if not priorities:
        return "Consider making a small concession to move the negotiation forward."
    
    # Analyze what each party wants
    you_wants = priorities.get("You", {})
    them_wants = priorities.get("Them", {})
    
    suggestions = []
    
    # Look for high-priority items for each party
    you_high_priority = [item for item, data in you_wants.items() if data.get("priority") == "high"]
    them_high_priority = [item for item, data in them_wants.items() if data.get("priority") == "high"]
    
    # Look for low-priority items that can be conceded
    you_low_priority = [item for item, data in you_wants.items() if data.get("priority") == "low"]
    them_low_priority = [item for item, data in them_wants.items() if data.get("priority") == "low"]
    
    # Generate specific concession suggestions with actual item names
    if you_high_priority and them_high_priority:
        # Both parties have high-priority items - suggest compromise
        for you_item in you_high_priority:
            for them_item in them_high_priority:
                if you_item != them_item:
                    you_item_name = you_wants[you_item].get("item_name", you_item)
                    them_item_name = them_wants[them_item].get("item_name", them_item)
                    you_count = you_wants[you_item].get("quantity", 1)
                    them_count = them_wants[them_item].get("quantity", 1)
                    
                    # Only suggest reduction if there's actually a reduction
                    if you_count > 1 and them_count > 1:
                        you_reduced = max(1, int(you_count * 0.7))
                        them_reduced = max(1, int(them_count * 0.7))
                        if you_reduced < you_count or them_reduced < them_count:
                            suggestions.append(f"Offer to reduce your {you_item_name} demand from {you_count} to {you_reduced} if they reduce their {them_item_name} demand from {them_count} to {them_reduced}")
                    elif you_count > 1:
                        you_reduced = max(1, int(you_count * 0.7))
                        if you_reduced < you_count:
                            suggestions.append(f"Offer to reduce your {you_item_name} demand from {you_count} to {you_reduced} to show flexibility")
                    elif them_count > 1:
                        them_reduced = max(1, int(them_count * 0.7))
                        if them_reduced < them_count:
                            suggestions.append(f"Ask them to reduce their {them_item_name} demand from {them_count} to {them_reduced} to move toward agreement")
                    else:
                        # Both want 1 item - suggest splitting or percentage-based approach
                        suggestions.append(f"Propose splitting the {you_item_name} and {them_item_name} 60-40 to find middle ground")
    
    elif you_high_priority and them_low_priority:
        # You have high priority, they have low priority items to concede
        for low_item in them_low_priority:
            item_name = them_wants[low_item].get("item_name", low_item)
            count = them_wants[low_item].get("quantity", 1)
            if count > 1:
                reduced_count = max(1, count // 2)
                suggestions.append(f"Ask them to reduce their {item_name} demand by 50% (from {count} to {reduced_count}) since it's not their priority")
            else:
                suggestions.append(f"Ask them to concede the {item_name} since it's not their priority")
    
    elif them_high_priority and you_low_priority:
        # They have high priority, you have low priority items to concede
        for low_item in you_low_priority:
            item_name = you_wants[low_item].get("item_name", low_item)
            count = you_wants[low_item].get("quantity", 1)
            if count > 1:
                reduced_count = max(1, count // 2)
                suggestions.append(f"Offer to reduce your {item_name} demand by 50% (from {count} to {reduced_count}) to meet their high-priority needs")
            else:
                suggestions.append(f"Offer to concede the {item_name} to meet their high-priority needs")
    
    # If no specific suggestions, provide general numerical advice with percentages
    if not suggestions:
        if you_wants and them_wants:
            # Check if we have any items with quantities > 1
            you_items_with_quantities = [(item, data.get("quantity", 1)) for item, data in you_wants.items() if data.get("quantity", 1) > 1]
            them_items_with_quantities = [(item, data.get("quantity", 1)) for item, data in them_wants.items() if data.get("quantity", 1) > 1]
            
            if you_items_with_quantities:
                item_name = you_items_with_quantities[0][0]
                count = you_items_with_quantities[0][1]
                reduced = max(1, int(count * 0.8))
                suggestions.append(f"Offer to reduce your {item_name} demand by 20% (from {count} to {reduced}) to show flexibility")
            elif them_items_with_quantities:
                item_name = them_items_with_quantities[0][0]
                count = them_items_with_quantities[0][1]
                reduced = max(1, int(count * 0.8))
                suggestions.append(f"Ask them to reduce their {item_name} demand by 20% (from {count} to {reduced}) to move toward agreement")
            else:
                # No specific quantities, suggest percentage-based concessions
                suggestions.append("Offer to reduce your demands by 25-30% to show flexibility")
                suggestions.append("Propose splitting contested items 60-40 to move toward agreement")
        else:
            suggestions.append("Make a small concession (15-20%) to demonstrate goodwill")
    
    return "; ".join(suggestions[:2])  # Return top 2 suggestions

def extract_current_offers(turns: List[Dict[str, Any]], model: str = "qwen3:latest") -> Dict[str, Dict[str, Any]]:
    """
    Extract current offers and positions from conversation.
    Now uses LLM for dynamic item identification.
    """
    offers = {"You": {}, "Them": {}}
    
    # Collect all text for item identification
    all_text = " ".join([turn.get("text", "") for turn in turns if turn.get("text")])
    
    # Use LLM to identify items dynamically
    item_mapping = _identify_items_with_llm(all_text, model)
    
    # If no items found, fall back to default mapping
    if not item_mapping:
        item_mapping = {
            "book": "item0", "books": "item0",
            "hat": "item1", "hats": "item1", 
            "ball": "item2", "balls": "item2", "basketball": "item2", "basketballs": "item2"
        }
    
    for turn in turns:
        speaker = turn.get("speaker", "")
        text = turn.get("text", "").lower()
        
        if not text or speaker not in offers:
            continue
        
        # Extract numerical offers
        import re
        for item_name, item_id in item_mapping.items():
            # Look for patterns like "2 hats", "3 balls", etc.
            pattern = r'(\d+)\s+' + re.escape(item_name)
            matches = re.findall(pattern, text)
            if matches:
                quantity = int(matches[0])
                if item_id not in offers[speaker]:
                    offers[speaker][item_id] = {"quantity": quantity, "mentions": 0, "item_name": item_name}
                offers[speaker][item_id]["quantity"] = max(offers[speaker][item_id]["quantity"], quantity)
                offers[speaker][item_id]["mentions"] += 1
    
    return offers

def _identify_items_with_llm(text: str, model: str = "qwen3:latest") -> Dict[str, str]:
    """
    Use LLM to dynamically identify items, goods, services, or resources from conversation text.
    Returns mapping of item names to generic IDs (item0, item1, etc.).
    """
    if not text.strip():
        return {}
    
    # Common negotiation items to prioritize
    common_items = [
        "hat", "hats", "ball", "balls", "basketball", "basketballs",
        "book", "books", "car", "cars", "house", "houses", "apartment", "apartments",
        "money", "cash", "dollar", "dollars", "euro", "euros", "pound", "pounds",
        "equity", "shares", "stock", "stocks", "ownership", "percentage", "percent",
        "service", "services", "consulting", "support", "maintenance", "training",
        "time", "hours", "days", "weeks", "months", "year", "years",
        "equipment", "tools", "machines", "devices", "computers", "phones",
        "space", "office", "room", "rooms", "area", "location", "venue"
    ]
    
    # Filter out common non-item words
    non_items = [
        "you", "your", "yours", "them", "their", "theirs", "we", "our", "ours",
        "i", "me", "my", "mine", "he", "she", "it", "its", "this", "that", "these", "those",
        "all", "everything", "everyone", "everybody", "anything", "anyone", "anybody",
        "nothing", "no", "none", "neither", "either", "both", "each", "every", "else", "everything else",
        "else", "other", "others", "another", "additional", "extra", "more", "less",
        "deal", "deals", "agreement", "agreements", "contract", "contracts",
        "offer", "offers", "proposal", "proposals", "suggestion", "suggestions",
        "price", "prices", "cost", "costs", "value", "values", "worth", "amount",
        # discourse verbs/auxiliaries often misread as items
        "like", "need", "want", "would", "could", "should", "might", "will", "have", "has", "had", "keep", "take", "give", "get", "can",
        "what", "when", "where", "why", "how", "which", "who", "whom", "whose"
    ]
    
    # Create a more focused prompt for the LLM
    prompt = f"""
Analyze the negotiation text and extract ONLY concrete items/goods/services that are being negotiated (things that can be counted, split, or traded).

Text: "{text}"

Return ONLY a valid JSON object:
{{
  "hats": "item0",
  "balls": "item1",
  "consulting_hours": "item2"
}}

Strict rules:
1) Include ONLY tangible or well-defined items (e.g., hats, books, consulting_hours).
2) EXCLUDE pronouns and discourse terms (e.g., you, them, else, everything, everything else).
3) EXCLUDE verbs/modals/auxiliaries (e.g., need, want, keep, take, give, get, have, will, can).
4) EXCLUDE abstract or meta words (deal, agreement, value, price) unless the text names a specific commodity (e.g., dollars, euros are OK).
5) Use exactly the item names from the text; don't invent new items.
6) Map each unique item to item0, item1, item2...; respond with ONLY the JSON object.
"""

    try:
        provider = get_provider_from_model(model)
        llm_client = create_llm_client(provider, model)
        
        response = llm_client.generate_response([
            {"role": "system", "content": "You are a negotiation analysis tool. Respond with ONLY valid JSON objects, no other text."},
            {"role": "user", "content": prompt}
        ], temperature=0.1, max_tokens=200)
        
        # Clean the response and extract JSON
        cleaned_response = response.strip()
        
        # Remove any think tags
        import re
        think_pattern = r'<think>.*?</think>'
        cleaned_response = re.sub(think_pattern, '', cleaned_response, flags=re.DOTALL)
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                items = json.loads(json_str)
                
                # Filter out non-items and validate
                filtered_items = {}
                for item_name, item_id in items.items():
                    # Skip if it's in our non-items list
                    if item_name.lower() in [word.lower() for word in non_items]:
                        continue
                    # Skip if it's too generic
                    if len(item_name) < 2 or item_name.lower() in ['all', 'none', 'some', 'any']:
                        continue
                    filtered_items[item_name] = item_id
                
                return filtered_items
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from LLM response: {e}")
                logger.warning(f"Response was: {cleaned_response}")
        
        # If LLM fails, fall back to regex-based extraction
        logger.info("LLM failed, using fallback item extraction")
        return _extract_items_fallback(text)
        
    except Exception as e:
        logger.warning(f"LLM item identification failed: {e}")
        return _extract_items_fallback(text)

def _extract_items_fallback(text: str) -> Dict[str, str]:
    """
    Fallback method to extract items using regex patterns when LLM fails.
    """
    import re
    
    # Common item patterns
    item_patterns = [
        r'\b(\d+)\s+(\w+s?)\b',  # "3 books", "2 hats"
        r'\b(all|some|few)\s+(\w+s?)\b',  # "all books", "some hats"
        r'\b(the)\s+(\w+s?)\b',  # "the books", "the hats"
        r'\b(\w+s?)\b'  # any plural/singular nouns
    ]
    
    items = {}
    text_lower = text.lower()
    
    # Common negotiation items (prioritize these)
    common_items = [
        "books", "hats", "balls", "cars", "houses", "money", "time", "hours", 
        "services", "products", "equipment", "space", "resources", "shares",
        "contracts", "licenses", "rights", "access", "data", "information",
        "equity", "property", "support", "consulting", "development", "cloud"
    ]
    
    # Check for common items first
    for item in common_items:
        if item in text_lower:
            items[item] = f"item{len(items)}"
    
    # Use regex patterns to find other items, but filter out common words and pronouns/abstracts
    common_words = {"the", "and", "or", "but", "for", "with", "from", "this", "that", "what", "when", "where", "why", "how"}
    banned = {"you","your","yours","we","our","ours","they","their","theirs","i","me","my","mine","everything","anything","nothing","deal","agreement","contract","offer","proposal","like","need","want","would","could","should","might","will","have","has","had","keep","take","give","get","can","else","everything else"}
    
    for pattern in item_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                item_name = match[1] if len(match) > 1 else match[0]
            else:
                item_name = match
            
            # Only add if it's plausible as an item
            if (item_name and item_name not in items and len(item_name) > 2 and 
                item_name not in common_words and item_name not in banned and not item_name.isdigit()):
                items[item_name] = f"item{len(items)}"
    
    return items

if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python -m coach <conv_id> <speaker>")
        sys.exit(1)
    
    conv_id = sys.argv[1]
    speaker = sys.argv[2]
    
    try:
        if VERBOSE_LOGGING:
            print(f"Getting advice for conversation {conv_id}, speaker {speaker}...")
        
        # Get basic advice
        advice = get_advice(conv_id, speaker)
        if VERBOSE_LOGGING:
            print(f"Advice: {advice['advice']}")
            print(f"Reply: {advice['reply']}")
        
        # Get comprehensive recommendations
        if VERBOSE_LOGGING:
            print("\nGetting comprehensive recommendations...")
        recommendations = get_negotiation_recommendations(conv_id, speaker)
        
        if VERBOSE_LOGGING:
            if "error" not in recommendations:
                print(f"Speaker Statistics: {recommendations['speaker_statistics']}")
                print(f"Conversation Patterns: {recommendations['conversation_patterns']}")
                print(f"Strategic Insights: {recommendations['strategic_insights']}")
            else:
                print(f"Error: {recommendations['error']}")
            
    except Exception as e:
        if VERBOSE_LOGGING:
            print(f"Error: {e}")
        sys.exit(1)
