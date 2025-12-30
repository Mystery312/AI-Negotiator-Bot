"""Auto-proposal functionality for negotiation bot."""

from typing import Dict, Sequence, Optional
from pareto import best_offer, utility
from preference import estimate_preferences

EPS = 1e-9  # avoid edge cases in ≥ checks

def _norm(ws: Sequence[float]) -> list[float]:
    """Normalise a weight vector to sum to 1 (clip negatives)."""
    ws = [max(0.0, float(x)) for x in ws]
    s = sum(ws) or 1.0
    return [w / s for w in ws]

def _utilities_for_split(
    counts: Dict[str, int],
    split_you: Dict[str, int],
    w_you: Sequence[float],
    w_them: Sequence[float],
) -> tuple[float, float]:
    """Return (u_you, u_them) for the given allocation."""
    u_you = utility(split_you, w_you)
    opp_split = {k: counts[k] - split_you[k] for k in counts}
    u_them = utility(opp_split, w_them)
    return u_you, u_them

def _proposal_for_bot(
    counts: Dict[str, int],
    w_you: Sequence[float],
    w_them: Sequence[float],
    base_you: float,
    base_them: float,
    ratio: float,
    use_pareto: bool,
) -> Dict[str, int]:
    """Return the bot's split (proposal)."""
    if not use_pareto:
        return {k: 0 for k in counts}  # bot stays with nothing

    prop = best_offer(
        counts=counts,
        w_you=w_you,
        w_them=w_them,
        last_split={k: 0 for k in counts},  # previous allocation (all zero)
        base_you=base_you,
        base_them=base_them,
        ratio=ratio,
    )
    return prop or {k: 0 for k in counts}

def generate_bot_proposal(
    turns: list,
    counts: Dict[str, int],
    use_pareto: bool = True,
    ratio: float = 1.0,
) -> Optional[Dict[str, int]]:
    """Generate a bot proposal based on conversation history."""
    try:
        # Estimate preferences from conversation turns
        w_you, w_them = estimate_preferences(turns)
        if w_you is None or w_them is None:
            return None
        
        # Normalize preferences
        w_you, w_them = _norm(w_you), _norm(w_them)
        
        # Calculate baseline utilities (equal split)
        equal_split = {k: v // 2 for k, v in counts.items()}
        base_you, base_them = _utilities_for_split(counts, equal_split, w_you, w_them)
        
        # Generate proposal
        proposal = _proposal_for_bot(counts, w_you, w_them, base_you, base_them, ratio, use_pareto)
        
        return proposal
    except Exception as e:
        print(f"Error generating bot proposal: {e}")
        return None

def format_proposal_message(proposal: Dict[str, int], counts: Dict[str, int]) -> str:
    """Format a proposal as a readable message."""
    if not proposal:
        return "🤖 Bot proposal: No proposal available"
    
    # Calculate what each party gets
    you_items = []
    them_items = []
    
    for item, total_qty in counts.items():
        you_qty = proposal.get(item, 0)
        them_qty = total_qty - you_qty
        
        if you_qty > 0:
            you_items.append(f"{item}: {you_qty}")
        if them_qty > 0:
            them_items.append(f"{item}: {them_qty}")
    
    message = "🤖 Bot proposal:\n"
    if you_items:
        message += f"• You get: {', '.join(you_items)}\n"
    if them_items:
        message += f"• They get: {', '.join(them_items)}"
    
    return message
