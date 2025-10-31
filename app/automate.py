"""Automated conversation system with negotiation and coach advice."""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
import random

import os
import sys

# Add parent directory to sys.path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from app package
from app.autoplay import generate_bot_proposal, format_proposal_message
from app.coach import get_advice
from app.dond_data import load_dond
from app.pareto import utility

def create_message(role: str, speaker: str, text: str) -> Dict:
    """Create a message dictionary with timestamp."""
    return {
        "role": role,
        "speaker": speaker,
        "text": text,
        "move": None,
        "pd": None,
        "ts": datetime.now().isoformat()
    }

def generate_opening(counts: Dict[str, int]) -> str:
    """Generate an opening message."""
    items = [f"{count} {item}" for item, count in counts.items()]
    return f"Hi! I see we have {', '.join(items)} to divide between us. What do you think would be a fair split?"

def generate_response(history: List[Dict], counts: Dict[str, int], is_party_a: bool) -> str:
    """Generate a contextual response based on conversation history."""
    last_msg = history[-1]["text"] if history else ""
    
    # Simple response templates
    responses = [
        "I understand your position. Let me think about that.",
        "That's an interesting proposal. Let me consider it.",
        "I appreciate your offer. Let me suggest something.",
        "Thank you for sharing your thoughts. Here's what I think.",
        "I see your point. Let me propose an alternative."
    ]
    
    return random.choice(responses)

def run_automated_conversation(
    counts: Dict[str, int] = None,
    max_turns: int = 10,
    model: str = "ollama:qwen3:latest"
) -> List[Dict]:
    """Run an automated conversation between two parties with coach advice."""
    
    # Default item counts if not provided
    if counts is None:
        counts = {"item0": 3, "item1": 2, "item2": 1}
    
    history = []
    
    # Start conversation
    opening = create_message("A", "Party A", generate_opening(counts))
    history.append(opening)
    
    # Get coach advice for opening
    advice = get_advice("auto", "Party A", model)
    if advice:
        history.append(create_message("Coach", "Coach", f"Advice: {advice}"))
    
    turns = 0
    while turns < max_turns:
        turns += 1
        
        # Party B's turn
        # First, get a response
        b_response = generate_response(history, counts, False)
        b_msg = create_message("B", "Party B", b_response)
        history.append(b_msg)
        
        # Get coach advice for B's message
        advice = get_advice("auto", "Party A", model)
        if advice:
            history.append(create_message("Coach", "Coach", f"Advice: {advice}"))
        
        # Generate bot proposal after B's message
        proposal = generate_bot_proposal([msg["text"] for msg in history], counts)
        if proposal:
            proposal_text = format_proposal_message(proposal, counts)
            history.append(create_message("Coach", "Bot", proposal_text))
        
        # Party A's turn
        a_response = generate_response(history, counts, True)
        a_msg = create_message("A", "Party A", a_response)
        history.append(a_msg)
        
        # Get coach advice for A's message
        advice = get_advice("auto", "Party B", model)
        if advice:
            history.append(create_message("Coach", "Coach", f"Advice: {advice}"))
        
        # Add some delay to make it more realistic
        time.sleep(1)
    
    return history

def run_dond_sample(
    sample_idx: int,
    model: str = "ollama:qwen3:latest"
) -> List[Dict]:
    """Run automation on a specific DoND sample."""
    samples = load_dond("validation")
    sample = samples[sample_idx]
    
    # Convert sample counts to dictionary
    counts = {f"item{i}": count for i, count in enumerate(sample.counts)}
    
    return run_automated_conversation(counts=counts, model=model)

def evaluate_conversation(
    history: List[Dict],
    counts: Dict[str, int],
    w_you: List[float],
    w_them: List[float]
) -> Tuple[float, float]:
    """Evaluate the final utilities achieved in the conversation."""
    # Find the last bot proposal
    last_proposal = None
    for msg in reversed(history):
        if msg["role"] == "Coach" and msg["speaker"] == "Bot":
            # Extract proposal from the message
            # Example format: "🤖 Bot proposal:\n• You get: item0: 2, item1: 1\n• They get: item0: 1, item2: 1"
            lines = msg["text"].split("\n")
            if len(lines) >= 3:
                # Parse the proposal
                proposal = {}
                for item in lines[1].split("You get: ")[1].split(", "):
                    item_name, count = item.split(": ")
                    proposal[item_name] = int(count)
                last_proposal = proposal
                break
    
    if last_proposal:
        # Calculate utilities
        u_you = utility(last_proposal, w_you)
        opp_split = {k: counts[k] - last_proposal[k] for k in counts}
        u_them = utility(opp_split, w_them)
        return u_you, u_them
    
    return 0.0, 0.0  # No proposal found

if __name__ == "__main__":
    # Example usage
    print("Running automated conversation...")
    history = run_automated_conversation()
    print("\nConversation history:")
    for msg in history:
        print(f"{msg['speaker']}: {msg['text']}\n")
