#!/usr/bin/env python3
"""
Debug the specific conversation to see why advice is still generic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.coach import (
    analyze_item_priorities, 
    suggest_numerical_concessions, 
    rule_based_advice_enhanced,
    score_turns,
    _identify_items_with_llm
)

def debug_conversation():
    """Debug the specific conversation that's giving generic advice."""
    
    # The conversation from user's feedback
    turns = [
        {"speaker": "You", "text": "what would you like", "pd": "INFO_GATHER"},
        {"speaker": "Them", "text": "i need the hats you can have everything else", "pd": "VALUE_CLAIMING"},
        {"speaker": "You", "text": "you take all balls and i keep everything", "pd": "VALUE_CLAIMING"},
        {"speaker": "Them", "text": "no deal . i need the hats or no deal .", "pd": "VALUE_CLAIMING"},
        {"speaker": "You", "text": "no deal", "pd": "VALUE_CLAIMING"},
        {"speaker": "Them", "text": "no deal .", "pd": "VALUE_CLAIMING"},
        {"speaker": "You", "text": "no deal", "pd": "VALUE_CLAIMING"},
        {"speaker": "Them", "text": "no deal .", "pd": "VALUE_CLAIMING"},
        {"speaker": "You", "text": "no", "pd": "VALUE_CLAIMING"},
        {"speaker": "Them", "text": "no deal .", "pd": "VALUE_CLAIMING"},
        {"speaker": "You", "text": "", "pd": "INFO_GATHER"},
    ]
    
    print("=== DEBUGGING CONVERSATION ===")
    print("Conversation:")
    for i, turn in enumerate(turns, 1):
        print(f"{i}. {turn['speaker']}: {turn['text']}")
    
    print("\n=== Step 1: Item Identification ===")
    all_text = " ".join([turn.get("text", "") for turn in turns if turn.get("text")])
    print(f"All text: '{all_text}'")
    
    items = _identify_items_with_llm(all_text, "qwen3:latest")
    print(f"Identified items: {items}")
    
    print("\n=== Step 2: Item Analysis ===")
    priorities = analyze_item_priorities(turns, "qwen3:latest")
    print(f"Priorities: {priorities}")
    
    print("\n=== Step 3: Numerical Concessions ===")
    numerical_advice = suggest_numerical_concessions(priorities)
    print(f"Numerical advice: {numerical_advice}")
    
    print("\n=== Step 4: Rule-based Advice ===")
    my_moves = [t["pd"] for t in turns if t["speaker"] == "You"]
    opp_moves = [t["pd"] for t in turns if t["speaker"] == "Them"]
    scores = score_turns(my_moves, opp_moves)
    print(f"Scores: {scores}")
    
    base_advice = rule_based_advice_enhanced(scores, priorities, {})
    print(f"Rule-based advice: {base_advice}")
    
    print("\n=== Step 5: Final Hint ===")
    if numerical_advice and numerical_advice != "Consider making a small concession to move the negotiation forward.":
        hint = f"{base_advice} Specifically: {numerical_advice}"
    else:
        hint = base_advice
    print(f"Final hint: {hint}")
    
    print("\n=== ANALYSIS ===")
    print("Problem: The conversation contains 'hats' and 'balls' but the advice is still generic.")
    print("Expected: Advice should mention 'hats', 'balls', 'concessions', 'percentages'")
    print("Actual: Check if the hint contains specific items")

if __name__ == "__main__":
    debug_conversation()
