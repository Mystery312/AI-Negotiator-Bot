"""Run automated negotiation experiments."""

import argparse
from typing import Dict, List
import json
import os
from datetime import datetime

import os
import sys

# Add parent directory to sys.path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import from app package
from app.automate import run_automated_conversation, run_dond_sample, evaluate_conversation
from app.dond_data import load_dond
from app.preference import estimate_preferences

def run_experiment(
    n_samples: int = 10,
    max_turns: int = 10,
    model: str = "ollama:qwen3:latest",
    output_dir: str = "experiments"
) -> None:
    """Run experiments on multiple samples and save results."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = []
    for i in range(n_samples):
        print(f"\nRunning experiment {i+1}/{n_samples}")
        
        try:
            # Run conversation
            history = run_dond_sample(i, model=model)
            
            # Get the item counts from the first bot proposal message
            counts = None
            for msg in history:
                if msg["role"] == "Coach" and msg["speaker"] == "Bot":
                    # Parse the counts from the proposal
                    lines = msg["text"].split("\n")
                    if len(lines) >= 3:
                        counts = {}
                        for line in lines[1:3]:  # "You get" and "They get" lines
                            items = line.split(": ", 1)[1].split(", ")
                            for item in items:
                                item_name, count = item.split(": ")
                                if item_name not in counts:
                                    counts[item_name] = 0
                                counts[item_name] += int(count)
                        break
            
            if counts:
                # Estimate final preferences
                turns = [msg["text"] for msg in history if msg["role"] in ["A", "B"]]
                w_you, w_them = estimate_preferences(turns)
                
                if w_you and w_them:
                    # Calculate final utilities
                    u_you, u_them = evaluate_conversation(history, counts, w_you, w_them)
                    
                    result = {
                        "sample_idx": i,
                        "n_turns": len([m for m in history if m["role"] in ["A", "B"]]),
                        "n_proposals": len([m for m in history if m["speaker"] == "Bot"]),
                        "utility_you": u_you,
                        "utility_them": u_them,
                        "final_preferences_you": w_you,
                        "final_preferences_them": w_them,
                        "item_counts": counts,
                        "history": history
                    }
                    results.append(result)
                    
                    # Save individual conversation
                    conv_file = os.path.join(output_dir, f"conversation_{timestamp}_{i}.json")
                    with open(conv_file, "w") as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"Saved conversation to {conv_file}")
                    print(f"Final utilities - You: {u_you:.3f}, Them: {u_them:.3f}")
        
        except Exception as e:
            print(f"Error in experiment {i}: {e}")
            continue
    
    # Save summary results
    summary = {
        "timestamp": timestamp,
        "n_samples": n_samples,
        "max_turns": max_turns,
        "model": model,
        "n_successful": len(results),
        "avg_turns": sum(r["n_turns"] for r in results) / len(results) if results else 0,
        "avg_proposals": sum(r["n_proposals"] for r in results) / len(results) if results else 0,
        "avg_utility_you": sum(r["utility_you"] for r in results) / len(results) if results else 0,
        "avg_utility_them": sum(r["utility_them"] for r in results) / len(results) if results else 0,
    }
    
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExperiment complete! Summary saved to {summary_file}")
    print(f"Successfully ran {len(results)}/{n_samples} experiments")
    print(f"Average utilities - You: {summary['avg_utility_you']:.3f}, Them: {summary['avg_utility_them']:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Run automated negotiation experiments")
    parser.add_argument("--n", type=int, default=10, help="Number of samples to run")
    parser.add_argument("--turns", type=int, default=10, help="Maximum turns per conversation")
    parser.add_argument("--model", type=str, default="ollama:qwen3:latest", help="Model to use for coach")
    parser.add_argument("--output", type=str, default="experiments", help="Output directory")
    
    args = parser.parse_args()
    run_experiment(
        n_samples=args.n,
        max_turns=args.turns,
        model=args.model,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()
