#!/usr/bin/env python3
"""
Create sample Deal-or-No-Deal data for testing the negotiation chatbot.

This creates a minimal validation.jsonl file with realistic negotiation samples
that match the format expected by the application.
"""

import json
import os
from pathlib import Path

def create_sample_data():
    """Create sample negotiation data in DoND format."""

    samples = [
        {
            "dialogue": "Hello! I'd like to discuss how we can split these items. <eos> Hi! I'm interested in the books and basketballs. <eos> I also need some books. How about I take 2 books and you take 1 book? <eos> That could work. What about the hats? <eos> You can have all the hats. I don't need them. <eos> Great! And the basketballs? <eos> How about we split them? I'll take 1 basketball and you take 1. <eos> That sounds fair. So I get 1 book, 2 hats, and 1 basketball? <eos> Yes, and I get 2 books and 1 basketball. Deal? <eos> Deal! <selection>",
            "input": {
                "count": [3, 2, 2],
                "value": [2, 1, 3]
            },
            "partner_input": {
                "value": [1, 3, 2]
            },
            "output": "2 0 1 1 2 1"
        },
        {
            "dialogue": "Hi there! Let's see what we have here. <eos> Hello! I see we have books, hats, and basketballs to divide. <eos> I really need the basketballs for my team. <eos> I understand. I'm more interested in the books and hats. <eos> How about you take all the books and hats, and I take all the basketballs? <eos> Hmm, that's too much for you. How about I keep 1 basketball? <eos> Ok, you can have 1 basketball. I'll take the other 2. <eos> Great! And I'll take all 3 books and both hats. <eos> Deal! <selection>",
            "input": {
                "count": [3, 2, 3],
                "value": [3, 2, 1]
            },
            "partner_input": {
                "value": [2, 1, 3]
            },
            "output": "3 2 1 0 0 2"
        },
        {
            "dialogue": "Good morning! Shall we negotiate? <eos> Good morning! Yes, let's figure this out. <eos> I value the hats highly. Can I have both? <eos> I need at least one hat. But you can have the other one. <eos> Fair enough. What about the books? <eos> I'd like 2 books if possible. <eos> I can do that. I'll take 1 book. <eos> And the basketball? <eos> We only have one. How about we flip for it? <eos> Or I could just give it to you since I'm getting most of the books. <eos> That works! Thank you! <selection>",
            "input": {
                "count": [3, 2, 1],
                "value": [1, 3, 2]
            },
            "partner_input": {
                "value": [2, 2, 1]
            },
            "output": "1 1 1 2 1 0"
        },
        {
            "dialogue": "Hello! <eos> Hi! <eos> I really need all the books. They're very important to me. <eos> I also need the books. Can we split them? <eos> No, I need all 3 books. <eos> That's not fair. I can't agree to that. <eos> Then take the hats and basketballs. <eos> I want some books too. <eos> I'm not willing to give up any books. <eos> Then we have no deal.",
            "input": {
                "count": [3, 2, 2],
                "value": [3, 1, 1]
            },
            "partner_input": {
                "value": [2, 2, 1]
            },
            "output": "0 0 0 0 0 0"
        },
        {
            "dialogue": "Hi! Let's make this quick. <eos> Sure, what do you propose? <eos> 50-50 split on everything. <eos> We can't split evenly - there are 3 books. <eos> Then let's do this: I take 2 books, you take 1 book. We split the rest evenly. <eos> That means I get 1 book, 1 hat, and 1 basketball? <eos> Yes, and I get 2 books, 1 hat, and 1 basketball. <eos> Sounds fair! <eos> Great, deal! <selection>",
            "input": {
                "count": [3, 2, 2],
                "value": [2, 2, 2]
            },
            "partner_input": {
                "value": [2, 2, 2]
            },
            "output": "2 1 1 1 1 1"
        },
        {
            "dialogue": "Hi! I'm hoping we can work something out. <eos> Me too! What are you most interested in? <eos> I really need the basketballs. <eos> I don't care much about basketballs. You can have them all. <eos> Thank you! What do you want in return? <eos> I'd like all the hats and at least 2 books. <eos> I can give you all the hats and 2 books. I'll keep 1 book. <eos> Perfect! That works for me. <eos> Deal! <selection>",
            "input": {
                "count": [3, 2, 2],
                "value": [1, 1, 3]
            },
            "partner_input": {
                "value": [3, 2, 1]
            },
            "output": "1 0 2 2 2 0"
        },
        {
            "dialogue": "Hello! <eos> Hi there! <eos> I need the books for school. <eos> I also need books. <eos> How about we each take what we value most? <eos> That makes sense. What do you value? <eos> Books first, then hats, then basketballs. <eos> I value hats most, then basketballs, then books. <eos> Great! So you take all the hats and basketballs, I take all the books? <eos> That's perfect! <eos> Deal! <selection>",
            "input": {
                "count": [3, 2, 2],
                "value": [3, 2, 1]
            },
            "partner_input": {
                "value": [1, 3, 2]
            },
            "output": "3 0 0 0 2 2"
        },
        {
            "dialogue": "Hi! <eos> Hello! <eos> What do you want? <eos> I want a fair split. <eos> Me too. <eos> How about 1 book each, 1 hat each, and 1 basketball each? <eos> But there are 3 books. <eos> You can have the extra book. <eos> Ok, deal! <selection>",
            "input": {
                "count": [3, 2, 2],
                "value": [1, 1, 1]
            },
            "partner_input": {
                "value": [1, 1, 1]
            },
            "output": "1 1 1 2 1 1"
        },
        {
            "dialogue": "Good day! <eos> Good day to you! <eos> I have a proposal. <eos> I'm listening. <eos> You seem reasonable. How about I take 2 books, 1 hat, you take 1 book, 1 hat, 2 basketballs? <eos> Why should you get more books? <eos> Because I'm giving you both basketballs. <eos> That's fair. I accept! <selection>",
            "input": {
                "count": [3, 2, 2],
                "value": [2, 1, 2]
            },
            "partner_input": {
                "value": [1, 2, 3]
            },
            "output": "2 1 0 1 1 2"
        },
        {
            "dialogue": "Hello! I hope we can reach an agreement. <eos> I hope so too. <eos> What are your priorities? <eos> I need books for my research. <eos> I also need books. How about we split everything equally? <eos> That won't work - odd numbers. <eos> True. How about you take 2 books, I take 1 book and all the hats and basketballs? <eos> No, I need more than that. <eos> What do you suggest? <eos> I need at least 2 books and 1 hat. <eos> Ok, you can have 2 books and 1 hat. I'll take 1 book, 1 hat, and 2 basketballs. <eos> Deal! <selection>",
            "input": {
                "count": [3, 2, 2],
                "value": [3, 1, 1]
            },
            "partner_input": {
                "value": [2, 2, 2]
            },
            "output": "2 1 0 1 1 2"
        }
    ]

    return samples

def main():
    """Create sample dataset files."""
    print("=" * 60)
    print("Creating Sample Deal-or-No-Deal Dataset")
    print("=" * 60)
    print()

    # Create directory
    data_dir = Path("deal_or_no_dialog/exported")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {data_dir}")

    # Generate samples
    samples = create_sample_data()
    print(f"✓ Generated {len(samples)} sample negotiations")

    # Write validation.jsonl (the one used by the visualizer)
    validation_file = data_dir / "validation.jsonl"
    with open(validation_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"✓ Created {validation_file}")
    print(f"  File size: {validation_file.stat().st_size:,} bytes")
    print(f"  Samples: {len(samples)}")

    # Also create train.jsonl and test.jsonl with the same data for completeness
    for split_name in ["train", "test"]:
        split_file = data_dir / f"{split_name}.jsonl"
        with open(split_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        print(f"✓ Created {split_file}")

    # Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    for split in ["train", "validation", "test"]:
        filepath = data_dir / f"{split}.jsonl"
        if filepath.exists():
            # Count lines
            with open(filepath, 'r') as f:
                line_count = sum(1 for _ in f)
            size = filepath.stat().st_size
            print(f"✓ {split}.jsonl: {size:,} bytes, {line_count} samples")
        else:
            print(f"✗ {split}.jsonl: NOT FOUND")

    print("\n✅ Setup complete!")
    print(f"\nDataset location: {data_dir.absolute()}")
    print("\nYou can now run the Gradio UI with DoND features enabled:")
    print("  python -m negotiation_chatbot.gradio_ui")
    print("\nThe DoND Conversation Visualizer will now work with these sample negotiations!")

if __name__ == "__main__":
    main()
