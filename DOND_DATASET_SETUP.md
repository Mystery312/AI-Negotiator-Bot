# Deal-or-No-Deal Dataset Setup

## Overview

The `deal_or_no_dialog` directory contains the Deal-or-No-Deal negotiation dataset used by the chatbot application for:
- **Coach Advice**: Training and testing the AI negotiation coach
- **Conversation Visualization**: Analyzing real negotiation patterns
- **Preference Estimation**: Learning negotiator preferences from dialogue
- **Pareto Optimization**: Testing optimal proposal generation

## Directory Structure

```
deal_or_no_dialog/
└── exported/
    ├── train.jsonl         # Training data (10 samples)
    ├── validation.jsonl    # Validation data (10 samples)
    └── test.jsonl          # Test data (10 samples)
```

## Dataset Format

Each line in the JSONL files contains a negotiation sample with:

```json
{
  "dialogue": "turn1 <eos> turn2 <eos> ... <selection>",
  "input": {
    "count": [3, 2, 2],  // Quantities of item0, item1, item2
    "value": [2, 1, 3]   // Your values for each item
  },
  "partner_input": {
    "value": [1, 3, 2]   // Partner's values for each item
  },
  "output": "2 0 1 1 2 1"  // Final allocation (your items, partner items)
}
```

### Example Negotiation

```
Input:
- Items: 3 books, 2 hats, 2 basketballs
- Your values: books=2, hats=1, basketballs=3
- Partner values: books=1, hats=3, basketballs=2

Dialogue:
- "Hello! I'd like to discuss how we can split these items."
- "Hi! I'm interested in the books and basketballs."
- "I also need some books. How about I take 2 books and you take 1 book?"
- "That could work. What about the hats?"
- "You can have all the hats. I don't need them."
- "Great! And the basketballs?"
- "How about we split them? I'll take 1 basketball and you take 1."
- "That sounds fair. So I get 1 book, 2 hats, and 1 basketball?"
- "Yes, and I get 2 books and 1 basketball. Deal?"
- "Deal!"

Output:
- You get: 2 books, 0 hats, 1 basketball
- Partner gets: 1 book, 2 hats, 1 basketball
```

## Setup Instructions

### Option 1: Use Sample Data (Quick Start)

Run the provided script to create sample negotiation data:

```bash
python create_sample_dond_data.py
```

This creates 10 realistic negotiation samples suitable for testing and demonstration.

### Option 2: Download Original Dataset

To get the full Facebook Research Deal-or-No-Deal dataset:

1. **Clone the original repository:**
   ```bash
   git clone https://github.com/facebookresearch/end-to-end-negotiator.git
   cd end-to-end-negotiator
   ```

2. **Extract the data:**
   ```bash
   # The data is in src/data/negotiate/
   mkdir -p deal_or_no_dialog/exported
   cp src/data/negotiate/*.txt deal_or_no_dialog/exported/
   ```

3. **Convert format (if needed):**
   ```bash
   python setup_dond_dataset.py
   ```

### Option 3: Manual Setup

1. Create the directory structure:
   ```bash
   mkdir -p deal_or_no_dialog/exported
   ```

2. Create or copy `train.jsonl`, `validation.jsonl`, and `test.jsonl` files

3. Verify the format matches the structure shown above

## Verification

Test that the dataset is properly loaded:

```bash
python -c "
from negotiation_chatbot.dond_data import load_dond
samples = load_dond('validation')
print(f'Loaded {len(samples)} samples')
print(f'First sample: {samples[0].turns[0][:50]}...')
"
```

Expected output:
```
Found data directory at primary path: /path/to/deal_or_no_dialog/exported
Looking for data file at: /path/to/deal_or_no_dialog/exported/validation.jsonl
Loaded 10 samples
First sample: Hello! I'd like to discuss how we can split these...
```

## Using the Dataset in the Application

### 1. DoND Conversation Visualizer

Access via the Gradio UI at http://localhost:7860:

1. Open the **"DoND Conversation Visualizer"** accordion
2. Use the slider to select a sample (0-9)
3. Click **"Load Sample"** to visualize the negotiation
4. View:
   - Item counts and values
   - Speaker statistics
   - Message timeline
   - Deal outcome analysis
   - Coach advice (when enabled)

### 2. Preference Estimator Training

Train a model to estimate negotiator preferences:

```bash
python -m negotiation_chatbot.train_prefs
```

### 3. Pareto Coach Effectiveness Simulator

Test how the AI coach improves negotiation outcomes:

1. Open the **"Pareto Coach Effectiveness Simulator"** accordion
2. Set parameters:
   - Number of samples (10-200)
   - Baseline strategy (equal, greedy, walkaway, statusquo)
   - Success threshold ratio (0.7-1.0)
3. Click **"Run Simulation"**
4. Review results showing rescue rate and transcripts

### 4. DoND Simulations

Run bot-vs-bot simulations:

```bash
python -m negotiation_chatbot.simulate_dond
```

## Features Enabled by Dataset

✅ **DoND Sample Loading**: Load pre-configured negotiation scenarios
✅ **Conversation Analysis**: Analyze dialogue patterns and outcomes
✅ **Deal Detection**: LLM-based or keyword-based outcome prediction
✅ **Item Mention Tracking**: Track which items are discussed
✅ **Speaker Statistics**: Analyze participation patterns
✅ **Coach Advice**: Get AI guidance for each negotiation turn
✅ **Pareto Simulations**: Test optimal proposal strategies

## Troubleshooting

### "No validation samples available"

**Cause**: The dataset files are missing or in the wrong location

**Fix**:
```bash
# Option 1: Run the sample data script
python create_sample_dond_data.py

# Option 2: Check environment variable
export DOND_DATA_DIR="/path/to/deal_or_no_dialog/exported"

# Option 3: Verify files exist
ls -lh deal_or_no_dialog/exported/
```

### "Invalid JSONL format"

**Cause**: Dataset files are corrupted or in wrong format

**Fix**:
```bash
# Re-run the sample data script
python create_sample_dond_data.py

# Or verify JSON format
python -c "
import json
with open('deal_or_no_dialog/exported/validation.jsonl', 'r') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            print(f'Line {i}: OK')
        except Exception as e:
            print(f'Line {i}: ERROR - {e}')
"
```

### "Sample X missing 'turns' list"

**Cause**: Data structure doesn't match expected format

**Fix**: Ensure each sample has:
- `dialogue` field with `<eos>` separated turns
- `input.count` with item quantities
- `input.value` with your values
- `partner_input.value` with partner values
- `output` with final allocations

## Dataset Statistics

Current sample dataset (10 negotiations):
- **Deal outcomes**: 9 deals, 1 no-deal
- **Average turns**: 9-11 turns per negotiation
- **Item types**: Books (item0), Hats (item1), Basketballs (item2)
- **Typical quantities**: 2-3 of each item type
- **Negotiation patterns**: Collaborative, competitive, compromising

## References

- **Original Paper**: [Deal or No Deal? End-to-End Learning for Negotiation Dialogues](https://arxiv.org/abs/1706.05125)
- **Facebook Research**: [End-to-End Negotiator](https://github.com/facebookresearch/end-to-end-negotiator)
- **Dataset**: Deal-or-No-Deal Negotiation Corpus

## Notes

- The sample dataset is for demonstration purposes
- For research or production use, download the full dataset (5,808 training samples)
- The application gracefully handles missing dataset files
- Coach advice and visualizations work with any valid JSONL format
