# AI Negotiation Chatbot: Comprehensive Project Summary

## 1. Project Overview

This project implements an AI-powered negotiation coaching system that provides real-time strategic advice during multi-party negotiations. The system combines several AI and game-theoretic techniques—large language model (LLM) prompting, retrieval-augmented generation (RAG), a DistilBERT-based preference estimator, Pareto-optimal allocation solving, and a Prisoner's Dilemma–inspired move taxonomy—to analyze live negotiation dialogues and generate actionable coaching suggestions. A Gradio web interface allows two human negotiators to exchange messages while receiving inline strategic guidance from an AI coach.

### Key Capabilities

- **Real-time negotiation coaching**: After each conversational turn, the system classifies the move, estimates both parties' preferences, computes a Pareto-efficient allocation, retrieves relevant prior negotiation examples, and synthesizes a concise strategic recommendation.
- **Rich move classification**: Every utterance is classified using a 40-move taxonomy spanning 10 categories (information, value creation, value claiming, concession, pressure, relationship, communication, strategic, defensive, closure), each with an associated intensity level (low/medium/high).
- **Prisoner's Dilemma labeling**: Each move is also mapped to a binary Cooperate (C) or Defect (D) label, enabling game-theoretic scoring of reciprocity, volatility, and momentum.
- **Pareto-optimal proposal generation**: A preference estimation model infers each party's item valuations from conversation text; these weights feed an exhaustive Pareto frontier solver that identifies Nash-product–optimal allocations subject to per-side fairness constraints.
- **Multi-provider LLM support**: The system supports Ollama (local models such as Qwen 3, LLaMA 3.2, Mistral), Google Gemini, and OpenAI, abstracted behind a unified `LLMClient` interface.
- **Dual RAG system**: A generic negotiation-tactics RAG (ChromaDB + Sentence Transformers) and a CaSiNo-corpus–specific RAG provide evidence-grounded advice with provenance tracking.
- **Deal-or-No-Deal dataset integration**: The Facebook Deal-or-No-Deal dialogue corpus is used for visualization, simulation benchmarking, and preference model training.

---

## 2. System Architecture

### 2.1 Two-Process Design

The application runs as two independent processes:

| Process | Framework | Default Port | Entry Point |
|---------|-----------|-------------|-------------|
| Backend API | FastAPI + Uvicorn | 8001 | `python -m negotiation_chatbot.main` |
| Frontend UI | Gradio 6.0 | 7860 | `python -m negotiation_chatbot.gradio_ui` |

The Gradio frontend communicates with the FastAPI backend exclusively via HTTP REST calls. On startup, the frontend probes a prioritized list of candidate URLs (Docker service name, localhost:8001, localhost:8000) and locks onto the first responding backend.

### 2.2 Request Pipeline

When a user sends a message through the Gradio UI, the following pipeline executes:

```
User Message (Gradio UI)
    │
    ▼
POST /chat  (FastAPI backend)
    │
    ├─ Step 1: Move Classification (ingest.py → GPT-4o-mini)
    │     Classifies utterance into move_type ∈ {concession, threat, info_share, cooperate, defect}
    │     Maps to Prisoner's Dilemma label: C or D
    │
    ├─ Step 2: Graph Storage (graph.py → Neo4j, optional)
    │     Stores turn as Neo4j node with retry + exponential backoff
    │     Records speaker, text, move, PD label, timestamp
    │
    ├─ Step 3: Coach Advice Generation (coach.py → get_advice_async)
    │     ├─ 3a: Fetch last 5 turns from Neo4j (or in-memory fallback)
    │     ├─ 3b: Guard checks (min 2 turns, both parties must have spoken)
    │     ├─ 3c: Preference estimation (DistilBERT → softmax weight vectors)
    │     ├─ 3d: Pareto-optimal split computation (exhaustive enumeration)
    │     ├─ 3e: LLM-based item priority analysis
    │     ├─ 3f: LLM-based current offers extraction
    │     ├─ 3g: Score turns → rich metrics (cooperation rate, competition rate,
    │     │       reciprocity, volatility, momentum, phase, risk level, etc.)
    │     ├─ 3h: Rule-based strategy selection (12 pluggable strategies)
    │     ├─ 3i: RAG context retrieval (CaSiNo corpus or generic tactics)
    │     ├─ 3j: LLM reply generation with hint injection
    │     └─ 3k: De-duplication against recent advice history
    │
    └─ Step 4: Response
          Returns: { advice, reply, rag_source, rag_context }
```

### 2.3 Async Execution Model

The `get_advice_async()` function uses `asyncio.gather()` to parallelize independent blocking operations through a `ThreadPoolExecutor(max_workers=4)`. The parallelized operations include:

- Neo4j graph queries
- DistilBERT preference inference
- LLM-based item analysis (priorities and offers)
- RAG vector search

A TTL-based caching layer (configurable per function: 60s–600s) reduces redundant computation. Cache keys are derived from MD5 hashes of serialized function arguments.

### 2.4 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat` | POST | Process a chat message and return coaching advice |
| `/health` | GET | Health check |
| `/label` | POST | Classify text into move type and PD label |
| `/v1/chat/completions` | POST | OpenAI-compatible endpoint for external tool integration |
| `/graph/{conv_id}` | GET | Fetch conversation graph data for visualization |
| `/stats/{conv_id}` | GET | Fetch conversation statistics |
| `/deal/outcome` | POST | Record deal/no-deal outcome |
| `/deal/mark-accepted/{conv_id}` | POST | Mark a conversation turn as accepted |
| `/casino/cache` | GET | CaSiNo RAG cache info |
| `/casino/cache/clear` | POST | Clear CaSiNo cache |
| `/casino/reload` | POST | Reload CaSiNo RAG system |

---

## 3. Core Components

### 3.1 Move Classification (`ingest.py`)

Uses OpenAI's GPT-4o-mini to classify each negotiation utterance into a structured JSON label:

```json
{
  "move_type": "concession | threat | info_share | cooperate | defect",
  "pd": "C | D"
}
```

The classification prompt instructs the model to map cooperative moves (concession, info_share, cooperate) to `C` and competitive moves (threat, defect) to `D`. The module also supports bulk ingestion from CSV and PDF files, with automatic speaker detection via regex patterns for PDF documents.

### 3.2 Negotiation Coach (`coach.py`)

The coach is the central intelligence module (~2,400 lines). Its major subsystems are:

#### 3.2.1 Rich Move Taxonomy

A 40-move taxonomy organized into 10 categories:

| Category | Moves | Intensity Examples |
|----------|-------|-------------------|
| Information | INFO_GATHER, INFO_SHARE, INFO_REQUEST, INFO_DISCLOSE | Low |
| Value Creation | EXPAND_PIE, INTEGRATIVE, CREATIVE_SOLUTION, MUTUAL_GAIN | Medium |
| Value Claiming | DISTRIBUTIVE, HARD_BALL, POSITIONAL, COMPETITIVE | High |
| Concession | CONCESSION, CONDITIONAL_CONCESSION, GRADUAL_CONCESSION, RECIPROCAL_CONCESSION | Medium |
| Pressure | DEADLINE, ULTIMATUM, WALK_AWAY, ESCALATION | High |
| Relationship | BUILD_TRUST, APPEAL_EMOTION, RELATIONSHIP_FOCUS, COLLABORATIVE | Low |
| Communication | SUMMARIZE, CLARIFY, REFORMULATE, ACTIVE_LISTEN | Low |
| Strategic | ANCHOR, FRAMING, NORM_APPEAL, PRECEDENT | Medium |
| Defensive | DEFEND_POSITION, COUNTER_OFFER, REJECT, STALL | Medium |
| Closure | ACCEPT, FINAL_OFFER, AGREEMENT, CLOSE_DEAL | High |

#### 3.2.2 Scoring Engine (`score_turns`)

Computes a rich set of metrics from the last 6 turns of each party:

- **Cooperation/Competition rates**: Fraction of moves in value-creation vs. value-claiming categories.
- **Reciprocity**: How often a player mirrors the opponent's previous move category.
- **Volatility**: Frequency of category switches by the opponent.
- **Momentum**: Direction of recent cooperation/competition trend (+1 toward cooperation, −1 toward competition).
- **Phase detection**: Opening (≤4 moves), Exploration (5–12), Bargaining (13–20), Closing (>20).
- **Risk assessment**: Composite score from high competition, high volatility, pressure tactics, and defensive posture.
- **Intensity distributions**: Per-party breakdown of low/medium/high intensity moves.
- **Additional metrics**: Concession ratio, pressure ratio, information balance, strategic advantage, communication quality, trust level, defensive ratio, value balance.
- **No-deal tracking**: Count of explicit "no deal" utterances in the last 6 turns.

#### 3.2.3 Strategy Selection

Twelve pluggable `Strategy` dataclass instances, each with a trigger predicate (a lambda function over the scoring dictionary) and an advice template:

| # | Strategy | Trigger Condition |
|---|----------|------------------|
| 1 | TitForTat | High reciprocity (>0.6), competitive opponent (>0.4), low volatility (<0.3) |
| 2 | EscalateDeadlock | High competition (>0.7), low volatility (<0.2), low reciprocity (<0.3) |
| 3 | PackageOffer | Cooperative opponent (>0.5) in bargaining phase |
| 4 | MirrorCoop | High opponent cooperation (>0.7), lower self-cooperation (<0.5) |
| 5 | StabilizeVolatile | High volatility (>0.6) |
| 6 | BuildMomentum | Balanced cooperation (0.4–0.6) on both sides |
| 7 | GatherInfo | Early conversation (≤2 moves) |
| 8 | TestFlexibility | Low opponent cooperation (<0.4), high volatility (>0.4) |
| 9 | CreateValue | High mutual cooperation (both >0.6) |
| 10 | BreakPattern | Low reciprocity (<0.3) |
| 11 | ManageIntensity | High opponent intensity (avg >2.5 on 1–3 scale) |
| 12 | NiceGuyRecovery | Self is blocking (comp_me >0.5), opponent cooperative (coop_opp >0.6), multiple "no deal" signals |

Each strategy includes a structured output with reasoning, example offer, example multi-turn dialogue, implementation steps, risk/benefit assessment, a confidence score (0.0–1.0), and a priority level (high/medium/low).

A "nice-guy" (tit-for-tat) bias is injected: when the opponent is cooperative (>0.6) and the user is at least somewhat cooperative (>0.4), the MirrorCoop strategy is prioritized regardless of trigger order.

#### 3.2.4 Numerical Offer Tracking

The coach extracts and tracks numerical offers using regex patterns for:
- Prices ($X, X dollars)
- Percentages (X% discount)
- Quantities (X units/items)
- Time commitments (X days/weeks/months)

Concession histories are tracked across rounds via `Offer` and `Concession` dataclasses, enabling analysis of concession rate (concessions/hour) and offer progression (trend, volatility, total change).

#### 3.2.5 LLM Reply Generation

The system prompt constrains the LLM to produce a single, crisp, context-specific suggestion (approximately 35 words maximum) grounded in the last exchange. The prompt template injects:
- The strategic hint (Pareto suggestion or rule-based advice)
- Recognized item names with quantities
- Conversation summary (compressed dialogue for token efficiency)
- RAG context (CaSiNo or generic)
- The last conversational turn
- Whether to use percentages (when no explicit item counts are mentioned in conversation)

A `<think>` tag stripping pass cleans chain-of-thought artifacts from models like Qwen 3. Responses are further truncated by a `_concise()` helper enforcing a 40-word maximum for the advice string.

#### 3.2.6 Advice De-duplication

An `ALT_CACHE` (a `defaultdict(list)`) maintains rolling variant pools per advice "idea." When repeated advice is detected (via Neo4j lookup or string comparison), the system generates synonym-based rephrasings using a verb-synonym table (e.g., "Consider" → "Try" / "Perhaps" / "You could"). This ensures variety across turns.

### 3.3 Preference Estimation (`preference.py`)

A lightweight neural model for inferring each party's item valuations from conversation text:

**Architecture:**
```
DistilBERT (distilbert-base-uncased)
    → [CLS] token embedding (768-dim)
    → Linear head_you (768 → n_issues) → softmax
    → Linear head_them (768 → n_issues) → softmax
```

- **Input**: Concatenation of the last ≤6 utterances, tokenized with the DistilBERT tokenizer.
- **Output**: Two probability distributions over `n_issues` items (default: 3), representing estimated importance weights for each party. These weights are used directly by the Pareto solver.
- **Training**: Supervised on Deal-or-No-Deal ground-truth value annotations using AdamW optimizer with linear warmup schedule (`train_prefs.py`). Training command: `python train_pref.py --model_out checkpoints/pref_estimator.pt --epochs 3 --batch_size 32 --lr 3e-5`.
- **Inference**: Cached for 5 minutes per conversation context (TTL = 300s). The model runs on CPU by default; GPU is used if available.

### 3.4 Pareto Solver (`pareto.py`)

Computes Pareto-optimal allocations via exhaustive enumeration:

1. **Enumeration**: Generates all possible splits of items between two parties using `itertools.product` over the range of each item's count. For 3 items with counts (3, 2, 1), this produces 4 × 3 × 2 = 24 candidate allocations.
2. **Dominance filtering**: Removes dominated allocations to construct the Pareto frontier. An allocation `A` dominates `B` if `A` is at least as good for both parties and strictly better for at least one.
3. **Best offer selection** (`best_offer`): Selects the frontier point satisfying per-side slack constraints:
   - Identifies the weaker side (lower baseline utility) and the stronger side.
   - The weaker side must retain at least `ratio × baseline_utility`.
   - The stronger side is allowed an additional `DELTA = 0.05` slack (dynamic slack for asymmetric negotiations).
   - Sweeps all frontier points; the first to satisfy both constraints is returned (early exit).
   - If no point meets constraints, falls back to max-min fairness (maximizes the minimum utility across both parties).
   - An equal-blend nudge is applied as a final attempt: averages the best max-min split with the equal split.
4. **Utility function**: Weighted sum: `u(split, weights) = Σ(quantity_i × weight_i)`.

### 3.5 Graph Database (`graph.py`)

Neo4j integration (optional, disabled by default via `ENABLE_NEO4J=false`) provides:

- **Turn storage**: Each utterance stored as a `Turn` node linked to `Conv` (conversation) and `Person` (speaker) nodes.
- **Advice history**: `Advice` nodes linked to `Person` nodes, enabling de-duplication checks over the last N pieces of advice.
- **Deal outcome tracking**: `Outcome` nodes recording deal/no-deal status and acceptance markers.
- **Graph queries**: `fetch_last_n` (recent turns), conversation stats (per-speaker turn counts, move distributions), and full graph data for visualization.
- **Resilience**: Connection pooling (50 max connections, 1-hour lifetime), exponential backoff retry (3 attempts, 0.1s initial delay, doubling each attempt), and separate service-unavailable/session-expired exception handling.
- **Graceful degradation**: When Neo4j is unavailable, the system continues operating without graph features; all graph operations fail silently with logged warnings.

### 3.6 RAG Systems

#### 3.6.1 Generic Negotiation Tactics RAG (`rag.py`)

- **Vector store**: ChromaDB with persistent storage (`./chroma_db/`).
- **Embedding model**: `all-MiniLM-L6-v2` (Sentence Transformers, 384-dim).
- **Content**: 8 curated negotiation tactic documents covering: aggressive negotiators, information sharing, concessions, trust building, threats/ultimatums, cooperative strategies, defection response, and deadlock resolution.
- **Retrieval**: Top-5 nearest neighbors by cosine similarity.
- **Initialization**: Lazy-loaded on first query; auto-populates the collection if empty.

#### 3.6.2 CaSiNo Corpus RAG (`casino_rag.py`)

- **Data source**: CaSiNo (Camp Site Negotiation) corpus from ConvoKit, containing multi-issue negotiations between human participants.
- **Vector store**: Separate ChromaDB collection (`casino_negotiations`) with cosine similarity.
- **Eval/Train partition**: 15 reserved dialogue IDs (5 per category) are held out for evaluation; only non-reserved dialogues are indexed.
- **Caching**: Pickle-serialized corpus cache + JSON eval ID list in `./cache/` directory for fast subsequent loads.
- **Strategy-based retrieval**: The coach maps its selected strategy to a CaSiNo retrieval category (escalation, concession, stabilization, mirroring, exploration, goodwill, information_gathering) and fetches relevant dialogue examples.
- **Few-shot examples**: Returns human-authored dialogue pairs for the matched strategy, formatted as `H1:` / `H2:` utterance pairs for in-context learning.

#### 3.6.3 Academic PDF RAG (`build_vector_db.py`)

A separate offline pipeline for indexing academic negotiation literature from PDF/TXT files:
- Processes documents from `./data_sources/` (includes texts on *Getting to Yes*, diplomatic negotiation, Pareto optimality, Axelrod's tournament).
- Uses `RecursiveCharacterTextSplitter` with chunk_size=1000, overlap=200.
- Indexes into the same ChromaDB `negotiation_tactics` collection.
- Run manually: `python negotiation_chatbot/build_vector_db.py`.

### 3.7 LLM Client (`llm_client.py`)

A provider-abstraction layer with a factory pattern:

```
LLMClient (unified interface)
  ├── OllamaProvider  → OpenAI-compatible API at OLLAMA_BASE_URL/v1
  └── GeminiProvider  → google.generativeai SDK (lazy-imported)
```

- **Ollama**: Uses a dummy API key with the OpenAI Python SDK; default model `qwen3:latest`; dynamically discovers available models via the `/api/tags` endpoint.
- **Gemini**: Requires `GOOGLE_API_KEY`; converts OpenAI message format to Gemini format (role mapping: "assistant" → "model"; system messages prepended to first user message since Gemini lacks a system role).
- **OpenAI**: Used directly (not through `LLMClient`) by `ingest.py` for GPT-4o-mini move classification.
- **Model resolution**: The UI represents models as `"provider:model"` strings (e.g., `"gemini:gemini-1.5-flash"`, `"ollama:qwen3:latest"`), which the Gradio frontend parses to extract provider and model name before sending to the API.

---

## 4. Frontend Interface (`gradio_ui.py`)

The Gradio UI (approximately 1,780 lines) provides a single-page negotiation workspace.

### 4.1 Item Configuration Panel
- Three configurable items with custom names and quantities (e.g., "Senior Engineers" × 5, "Budget ($K)" × 200, "Timeline (weeks)" × 12).
- Validation: all three names required, all quantities must be positive.
- Configuration is stored in Gradio session state and passed to every API call.
- Chat is blocked until items are configured (system error message displayed).

### 4.2 Chat Interface
- Role selector (You / Other Party) with auto-switching after each message.
- Custom speaker names for both parties.
- Model selector dropdown dynamically populated from available Ollama and Gemini models.
- HTML-rendered chat bubbles with role chips (color-coded by party) and metadata badges (move type, PD label).
- Coach advice appears as inline assistant messages after each user/opponent turn.
- Conversations auto-saved as JSON to `./data/`.

### 4.3 Deal-or-No-Deal Visualizer
Accessible via an expandable accordion:
- Loads samples from the validation split of the Deal-or-No-Deal dataset (up to 1,500 samples).
- **Deal detection**: Dual-mode detection—a keyword-based heuristic (weighted scoring across the last 3 turns, with 3× weight on the final turn) and LLM-based analysis (prompts the selected model with the last 5 turns).
- **Early outcome markers**: Detects the first "no deal" signal and the first Nash-like (50-50) proposal using regex patterns.
- **Timeline dataframe**: Turn-by-turn display with columns for turn number, speaker, message, items mentioned, deal outcome, and RAG source.
- **Inline coach advice**: Optionally replays each turn through the full coaching pipeline, appending coach advice rows to the timeline.
- **No-deal filtering**: Option to show only conversations that ended without a deal.
- **Custom item names**: Override default DOND item names (book/hat/ball) with user-specified names.
- **Visualizations**: Speaker activity bar chart and item mentions over time line chart (Plotly).

### 4.4 Pareto Coach Effectiveness Simulator
Also in an expandable accordion:
- Runs batch simulations comparing Pareto-guided coaching vs. no coaching on Deal-or-No-Deal validation data.
- Configurable parameters: sample count (10–200), baseline strategy (equal/greedy/walkaway/statusquo), success threshold ratio (0.7–1.0).
- Outputs: rescue rate (what fraction of previously failing negotiations are saved by the coach), overall success rate with coaching, and full per-sample transcripts rendered as HTML with coach advice annotations.

### 4.5 Auto-Proposal (`autoplay.py`)
When the other party (role B) sends a message, the system automatically generates a Pareto-optimal counter-proposal using estimated preferences and the `best_offer` solver. The proposal is formatted with item names and quantities (e.g., "I suggest you take 2 Senior Engineers and I'll take 3 Budget ($K)") and displayed as a coach message.

---

## 5. Simulation & Evaluation (`simulate_dond.py`)

A benchmarking framework that compares two strategies on Deal-or-No-Deal validation data:

1. **Pareto-bot**: Uses `best_offer()` to propose a Nash-product–optimal allocation based on estimated preferences.
2. **No-Pareto bot**: Keeps the zero allocation (never proposes).

Both are evaluated against a configurable baseline allocation:
- **equal**: 50-50 integer-division split of each item.
- **greedy**: One side gets everything (the other gets zero).
- **walkaway**: Zero allocation for the bot (initial DOND state where neither side has anything).
- **statusquo**: Alias for equal split, called out separately for semantic clarity.

**Success criterion**: Both parties' utilities must reach at least `ratio × baseline_utility`, where ratio is configurable (default: 1.0).

**Coach-rescue simulation** (`simulate_with_coach`): A two-stage process for each of N samples:
1. Evaluate the no-Pareto bot; if the zero-allocation already satisfies both parties' utility thresholds, mark as "success without coach."
2. If unsuccessful, invoke the Pareto-bot (coach) to propose a `best_offer` and re-evaluate utilities. If the proposal now meets the threshold, count as "rescued by coach."

Output dictionary:
- `total`: Number of evaluated samples (skipping those where preference estimation fails).
- `success_without_coach`: Count of samples that succeed even without coaching.
- `rescued_by_coach`: Count of samples rescued by the Pareto-guided coach.
- `rescue_rate`: `rescued_by_coach / (total - success_without_coach)`.
- `overall_success_with_coach`: `(success_without_coach + rescued_by_coach) / total`.
- `transcripts`: Full dialogue transcripts with coach proposals and outcome annotations.

CLI usage:
```bash
python scripts/simulate_dond.py --n 100 --baseline equal
python scripts/simulate_dond.py --baseline walkaway
python scripts/simulate_dond.py --baseline statusquo --opp_ratio 0.95
```

---

## 6. Data Pipeline

### 6.1 Deal-or-No-Deal Dataset (`dond_data.py`)

Loads the Facebook Deal-or-No-Deal dialogue corpus from JSONL files in `deal_or_no_dialog/exported/`:

```python
@dataclass
class DialogSample:
    turns: List[str]          # Cleaned dialogue turns (split on <eos>, <selection> markers removed)
    counts: List[int]         # Item counts per type (e.g., [3, 2, 1])
    my_values: List[int]      # My value-per-item weights
    partner_values: List[int] # Partner's value-per-item weights
    my_final: List[int]       # My final allocation
    partner_final: List[int]  # Partner's final allocation
```

Supports train/validation/test splits. Path resolution follows a priority chain: `DOND_DATA_DIR` environment variable → `deal_or_no_dialog/exported/` → several fallback paths. The output field is parsed to extract integer allocations regardless of whether the format uses space-separated numbers or `item0=N` key-value pairs.

### 6.2 Document Ingestion (`ingest.py`)

Supports ingesting conversations from external files:
- **CSV**: Expects `speaker` and `text` columns; each row becomes a labeled turn.
- **PDF**: Extracts text via `pdfplumber`, splits on all-caps speaker patterns (e.g., `SPEAKER A:`), and labels each segment.
- Each extracted turn is classified by GPT-4o-mini and optionally stored in Neo4j via `upsert_turn`.

### 6.3 Conversation Storage

- Runtime conversations are saved as JSON in `./data/` with auto-save after each turn.
- Format: `{ conv_id, history: [{role, speaker, display_name, text, move, pd, ts}], last_updated }`.
- Markdown export functionality for transcript archiving.
- Files matching `data/dond_viz_*.json` and `data/conv_*.json` are gitignored.

---

## 7. Technology Stack

| Layer | Technology | Version / Notes |
|-------|-----------|----------------|
| Backend Framework | FastAPI + Uvicorn | Async ASGI server |
| Frontend Framework | Gradio | 6.0 |
| LLM Providers | Ollama (local), Google Gemini API, OpenAI API | Multi-provider abstraction |
| Vector Database | ChromaDB | Persistent client, cosine similarity |
| Embedding Model | all-MiniLM-L6-v2 | Sentence Transformers, 384-dim |
| Preference Model | DistilBERT | distilbert-base-uncased + 2 linear heads |
| Graph Database | Neo4j | v5+, optional |
| Visualization | Plotly | Bar charts, line charts, pie charts, subplots |
| NLP Corpus | ConvoKit | CaSiNo corpus |
| Dataset | Facebook Deal-or-No-Deal | JSONL format, train/validation/test splits |
| ML Framework | PyTorch + Hugging Face Transformers | CPU/GPU inference |
| Data Validation | Pydantic | v2, request/response models |
| Language | Python | 3.13+ |
| PDF Processing | pdfplumber | Text extraction from academic papers |

---

## 8. Configuration & Environment

Key environment variables (configured in `.env`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEFAULT_MODEL` | `qwen3:latest` | Default LLM model for advice generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `ENABLE_NEO4J` | `false` | Enable/disable Neo4j graph features |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | (configured) | Neo4j password |
| `OPENAI_API_KEY` | — | Required for GPT-4o-mini move classification |
| `GOOGLE_API_KEY` | — | Required for Gemini provider |
| `PRELOAD_RAG` | `false` | Preload RAG system at startup vs. lazy-load |
| `DOND_DATA_DIR` | `deal_or_no_dialog/exported` | Deal-or-No-Deal dataset location |
| `PREF_MODEL_CKPT` | `checkpoints/pref_estimator.pt` | Preference model checkpoint path |
| `VERBOSE_LOGGING` | `false` | Enable detailed debug logging |
| `API_BASE_URL` | (auto-detected) | Override backend URL for frontend |

---

## 9. Performance Optimizations

The system implements several performance optimizations across multiple phases:

1. **Async parallelization**: `get_advice_async()` runs Neo4j queries, ML inference, LLM calls, and RAG retrieval concurrently via `asyncio.gather()` and a 4-worker `ThreadPoolExecutor`. Expected latency reduction: 40–50%.
2. **TTL caching**: Function-level caching with configurable TTL—60s for general functions, 180s for LLM item analysis, 300s for preference estimation, 600s for RAG context. Cache keys use MD5 hashes of JSON-serialized arguments.
3. **Lazy initialization**: RAG system, embedding models, Gemini SDK, and the CaSiNo corpus are loaded on first use rather than at startup, reducing cold-start time.
4. **Early exit guards**: Skip expensive multi-step analysis when fewer than 2 turns exist, only one speaker is present, or a deal has already been reached.
5. **Neo4j connection pooling**: 50-connection pool with 1-hour lifetime, 30-second acquisition timeout, and 10-second socket timeout.
6. **Startup optimization**: Dependency checks and RAG preloading skipped by default (30–50 second cold start for PyTorch + DistilBERT model loading; set `PRELOAD_RAG=true` to front-load).
7. **Response conciseness**: LLM outputs are constrained to 35 words via system prompt and post-processed with a 40-word hard cap, reducing token generation time.

---

## 10. Limitations and Design Tradeoffs

- **Fixed to 3 items**: The preference estimator and Pareto solver are hard-coded for 3 negotiation items. Supporting N items would require architectural changes to the `PreferenceEstimator` model.
- **Short context window**: The coach considers only the last 5 turns from Neo4j, which may miss important context in longer negotiations.
- **Exhaustive Pareto enumeration**: The `itertools.product` approach scales as O(∏ counts_i), which is tractable for small item counts but would become expensive for items with large quantity ranges.
- **Move classification dependency on OpenAI**: The `ingest.py` module requires an OpenAI API key for GPT-4o-mini classification, introducing an external dependency even when using local Ollama models for advice generation.
- **Neo4j as optional**: While Neo4j provides conversation persistence and graph-based analysis, the system operates without it (losing conversation history across restarts and advice de-duplication).
- **No formal test suite**: The project lacks automated unit/integration tests; validation relies on manual testing and the simulation benchmarking framework.
