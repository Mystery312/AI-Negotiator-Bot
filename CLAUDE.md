# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

The app is a two-process system: a FastAPI backend and a Gradio frontend.

```bash
# Start both processes (backend + UI), opens browser automatically
./start_demo.sh

# Stop all processes
./stop_demo.sh

# Run backend only (port 8001)
python -m negotiation_chatbot.main

# Run Gradio UI only (port 7860, requires backend running)
python -m negotiation_chatbot.gradio_ui
```

**Ports:** Backend defaults to 8001, Gradio UI to 7860. API docs at `http://localhost:8001/docs`.

## Dependencies & Environment

- Python 3.13+, virtualenv in `.venv/`
- `pip install -r requirements.txt`
- `.env` file configures: `OLLAMA_BASE_URL`, `ENABLE_NEO4J`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `PRELOAD_RAG`, `DOND_DATA_DIR`

## Architecture

### Two-Process Design
- **`negotiation_chatbot/main.py`** — FastAPI backend. Exposes `/chat`, `/label`, `/health`, `/v1/chat/completions` (OpenAI-compatible), graph/stats endpoints. All module-as-package: `python -m negotiation_chatbot.main`.
- **`negotiation_chatbot/gradio_ui.py`** — Gradio frontend. Communicates with backend via HTTP (`requests`). Resolves backend URL by probing candidates (docker service name, localhost:8001, fallbacks).

### Request Pipeline (POST /chat)
1. **`ingest.py`** `label_text()` — Classifies negotiation speech via GPT-4o-mini into move type (concession/threat/info_share/cooperate/defect) and Prisoner's Dilemma label (C/D).
2. **`graph.py`** `upsert_turn()` — Stores turn in Neo4j (optional, graceful failure if `ENABLE_NEO4J=false`). Neo4j driver uses retry with exponential backoff.
3. **`coach.py`** `get_advice_async()` — Core logic. Fetches conversation history, estimates preferences, retrieves RAG context, generates strategic advice and a reply via LLM. Uses `async_helpers.py` thread pool for parallel execution of blocking operations (Neo4j, ML inference, LLM calls).
4. Response returns advice, reply, RAG source/context.

### LLM Providers (`llm_client.py`)
- **Ollama** (default) — Uses OpenAI-compatible API at `OLLAMA_BASE_URL/v1`. Default model: `qwen3:latest`.
- **Google Gemini** — Requires `GOOGLE_API_KEY`. Lazy-imports `google.generativeai`.
- **OpenAI** — Used only by `ingest.py` for move classification (GPT-4o-mini).
- Factory: `create_llm_client(provider, model_name)`. UI passes models as `"provider:model"` strings (e.g., `"gemini:gemini-1.5-flash"`).

### RAG Systems
- **Generic RAG** (`rag.py`) — ChromaDB + `all-MiniLM-L6-v2` embeddings. Stores negotiation tactics. Lazy-initialized on first use.
- **CaSiNo RAG** (`casino_rag.py`) — ChromaDB collection from CaSiNo negotiation corpus (via `convokit`). Has eval/train partition and caching (`./cache/`). Set `PRELOAD_RAG=true` to preload at startup.

### ML & Game Theory
- **`preference.py`** — DistilBERT-based `PreferenceEstimator` model. Takes conversation text, outputs softmax weight vectors for both parties (3 issues). Used by coach for Pareto-optimal proposals.
- **`train_prefs.py`** — Fine-tunes `PreferenceEstimator` on Deal-or-No-Dialog data. Run: `python train_pref.py --model_out checkpoints/pref_estimator.pt`.
- **`pareto.py`** — Enumerates all allocations, computes Pareto frontier, finds best offer (Nash product). Uses dynamic slack for asymmetric negotiations.
- **`autoplay.py`** — Auto-generates bot proposals using preference estimation + Pareto solver.
- **`simulate_dond.py`** — Simulates Pareto-bot vs no-Pareto-bot on validation data. Run: `python scripts/simulate_dond.py --n 100 --baseline equal`.

### Data
- **Deal-or-No-Dialog** (`deal_or_no_dialog/exported/`) — JSONL files (train/validation/test). Loaded by `dond_data.py`. Set `DOND_DATA_DIR` to override path.
- **`data/`** — Runtime conversation JSON storage (gitignored for viz/conv files).
- **`chroma_db/`** — Persistent ChromaDB vector store (gitignored).

### Async Architecture (`async_helpers.py`)
Wraps blocking operations (Neo4j, ML inference, LLM calls) in a `ThreadPoolExecutor(max_workers=4)` for concurrent execution within the async FastAPI handlers. Coach uses `asyncio.gather()` to parallelize independent steps.

## Import Pattern

Modules use dual import pattern for package vs script execution:
```python
try:
    from negotiation_chatbot.module import func  # package mode
except ImportError:
    from module import func  # script mode
```
