# Chatbot Application - Quick Start Guide

## Overview

This is a negotiation chatbot application built with FastAPI and Gradio that provides AI-powered coaching advice for multi-party negotiations. The system analyzes conversation dynamics, provides strategic recommendations, and uses RAG (Retrieval-Augmented Generation) to offer contextual negotiation tactics.

## Prerequisites

- **Python 3.8+** (Python 3.13 is currently being used)
- **Neo4j Database** (for conversation graph storage)
- **Ollama** (optional, for local LLM models)
- **Google Gemini API key** (optional, for Gemini models)

## Installation

### 1. Clone and Setup

```bash
cd /Users/yeonjune.kim.27/Desktop/chatbot
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create or verify your `.env` file in the project root with the following variables:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687  # or your Neo4j URI
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# Optional: API Keys
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional: API Configuration
PORT=8000
API_BASE_URL=http://localhost:8000
OLLAMA_BASE_URL=http://localhost:11434
```

## Running the Application

### Option 1: Docker Compose (Recommended)

If you have Docker installed, this is the easiest way to run everything:

```bash
docker-compose up --build
```

This will start:
- FastAPI backend on `http://localhost:8000`
- Gradio UI on `http://localhost:7860`
- Neo4j database
- Ollama (if configured)

### Option 2: Local Development

#### Step 1: Start Neo4j Database

Make sure Neo4j is running on your system. You can:
- Use Docker: `docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your_password neo4j:latest`
- Or install Neo4j Desktop and start a local database

#### Step 2: Start Ollama (Optional)

If you want to use local LLM models:

```bash
# Install Ollama from https://ollama.ai
ollama serve

# Pull a model (e.g., qwen3)
ollama pull qwen3:latest
```

#### Step 3: Start the FastAPI Backend

```bash
cd app
python main.py
```

The API will be available at `http://localhost:8000`

You can verify it's running:
```bash
curl http://localhost:8000/health
```

#### Step 4: Start the Gradio UI

In a new terminal:

```bash
cd app
python gradio_ui.py
```

The UI will be available at `http://localhost:7860`

## Using the Application

### Web Interface (Gradio)

1. **Open your browser** to `http://localhost:7860`

2. **Start a conversation:**
   - The UI auto-generates a conversation ID (or you can enter your own)
   - Enter names for both negotiating parties
   - Select an AI model from the dropdown

3. **Send messages:**
   - Type messages in the chat input
   - Toggle between "You" and "Other Party" using the role selector
   - The system provides coaching advice after each message
   - Bot proposals are automatically generated when simulating the other party

4. **View insights:**
   - Check the Statistics panel for conversation metrics
   - Use the visualization tools to see negotiation graphs
   - Export conversations as markdown for later review

### API Endpoints

The FastAPI backend provides several endpoints:

- **`POST /chat`** - Process a chat message and get advice
  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{
      "conv_id": "demo123",
      "speaker": "Alice",
      "text": "I would like to discuss the terms",
      "model": "qwen3:latest"
    }'
  ```

- **`GET /health`** - Health check endpoint
  ```bash
  curl http://localhost:8000/health
  ```

- **`POST /v1/chat/completions`** - OpenAI-compatible endpoint

- **`GET /graph/{conv_id}`** - Get conversation graph visualization data

- **`GET /stats/{conv_id}`** - Get conversation statistics

## Project Structure

```
chatbot/
├── app/                        # Main application code
│   ├── main.py                # FastAPI backend server
│   ├── gradio_ui.py           # Gradio web interface
│   ├── coach.py               # AI coaching logic
│   ├── ingest.py              # Text labeling utilities
│   ├── graph.py               # Neo4j graph operations
│   ├── rag.py                 # RAG utilities
│   ├── casino_rag.py          # CaSiNo corpus RAG
│   ├── llm_client.py          # LLM provider abstraction
│   ├── pareto.py              # Pareto optimization
│   ├── preference.py          # Preference estimation
│   ├── autoplay.py            # Auto-proposal generation
│   ├── automate.py            # Automation scripts
│   ├── dond_data.py           # Deal-or-No-Deal data utilities
│   ├── simulate_dond.py       # Negotiation simulations
│   ├── run_experiments.py     # Experiment runner
│   ├── train_prefs.py         # Preference model training
│   ├── build_vector_db.py     # Vector DB builder
│   └── style.css              # UI styling
├── data/                       # Data storage
│   └── casino.json            # CaSiNo corpus data
├── chroma_db/                  # Vector database storage
├── .env                        # Environment variables (create this)
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker configuration
├── Dockerfile                  # Docker image definition
├── README.md                   # Comprehensive documentation
└── RESOURCE_MANAGEMENT_SYSTEM_PLAN.md  # Future plans
```

## Key Features

1. **AI Coaching**: Get strategic advice for negotiation moves based on conversation history
2. **RAG-Enhanced Advice**: Retrieval-augmented generation using negotiation tactics from the CaSiNo corpus
3. **Multi-Model Support**: Works with Ollama (local) and Gemini (cloud) models
4. **Graph-Based Analysis**: Conversation tracking and analysis using Neo4j
5. **Pareto Optimization**: Generate optimal proposals using preference estimation
6. **Web UI**: User-friendly Gradio interface for interactive negotiations
7. **API Access**: RESTful API for programmatic access

## Troubleshooting

### Cannot connect to Neo4j

- Ensure Neo4j is running
- Check the `NEO4J_URI` in your `.env` file
- Verify credentials are correct

### Ollama models not loading

- Ensure Ollama is installed and running: `ollama serve`
- Pull the model you want to use: `ollama pull qwen3:latest`
- Check that `OLLAMA_BASE_URL` in `.env` is correct

### API health check fails

- Verify the FastAPI server is running: `python app/main.py`
- Check port 8000 is not in use by another application
- Review logs for any startup errors

### Gradio UI not loading

- Ensure port 7860 is available
- Check that the API backend is running first
- Verify `API_BASE_URL` in the environment points to the correct backend

### Import errors

If you get "No module named 'app'" errors:

```bash
# Add the project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/Users/yeonjune.kim.27/Desktop/chatbot"
```

## Additional Resources

- Full documentation: See [README.md](README.md)
- For advanced configuration and features, refer to the comprehensive README
- CaSiNo corpus: The system uses negotiation data for RAG-based coaching

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review error logs when troubleshooting
- Ensure all prerequisites are installed and configured correctly
