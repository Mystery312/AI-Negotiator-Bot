# AI Negotiation Chatbot

An AI-powered negotiation coach that helps you practice and improve your negotiation skills through real-time coaching, move classification, and strategy analysis.

## Features

- **Real-time AI Coaching**: Get advice during negotiations based on conversation context
- **Move Classification**: Automatic labeling of negotiation moves (cooperate, compete, defer)
- **Prisoner's Dilemma Analysis**: Strategic framework for analyzing negotiation positions
- **Deal-or-No-Deal Visualizer**: Visualize negotiation dialogues and outcomes
- **RAG System**: Context-aware advice from the CaSiNo negotiation corpus
- **Multi-LLM Support**: Works with OpenAI, Google Gemini, or Ollama

## Quick Start

```bash
./start_demo.sh
```

The demo will open automatically at http://localhost:7860

## Installation

### Prerequisites
- Python 3.13+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Start the Demo
```bash
./start_demo.sh
```

Access points:
- **Gradio UI**: http://localhost:7860
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Stop the Demo
```bash
./stop_demo.sh
```

## Project Structure

```
.
├── negotiation_chatbot/     # Core chatbot logic
│   ├── main.py             # FastAPI backend
│   ├── gradio_ui.py        # Gradio web interface
│   ├── coach.py            # AI negotiation coach
│   ├── rag.py              # RAG system
│   ├── graph.py            # Neo4j integration
│   └── pareto.py           # Pareto analysis
├── deal_or_no_dialog/      # DoND dataset
├── data/                   # Conversation storage
├── chroma_db/              # Vector database
├── cache/                  # Cache files
├── scripts/                # Utility scripts
├── start_demo.sh           # Start script
├── stop_demo.sh            # Stop script
└── requirements.txt        # Python dependencies
```

## Configuration

See `.env` file for optional configuration:
- Neo4j (conversation graphs)
- LLM API keys (Google, OpenAI)
- RAG preloading

## Tech Stack

- FastAPI, Python 3.13, Gradio 6.0
- OpenAI, Google Gemini, Ollama
- ChromaDB, Sentence Transformers
- Neo4j (optional)
