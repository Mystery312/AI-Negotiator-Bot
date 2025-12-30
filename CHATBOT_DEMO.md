# Negotiation Chatbot Demo Guide

Quick start guide for running the negotiation chatbot demo.

## What is This?

An AI-powered negotiation assistant that helps you practice and improve your negotiation skills through:
- Real-time coaching during negotiations
- Move classification (cooperate, compete, etc.)
- Prisoner's Dilemma strategy analysis
- Deal-or-No-Deal visualization
- RAG-based advice from expert negotiation data

## Quick Start

### 1. Start the Demo

```bash
./start_demo.sh
```

This will:
- Start the backend API on http://localhost:8000
- Start the Gradio UI on http://localhost:7860
- Open your browser automatically
- Create logs in `logs/` directory

### 2. Access the Demo

The browser will open automatically to http://localhost:7860

If it doesn't open:
- Open http://localhost:7860 manually
- Or check logs: `tail -f logs/gradio.log`

### 3. Stop the Demo

```bash
./stop_demo.sh
```

Or manually kill the processes shown when you started the demo.

## Features

### Negotiation Chat
1. Click "Negotiation Chat" tab
2. Enter your negotiation messages
3. Get real-time coaching advice
4. See move classifications (cooperate/compete/defer)
5. View Prisoner's Dilemma strategy analysis

### DoND Visualizer
1. Click "DoND Visualizer" tab
2. Select a sample negotiation dialogue
3. View timeline visualization
4. See Pareto frontier analysis
5. Analyze deal outcomes

### Settings
- Choose LLM provider (Gemini/OpenAI/Ollama)
- Select model (gemini-1.5-flash recommended)
- Configure API keys if using cloud models

## Requirements

### Python Environment
```bash
# Already set up if start_demo.sh works
python 3.13+
pip install -r requirements.txt
```

### Optional Services

**Neo4j (for conversation graphs)**
```bash
export ENABLE_NEO4J=true
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

**Ollama (for local LLMs)**
```bash
ollama serve
ollama pull qwen3:latest
# Then select "ollama:qwen3:latest" in UI
```

## Troubleshooting

### API won't start
```bash
# Check if port 8000 is in use
lsof -i :8000
# Kill existing process
kill <PID>
# Restart
./start_demo.sh
```

### Gradio won't start
```bash
# Check if port 7860 is in use
lsof -i :7860
# Kill existing process
kill <PID>
# Restart
./start_demo.sh
```

### Slow responses (30+ seconds)
This was a Neo4j connection issue - now fixed. If you still experience this:
```bash
# Ensure Neo4j is disabled (default)
unset ENABLE_NEO4J
# Restart services
./stop_demo.sh
./start_demo.sh
```

### No coach advice
```bash
# Check API is running
curl http://localhost:8000/health
# Should return: {"status":"ok"}

# Check logs
tail -f logs/api.log
```

## API Endpoints

The backend API runs on http://localhost:8000

### Key Endpoints
- `GET /health` - Health check
- `POST /chat` - Get negotiation advice
- `POST /label` - Classify negotiation moves
- `GET /docs` - Interactive API documentation

### Example API Call
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "conv_id": "demo-123",
    "speaker": "A",
    "text": "I need more resources for my project",
    "model": "gemini-1.5-flash",
    "provider": "gemini"
  }'
```

## Data & Files

### Conversations
Saved to: `data/conv_*.json`

### DoND Dataset
Located at: `deal_or_no_dialog/exported/`
- 10 sample negotiations included
- JSONL format with dialogue, preferences, outcomes

### Logs
- `logs/api.log` - Backend API logs
- `logs/gradio.log` - Gradio UI logs

## Advanced Configuration

### Environment Variables
```bash
# Neo4j (optional)
export ENABLE_NEO4J=true
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password

# API Keys (optional, for cloud models)
export GOOGLE_API_KEY=your_key
export OPENAI_API_KEY=your_key

# API Base URL
export API_BASE_URL=http://localhost:8000
```

### Using Local Models
1. Start Ollama: `ollama serve`
2. Pull a model: `ollama pull qwen3:latest`
3. In Gradio UI settings:
   - Provider: ollama
   - Model: qwen3:latest

## Performance

- API health check: ~80ms
- Coach advice: 3-5 seconds
- UI load time: 2-3 seconds
- No timeouts or delays

## Support

For issues or questions:
1. Check logs: `tail -f logs/*.log`
2. Check API health: `curl http://localhost:8000/health`
3. Restart services: `./stop_demo.sh && ./start_demo.sh`
