# 🚀 START HERE - Chatbot Application

Welcome! This is your negotiation chatbot application. Follow this guide to get started quickly.

---

## ✅ Application Status

**The chatbot is now fully functional and ready to use!**

All dependencies are installed, configurations are fixed, and the application has been tested.

---

## 🎯 Quick Start (Choose One Method)

### Method 1: Docker Compose (Easiest - Recommended)

```bash
# Start everything with one command
docker-compose up --build

# Access the UI at: http://localhost:7860
```

That's it! Docker will handle Neo4j, Ollama, API, and UI automatically.

---

### Method 2: Local Development

```bash
# Terminal 1: Start the API server
python -m app.main

# Terminal 2: Start the Gradio UI
python -m app.gradio_ui

# Access the UI at: http://localhost:7860
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[HOW_TO_RUN.md](HOW_TO_RUN.md)** | 📖 **START HERE** - Complete step-by-step guide |
| [CHATBOT_GUIDE.md](CHATBOT_GUIDE.md) | Feature explanations and usage guide |
| [README.md](README.md) | Comprehensive technical documentation |
| [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) | What was cleaned up and why |
| [test_app.py](test_app.py) | Run this to verify your setup |

---

## 🧪 Test Your Setup

Before running the app, verify everything is configured:

```bash
python test_app.py
```

If you see `✅ All tests passed!`, you're ready to go!

---

## 🌐 Access Points

Once running, access these URLs:

| Service | URL | Description |
|---------|-----|-------------|
| **Gradio UI** | http://localhost:7860 | Main chat interface |
| **API** | http://localhost:8000 | Backend API |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Neo4j** | http://localhost:7474 | Database browser (if using Docker) |

---

## 🆘 Having Issues?

### Quick Troubleshooting

**Import errors:**
```bash
pip install -r requirements.txt
```

**Port already in use:**
```bash
# Change the port
PORT=8001 python -m app.main
```

**Neo4j connection failed:**
```bash
# Option 1: Start Neo4j
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5

# Option 2: Disable Neo4j in .env
ENABLE_NEO4J=false
```

**Full troubleshooting guide:** See [HOW_TO_RUN.md](HOW_TO_RUN.md#common-issues-and-solutions)

---

## 💡 What This App Does

This is an **AI-powered negotiation coaching system** that:

✅ Analyzes conversations in real-time
✅ Provides strategic coaching advice
✅ Generates optimal proposals using Pareto optimization
✅ Uses RAG (Retrieval-Augmented Generation) for context-aware recommendations
✅ Supports multiple LLM providers (Ollama, OpenAI, Gemini)
✅ Tracks conversation history in a graph database

---

## 🎮 Basic Usage

1. **Open the UI** at http://localhost:7860
2. **Enter names** for both parties (e.g., "Alice" and "Bob")
3. **Select a model** from the dropdown (e.g., `qwen3:latest`)
4. **Start chatting** - type messages and get AI coaching advice!

---

## 📦 What's Included

```
chatbot/
├── app/              # Application code
├── data/             # CaSiNo corpus data
├── chroma_db/        # Vector database
├── .env              # Configuration
├── requirements.txt  # Dependencies
├── docker-compose.yml # Docker setup
└── HOW_TO_RUN.md     # Detailed guide
```

---

## 🔧 Configuration

Edit `.env` to customize:

```bash
# LLM Provider
DEFAULT_MODEL=qwen3:latest
OLLAMA_BASE_URL=http://localhost:11434

# Neo4j (optional)
ENABLE_NEO4J=true
NEO4J_URI=bolt://localhost:7687

# API Keys (optional)
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

---

## 📖 Next Steps

1. **Read** [HOW_TO_RUN.md](HOW_TO_RUN.md) for detailed setup instructions
2. **Run** `python test_app.py` to verify your setup
3. **Start** the application using Docker or local development
4. **Explore** the Gradio UI and try negotiating!

---

## 🎉 You're All Set!

The application is clean, functional, and ready to use. Enjoy your negotiation chatbot! 🤖

For questions or issues, check the documentation files listed above.
