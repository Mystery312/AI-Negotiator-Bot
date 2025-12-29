# Multi-Project Repository

This repository contains two independent systems:

1. **Negotiation Chatbot** - AI-powered multi-party negotiation system
2. **Resource Management Backend** - REST API for resource management system

---

## 📁 Project Structure

```
chatbot/
├── negotiation_chatbot/          # AI Negotiation Chatbot Application
│   ├── main.py                   # FastAPI server
│   ├── gradio_ui.py              # Gradio UI
│   ├── coach.py                  # AI coaching logic
│   ├── casino_rag.py             # RAG implementation
│   └── ...                       # Other modules
│
├── backend/                      # Resource Management Backend
│   ├── app/                      # Backend application code
│   │   ├── main.py               # FastAPI server
│   │   ├── frontend_routes.py    # Frontend API endpoints
│   │   ├── frontend_storage.py   # In-memory storage
│   │   ├── api_routes.py         # Negotiation API
│   │   └── models.py             # Data models
│   ├── FRONTEND_INTEGRATION_GUIDE.md
│   ├── API_QUICK_REFERENCE.md
│   └── start_backend.sh          # Easy startup script
│
├── docs/                         # Documentation
│   ├── START_HERE.md             # Quick start for negotiation chatbot
│   ├── NEGOTIATION_CHATBOT.md    # Chatbot documentation
│   ├── CHATBOT_GUIDE.md          # Feature guide
│   ├── HOW_TO_RUN.md             # Setup instructions
│   ├── BACKEND_INTEGRATION_COMPLETE.md
│   └── ...                       # Other documentation
│
├── data/                         # Negotiation data (CaSiNo corpus)
├── chroma_db/                    # Vector database
├── .env                          # Environment configuration
├── docker-compose.yml            # Docker setup (for negotiation chatbot)
├── requirements.txt              # Python dependencies
└── test_app.py                   # Test script
```

---

## 🚀 Quick Start

### Option 1: Negotiation Chatbot

**Start with Docker Compose (Recommended):**
```bash
docker-compose up --build
# Access UI at http://localhost:7860
```

**Or start locally:**
```bash
# Terminal 1: API Server
python -m negotiation_chatbot.main

# Terminal 2: Gradio UI
python -m negotiation_chatbot.gradio_ui
```

**📖 Documentation:**
- [docs/START_HERE.md](docs/START_HERE.md) - Quick start guide
- [docs/NEGOTIATION_CHATBOT.md](docs/NEGOTIATION_CHATBOT.md) - Full documentation
- [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) - Setup instructions

---

### Option 2: Resource Management Backend

**Start the backend:**
```bash
cd backend
./start_backend.sh
# Access API at http://localhost:8000
```

**Test the API:**
```bash
# Dashboard stats
curl http://localhost:8000/api/dashboard/stats

# Get employees
curl http://localhost:8000/api/employees

# Interactive docs
open http://localhost:8000/docs
```

**📖 Documentation:**
- [backend/FRONTEND_INTEGRATION_GUIDE.md](backend/FRONTEND_INTEGRATION_GUIDE.md) - Complete API guide
- [backend/API_QUICK_REFERENCE.md](backend/API_QUICK_REFERENCE.md) - Quick reference
- [backend/ARCHITECTURE.md](backend/ARCHITECTURE.md) - System architecture

---

## 🎯 What Each Project Does

### Negotiation Chatbot

An AI-powered negotiation coaching system that:

✅ Analyzes conversations in real-time
✅ Provides strategic coaching advice
✅ Generates optimal proposals using Pareto optimization
✅ Uses RAG for context-aware recommendations
✅ Supports multiple LLM providers (Ollama, OpenAI, Gemini)
✅ Tracks conversation history in Neo4j graph database

**Tech Stack:** FastAPI, Gradio, Neo4j, ChromaDB, LangChain

**Ports:**
- UI: http://localhost:7860
- API: http://localhost:8000
- Neo4j: http://localhost:7474

---

### Resource Management Backend

A REST API backend for resource management with:

✅ Full CRUD for Employees, Equipment, Inventory, Rooms, Bookings
✅ Dashboard statistics endpoint
✅ In-memory storage (easily swappable to database)
✅ Sample data included
✅ Auto-generated API documentation
✅ CORS enabled for frontend integration

**Tech Stack:** FastAPI, Pydantic, Uvicorn

**Ports:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

**API Endpoints:**
- `/api/dashboard/stats` - Dashboard statistics
- `/api/employees` - Employee CRUD
- `/api/equipment` - Equipment CRUD
- `/api/inventory` - Inventory CRUD
- `/api/rooms` - Room CRUD
- `/api/bookings` - Booking CRUD

---

## 🧪 Testing

### Test Negotiation Chatbot
```bash
python test_app.py
```

### Test Resource Management Backend
```bash
cd backend
curl http://localhost:8000/api/dashboard/stats
```

---

## 📋 Requirements

- Python 3.8+
- Docker & Docker Compose (for negotiation chatbot)
- Ollama (optional, for local LLM models)

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

Edit `.env` to configure:

```bash
# Negotiation Chatbot
DEFAULT_MODEL=qwen3:latest
OLLAMA_BASE_URL=http://localhost:11434
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# API Keys (optional)
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# Features
ENABLE_NEO4J=true
```

---

## 📚 Documentation Index

### Negotiation Chatbot
- [docs/START_HERE.md](docs/START_HERE.md) - **START HERE** for chatbot
- [docs/NEGOTIATION_CHATBOT.md](docs/NEGOTIATION_CHATBOT.md) - Complete chatbot documentation
- [docs/CHATBOT_GUIDE.md](docs/CHATBOT_GUIDE.md) - Feature guide
- [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) - Setup instructions
- [docs/CLEANUP_SUMMARY.md](docs/CLEANUP_SUMMARY.md) - What was cleaned up

### Resource Management Backend
- [backend/FRONTEND_INTEGRATION_GUIDE.md](backend/FRONTEND_INTEGRATION_GUIDE.md) - **START HERE** for backend
- [backend/INTEGRATION_SUMMARY.md](backend/INTEGRATION_SUMMARY.md) - Overview
- [backend/API_QUICK_REFERENCE.md](backend/API_QUICK_REFERENCE.md) - Quick reference
- [backend/ARCHITECTURE.md](backend/ARCHITECTURE.md) - Architecture details
- [docs/BACKEND_INTEGRATION_COMPLETE.md](docs/BACKEND_INTEGRATION_COMPLETE.md) - Integration summary

---

## 🗂️ Directory Guide

| Directory | Purpose |
|-----------|---------|
| `negotiation_chatbot/` | AI negotiation chatbot source code |
| `backend/` | Resource management REST API |
| `docs/` | All documentation files |
| `data/` | CaSiNo corpus data for negotiation training |
| `chroma_db/` | Vector database for RAG |
| `.venv/` | Python virtual environment |

---

## 🛠️ Common Tasks

### Start Negotiation Chatbot
```bash
# With Docker
docker-compose up --build

# Or locally
python -m negotiation_chatbot.main
python -m negotiation_chatbot.gradio_ui
```

### Start Resource Management Backend
```bash
cd backend
./start_backend.sh
```

### Run Tests
```bash
python test_app.py
```

### View API Documentation
- Negotiation Chatbot: http://localhost:8000/docs
- Resource Management: http://localhost:8000/docs

---

## 📞 Support

### Troubleshooting

**Import errors:**
```bash
pip install -r requirements.txt
```

**Port conflicts:**
```bash
# Change port
PORT=8001 python -m negotiation_chatbot.main
```

**Neo4j connection issues:**
```bash
# Start Neo4j in Docker
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5

# Or disable in .env
ENABLE_NEO4J=false
```

### Getting Help

1. Check the documentation in `docs/`
2. Review the specific project's README
3. Check the logs for error messages
4. Open an issue with:
   - Description of the problem
   - Steps to reproduce
   - Error messages/logs
   - System information

---

## 📝 Notes

- **Negotiation Chatbot** runs on port 8000 (API) and 7860 (UI)
- **Resource Management Backend** also runs on port 8000
- **Cannot run both simultaneously** on default ports
- Use different ports if you need to run both:
  ```bash
  # Backend on 8001
  cd backend
  uvicorn app.main:app --port 8001
  ```

---

## 🎉 Get Started!

1. **For Negotiation Chatbot:** Read [docs/START_HERE.md](docs/START_HERE.md)
2. **For Resource Management:** Read [backend/FRONTEND_INTEGRATION_GUIDE.md](backend/FRONTEND_INTEGRATION_GUIDE.md)
3. Run `python test_app.py` to verify setup
4. Start your preferred project!

---

**Last Updated:** 2025-12-29
