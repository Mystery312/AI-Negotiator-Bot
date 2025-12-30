# AI Negotiation Chatbot System

A comprehensive multi-party negotiation system with AI-powered coaching, real-time analysis, Pareto-optimal proposal generation, and resource management capabilities.

## 🚀 Quick Start

### Launch the Application

```bash
# Quick Demo (recommended for first-time users)
./start_demo.sh

# Or manually
python -m negotiation_chatbot.gradio_ui
```

**Access**: http://localhost:7860

For detailed instructions, see **[RUN_DEMO.md](RUN_DEMO.md)** or **[START_HERE.md](START_HERE.md)**

### Quick Reference

| Component | URL | Purpose |
|-----------|-----|---------|
| **Gradio UI** | http://localhost:7860 | Main negotiation interface |
| **Backend API** | http://localhost:8000 | REST API endpoints |
| **Frontend** | http://localhost:3000 | React resource management UI |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |

## 📋 Project Overview

This system combines two powerful applications:

1. **Negotiation Chatbot** - AI-powered negotiation assistance with coach advice, conversation analysis, and optimal proposal generation
2. **Resource Management System** - Multi-department resource allocation with API-driven negotiation engine

## 🏗️ Project Structure

```
chatbot/
├── 📄 README.md                       # This file - main entry point
├── 📄 START_HERE.md                   # Getting started guide
├── 📄 RUN_DEMO.md                     # Demo launch instructions
├── 📄 QUICK_START.md                  # Quick reference guide
├── 📄 DOND_DATASET_SETUP.md          # Dataset setup documentation
│
├── 🚀 start_demo.sh                   # Demo launcher script
├── 🛑 stop_demo.sh                    # Demo stopper script
├── 📦 requirements.txt                # Python dependencies
├── 🔒 .gitignore                      # Git ignore rules
│
├── 📁 scripts/                        # Utility scripts
│   ├── create_sample_dond_data.py    # Generate sample negotiations
│   ├── setup_dond_dataset.py         # Download full dataset
│   └── test_app.py                   # Test application setup
│
├── 📁 negotiation_chatbot/            # Main chatbot application
│   ├── gradio_ui.py                  # Gradio web interface
│   ├── main.py                       # FastAPI backend
│   ├── coach.py                      # AI coaching logic
│   ├── rag.py                        # Retrieval-augmented generation
│   ├── pareto.py                     # Pareto optimization
│   ├── preference.py                 # Preference estimation
│   ├── autoplay.py                   # Auto-proposal generation
│   ├── graph.py                      # Neo4j graph operations
│   ├── llm_client.py                 # Multi-LLM provider support
│   ├── dond_data.py                  # Dataset utilities
│   ├── simulate_dond.py              # Bot simulations
│   └── ...                           # Additional modules
│
├── 📁 backend/                        # Resource management backend
│   └── app/
│       ├── main.py                   # FastAPI server
│       ├── api_routes.py             # API endpoints
│       ├── auth_routes.py            # Authentication
│       ├── models.py                 # Pydantic models
│       └── ...                       # Logic engines, orchestrators
│
├── 📁 resource-hub-main/              # Frontend React application
│   ├── src/
│   │   ├── components/               # React components
│   │   ├── services/                 # API services
│   │   └── ...
│   ├── public/
│   └── package.json
│
├── 📁 deal_or_no_dialog/              # Negotiation dataset
│   └── exported/
│       ├── train.jsonl               # Training data
│       ├── validation.jsonl          # Validation data
│       └── test.jsonl                # Test data
│
├── 📁 data/                           # Application data
│   └── casino.json                   # CaSiNo corpus (4.1MB)
│
├── 📁 docs/                           # Documentation
│   ├── NEGOTIATION_CHATBOT.md        # Chatbot feature guide
│   ├── HOW_TO_RUN.md                 # Detailed setup guide
│   ├── RESOURCE_MANAGEMENT_SYSTEM_PLAN.md
│   └── archive/                      # Archived documentation
│
├── 📁 chroma_db/                      # Vector database (gitignored)
└── 📁 cache/                          # Cache files (gitignored)
```

## 🎯 Key Features

### Negotiation Chatbot

✅ **AI Coach Advice** - Strategic guidance using RAG and LLMs
✅ **Conversation Analysis** - Track moves, power dynamics, speaker patterns
✅ **Pareto Optimization** - Generate optimal proposals
✅ **Deal Detection** - LLM-based outcome prediction
✅ **DoND Visualizer** - Analyze real negotiation samples
✅ **Multi-Model Support** - Ollama, Gemini, OpenAI
✅ **Graph Storage** - Neo4j conversation tracking

### Resource Management System

✅ **Department Management** - Create and manage departments
✅ **Resource Requests** - Submit allocation requests
✅ **Negotiation Engine** - Multi-round negotiation with proposals
✅ **Counter-Proposals** - Automated counter-offer generation
✅ **Pareto Analysis** - Find optimal allocations
✅ **REST API** - Full API with authentication
✅ **React Frontend** - Modern UI for resource management

## 💻 Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- (Optional) Docker & Docker Compose
- (Optional) Ollama for local LLMs
- (Optional) Neo4j for graph features

### Setup

1. **Clone and install dependencies**:
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt

   # Install frontend dependencies (optional)
   cd resource-hub-main
   npm install
   cd ..
   ```

2. **Set up environment**:
   ```bash
   # Create .env file (optional)
   # Edit with your API keys if needed
   ```

3. **Set up dataset** (optional but recommended):
   ```bash
   python scripts/create_sample_dond_data.py
   ```

## 🎮 Usage

### Launch Everything

```bash
./start_demo.sh
```

This starts:
- Gradio UI on port 7860
- Backend API on port 8000 (if configured)
- Frontend on port 3000 (if configured)

### Individual Components

```bash
# Gradio Negotiation UI only
python -m negotiation_chatbot.gradio_ui

# Backend API only
python -m backend.app.main

# Frontend only
cd resource-hub-main && npm run dev
```

### Using the Gradio UI

1. **Start a Conversation**
   - Enter names for negotiating parties
   - Select AI model (Ollama or Gemini)
   - Begin chatting

2. **Get Coach Advice**
   - System automatically provides advice after each message
   - Uses RAG to find relevant negotiation tactics
   - Displays strategic recommendations

3. **Analyze Negotiations**
   - Open "DoND Conversation Visualizer"
   - Load sample negotiations (0-9)
   - View statistics, timelines, and outcomes
   - Enable coach advice for turn-by-turn guidance

4. **Run Simulations**
   - Open "Pareto Coach Effectiveness Simulator"
   - Set parameters and baseline
   - See how AI coaching improves outcomes

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[START_HERE.md](START_HERE.md)** | Complete getting started guide |
| **[RUN_DEMO.md](RUN_DEMO.md)** | Demo launch instructions |
| **[QUICK_START.md](QUICK_START.md)** | Quick reference |
| **[DOND_DATASET_SETUP.md](DOND_DATASET_SETUP.md)** | Dataset setup guide |
| **[docs/NEGOTIATION_CHATBOT.md](docs/NEGOTIATION_CHATBOT.md)** | Chatbot features |
| **[docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md)** | Detailed setup |

## 🔧 Configuration

### Environment Variables

```bash
# LLM Providers (optional)
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# API Endpoints
API_BASE_URL=http://localhost:8000
OLLAMA_BASE_URL=http://localhost:11434

# Neo4j (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Dataset
DOND_DATA_DIR=deal_or_no_dialog/exported
```

### Model Selection

The system supports multiple LLM providers:

- **Ollama** (local): qwen3:latest, llama3.2:latest, mistral:latest
- **Gemini** (cloud): gemini-1.5-flash, gemini-2.0-flash, gemini-2.5-pro
- **OpenAI** (cloud): gpt-4, gpt-3.5-turbo

Models are auto-detected and listed in the UI dropdown.

## 🧪 Testing

```bash
# Test application setup
python scripts/test_app.py

# Test dataset loading
python -c "from negotiation_chatbot.dond_data import load_dond; print(f'Loaded {len(load_dond(\"validation\"))} samples')"

# Test API health
curl http://localhost:8000/health
```

## 🐛 Troubleshooting

### Gradio UI won't start
- Check if port 7860 is available: `lsof -i :7860`
- Verify dependencies: `pip install -r requirements.txt`
- Check logs for errors

### No coach advice
- Verify API is running: `curl http://localhost:8000/health`
- Check LLM provider connection (Ollama/Gemini/OpenAI)
- Review browser console for errors

### Dataset not loading
- Run setup: `python scripts/create_sample_dond_data.py`
- Check `deal_or_no_dialog/exported/` directory exists
- Verify JSONL files are present

### Frontend connection issues
- Ensure backend is running on port 8000
- Check CORS settings in backend
- Verify API_BASE_URL in frontend config

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface Layer                 │
├─────────────────────────────────────────────────────────┤
│  Gradio UI (7860)  │  React Frontend (3000)             │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
├─────────────────────────────────────────────────────────┤
│  Coach  │  RAG  │  Pareto  │  Preference  │  Autoplay   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                     Backend API Layer                    │
├─────────────────────────────────────────────────────────┤
│         FastAPI (8000) - REST API Endpoints             │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    Integration Layer                     │
├─────────────────────────────────────────────────────────┤
│  LLM Client  │  Neo4j  │  ChromaDB  │  Dataset Loader   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   External Services                      │
├─────────────────────────────────────────────────────────┤
│  Ollama  │  Gemini  │  OpenAI  │  Neo4j  │  ChromaDB   │
└─────────────────────────────────────────────────────────┘
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **Facebook Research** - Deal-or-No-Deal dataset
- **CaSiNo Corpus** - Negotiation dialogue corpus
- **Gradio** - Web UI framework
- **FastAPI** - Backend framework
- **Ollama** - Local LLM runtime
- **Google Gemini** - Cloud LLM API

## 📞 Support

For issues and questions:
1. Check documentation in `docs/`
2. Review troubleshooting section above
3. Check logs for error messages
4. Open an issue with details

---

**Last Updated**: December 29, 2025
**Status**: ✅ Fully functional and tested
**Version**: 2.0
