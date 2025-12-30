# AI Negotiation & Resource Management Platform

Complete AI-powered negotiation system with two main applications:
1. **Negotiation Chatbot** - Practice negotiations with AI coaching
2. **Resource Management System** - Multi-department resource allocation with AI negotiation agents

## Quick Start

### Run Negotiation Chatbot Demo
```bash
./start_demo.sh
# Opens at http://localhost:7860
```

### Run Resource Management Website
```bash
# Terminal 1 - Backend
cd backend
python -m app.main

# Terminal 2 - Frontend
cd resource-hub-main
npm run dev
# Opens at http://localhost:5173
```

## Documentation

- [CHATBOT_DEMO.md](CHATBOT_DEMO.md) - Complete guide for negotiation chatbot
- [RESOURCE_MANAGEMENT.md](RESOURCE_MANAGEMENT.md) - Complete guide for resource management system

## What's Included

### 1. Negotiation Chatbot
Practice and improve negotiation skills with:
- Real-time AI coaching during negotiations
- Move classification (cooperate, compete, defer)
- Prisoner's Dilemma strategy analysis
- Deal-or-No-Deal (DoND) visualization
- RAG-based advice from expert negotiation corpus

**Tech Stack:**
- Gradio UI (Python web interface)
- FastAPI backend
- ChromaDB for RAG
- Sentence Transformers for embeddings
- OpenAI/Gemini/Ollama for LLM

### 2. Resource Management System
Multi-department resource allocation with:
- Department CRUD operations
- AI negotiation agents per department
- Automated negotiation orchestration
- Consensus detection and validation
- Real-time negotiation visualization

**Tech Stack:**
- React + Vite + TypeScript frontend
- FastAPI backend
- Tailwind CSS + Radix UI
- JWT authentication
- RESTful API architecture

## Project Structure

```
.
├── negotiation_chatbot/        # Chatbot core logic
│   ├── main.py                # Chatbot API entry point
│   ├── gradio_ui.py           # Gradio web interface
│   ├── coach.py               # AI negotiation coach
│   ├── rag.py                 # RAG system for advice
│   ├── graph.py               # Neo4j conversation graphs
│   └── pareto.py              # Pareto optimality analysis
│
├── backend/                    # Resource management backend
│   ├── app/
│   │   ├── main.py            # Backend API entry point
│   │   ├── models.py          # Data models
│   │   ├── api_routes.py      # Negotiation API
│   │   ├── frontend_routes.py # CRUD API
│   │   └── negotiation_orchestrator.py  # Multi-agent orchestration
│   └── requirements.txt
│
├── resource-hub-main/          # Resource management frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/             # Page components
│   │   └── services/          # API clients
│   └── package.json
│
├── data/                       # Data storage
│   ├── casino.json            # CaSiNo negotiation corpus
│   └── conv_*.json            # Saved conversations
│
├── deal_or_no_dialog/         # DoND dataset
│   └── exported/              # Sample negotiations
│
├── start_demo.sh              # Quick start chatbot
└── stop_demo.sh               # Stop chatbot
```

## Installation

### Prerequisites
- Python 3.13+
- Node.js 18+ (for resource management frontend)
- pip and npm

### Install Chatbot
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start demo
./start_demo.sh
```

### Install Resource Management System
```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd resource-hub-main
npm install
```

## Features

### Negotiation Chatbot Features
- **Real-time Coaching**: Get AI advice during negotiations
- **Move Classification**: Automatic labeling of negotiation moves
- **Strategy Analysis**: Prisoner's Dilemma framework analysis
- **DoND Visualizer**: Visualize deal-or-no-deal negotiations
- **RAG System**: Context-aware advice from CaSiNo corpus
- **Multi-LLM Support**: OpenAI, Google Gemini, Ollama

### Resource Management Features
- **Department Management**: Create, read, update, delete departments
- **AI Negotiation Agents**: Each department has intelligent agent
- **Multi-Agent Orchestration**: Automated negotiation between departments
- **Resource Types**: Budget, personnel, equipment, time
- **Consensus Detection**: Automatic agreement identification
- **Deadlock Resolution**: Detect and handle negotiation deadlocks
- **Modern UI**: Responsive React interface with real-time updates

## API Endpoints

### Chatbot API (Port 8000)
```bash
GET  /health                    # Health check
POST /chat                      # Get negotiation advice
POST /label                     # Classify negotiation move
GET  /docs                      # API documentation
```

### Resource Management API (Port 8000)
```bash
# Departments
GET    /api/departments         # List all departments
POST   /api/departments         # Create department
GET    /api/departments/{id}    # Get department
PUT    /api/departments/{id}    # Update department
DELETE /api/departments/{id}    # Delete department

# Negotiation
POST   /api/start-negotiation   # Start new negotiation
GET    /api/negotiation/{id}/status    # Get status
GET    /api/negotiation/{id}/messages  # Get messages

# Auth
POST   /auth/login              # Login
POST   /auth/register           # Register
```

## Configuration

### Environment Variables

**Chatbot (.env or export):**
```bash
# Neo4j (optional - for conversation graphs)
ENABLE_NEO4J=false
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM API Keys (optional - for cloud models)
GOOGLE_API_KEY=your_key
OPENAI_API_KEY=your_key
```

**Resource Management Backend (backend/.env):**
```bash
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:8000
```

**Resource Management Frontend (resource-hub-main/.env):**
```bash
VITE_API_BASE_URL=http://localhost:8000
```

## Optional Services

### Neo4j (for conversation graphs)
```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5

export ENABLE_NEO4J=true
```

### Ollama (for local LLMs)
```bash
ollama serve
ollama pull qwen3:latest
# Select in UI: Provider=ollama, Model=qwen3:latest
```

## Usage Examples

### Chatbot Example
```python
# Using API directly
import requests

response = requests.post('http://localhost:8000/chat', json={
    'conv_id': 'demo-123',
    'speaker': 'A',
    'text': 'I think we should split the resources 50-50',
    'model': 'gemini-1.5-flash',
    'provider': 'gemini'
})

print(response.json())
# {
#   "advice": "Consider their priorities...",
#   "move": "cooperate",
#   "pd": "C",
#   "rag_source": "casino",
#   "rag_context": "..."
# }
```

### Resource Management Example
```bash
# Create a department
curl -X POST http://localhost:8000/api/departments \
  -H "Content-Type: application/json" \
  -d '{
    "department_id": "eng-001",
    "department_name": "Engineering",
    "resource_priorities": {"budget": 0.4, "personnel": 0.6},
    "strategic_objectives": ["Build AI platform"]
  }'

# Start negotiation
curl -X POST http://localhost:8000/api/start-negotiation \
  -H "Content-Type: application/json" \
  -d '{
    "departments": ["eng-001", "sales-001"],
    "total_resources": {"budget": 1000000, "personnel": 50}
  }'
```

## Performance

### Chatbot
- API health check: ~80ms
- Coach advice: 3-5 seconds
- UI load time: 2-3 seconds
- No timeouts or delays

### Resource Management
- API response: < 100ms
- Negotiation round: 2-5 seconds per agent
- Frontend load: < 2 seconds
- Concurrent negotiations: Up to 10

## Troubleshooting

### Chatbot Issues
```bash
# API won't start
lsof -i :8000
kill <PID>
./start_demo.sh

# Slow responses
# Ensure Neo4j is disabled (default)
unset ENABLE_NEO4J
./stop_demo.sh && ./start_demo.sh

# Check logs
tail -f logs/api.log
tail -f logs/gradio.log
```

### Resource Management Issues
```bash
# Backend port conflict
lsof -i :8000
kill <PID>

# Frontend port conflict
lsof -i :5173
kill <PID>

# CORS errors
# Check backend/app/main.py CORS settings
# Ensure frontend URL in allow_origins

# Build errors
cd resource-hub-main
rm -rf node_modules dist
npm install
npm run build
```

## Development

### Chatbot Development
```bash
# Backend with auto-reload
cd negotiation_chatbot
python -m uvicorn negotiation_chatbot.main:app --reload

# Gradio UI
python -m negotiation_chatbot.gradio_ui
```

### Resource Management Development
```bash
# Backend with auto-reload
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend with hot reload
cd resource-hub-main
npm run dev
```

## Testing

### Chatbot Tests
```bash
# Run comprehensive tests
bash comprehensive_test.sh

# Test specific endpoint
curl http://localhost:8000/health
curl -X POST http://localhost:8000/label \
  -H "Content-Type: application/json" \
  -d '{"text": "Let us split equally"}'
```

### Resource Management Tests
```bash
# Test department creation
curl -X POST http://localhost:8000/api/departments \
  -H "Content-Type: application/json" \
  -d '{"department_id":"test","department_name":"Test"}'

# Test department retrieval
curl http://localhost:8000/api/departments
```

## Production Deployment

### Chatbot
```bash
# Use production ASGI server
gunicorn negotiation_chatbot.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:8000
```

### Resource Management
```bash
# Build frontend
cd resource-hub-main
npm run build

# Backend serves frontend automatically
cd backend
gunicorn app.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:8000
```

## Tech Stack Summary

### Chatbot
- **Backend**: FastAPI, Python 3.13
- **UI**: Gradio 6.0
- **LLM**: OpenAI, Google Gemini, Ollama
- **RAG**: ChromaDB, Sentence Transformers
- **Graph**: Neo4j (optional)
- **Data**: CaSiNo corpus, DoND dataset

### Resource Management
- **Frontend**: React 18, TypeScript, Vite
- **UI**: Tailwind CSS, Radix UI, Lucide icons
- **Backend**: FastAPI, Python 3.13
- **Auth**: JWT, bcrypt
- **API**: RESTful, OpenAPI/Swagger

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is for educational and research purposes.

## Support

- **Chatbot Issues**: Check [CHATBOT_DEMO.md](CHATBOT_DEMO.md)
- **Resource Management Issues**: Check [RESOURCE_MANAGEMENT.md](RESOURCE_MANAGEMENT.md)
- **Logs**: `tail -f logs/*.log`
- **API Docs**: http://localhost:8000/docs

## Acknowledgments

- CaSiNo negotiation corpus for RAG system
- Deal-or-No-Deal dataset for visualization
- FastAPI and Gradio frameworks
- React and Vite communities
