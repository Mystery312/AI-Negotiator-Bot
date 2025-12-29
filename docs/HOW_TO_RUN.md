# How to Run the Chatbot Application

This guide provides step-by-step instructions for running the negotiation chatbot application.

## Quick Status Check

Before starting, verify your setup:

```bash
# Run the test script to check if everything is configured correctly
python test_app.py
```

If all tests pass, you're ready to run the application!

---

## Method 1: Docker Compose (Recommended - Easiest Setup)

Docker Compose automatically sets up all required services (Neo4j, Ollama, API, Gradio UI).

### Prerequisites
- Docker and Docker Compose installed
- (Optional) NVIDIA GPU and drivers for GPU acceleration

### Steps

1. **Start all services:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - **Gradio UI**: http://localhost:7860
   - **API**: http://localhost:8000
   - **Neo4j Browser**: http://localhost:7474 (username: neo4j, password: 6xlBSIDu8Nc8gjXrpt3kNuwM7AZHGI3WJrfpN2fFDXE)
   - **Ollama**: http://localhost:11434

3. **Pull an Ollama model (first time only):**
   ```bash
   # In a new terminal while Docker is running
   docker exec -it ollama ollama pull qwen3:latest

   # Or try other models:
   # docker exec -it ollama ollama pull llama3.2
   # docker exec -it ollama ollama pull mistral
   ```

4. **Stop the services:**
   ```bash
   # Press Ctrl+C in the terminal running docker-compose
   # Or run:
   docker-compose down
   ```

### Troubleshooting Docker

- **Ports already in use**: Change port mappings in `docker-compose.yml`
- **GPU not detected**: Remove the GPU sections from `docker-compose.yml` if you don't have NVIDIA GPU
- **Services won't start**: Check logs with `docker-compose logs <service-name>`

---

## Method 2: Local Development (Full Control)

Run each component separately on your local machine.

### Prerequisites

1. **Python 3.8+** (tested with Python 3.13)
2. **Neo4j Database** (running locally or via Docker)
3. **Ollama** (optional, for local LLM models)
4. **Virtual Environment** (recommended)

### Steps

#### 1. Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# Install all required packages
pip install -r requirements.txt
```

#### 2. Start Neo4j Database

**Option A: Using Docker**
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

**Option B: Using Neo4j Desktop**
- Download from https://neo4j.com/download/
- Create a new database
- Start it and note the credentials

**Option C: Skip Neo4j (Limited Functionality)**
- The app will still work without Neo4j, but graph features will be disabled
- Just ensure `ENABLE_NEO4J=false` in your `.env` file

#### 3. Start Ollama (Optional)

**Install Ollama:**
```bash
# macOS/Linux - visit https://ollama.ai to download
# Or use Docker:
docker run -d --name ollama -p 11434:11434 ollama/ollama:latest
```

**Pull a model:**
```bash
ollama pull qwen3:latest
# Or other models: llama3.2, mistral, gemma2, etc.
```

**Verify Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

#### 4. Configure Environment Variables

Edit `.env` file in the project root:

```bash
# LLM Provider settings
DEFAULT_MODEL=qwen3:latest
OLLAMA_BASE_URL=http://localhost:11434

# Neo4j settings (set ENABLE_NEO4J=false to skip Neo4j)
ENABLE_NEO4J=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# API Keys (optional - only if using cloud LLMs)
# OPENAI_API_KEY=your_openai_key_here
# GOOGLE_API_KEY=your_google_api_key_here
```

#### 5. Start the FastAPI Backend

```bash
# From the project root
python -m app.main

# The API will start on http://localhost:8000
```

**Verify it's running:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok"}
```

#### 6. Start the Gradio UI

In a **new terminal** (keep the API running):

```bash
# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux

# Start Gradio
python -m app.gradio_ui

# The UI will start on http://localhost:7860
```

#### 7. Open the Application

Open your browser to **http://localhost:7860**

---

## Method 3: Minimal Setup (No Neo4j, No Ollama)

If you just want to test the application with cloud LLMs (OpenAI/Gemini):

### Steps

1. **Install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure `.env`:**
   ```bash
   # Disable Neo4j
   ENABLE_NEO4J=false

   # Add API keys for cloud LLMs
   OPENAI_API_KEY=your_openai_key
   GOOGLE_API_KEY=your_google_key
   ```

3. **Start the API:**
   ```bash
   python -m app.main
   ```

4. **Start the UI:**
   ```bash
   # In a new terminal
   python -m app.gradio_ui
   ```

5. **Use cloud models** in the UI (e.g., `gpt-4`, `gemini-pro`)

---

## Using the Application

### Web Interface (Gradio)

1. **Open** http://localhost:7860 in your browser

2. **Basic Setup:**
   - Enter names for both negotiating parties (e.g., "Alice" and "Bob")
   - Select an AI model from the dropdown (or enter a custom model name)
   - The conversation ID is auto-generated (or you can enter your own)

3. **Start Chatting:**
   - Type a message in the chat input
   - Click "Send" or press Enter
   - The system will:
     - Display your message
     - Provide AI coaching advice
     - Optionally generate bot proposals

4. **Features:**
   - **Coach Advice**: Strategic recommendations after each turn
   - **Bot Proposals**: Auto-generated when simulating the other party
   - **Export**: Save conversations as Markdown files
   - **Visualizations**: View negotiation graphs and statistics

### API Usage

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Send a chat message:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "conv_id": "demo123",
    "speaker": "Alice",
    "text": "I would like to negotiate the terms",
    "model": "qwen3:latest"
  }'
```

**Label text:**
```bash
curl -X POST http://localhost:8000/label \
  -H "Content-Type: application/json" \
  -d '{"text": "I propose we split the items equally"}'
```

**Get conversation graph:**
```bash
curl http://localhost:8000/graph/demo123
```

---

## Common Issues and Solutions

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'openai'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Neo4j Connection Failed

**Problem:** `ServiceUnavailable: Couldn't connect to localhost:7687`

**Solutions:**
1. Check if Neo4j is running:
   ```bash
   docker ps | grep neo4j
   ```

2. Start Neo4j:
   ```bash
   docker start neo4j
   ```

3. Or disable Neo4j in `.env`:
   ```bash
   ENABLE_NEO4J=false
   ```

### Ollama Not Found

**Problem:** `Cannot connect to Ollama at http://localhost:11434`

**Solutions:**
1. Check if Ollama is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. Start Ollama:
   ```bash
   ollama serve
   # Or
   docker start ollama
   ```

3. Use cloud LLMs instead (set API keys in `.env`)

### Port Already in Use

**Problem:** `Address already in use` error

**Solutions:**
1. Change the port in your startup command:
   ```bash
   PORT=8001 python -m app.main
   ```

2. Find and kill the process using the port:
   ```bash
   # macOS/Linux
   lsof -ti:8000 | xargs kill

   # Or specify a different port in docker-compose.yml
   ```

### Gradio UI Won't Connect to API

**Problem:** UI shows "API connection failed"

**Solutions:**
1. Verify API is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. Check `API_BASE_URL` environment variable:
   ```bash
   export API_BASE_URL=http://localhost:8000
   python -m app.gradio_ui
   ```

3. Check firewall settings

### Missing Dependencies

**Problem:** Various import errors

**Solution:**
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Or install specific missing packages:
pip install fastapi uvicorn gradio neo4j openai chromadb
```

---

## Stopping the Application

### Docker Compose:
```bash
# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes (clears all data)
docker-compose down -v
```

### Local Development:
```bash
# Press Ctrl+C in each terminal running the services

# Stop Neo4j (if using Docker)
docker stop neo4j

# Stop Ollama (if using Docker)
docker stop ollama
```

---

## Verifying Everything Works

Run the test script:
```bash
python test_app.py
```

This will check:
- All modules can be imported
- API responds to health checks
- Dependencies are correctly installed

If all tests pass, your application is ready to use!

---

## Next Steps

- See [README.md](README.md) for comprehensive documentation
- Check [CHATBOT_GUIDE.md](CHATBOT_GUIDE.md) for detailed feature explanations
- Explore the API documentation at http://localhost:8000/docs (when API is running)

---

## Quick Reference

| Component | Default URL | Purpose |
|-----------|-------------|---------|
| Gradio UI | http://localhost:7860 | Web interface for chat |
| FastAPI | http://localhost:8000 | Backend API server |
| Neo4j Browser | http://localhost:7474 | Database admin interface |
| Ollama | http://localhost:11434 | Local LLM server |
| API Docs | http://localhost:8000/docs | Interactive API documentation |

### Default Credentials

- **Neo4j** (Docker): neo4j / 6xlBSIDu8Nc8gjXrpt3kNuwM7AZHGI3WJrfpN2fFDXE
- **Neo4j** (Local): neo4j / password (or as configured in `.env`)
