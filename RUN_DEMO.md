# 🚀 Run Negotiation Chatbot Demo - Local Setup

This guide will get your negotiation chatbot running locally in just a few minutes.

---

## ✅ Pre-Flight Check

First, verify everything is ready:

```bash
python test_app.py
```

If you see `✅ All tests passed!`, you're good to go!

---

## 🎯 Quick Start (Recommended)

The easiest way to run a demo using **cloud LLMs** (no Ollama needed):

### Step 1: Ensure Dependencies are Installed

```bash
pip install -r requirements.txt
```

### Step 2: Start the Backend API

```bash
python -m negotiation_chatbot.main
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ **API is now running at http://localhost:8000**

### Step 3: Start the Gradio UI

Open a **NEW terminal** and run:

```bash
python -m negotiation_chatbot.gradio_ui
```

You should see:
```
Running on local URL:  http://127.0.0.1:7860
```

✅ **Open your browser to http://localhost:7860**

### Step 4: Try the Demo!

1. **In the Gradio UI:**
   - Enter names for both parties (e.g., "Alice" and "Bob")
   - For the model, select or enter: `gemini-2.0-flash-exp` (uses your Google API key)
   - Start chatting!

2. **Example conversation:**
   ```
   Alice: "I'd like to negotiate the distribution of resources."
   → Get AI coaching advice

   Bob: "What do you propose?"
   → Get strategic recommendations
   ```

---

## 🔧 Alternative: Run with Ollama (Local LLM)

If you want to use local models:

### Step 1: Install and Start Ollama

**Option A: Native Ollama**
```bash
# Install from https://ollama.ai
# Then start the service
ollama serve
```

**Option B: Docker Ollama**
```bash
docker run -d --name ollama -p 11434:11434 ollama/ollama:latest
```

### Step 2: Pull a Model

```bash
# Recommended model for chat
ollama pull qwen3:latest

# Or other options:
# ollama pull llama3.2
# ollama pull mistral
# ollama pull gemma2
```

### Step 3: Verify Ollama is Running

```bash
curl http://localhost:11434/api/tags
```

You should see a list of available models.

### Step 4: Start Backend and UI

```bash
# Terminal 1: Backend
python -m negotiation_chatbot.main

# Terminal 2: UI
python -m negotiation_chatbot.gradio_ui
```

### Step 5: Use Local Model in UI

In the Gradio UI, select `qwen3:latest` (or your installed model) from the dropdown.

---

## 🐳 Docker Compose (Full Stack)

Run everything with one command:

```bash
docker-compose up --build
```

This starts:
- ✅ Neo4j database
- ✅ Ollama with models
- ✅ FastAPI backend
- ✅ Gradio UI

**Access the UI at:** http://localhost:7860

**First time setup:**
```bash
# Pull a model into Ollama container
docker exec -it ollama ollama pull qwen3:latest
```

---

## 📊 Testing the API Directly

### Health Check
```bash
curl http://localhost:8000/health
```

Response: `{"status":"ok"}`

### Send a Chat Message
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "conv_id": "demo-123",
    "speaker": "Alice",
    "text": "I would like to propose a fair distribution",
    "model": "gemini-2.0-flash-exp",
    "provider": "gemini"
  }'
```

### Label Text
```bash
curl -X POST http://localhost:8000/label \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I think we should split the items equally"
  }'
```

---

## 🎮 Demo Scenarios

### Scenario 1: Basic Negotiation

1. **Setup:**
   - Name 1: "Alice"
   - Name 2: "Bob"
   - Model: `gemini-2.0-flash-exp`

2. **Conversation:**
   ```
   Alice: "Hi, I'd like to negotiate the terms of our agreement."
   → See coaching advice

   Bob: "What are you proposing?"
   → See strategic guidance

   Alice: "I propose we split the resources 60-40."
   → Get recommendations
   ```

### Scenario 2: Multi-Item Negotiation

1. **Setup:**
   - Enable "DoND Conversation Visualizer" accordion
   - Load a sample conversation
   - See item counts and values

2. **Experiment:**
   - Try different allocation strategies
   - See Pareto-optimal proposals
   - Get coaching on each move

### Scenario 3: API Integration

```python
import requests

API_URL = "http://localhost:8000"

# Send a message
response = requests.post(f"{API_URL}/chat", json={
    "conv_id": "python-demo",
    "speaker": "Alice",
    "text": "Let's negotiate",
    "model": "gemini-2.0-flash-exp"
})

print("Advice:", response.json()["advice"])
print("Reply:", response.json()["reply"])
```

---

## 🛠️ Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: API won't start (port 8000 in use)

**Solution:**
```bash
# Option 1: Kill the process using port 8000
lsof -ti:8000 | xargs kill

# Option 2: Use a different port
PORT=8001 python -m negotiation_chatbot.main
```

### Issue: Gradio UI can't connect to API

**Solution:**
```bash
# Check API is running
curl http://localhost:8000/health

# If not, start it:
python -m negotiation_chatbot.main
```

### Issue: Ollama models not showing in dropdown

**Solution:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If not running:
ollama serve

# Or use cloud models instead (Gemini/OpenAI)
```

### Issue: Neo4j connection errors

**Don't worry!** Neo4j is optional. It's already disabled in `.env`:
```bash
ENABLE_NEO4J=false
```

The app works fine without it. Graph features will be skipped.

### Issue: Google API rate limits

**Solution:**
Use Ollama instead:
1. Install Ollama
2. Pull a model: `ollama pull qwen3:latest`
3. Select `qwen3:latest` in the UI

---

## 📸 What You'll See

### Gradio UI Main Features:

1. **Chat Panel** (left)
   - Message history
   - Your messages and bot responses
   - Coaching advice after each turn

2. **Control Panel** (right)
   - Conversation ID
   - Negotiator names
   - Model selection
   - Role selector (You/Other Party)

3. **Statistics Panel**
   - Conversation metrics
   - Turn counts
   - Move types

4. **Visualization Tools**
   - Export conversations as Markdown
   - View conversation graphs
   - Analyze negotiation patterns

---

## 🎯 Key Features to Demo

### 1. AI Coaching
Every message gets strategic advice:
- What move was made
- Power dynamics analysis
- Suggested responses

### 2. RAG-Enhanced Recommendations
Uses negotiation tactics from academic literature:
- CaSiNo corpus integration
- Context-aware suggestions
- Evidence-based strategies

### 3. Multi-Model Support
Switch between models on the fly:
- Local: `qwen3:latest`, `llama3.2`, `mistral`
- Cloud: `gemini-2.0-flash-exp`, `gpt-4`, `gpt-3.5-turbo`

### 4. Pareto Optimization
For multi-item negotiations:
- Auto-generate optimal proposals
- Maximize joint utility
- Find win-win solutions

---

## 📈 Performance Tips

### For Fastest Response:
1. Use `gemini-2.0-flash-exp` (fastest cloud model)
2. Or use small local model: `qwen3:4b`

### For Best Quality:
1. Use `gpt-4` (requires OpenAI API key)
2. Or `gemini-1.5-pro`

### For Privacy:
1. Use Ollama with local models
2. No data leaves your machine

---

## 🔗 Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| **Gradio UI** | http://localhost:7860 | Main chat interface |
| **API Server** | http://localhost:8000 | Backend API |
| **API Docs** | http://localhost:8000/docs | Interactive API docs |
| **Health Check** | http://localhost:8000/health | Server status |

---

## 🎬 Quick Demo Script

**For a 5-minute demo:**

```bash
# 1. Start the backend (Terminal 1)
python -m negotiation_chatbot.main

# 2. Start the UI (Terminal 2)
python -m negotiation_chatbot.gradio_ui

# 3. Open browser to http://localhost:7860

# 4. In the UI:
#    - Names: "Alice" and "Bob"
#    - Model: "gemini-2.0-flash-exp"
#    - Send: "I'd like to negotiate fairly"
#    - See coaching advice!

# 5. Try switching to "Other Party" and respond
#    - See how the system adapts strategy

# 6. Export the conversation as Markdown
#    - Click "Export Markdown" button
```

---

## 🧹 Stopping the Demo

### Local Development:
```bash
# In each terminal, press Ctrl+C
```

### Docker Compose:
```bash
docker-compose down
```

---

## 📚 Next Steps

After the demo:
- Check out [README.md](README.md) for comprehensive documentation
- See [INTEGRATION_QUICK_START.md](INTEGRATION_QUICK_START.md) for frontend integration
- Review [FRONTEND_BACKEND_INTEGRATION_GUIDE.md](FRONTEND_BACKEND_INTEGRATION_GUIDE.md) for full features

---

## ✅ Demo Checklist

Before showing to others:

- [ ] Run `python test_app.py` - All tests pass
- [ ] Backend started - http://localhost:8000/health returns OK
- [ ] UI started - http://localhost:7860 loads
- [ ] Model selected - Shows in dropdown or manually entered
- [ ] Test message sent - Gets coaching advice
- [ ] Export works - Can download Markdown

---

## 🎉 You're Ready!

The negotiation chatbot is now running locally. Start negotiating! 🤝
