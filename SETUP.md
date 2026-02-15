# Setup Instructions

## First Time Setup

1. **Create Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate  # On Windows
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Environment (Optional)**
Edit `.env` file to configure:
- LLM API keys (Google, OpenAI)
- Neo4j settings (if using conversation graphs)
- RAG preloading (faster startup vs faster first request)

4. **Run the Demo**
```bash
./start_demo.sh
```

## Troubleshooting First Run

If you get import errors:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If the demo doesn't start:
```bash
# Check logs
tail -f logs/api.log
tail -f logs/gradio.log
```

If port conflicts occur:
```bash
# Kill processes on ports 8000 and 7860
./stop_demo.sh
# Then restart
./start_demo.sh
```
