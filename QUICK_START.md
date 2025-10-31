# 🚀 Quick Start Guide

## Essential Commands

### Start the System
```bash
# Option 1: Docker (recommended)
docker-compose up --build

# Option 2: Local development
cd app
python gradio_ui.py
```

### Test the System
```bash
# Test API health
curl http://localhost:8000/health

# Test Ollama connection
curl http://localhost:11434/api/tags

# Test UI (open in browser)
http://localhost:7860
```

### Debug Common Issues

#### Import Errors
```bash
# Fix Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from project root
cd /path/to/chat-negotiator
python -m app.gradio_ui
```

#### Ollama Issues
```bash
# Start Ollama
ollama serve

# Pull a model
ollama pull qwen3:latest

# Check available models
ollama list
```

#### Port Conflicts
```bash
# Check what's using port 7860
lsof -i :7860

# Kill process if needed
kill -9 <PID>
```

### Development Commands

#### Run Tests
```bash
# All tests
python -m pytest tests/

# Specific test
python -m pytest tests/test_pareto.py -v

# With coverage
python -m pytest --cov=app tests/
```

#### Run Simulations
```bash
# Test Pareto vs No-Pareto
python app/simulate_dond.py --n 100 --baseline equal

# Test with different baselines
python app/simulate_dond.py --baseline walkaway
python app/simulate_dond.py --baseline statusquo
```

#### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python app/gradio_ui.py

# Or set in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### UI Testing Checklist

1. **Basic Functionality**
   - [ ] UI loads at http://localhost:7860
   - [ ] Conversation ID is auto-generated
   - [ ] Model dropdown shows available models
   - [ ] Can send messages as "You"
   - [ ] Can send messages as "Other Party"

2. **Auto-Proposal Feature**
   - [ ] Send message as "Other Party"
   - [ ] Bot proposal appears automatically
   - [ ] Proposal shows item allocations
   - [ ] Coach advice appears for both roles

3. **DoND Sample Loading**
   - [ ] Move sample slider
   - [ ] Conversation loads with sample data
   - [ ] Item counts display in Statistics panel
   - [ ] Can continue conversation after loading

4. **Debug Panels**
   - [ ] Debug/Status panel shows API status
   - [ ] Conversation Inspector shows raw data
   - [ ] Model & API panel shows connections
   - [ ] Tools & Exports work correctly

### Common Error Messages

#### "No module named 'app'"
```bash
# Solution: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### "Connection refused" for Ollama
```bash
# Solution: Start Ollama
ollama serve
```

#### "Port already in use"
```bash
# Solution: Kill existing process
lsof -i :7860
kill -9 <PID>
```

#### "Failed to fetch stats"
```bash
# Solution: Check API is running
curl http://localhost:8000/health
```

### Performance Tips

1. **Use Docker** for consistent environment
2. **Pull models** before starting: `ollama pull qwen3:latest`
3. **Monitor logs** for performance issues
4. **Use debug mode** for troubleshooting

### Emergency Reset

```bash
# Stop all services
docker-compose down

# Clear data (if needed)
rm -rf app/data/*

# Restart
docker-compose up --build
```
