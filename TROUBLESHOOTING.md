# 🐛 Troubleshooting Guide

## Auto-Proposal Issues

### Bot Proposal Not Appearing

**Symptoms**: Send message as "Other Party" but no bot proposal appears

**Diagnosis**:
1. Check browser console for JavaScript errors
2. Check application logs for Python errors
3. Verify `autoplay.py` is properly imported

**Solutions**:
```bash
# 1. Check if autoplay module exists
ls app/autoplay.py

# 2. Test import manually
cd app
python -c "from autoplay import generate_bot_proposal; print('Import successful')"

# 3. Check preference estimation
python -c "from preference import estimate_preferences; print('Preference module loaded')"
```

**Common Causes**:
- Missing `autoplay.py` file
- Import path issues
- Preference estimation model not loaded
- Insufficient conversation history for preference estimation

### Proposal Format Issues

**Symptoms**: Bot proposal appears but formatting is wrong

**Check**:
1. Verify `format_proposal_message()` function
2. Check item counts are correct
3. Ensure proposal structure is valid

**Debug**:
```python
# Add debug prints to autoplay.py
print(f"Generated proposal: {proposal}")
print(f"Item counts: {counts}")
```

### Preference Estimation Failures

**Symptoms**: "No proposal available" message

**Causes**:
- Conversation too short for preference estimation
- Preference model not loaded
- API errors in estimation

**Solutions**:
```bash
# 1. Check preference model
python -c "from preference import load_pref_model; load_pref_model()"

# 2. Test with longer conversation
# Send multiple messages before testing auto-proposal

# 3. Check model files exist
ls models/preference_estimator*
```

## UI Issues

### Gradio Interface Not Loading

**Symptoms**: Browser shows error or blank page

**Diagnosis**:
```bash
# 1. Check if Gradio is running
curl http://localhost:7860

# 2. Check port availability
lsof -i :7860

# 3. Check Python dependencies
pip list | grep gradio
```

**Solutions**:
```bash
# 1. Kill existing process
pkill -f gradio

# 2. Reinstall dependencies
pip install -r requirements.txt

# 3. Start with debug mode
python app/gradio_ui.py --debug
```

### Model Dropdown Empty

**Symptoms**: No models available in dropdown

**Diagnosis**:
```bash
# 1. Check Ollama connection
curl http://localhost:11434/api/tags

# 2. Check available models
ollama list

# 3. Test model detection
python -c "from app.llm_client import get_available_providers; print(get_available_providers())"
```

**Solutions**:
```bash
# 1. Start Ollama
ollama serve

# 2. Pull a model
ollama pull qwen3:latest

# 3. Refresh models in UI
# Click "Refresh Models" button
```

## API Issues

### API Connection Failed

**Symptoms**: "Cannot connect to API" errors

**Diagnosis**:
```bash
# 1. Check API health
curl http://localhost:8000/health

# 2. Check API logs
docker-compose logs api

# 3. Check if API is running
ps aux | grep api_server
```

**Solutions**:
```bash
# 1. Start API server
cd app
python api_server.py

# 2. Or restart Docker services
docker-compose restart api

# 3. Check environment variables
echo $API_BASE_URL
```

### Coach Advice Not Working

**Symptoms**: No coach advice appears after messages

**Diagnosis**:
1. Check API endpoint `/chat`
2. Verify model configuration
3. Check request/response format

**Debug**:
```bash
# Test coach endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conv_id":"test","speaker":"You","text":"","model":"qwen3:latest","provider":"ollama"}'
```

## Data Issues

### DoND Sample Loading Failed

**Symptoms**: Slider doesn't load conversation data

**Diagnosis**:
```bash
# 1. Check if data files exist
ls deal_or_no_dialog/exported/validation.jsonl

# 2. Test data loading
python -c "from dond_data import load_dond; print(len(load_dond('validation')))"

# 3. Check file permissions
ls -la deal_or_no_dialog/exported/

# 4. Check environment variable
echo $DOND_DATA_DIR
```

**Solutions**:
```bash
# 1. Set environment variable to point to your data directory
export DOND_DATA_DIR="/path/to/your/data/directory"

# 2. Create the expected directory structure
mkdir -p deal_or_no_dialog/exported
# Copy your .jsonl files to deal_or_no_dialog/exported/

# 3. Fix permissions (if needed)
chmod 644 deal_or_no_dialog/exported/*.jsonl

# 4. Test data loading
python -c "from dond_data import load_dond; samples = load_dond('validation'); print(f'Loaded {len(samples)} samples')"
```

### Conversation Not Saving

**Symptoms**: Conversations disappear after refresh

**Diagnosis**:
```bash
# 1. Check data directory
ls -la app/data/

# 2. Check write permissions
ls -la app/data/

# 3. Test file writing
python -c "import json; json.dump({'test': 'data'}, open('app/data/test.json', 'w'))"
```

**Solutions**:
```bash
# 1. Create data directory
mkdir -p app/data

# 2. Fix permissions
chmod 755 app/data

# 3. Check disk space
df -h
```

## Performance Issues

### Slow Response Times

**Symptoms**: UI is slow, proposals take long to generate

**Diagnosis**:
1. Check CPU/memory usage
2. Monitor API response times
3. Check model loading times

**Solutions**:
```bash
# 1. Monitor system resources
htop

# 2. Check API performance
time curl http://localhost:8000/health

# 3. Use smaller models
ollama pull qwen3:1b  # Instead of qwen3:latest
```

### Memory Issues

**Symptoms**: Out of memory errors

**Solutions**:
```bash
# 1. Use smaller models
ollama pull qwen3:1b

# 2. Limit concurrent requests
# Add rate limiting to API

# 3. Monitor memory usage
docker stats
```

## Debug Commands

### Enable Verbose Logging

```bash
# Set debug level
export LOG_LEVEL=DEBUG

# Start with debug output
python app/gradio_ui.py --debug

# Check logs
tail -f app/logs/app.log
```

### Test Individual Components

```bash
# Test Pareto optimization
python -c "from app.pareto import best_offer; print('Pareto OK')"

# Test preference estimation
python -c "from app.preference import estimate_preferences; print('Preference OK')"

# Test autoplay
python -c "from app.autoplay import generate_bot_proposal; print('Autoplay OK')"
```

### Network Diagnostics

```bash
# Check all required ports
netstat -tulpn | grep -E ':(7860|8000|11434)'

# Test API endpoints
curl -v http://localhost:8000/health

# Test Ollama
curl -v http://localhost:11434/api/tags
```

## Emergency Procedures

### Complete Reset

```bash
# Stop all services
docker-compose down

# Clear all data
rm -rf app/data/*
rm -rf ~/.ollama/models/*

# Restart from scratch
docker-compose up --build
```

### Manual Recovery

```bash
# 1. Check all services
docker-compose ps

# 2. Restart specific service
docker-compose restart api

# 3. Check logs
docker-compose logs -f

# 4. Rebuild if needed
docker-compose up --build --force-recreate
```

## Getting Help

### Collect Debug Information

```bash
# System info
uname -a
python --version
docker --version

# Service status
docker-compose ps
curl http://localhost:8000/health
curl http://localhost:11434/api/tags

# Logs
docker-compose logs > debug_logs.txt
```

### Report Issues

When reporting issues, include:
1. Error messages and stack traces
2. System information (OS, Python version, etc.)
3. Steps to reproduce
4. Debug logs
5. Expected vs actual behavior
