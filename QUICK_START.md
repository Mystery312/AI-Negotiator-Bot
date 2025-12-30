# ⚡ Quick Start - Run the Demo NOW

## 🎯 The Absolute Fastest Way to Run

### Method 1: Automated Script (Easiest)

```bash
./start_demo.sh
```

That's it! The script will:
- ✅ Check dependencies
- ✅ Start the API server
- ✅ Start the Gradio UI
- ✅ Open your browser automatically

**To stop:**
```bash
./stop_demo.sh
```

---

### Method 2: Manual (2 Commands)

**Terminal 1:**
```bash
python -m negotiation_chatbot.main
```

**Terminal 2:**
```bash
python -m negotiation_chatbot.gradio_ui
```

**Browser:**
Open http://localhost:7860

**To stop:**
Press `Ctrl+C` in both terminals

---

## 🎮 Using the Demo

1. **Enter names** for both parties (e.g., "Alice" and "Bob")
2. **Select model**:
   - `gemini-2.0-flash-exp` (recommended - uses Google API)
   - Or any Ollama model if you have it running
3. **Start chatting!**
   - Type a message
   - See AI coaching advice
   - Get strategic recommendations

---

## ✅ Before You Start

**Quick check:**
```bash
python test_app.py
```

Should show: `✅ All tests passed!`

**If you see errors:**
```bash
pip install -r requirements.txt
```

---

## 🔧 Common Issues

### Port 8000 already in use
```bash
# Kill the process
lsof -ti:8000 | xargs kill
```

### Can't connect to API
```bash
# Check API is running
curl http://localhost:8000/health
```

### Missing dependencies
```bash
pip install -r requirements.txt
```

---

## 📚 Need More Help?

- **Full guide:** See [RUN_DEMO.md](RUN_DEMO.md)
- **Troubleshooting:** See [RUN_DEMO.md#troubleshooting](RUN_DEMO.md#troubleshooting)
- **API usage:** See [RUN_DEMO.md#testing-the-api-directly](RUN_DEMO.md#testing-the-api-directly)

---

## 🎉 That's It!

Your negotiation chatbot should now be running at:
- **UI:** http://localhost:7860
- **API:** http://localhost:8000

Happy negotiating! 🤝
