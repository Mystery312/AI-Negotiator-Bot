# Quick Start Guide - 5 Minutes to Demo

## 🚀 Fastest Way to Run the Demo

### Option 1: Using the Start Script (Easiest)

```bash
cd "Resource Management Program"
./start_demo.sh
```

Then open your browser and visit:
- **🎯 Live Demo**: http://localhost:8000/demo
- **📖 API Docs**: http://localhost:8000/docs
- **🏠 Home**: http://localhost:8000

### Option 2: Manual Start

1. **Install dependencies** (one time):
   ```bash
   cd "Resource Management Program"
   pip install -r requirements.txt
   ```

2. **Start the server**:
   ```bash
   python -m app.main
   ```

3. **Open the demo**:
   - Open `demo.html` in your web browser
   - Or visit http://localhost:8000/docs for API documentation

## 📋 Demo Checklist

- [ ] Server running on http://localhost:8000
- [ ] Can access http://localhost:8000/health (returns {"status": "ok"})
- [ ] demo.html opens in browser
- [ ] "Check Connection" shows "Connected"

## 🎯 2-Minute Demo Flow

1. **Get API Key** (Authentication tab)
   - Click "Register & Get API Key"
   - Copy the key

2. **Create Resource Pool** (Resource Pools tab)
   - Type: Budget
   - Total: 1000000
   - Click "Create Resource Pool"
   - **Copy the pool_id from the result**

3. **Create Departments** (Departments tab)
   - Name: Engineering
   - Priorities: `{"budget": 0.8, "personnel": 0.9}`
   - Objectives: Product development
   - Click "Create Department"
   - Repeat for Marketing and Sales

4. **Start Negotiation** (Negotiations tab)
   - Participants: engineering, marketing, sales
   - Pool ID: [paste from step 2]
   - Click "Start Negotiation"
   - View the negotiation!

## 🔧 Troubleshooting

**Port 8000 in use?**
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9
```

**Import errors?**
```bash
pip install -r requirements.txt
```

**Can't connect?**
- Make sure server is running
- Check http://localhost:8000/health
- Try http://127.0.0.1:8000 instead

## 📱 For Live Presentation

1. Start server in terminal (keep it visible)
2. Open demo.html in browser (full screen)
3. Have the demo flow ready (see above)
4. Show API docs at http://localhost:8000/docs as backup

## 🎬 Presentation Tips

- Start with the "Check Connection" to show it's working
- Create departments first to show the system
- Create a realistic budget (e.g., $1M)
- Show the negotiation in real-time
- Use the API docs page to show the technical side

