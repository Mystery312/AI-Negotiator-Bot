# Live Demo Guide - Resource Management System & Chatbot

This guide will help you run both the Resource Management System and the Chatbot system for a live demonstration.

## Prerequisites

1. **Python 3.8+** installed
2. **All dependencies installed** (see installation steps below)

## Step 1: Install Dependencies

### For Resource Management System:
```bash
cd "Resource Management Program"
pip install -r requirements.txt
```

### For Chatbot System:
```bash
cd ..  # Go back to main chatbot directory
pip install -r requirements.txt
```

## Step 2: Start the Resource Management System API

Open a terminal and run:

```bash
cd "Resource Management Program"
python -m app.main
```

Or alternatively:
```bash
cd "Resource Management Program"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: **http://localhost:8000**

You can verify it's running by visiting:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Step 3: Start the Chatbot System (Optional - if you want both)

Open a **second terminal** and run:

```bash
cd app  # From the main chatbot directory
python main.py
```

Or if using uvicorn:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

The Chatbot API will be available at: **http://localhost:8001**

## Step 4: Open the Demo Interface

1. Open the `demo.html` file in your web browser:
   ```bash
   # On macOS:
   open "Resource Management Program/demo.html"
   
   # On Linux:
   xdg-open "Resource Management Program/demo.html"
   
   # On Windows:
   start "Resource Management Program/demo.html"
   ```

   Or simply double-click the `demo.html` file in your file explorer.

2. The demo interface will open in your browser.

## Step 5: Demo Workflow

### Quick Demo Flow:

1. **Check Connection**
   - Click "Check Connection" button
   - Should show "Connected" in green

2. **Get API Key**
   - Go to "Authentication" tab
   - Fill in organization name and email
   - Click "Register & Get API Key"
   - Copy the API key (it's also saved in browser)

3. **Create Departments**
   - Go to "Departments" tab
   - Create departments like:
     - Engineering (budget: 0.8, personnel: 0.9)
     - Marketing (budget: 0.9, personnel: 0.7)
     - Sales (budget: 0.85, personnel: 0.8)

4. **Create Resource Pool**
   - Go to "Resource Pools" tab
   - Create a budget pool (e.g., $1,000,000)
   - Note the Pool ID that gets created

5. **Start Negotiation**
   - Go to "Negotiations" tab
   - Enter participant department IDs (comma-separated)
   - Enter the Resource Pool ID from step 4
   - Click "Start Negotiation"
   - View the negotiation status and messages

### Example Demo Script:

```
1. "Let me show you our Resource Management System where AI chatbots 
   negotiate for resources."

2. [Get API Key] "First, I'll authenticate..."

3. [Create Engineering Department] "Here's our Engineering department 
   with their priorities..."

4. [Create Marketing Department] "And Marketing with their needs..."

5. [Create Resource Pool] "We have a $1M budget pool to allocate..."

6. [Start Negotiation] "Now let's watch the AI chatbots negotiate 
   for resources..."

7. [Show Negotiation] "You can see the negotiation in progress, with 
   proposals and counter-proposals..."
```

## Troubleshooting

### Port Already in Use

If port 8000 is already in use:
```bash
# Find what's using the port
lsof -i :8000

# Or change the port in app/main.py:
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Import Errors

If you get import errors:
```bash
# Make sure you're in the right directory
cd "Resource Management Program"

# Install dependencies
pip install -r requirements.txt

# Try running with Python module syntax
python -m app.main
```

### CORS Errors in Browser

If you see CORS errors, the API is already configured to allow all origins. Make sure:
- The API server is running
- You're accessing demo.html via file:// or http:// (not blocked by browser)

### API Not Responding

1. Check if the server is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. Check server logs for errors

3. Make sure no firewall is blocking port 8000

## Using the Chatbot System Separately

If you want to demo the chatbot system:

1. Start the chatbot API (port 8001 or different port)
2. Use the existing Gradio UI:
   ```bash
   cd app
   python gradio_ui.py
   ```
   This will open at http://localhost:7860

3. Or use the API directly via curl or Postman

## Integration Ideas

You can integrate both systems by:
- Having the Resource Management System use the chatbot's LLM for reasoning
- Using the chatbot's negotiation advice in resource negotiations
- Combining both UIs into a unified dashboard

## Next Steps

- Add real-time WebSocket updates for negotiations
- Create a unified dashboard combining both systems
- Add more visualization and analytics
- Integrate with the existing chatbot's LLM capabilities

## Support

If you encounter issues:
1. Check the server logs
2. Verify all dependencies are installed
3. Check that ports are not in use
4. Review the API documentation at http://localhost:8000/docs

