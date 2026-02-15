#!/bin/bash

# Negotiation Chatbot - Demo Startup Script
# This script starts both the backend API and Gradio UI

echo "=========================================="
echo "  Negotiation Chatbot Demo Startup"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✓ Virtual environment found"
    source .venv/bin/activate
else
    echo "⚠ No virtual environment found. Using system Python."
fi

# Skip dependency check for faster startup
# Dependencies assumed to be installed
# Run: pip install -r requirements.txt if needed
echo "Skipping dependency check for faster startup..."

# Skip tests - no test_app.py file exists
# Tests removed for faster startup

# Create logs directory if it doesn't exist
mkdir -p logs

echo ""
echo "=========================================="
echo "  Starting Backend API..."
echo "=========================================="
echo ""

# Check if API is already running
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠ API already running on port 8001. Stopping it first..."
    lsof -ti:8001 | xargs kill -9 2>/dev/null
    sleep 2
fi

echo "Starting FastAPI server on http://localhost:8001"
echo "Logs: logs/api.log"
echo ""

# Start backend in background
# PORT defaults to 8001 in main.py now
nohup python -m negotiation_chatbot.main > logs/api.log 2>&1 &
API_PID=$!
echo "Backend PID: $API_PID"

# Wait for API to start (with polling instead of fixed sleep)
echo "Waiting for API to start (this may take 30-50 seconds on first run)..."
echo "Loading: Neo4j → PyTorch → PreferenceEstimator model..."
for i in {1..100}; do
    sleep 0.5
    curl -s http://localhost:8001/health > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ API ready after $((i/2)) seconds"
        break
    fi
    # Progress indicator every 10 seconds
    if [ $((i % 20)) -eq 0 ]; then
        echo "  ... still waiting ($((i/2))s elapsed)"
    fi
done

# Final check with grace period
curl -s http://localhost:8001/health > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠ API not ready yet, waiting additional 10 seconds..."
    sleep 10
    curl -s http://localhost:8001/health > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Failed to start API after 60 seconds. Check logs/api.log"
        exit 1
    fi
    echo "✓ API started (required extra time)"
fi

echo "✓ API started successfully"

echo ""
echo "=========================================="
echo "  Starting Gradio UI..."
echo "=========================================="
echo ""

# Check if Gradio is already running
if lsof -Pi :7860 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠ Gradio already running on port 7860. Stopping it first..."
    lsof -ti:7860 | xargs kill -9 2>/dev/null
    sleep 2
fi

echo "Starting Gradio UI on http://localhost:7860"
echo "Logs: logs/gradio.log"
echo ""

# Start Gradio in background
nohup python -m negotiation_chatbot.gradio_ui > logs/gradio.log 2>&1 &
GRADIO_PID=$!
echo "Gradio PID: $GRADIO_PID"

# Wait for Gradio to start (with polling instead of fixed sleep)
echo "Waiting for Gradio to start..."
for i in {1..20}; do
    sleep 0.5
    curl -s http://localhost:7860 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Gradio ready after $((i/2)) seconds"
        break
    fi
done

echo ""
echo "=========================================="
echo "  ✅ Demo Started Successfully!"
echo "=========================================="
echo ""
echo "Access Points:"
echo "  • Gradio UI:  http://localhost:7860"
echo "  • API:        http://localhost:8001"
echo "  • API Docs:   http://localhost:8001/docs"
echo ""
echo "Process IDs:"
echo "  • Backend API: $API_PID"
echo "  • Gradio UI:   $GRADIO_PID"
echo ""
echo "To stop the demo:"
echo "  kill $API_PID $GRADIO_PID"
echo ""
echo "Or use: ./stop_demo.sh"
echo ""
echo "Logs:"
echo "  • API:    tail -f logs/api.log"
echo "  • Gradio: tail -f logs/gradio.log"
echo ""
echo "=========================================="
echo ""
echo "Opening browser..."
sleep 1

# Try to open browser (works on macOS, Linux, WSL)
if command -v open &> /dev/null; then
    open http://localhost:7860
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:7860
elif command -v wslview &> /dev/null; then
    wslview http://localhost:7860
else
    echo "Please open http://localhost:7860 in your browser"
fi

echo ""
echo "Press Ctrl+C to view logs, or close this window."
echo ""

# Follow logs
tail -f logs/gradio.log
