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

# Check dependencies
echo ""
echo "Checking dependencies..."
python -c "import fastapi, gradio, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
else
    echo "✓ Dependencies OK"
fi

# Run test
echo ""
echo "Running pre-flight tests..."
python test_app.py > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠ Warning: Some tests failed, but continuing..."
else
    echo "✓ All tests passed"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

echo ""
echo "=========================================="
echo "  Starting Backend API..."
echo "=========================================="
echo ""
echo "Starting FastAPI server on http://localhost:8000"
echo "Logs: logs/api.log"
echo ""

# Start backend in background
nohup python -m negotiation_chatbot.main > logs/api.log 2>&1 &
API_PID=$!
echo "Backend PID: $API_PID"

# Wait for API to start
echo "Waiting for API to start..."
sleep 3

# Check if API is running
curl -s http://localhost:8000/health > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Failed to start API. Check logs/api.log"
    exit 1
fi

echo "✓ API started successfully"

echo ""
echo "=========================================="
echo "  Starting Gradio UI..."
echo "=========================================="
echo ""
echo "Starting Gradio UI on http://localhost:7860"
echo "Logs: logs/gradio.log"
echo ""

# Start Gradio in background
nohup python -m negotiation_chatbot.gradio_ui > logs/gradio.log 2>&1 &
GRADIO_PID=$!
echo "Gradio PID: $GRADIO_PID"

# Wait for Gradio to start
echo "Waiting for Gradio to start..."
sleep 5

echo ""
echo "=========================================="
echo "  ✅ Demo Started Successfully!"
echo "=========================================="
echo ""
echo "Access Points:"
echo "  • Gradio UI:  http://localhost:7860"
echo "  • API:        http://localhost:8000"
echo "  • API Docs:   http://localhost:8000/docs"
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
echo "Opening browser in 3 seconds..."
sleep 3

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
