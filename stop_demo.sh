#!/bin/bash

# Negotiation Chatbot - Demo Shutdown Script

echo "=========================================="
echo "  Stopping Negotiation Chatbot Demo"
echo "=========================================="
echo ""

# Find and kill processes
echo "Stopping backend API (port 8000)..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Backend stopped"
else
    echo "ℹ Backend not running"
fi

echo ""
echo "Stopping Gradio UI (port 7860)..."
lsof -ti:7860 | xargs kill -9 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Gradio stopped"
else
    echo "ℹ Gradio not running"
fi

echo ""
echo "=========================================="
echo "  ✅ Demo Stopped"
echo "=========================================="
echo ""
