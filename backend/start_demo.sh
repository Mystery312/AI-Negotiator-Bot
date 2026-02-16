#!/bin/bash

echo "🚀 Starting Resource Management System Demo..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Please run this script from the 'Resource Management Program' directory"
    exit 1
fi

# Check if dependencies are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠️  Dependencies not installed. Installing now..."
    pip install -r requirements.txt
fi

# Check if port 8000 is available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8000 is already in use. Please stop the other service or use a different port."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Starting API server on http://localhost:8000"
echo ""
echo "🌐 Open these URLs in your browser:"
echo "   🎯 Live Demo:     http://localhost:8000/demo"
echo "   📖 API Docs:      http://localhost:8000/docs"
echo "   🏠 Home Page:     http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 -m app.main

