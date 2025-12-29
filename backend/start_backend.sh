#!/bin/bash

# Start the Resource Management Backend Server
# This script starts the FastAPI backend with frontend integration

echo "🚀 Starting Resource Management Backend Server..."
echo ""
echo "📍 Server will be available at: http://localhost:8000"
echo "📚 API Documentation at: http://localhost:8000/docs"
echo "🎯 Frontend endpoints: http://localhost:8000/api/*"
echo "🤖 Negotiation endpoints: http://localhost:8000/api/v1/*"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

# Change to the script's directory
cd "$(dirname "$0")"

# Start the server
python -m app.main
