# Resource Management System Guide

Complete guide for running the full resource management web application with frontend and backend.

## What is This?

An AI-powered multi-department resource allocation system featuring:
- Department management with CRUD operations
- AI negotiation agents for resource allocation
- Real-time negotiation orchestration
- Consensus validation and deadlock detection
- Modern React frontend with real-time updates
- RESTful API backend with FastAPI

## Architecture

```
┌─────────────────────────────────────────────┐
│         React Frontend (Vite + React)       │
│         Port: 5173 (dev) / served by nginx  │
│         Location: resource-hub-main/        │
└──────────────────┬──────────────────────────┘
                   │ HTTP/REST API
┌──────────────────▼──────────────────────────┐
│         FastAPI Backend                     │
│         Port: 8000                          │
│         Location: backend/                  │
├─────────────────────────────────────────────┤
│  • Department Management (CRUD)             │
│  • Negotiation Orchestrator                 │
│  • AI Chatbot Agents                        │
│  • Resource Manager                         │
│  • Consensus Validator                      │
└─────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Development Mode (Recommended for Testing)

**Terminal 1 - Start Backend:**
```bash
cd backend
python -m app.main
# Backend runs on http://localhost:8000
```

**Terminal 2 - Start Frontend:**
```bash
cd resource-hub-main
npm run dev
# Frontend runs on http://localhost:5173
```

**Access:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 2: Production Build

**Build Frontend:**
```bash
cd resource-hub-main
npm run build
# Creates dist/ folder with production build
```

**Serve with Backend:**
```bash
cd backend
# Backend serves both API and frontend
python -m app.main
# Access at http://localhost:8000
```

## Requirements

### Backend (Python)
```bash
cd backend
pip install -r requirements.txt

# Required packages:
# - fastapi
# - uvicorn
# - pydantic
# - python-multipart
# - python-jose[cryptography]
# - passlib[bcrypt]
```

### Frontend (Node.js)
```bash
cd resource-hub-main
npm install

# Key dependencies:
# - react
# - vite
# - tailwindcss
# - @radix-ui/* (UI components)
# - lucide-react (icons)
```

## Features

### Department Management
1. Create departments with resource needs
2. View all departments and their status
3. Update department profiles and priorities
4. Delete departments
5. Assign projects to departments

### AI Negotiation
1. Define total available resources
2. Add departments with resource requirements
3. Start negotiation between AI agents
4. View real-time negotiation messages
5. See final resource allocation consensus

### Resource Types
- **Budget**: Financial resources
- **Personnel**: Human resources/headcount
- **Equipment**: Hardware/tools/assets
- **Time**: Timeline/schedule allocations

### Negotiation Features
- Multi-agent negotiation orchestration
- Proposal and counter-proposal generation
- Automatic consensus detection
- Deadlock detection and resolution
- Message history and reasoning tracking
- Pareto optimality checking

## API Endpoints

### Department Management
```bash
# Create department
POST /api/departments
{
  "department_id": "dept-001",
  "department_name": "Engineering",
  "resource_priorities": {"budget": 0.4, "personnel": 0.6},
  "current_projects": [...],
  "strategic_objectives": ["Build new platform"]
}

# Get all departments
GET /api/departments

# Get specific department
GET /api/departments/{department_id}

# Update department
PUT /api/departments/{department_id}

# Delete department
DELETE /api/departments/{department_id}
```

### Negotiation
```bash
# Start negotiation
POST /api/start-negotiation
{
  "departments": ["dept-001", "dept-002"],
  "total_resources": {"budget": 1000000, "personnel": 50}
}

# Get negotiation status
GET /api/negotiation/{negotiation_id}/status

# Get messages
GET /api/negotiation/{negotiation_id}/messages

# Submit manual message
POST /api/negotiation/{negotiation_id}/message
```

### Authentication
```bash
# Login
POST /auth/login
{
  "username": "admin",
  "password": "password"
}

# Register
POST /auth/register
{
  "username": "newuser",
  "password": "securepass",
  "email": "user@example.com"
}
```

## Configuration

### Backend Environment Variables

Create `backend/.env`:
```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:8000

# AI Models (for negotiation agents)
OPENAI_API_KEY=your-key
GOOGLE_API_KEY=your-key
```

### Frontend Environment Variables

Create `resource-hub-main/.env`:
```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```

## Frontend Structure

```
resource-hub-main/
├── src/
│   ├── components/      # React components
│   │   ├── ui/         # Reusable UI components
│   │   └── features/   # Feature-specific components
│   ├── pages/          # Page components
│   ├── services/       # API client services
│   ├── hooks/          # Custom React hooks
│   ├── lib/            # Utility functions
│   └── App.tsx         # Main app component
├── public/             # Static assets
└── index.html          # Entry point
```

## Backend Structure

```
backend/
├── app/
│   ├── main.py                      # FastAPI app entry point
│   ├── models.py                    # Pydantic models
│   ├── api_routes.py                # Negotiation API routes
│   ├── frontend_routes.py           # CRUD API routes
│   ├── auth_routes.py               # Authentication routes
│   ├── negotiation_orchestrator.py  # Negotiation logic
│   ├── department_chatbot.py        # AI agent for each dept
│   ├── resource_manager.py          # Resource allocation logic
│   ├── consensus_validator.py       # Consensus detection
│   └── base_logic_engine.py         # Core negotiation engine
└── requirements.txt
```

## Development Workflow

### 1. Start Backend Development
```bash
cd backend
# Watch mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start Frontend Development
```bash
cd resource-hub-main
npm run dev
# Hot reload enabled
```

### 3. Make Changes
- Frontend changes auto-reload in browser
- Backend changes auto-reload with uvicorn --reload
- API changes visible at http://localhost:8000/docs

### 4. Test API
```bash
# Test department creation
curl -X POST http://localhost:8000/api/departments \
  -H "Content-Type: application/json" \
  -d '{
    "department_id": "eng-001",
    "department_name": "Engineering",
    "resource_priorities": {"budget": 0.5, "personnel": 0.5}
  }'

# Test department retrieval
curl http://localhost:8000/api/departments
```

## Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
lsof -i :8000
kill <PID>
# Or use different port:
uvicorn app.main:app --port 8001
```

**Import errors:**
```bash
# Ensure you're in backend directory
cd backend
# Reinstall dependencies
pip install -r requirements.txt
```

**CORS errors:**
```bash
# Check CORS middleware in backend/app/main.py
# Ensure frontend URL is in allow_origins
```

### Frontend Issues

**Port 5173 already in use:**
```bash
lsof -i :5173
kill <PID>
# Or Vite will auto-select next available port
```

**API connection errors:**
```bash
# Check .env file has correct API URL
echo "VITE_API_BASE_URL=http://localhost:8000" > .env

# Restart dev server
npm run dev
```

**Build errors:**
```bash
# Clear cache and rebuild
rm -rf node_modules dist
npm install
npm run build
```

## Production Deployment

### 1. Build Frontend
```bash
cd resource-hub-main
npm run build
# Output in dist/ directory
```

### 2. Deploy Backend
```bash
cd backend
# Use production ASGI server
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### 3. Serve Frontend
Option A - Use nginx to serve frontend and proxy API:
```nginx
server {
    listen 80;

    # Serve frontend
    location / {
        root /path/to/resource-hub-main/dist;
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests
    location /api/ {
        proxy_pass http://localhost:8000;
    }

    location /auth/ {
        proxy_pass http://localhost:8000;
    }
}
```

Option B - Backend serves frontend (single deployment):
```python
# Backend automatically serves from dist/ if available
# Just ensure dist/ is in correct location
```

## API Documentation

Interactive API documentation available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Data Storage

The system uses in-memory storage by default. For persistence:

1. Departments stored in: `FrontendStorage` class
2. Negotiation state in: `NegotiationOrchestrator` class
3. Messages in: `MessageBroker` class

To add database persistence, integrate:
- PostgreSQL for relational data
- MongoDB for document storage
- Redis for caching and real-time updates

## Performance

### Backend
- API response time: < 100ms
- Negotiation round: 2-5 seconds per agent
- Concurrent negotiations: Up to 10 simultaneously

### Frontend
- Initial load: < 2 seconds
- Navigation: < 100ms
- Real-time updates: < 500ms latency

## Security

### Authentication
- JWT token-based authentication
- Password hashing with bcrypt
- Token expiration (configurable)

### API Security
- CORS configured for specific origins
- Input validation with Pydantic
- SQL injection prevention (when using DB)

### Best Practices
- Keep SECRET_KEY secure
- Use HTTPS in production
- Regularly update dependencies
- Implement rate limiting for API

## Support

For issues:
1. Check backend logs: Console output from `python -m app.main`
2. Check frontend logs: Browser console (F12)
3. Check API docs: http://localhost:8000/docs
4. Test endpoints: Use curl or Postman
