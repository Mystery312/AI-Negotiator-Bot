# Resource Management System

A multi-department resource management system where autonomous AI chatbots, each representing a different department, negotiate with each other to determine optimal resource allocation.

## Features

- **Department Chatbots**: Autonomous agents representing departments
- **Base Thinking Logic**: Shared reasoning framework for all chatbots
- **Negotiation Protocol**: Structured communication and argumentation
- **Resource Pool Management**: Centralized resource tracking
- **REST API**: Complete API for frontend integration
- **Real-time Updates**: Support for SSE/WebSocket (to be implemented)

## Installation

1. **Clone or navigate to the directory**
   ```bash
   cd "Resource Management Program"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Quick Start

### Start the API Server

```bash
cd app
python main.py
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Get an API Key

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "organization": "My Company",
    "email": "admin@company.com",
    "role": "admin"
  }'
```

### Create a Resource Pool

```bash
curl -X POST "http://localhost:8000/api/v1/resource-pools" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "resource_type": "budget",
    "total_available": 1000000.0,
    "description": "Q1 2024 Budget"
  }'
```

### Create a Negotiation

```bash
curl -X POST "http://localhost:8000/api/v1/negotiations" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "participants": ["engineering", "marketing", "sales"],
    "resource_pool_id": "pool_id_here",
    "negotiation_type": "budget"
  }'
```

## Project Structure

```
Resource Management Program/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── models.py               # Data models
│   ├── base_logic_engine.py    # Base thinking logic
│   ├── department_chatbot.py  # Department chatbot
│   ├── resource_manager.py    # Resource pool manager
│   ├── negotiation_orchestrator.py  # Negotiation coordination
│   ├── consensus_validator.py  # Consensus validation
│   ├── message_broker.py      # Inter-chatbot communication
│   ├── api_auth.py             # API authentication
│   └── api_routes.py           # API endpoints
├── tests/                      # Test files
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── RESOURCE_MANAGEMENT_SYSTEM_PLAN.md  # Implementation plan
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new API key
- `POST /api/v1/auth/validate` - Validate API key

### Departments
- `GET /api/v1/departments` - List departments
- `GET /api/v1/departments/{id}` - Get department
- `POST /api/v1/departments` - Create department
- `PUT /api/v1/departments/{id}` - Update department
- `DELETE /api/v1/departments/{id}` - Delete department

### Resource Pools
- `GET /api/v1/resource-pools` - List pools
- `GET /api/v1/resource-pools/{id}` - Get pool
- `POST /api/v1/resource-pools` - Create pool
- `PUT /api/v1/resource-pools/{id}` - Update pool

### Negotiations
- `POST /api/v1/negotiations` - Create negotiation
- `GET /api/v1/negotiations` - List negotiations
- `GET /api/v1/negotiations/{id}` - Get negotiation
- `GET /api/v1/negotiations/{id}/messages` - Get messages
- `POST /api/v1/negotiations/{id}/intervene` - Manual intervention
- `POST /api/v1/negotiations/{id}/cancel` - Cancel negotiation

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

Follow PEP 8 style guidelines.

## Next Steps

See `RESOURCE_MANAGEMENT_SYSTEM_PLAN.md` for:
- Complete implementation plan
- Architecture details
- Frontend integration guide
- Advanced features

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

