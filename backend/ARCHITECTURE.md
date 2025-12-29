# System Architecture

## Overview

The Resource Management System consists of a **React frontend** and a **Python FastAPI backend** with two independent API layers.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     REACT FRONTEND                              │
│                   (Port 5173 - Vite)                            │
├─────────────────────────────────────────────────────────────────┤
│  Components:                                                    │
│  - Dashboard.tsx          - EmployeeList.tsx                    │
│  - EquipmentList.tsx      - InventoryList.tsx                   │
│  - RoomList.tsx           - BookingList.tsx                     │
│  - DataTable.tsx (Generic CRUD component)                       │
│                                                                 │
│  Services:                                                      │
│  - api.ts (API service layer)                                   │
│  - AuthContext (Auth state management)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ HTTP/JSON
                         │ (CORS enabled)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              PYTHON FASTAPI BACKEND                             │
│                (Port 8000 - Uvicorn)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────┐    ┌──────────────────────┐         │
│  │  Frontend API         │    │  Negotiation API     │         │
│  │  (/api/*)             │    │  (/api/v1/*)         │         │
│  ├───────────────────────┤    ├──────────────────────┤         │
│  │ Routes:               │    │ Routes:              │         │
│  │ - /employees          │    │ - /departments       │         │
│  │ - /equipment          │    │ - /resource-pools    │         │
│  │ - /inventory          │    │ - /negotiations      │         │
│  │ - /rooms              │    │                      │         │
│  │ - /bookings           │    │                      │         │
│  │ - /dashboard/stats    │    │                      │         │
│  │                       │    │                      │         │
│  │ Storage:              │    │ Storage:             │         │
│  │ FrontendStorage       │    │ ResourceManager      │         │
│  │ (In-memory)           │    │ (In-memory)          │         │
│  └───────────────────────┘    └──────────────────────┘         │
│                                                                 │
│  Common:                                                        │
│  - models.py (Pydantic data models)                             │
│  - CORS middleware                                              │
│  - Auto-generated OpenAPI docs (/docs)                          │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Frontend CRUD Operations

```
User Action (e.g., "Create Employee")
    │
    ▼
React Component (EmployeeList.tsx)
    │
    ▼
API Service (api.ts - createEmployee())
    │
    ▼
HTTP POST /api/employees
    │
    ▼
FastAPI Router (frontend_routes.py)
    │
    ▼
FrontendStorage (frontend_storage.py)
    │
    ▼
In-Memory Storage (Dict)
    │
    ▼
HTTP 200 Response (New Employee JSON)
    │
    ▼
React Component Updates State
    │
    ▼
UI Re-renders with New Data
```

## Component Details

### Frontend Layer

**Technology Stack:**
- React 18
- TypeScript
- Vite (build tool)
- TailwindCSS (styling)
- React Router (navigation)

**Key Components:**
- **DataTable** - Generic CRUD table with search, add, edit, delete
- **Dashboard** - Shows statistics and metrics
- **Resource Lists** - Dedicated pages for each resource type

**State Management:**
- React Context for authentication
- Component-level state for data
- useState/useEffect hooks

### Backend Layer - Frontend API

**Technology Stack:**
- FastAPI (web framework)
- Pydantic (data validation)
- Python 3.8+
- Uvicorn (ASGI server)

**Structure:**
```
app/
├── frontend_routes.py      # API endpoints (23 routes)
├── frontend_storage.py     # In-memory storage manager
├── models.py              # Data models (Pydantic)
└── main.py                # FastAPI app initialization
```

**Endpoints:**
- 5 resource types × 4 CRUD operations = 20 endpoints
- 1 dashboard stats endpoint
- 2 endpoints for GET by ID

### Backend Layer - Negotiation API

**Purpose:**
Original AI-powered multi-agent negotiation system

**Structure:**
```
app/
├── api_routes.py              # Negotiation API endpoints
├── resource_manager.py        # Resource pool management
├── negotiation_orchestrator.py # Negotiation coordination
├── department_chatbot.py      # AI agent chatbots
└── consensus_validator.py     # Agreement validation
```

**Features:**
- Multi-department resource negotiation
- AI-powered chatbot agents
- Consensus-based allocation
- Historical tracking

## Data Models

### Frontend Resources

```python
# Employee
Employee {
    id: str
    name: str
    email: str
    department: str
    position: str
    status: EmployeeStatus  # active | inactive | on-leave
    hireDate: str
}

# Equipment
Equipment {
    id: str
    name: str
    category: str
    status: EquipmentStatus  # available | in-use | maintenance | retired
    assignedTo: Optional[str]
    location: str
    serialNumber: str
}

# Inventory
InventoryItem {
    id: str
    name: str
    category: str
    quantity: int
    reorderLevel: int
    location: str
    unit: str
}

# Room
Room {
    id: str
    name: str
    capacity: int
    floor: str
    amenities: List[str]
    status: RoomStatus  # available | occupied | maintenance
}

# Booking
Booking {
    id: str
    resourceType: BookingResourceType  # room | equipment
    resourceId: str
    resourceName: str
    bookedBy: str
    startTime: str  # ISO format
    endTime: str    # ISO format
    purpose: str
    status: BookingStatus  # confirmed | pending | cancelled
}

# Dashboard
DashboardStats {
    totalEmployees: int
    activeEmployees: int
    totalEquipment: int
    availableEquipment: int
    lowStockItems: int
    totalRooms: int
    roomsInUse: int
    todayBookings: int
}
```

## Request/Response Flow

### Example: Create Employee

**1. Frontend Request**
```typescript
// api.ts
const response = await fetch(`${API_BASE_URL}/api/employees`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: "Jane Doe",
    email: "jane@company.com",
    department: "Sales",
    position: "Manager",
    status: "active",
    hireDate: "2024-01-15"
  })
});
```

**2. Backend Processing**
```python
# frontend_routes.py
@router.post("/api/employees")
async def create_employee(employee_data: EmployeeCreate):
    employee = Employee(
        id=str(uuid.uuid4()),
        **employee_data.dict()
    )
    created = storage.create_employee(employee)
    return created.dict()
```

**3. Backend Response**
```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "Jane Doe",
  "email": "jane@company.com",
  "department": "Sales",
  "position": "Manager",
  "status": "active",
  "hireDate": "2024-01-15"
}
```

**4. Frontend Updates**
```typescript
// Component state updates
setEmployees([...employees, newEmployee]);
```

## Storage Layer

### Current Implementation: In-Memory

```python
class FrontendStorage:
    def __init__(self):
        self.employees: Dict[str, Employee] = {}
        self.equipment: Dict[str, Equipment] = {}
        self.inventory: Dict[str, InventoryItem] = {}
        self.rooms: Dict[str, Room] = {}
        self.bookings: Dict[str, Booking] = {}
```

**Pros:**
- ✅ Fast
- ✅ Simple
- ✅ No setup required
- ✅ Perfect for development

**Cons:**
- ❌ Data lost on restart
- ❌ Not scalable
- ❌ No persistence

### Future Options

**SQLite** (Recommended for small-to-medium scale)
```python
# Easy migration path
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db():
    conn = sqlite3.connect('database.db')
    try:
        yield conn
    finally:
        conn.close()
```

**PostgreSQL/MySQL** (For production)
```python
# Using SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('postgresql://user:pass@localhost/dbname')
SessionLocal = sessionmaker(bind=engine)
```

## API Communication

### CORS Configuration

```python
# main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Development: allow all
    # allow_origins=["http://localhost:5173"],  # Production: specific
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Error Handling

```python
# 404 - Not Found
@router.get("/api/employees/{employee_id}")
async def get_employee(employee_id: str):
    employee = storage.get_employee(employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee.dict()

# 422 - Validation Error (automatic via Pydantic)
class EmployeeCreate(BaseModel):
    name: str  # Required - will return 422 if missing
    email: str
```

### Request Validation

FastAPI + Pydantic automatically validates:
- Required fields
- Data types
- Enum values
- Field constraints

```python
class EmployeeCreate(BaseModel):
    name: str                    # Must be string
    email: str                   # Must be string
    status: EmployeeStatus       # Must be valid enum value
    hireDate: str                # Must be string (could add date validation)
```

## Scalability Considerations

### Current Limitations
- In-memory storage (data not persisted)
- Single server instance
- No authentication/authorization
- No rate limiting
- No caching

### Production Enhancements

**1. Add Database**
```python
# Use SQLAlchemy or similar ORM
from sqlalchemy.ext.asyncio import AsyncSession

@router.get("/api/employees")
async def get_employees(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Employee))
    return result.scalars().all()
```

**2. Add Authentication**
```python
# Use existing API key system
from app.api_auth import verify_api_key

@router.post("/api/employees")
async def create_employee(
    employee_data: EmployeeCreate,
    credential: APICredential = Depends(verify_api_key)
):
    # Only authenticated users can create
```

**3. Add Caching**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_stats():
    return storage.get_dashboard_stats()
```

**4. Add Rate Limiting**
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@router.get("/api/employees")
@limiter.limit("100/minute")
async def get_employees():
    ...
```

## Deployment Architecture

### Development (Current)
```
Frontend (localhost:5173) ←→ Backend (localhost:8000)
```

### Production (Recommended)
```
                    ┌─────────────┐
User ──HTTPS──→     │  Nginx      │
                    │  Reverse    │
                    │  Proxy      │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         ┌────▼─────┐            ┌─────▼────┐
         │ Frontend │            │ Backend  │
         │ (Static) │            │ (FastAPI)│
         │ Files    │            │          │
         └──────────┘            └────┬─────┘
                                      │
                                 ┌────▼─────┐
                                 │ Database │
                                 │ (Postgres)│
                                 └──────────┘
```

## Summary

**Two Independent Systems:**
1. **Frontend CRUD API** (`/api/*`) - New, simple resource management
2. **Negotiation API** (`/api/v1/*`) - Original, complex AI negotiation

**Clean Separation:**
- Different route prefixes
- Different storage managers
- No shared state
- Can be used independently or together

**Easy to Extend:**
- Add new resource types (copy existing pattern)
- Swap storage backend (modify `FrontendStorage`)
- Add authentication (use existing `api_auth`)
- Add business logic (in route handlers)

**Production Ready:**
- Type-safe with Pydantic
- Auto-generated documentation
- CORS configured
- Error handling
- REST best practices
