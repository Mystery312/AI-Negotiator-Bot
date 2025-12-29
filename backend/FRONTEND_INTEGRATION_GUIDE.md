# Frontend Integration Guide

## Overview

The Resource Management Program backend has been successfully extended to support the React frontend application. This guide explains how to connect the frontend to the backend and use all available API endpoints.

## What Was Added

### New Files Created

1. **`app/frontend_storage.py`** - In-memory storage manager for frontend resources
   - Manages Employees, Equipment, Inventory, Rooms, and Bookings
   - Includes sample data for testing
   - Provides CRUD operations for all resource types

2. **`app/frontend_routes.py`** - REST API endpoints for frontend
   - Complete CRUD operations for all resource types
   - Dashboard statistics endpoint
   - Fully compatible with the frontend's TypeScript interfaces

### Updated Files

1. **`app/models.py`** - Added new data models
   - `Employee`, `Equipment`, `InventoryItem`, `Room`, `Booking`
   - `DashboardStats` for analytics
   - Status enums for each resource type

2. **`app/main.py`** - Integrated new routes
   - Added frontend_router to the FastAPI app
   - Both original negotiation API and new frontend API are available

## API Endpoints

All endpoints use the `/api` prefix to match the frontend expectations.

### Dashboard
- **GET** `/api/dashboard/stats` - Get dashboard statistics

### Employees
- **GET** `/api/employees` - Get all employees
- **GET** `/api/employees/{id}` - Get specific employee
- **POST** `/api/employees` - Create new employee
- **PUT** `/api/employees/{id}` - Update employee
- **DELETE** `/api/employees/{id}` - Delete employee

### Equipment
- **GET** `/api/equipment` - Get all equipment
- **GET** `/api/equipment/{id}` - Get specific equipment
- **POST** `/api/equipment` - Create new equipment
- **PUT** `/api/equipment/{id}` - Update equipment
- **DELETE** `/api/equipment/{id}` - Delete equipment

### Inventory
- **GET** `/api/inventory` - Get all inventory items
- **GET** `/api/inventory/{id}` - Get specific inventory item
- **POST** `/api/inventory` - Create new inventory item
- **PUT** `/api/inventory/{id}` - Update inventory item
- **DELETE** `/api/inventory/{id}` - Delete inventory item

### Rooms
- **GET** `/api/rooms` - Get all rooms
- **GET** `/api/rooms/{id}` - Get specific room
- **POST** `/api/rooms` - Create new room
- **PUT** `/api/rooms/{id}` - Update room
- **DELETE** `/api/rooms/{id}` - Delete room

### Bookings
- **GET** `/api/bookings` - Get all bookings
- **GET** `/api/bookings/{id}` - Get specific booking
- **POST** `/api/bookings` - Create new booking
- **PUT** `/api/bookings/{id}` - Update booking
- **DELETE** `/api/bookings/{id}` - Delete booking

## Starting the Backend Server

### Option 1: Using Python directly
```bash
cd "Resource Management Program"
python -m app.main
```

The server will start on `http://localhost:8000`

### Option 2: Using uvicorn directly
```bash
cd "Resource Management Program"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-reload during development.

## Connecting the Frontend

### Update Frontend Environment Variables

Create or update `.env` in your React frontend project:

```env
VITE_API_URL=http://localhost:8000
```

### Update API Service (if needed)

The frontend's `src/services/api.ts` should use this base URL:

```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

Then replace all placeholder API calls with actual fetch calls:

```typescript
// Example: Get all employees
export async function getEmployees(): Promise<Employee[]> {
  const response = await fetch(`${API_BASE_URL}/api/employees`);
  if (!response.ok) throw new Error('Failed to fetch employees');
  return response.json();
}

// Example: Create employee
export async function createEmployee(employee: Omit<Employee, 'id'>): Promise<Employee> {
  const response = await fetch(`${API_BASE_URL}/api/employees`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(employee)
  });
  if (!response.ok) throw new Error('Failed to create employee');
  return response.json();
}

// Example: Update employee
export async function updateEmployee(id: string, employee: Partial<Employee>): Promise<Employee> {
  const response = await fetch(`${API_BASE_URL}/api/employees/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(employee)
  });
  if (!response.ok) throw new Error('Failed to update employee');
  return response.json();
}

// Example: Delete employee
export async function deleteEmployee(id: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/employees/${id}`, {
    method: 'DELETE'
  });
  if (!response.ok) throw new Error('Failed to delete employee');
}
```

Repeat this pattern for all resource types (equipment, inventory, rooms, bookings).

## Running Both Frontend and Backend

### Terminal 1 - Backend
```bash
cd "Resource Management Program"
python -m app.main
```

### Terminal 2 - Frontend
```bash
cd "Resource Management Program"  # or your frontend directory
npm run dev
```

The frontend will typically run on `http://localhost:5173` (Vite default).

## Sample Data

The backend includes sample data for testing:

- **4 Employees** - Including active, inactive, and on-leave statuses
- **4 Equipment Items** - Computers, monitors, and printers with various statuses
- **4 Inventory Items** - Office supplies and electronics (some below reorder level)
- **4 Rooms** - Conference rooms and focus rooms with different capacities
- **3 Bookings** - Sample room and equipment bookings for today

## Testing the API

### Using curl

```bash
# Get dashboard stats
curl http://localhost:8000/api/dashboard/stats

# Get all employees
curl http://localhost:8000/api/employees

# Create a new employee
curl -X POST http://localhost:8000/api/employees \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Jane Doe",
    "email": "jane@company.com",
    "department": "Sales",
    "position": "Sales Manager",
    "status": "active",
    "hireDate": "2024-01-15"
  }'

# Update an employee
curl -X PUT http://localhost:8000/api/employees/1 \
  -H "Content-Type: application/json" \
  -d '{"position": "Senior Developer"}'

# Delete an employee
curl -X DELETE http://localhost:8000/api/employees/1
```

### Using Browser

Navigate to `http://localhost:8000/docs` to access the auto-generated FastAPI Swagger UI documentation where you can test all endpoints interactively.

## Data Models

### Employee
```typescript
{
  id: string;
  name: string;
  email: string;
  department: string;
  position: string;
  status: 'active' | 'inactive' | 'on-leave';
  hireDate: string;  // Format: 'YYYY-MM-DD'
}
```

### Equipment
```typescript
{
  id: string;
  name: string;
  category: string;
  status: 'available' | 'in-use' | 'maintenance' | 'retired';
  assignedTo: string | null;
  location: string;
  serialNumber: string;
}
```

### InventoryItem
```typescript
{
  id: string;
  name: string;
  category: string;
  quantity: number;
  reorderLevel: number;
  location: string;
  unit: string;
}
```

### Room
```typescript
{
  id: string;
  name: string;
  capacity: number;
  floor: string;
  amenities: string[];
  status: 'available' | 'occupied' | 'maintenance';
}
```

### Booking
```typescript
{
  id: string;
  resourceType: 'room' | 'equipment';
  resourceId: string;
  resourceName: string;
  bookedBy: string;
  startTime: string;  // ISO format
  endTime: string;    // ISO format
  purpose: string;
  status: 'confirmed' | 'pending' | 'cancelled';
}
```

### DashboardStats
```typescript
{
  totalEmployees: number;
  activeEmployees: number;
  totalEquipment: number;
  availableEquipment: number;
  lowStockItems: number;
  totalRooms: number;
  roomsInUse: number;
  todayBookings: number;
}
```

## CORS Configuration

The backend is already configured to accept requests from any origin:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, you should restrict this to your specific frontend domain:

```python
allow_origins=["http://localhost:5173", "https://your-production-domain.com"]
```

## Architecture

The backend maintains two separate systems:

1. **Original Negotiation System** (`/api/v1/*`)
   - Department management
   - Resource pool management
   - Negotiation orchestration
   - Multi-agent chatbot system

2. **Frontend CRUD System** (`/api/*`)
   - Employee, Equipment, Inventory, Room, Booking management
   - Dashboard statistics
   - Simple in-memory storage

Both systems run independently and can be used simultaneously.

## Storage

Currently using **in-memory storage** via the `FrontendStorage` class. This means:
- Data is lost when the server restarts
- Perfect for development and testing
- Fast and simple

### Future Enhancements

To persist data, you can:

1. Add SQLite database support
2. Use PostgreSQL or MySQL
3. Integrate with the existing resource pool system
4. Add file-based JSON storage

The storage layer is cleanly separated in `frontend_storage.py`, making it easy to swap implementations.

## Troubleshooting

### CORS Errors
If you see CORS errors in the browser console, ensure:
- The backend is running on port 8000
- The frontend is configured with the correct API URL
- The CORS middleware is properly configured

### Port Already in Use
If port 8000 is already in use:
```bash
# Find the process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use a different port
uvicorn app.main:app --port 8001
```

### Module Import Errors
Make sure you're running from the correct directory:
```bash
cd "Resource Management Program"
python -m app.main
```

## Next Steps

1. Start the backend server
2. Update your frontend's API service layer
3. Test the connection with the dashboard stats endpoint
4. Implement the remaining API calls for all resources
5. Test CRUD operations from the frontend UI
6. Customize the sample data or storage implementation as needed

## Support

For the original negotiation system features, refer to:
- `RESOURCE_MANAGEMENT_SYSTEM_PLAN.md`
- `DEMO_GUIDE.md`
- `README.md`

For frontend-specific issues, check:
- `FRONTEND_INTEGRATION_GUIDE.md` (this file)
- FastAPI docs at `http://localhost:8000/docs`
