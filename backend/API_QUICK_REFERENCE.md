# API Quick Reference

## Base URL
```
http://localhost:8000/api
```

## Quick Start
```bash
# Start backend
./start_backend.sh

# Test connection
curl http://localhost:8000/api/dashboard/stats
```

## Frontend Environment Variable
```env
VITE_API_URL=http://localhost:8000
```

## Endpoints Cheat Sheet

### Dashboard
```bash
GET    /api/dashboard/stats         # Get all statistics
```

### Employees
```bash
GET    /api/employees                # List all
GET    /api/employees/{id}           # Get one
POST   /api/employees                # Create
PUT    /api/employees/{id}           # Update
DELETE /api/employees/{id}           # Delete
```

### Equipment
```bash
GET    /api/equipment                # List all
GET    /api/equipment/{id}           # Get one
POST   /api/equipment                # Create
PUT    /api/equipment/{id}           # Update
DELETE /api/equipment/{id}           # Delete
```

### Inventory
```bash
GET    /api/inventory                # List all
GET    /api/inventory/{id}           # Get one
POST   /api/inventory                # Create
PUT    /api/inventory/{id}           # Update
DELETE /api/inventory/{id}           # Delete
```

### Rooms
```bash
GET    /api/rooms                    # List all
GET    /api/rooms/{id}               # Get one
POST   /api/rooms                    # Create
PUT    /api/rooms/{id}               # Update
DELETE /api/rooms/{id}               # Delete
```

### Bookings
```bash
GET    /api/bookings                 # List all
GET    /api/bookings/{id}            # Get one
POST   /api/bookings                 # Create
PUT    /api/bookings/{id}            # Update
DELETE /api/bookings/{id}            # Delete
```

## Sample Requests

### Create Employee
```bash
curl -X POST http://localhost:8000/api/employees \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@company.com",
    "department": "Engineering",
    "position": "Developer",
    "status": "active",
    "hireDate": "2024-01-01"
  }'
```

### Update Employee
```bash
curl -X PUT http://localhost:8000/api/employees/1 \
  -H "Content-Type: application/json" \
  -d '{"position": "Senior Developer"}'
```

### Create Equipment
```bash
curl -X POST http://localhost:8000/api/equipment \
  -H "Content-Type: application/json" \
  -d '{
    "name": "MacBook Pro",
    "category": "Computer",
    "status": "available",
    "assignedTo": null,
    "location": "IT Storage",
    "serialNumber": "MB-001"
  }'
```

### Create Room
```bash
curl -X POST http://localhost:8000/api/rooms \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Meeting Room A",
    "capacity": 8,
    "floor": "Floor 1",
    "amenities": ["Projector", "Whiteboard"],
    "status": "available"
  }'
```

### Create Booking
```bash
curl -X POST http://localhost:8000/api/bookings \
  -H "Content-Type: application/json" \
  -d '{
    "resourceType": "room",
    "resourceId": "1",
    "resourceName": "Conference Room A",
    "bookedBy": "John Smith",
    "startTime": "2024-01-15T09:00",
    "endTime": "2024-01-15T10:00",
    "purpose": "Team Meeting",
    "status": "confirmed"
  }'
```

## TypeScript Integration

### API Base Setup
```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

### Generic GET
```typescript
export async function getEmployees(): Promise<Employee[]> {
  const response = await fetch(`${API_BASE_URL}/api/employees`);
  if (!response.ok) throw new Error('Failed to fetch');
  return response.json();
}
```

### Generic POST
```typescript
export async function createEmployee(data: Omit<Employee, 'id'>): Promise<Employee> {
  const response = await fetch(`${API_BASE_URL}/api/employees`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  if (!response.ok) throw new Error('Failed to create');
  return response.json();
}
```

### Generic PUT
```typescript
export async function updateEmployee(id: string, data: Partial<Employee>): Promise<Employee> {
  const response = await fetch(`${API_BASE_URL}/api/employees/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  if (!response.ok) throw new Error('Failed to update');
  return response.json();
}
```

### Generic DELETE
```typescript
export async function deleteEmployee(id: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/employees/${id}`, {
    method: 'DELETE'
  });
  if (!response.ok) throw new Error('Failed to delete');
}
```

## Response Formats

### Dashboard Stats
```json
{
  "totalEmployees": 4,
  "activeEmployees": 3,
  "totalEquipment": 4,
  "availableEquipment": 1,
  "lowStockItems": 2,
  "totalRooms": 4,
  "roomsInUse": 1,
  "todayBookings": 3
}
```

### Employee
```json
{
  "id": "1",
  "name": "John Smith",
  "email": "john@company.com",
  "department": "Engineering",
  "position": "Developer",
  "status": "active",
  "hireDate": "2022-03-15"
}
```

### Equipment
```json
{
  "id": "1",
  "name": "Dell Laptop XPS 15",
  "category": "Computer",
  "status": "in-use",
  "assignedTo": "John Smith",
  "location": "Floor 2, Desk 42",
  "serialNumber": "DL-2024-001"
}
```

### Room
```json
{
  "id": "1",
  "name": "Conference Room A",
  "capacity": 10,
  "floor": "Floor 1",
  "amenities": ["Projector", "Whiteboard"],
  "status": "available"
}
```

### Booking
```json
{
  "id": "1",
  "resourceType": "room",
  "resourceId": "1",
  "resourceName": "Conference Room A",
  "bookedBy": "John Smith",
  "startTime": "2024-01-15T09:00",
  "endTime": "2024-01-15T10:00",
  "purpose": "Team Meeting",
  "status": "confirmed"
}
```

## Status Values

### Employee Status
- `active`
- `inactive`
- `on-leave`

### Equipment Status
- `available`
- `in-use`
- `maintenance`
- `retired`

### Room Status
- `available`
- `occupied`
- `maintenance`

### Booking Status
- `confirmed`
- `pending`
- `cancelled`

### Booking Resource Type
- `room`
- `equipment`

## Error Responses

### 404 Not Found
```json
{
  "detail": "Employee not found"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

## Interactive Documentation

Visit `http://localhost:8000/docs` when the server is running to access the auto-generated Swagger UI with:
- Complete API reference
- Request/response schemas
- Interactive "Try it out" feature
- Model definitions

## Common Tasks

### Check if server is running
```bash
curl http://localhost:8000/health
# Response: {"status": "ok"}
```

### Get all resources
```bash
curl http://localhost:8000/api/employees
curl http://localhost:8000/api/equipment
curl http://localhost:8000/api/inventory
curl http://localhost:8000/api/rooms
curl http://localhost:8000/api/bookings
```

### Test dashboard integration
```bash
curl http://localhost:8000/api/dashboard/stats
```

## Port Configuration

Default: `8000`

To use a different port:
```bash
uvicorn app.main:app --port 8001
```

Update frontend `.env`:
```env
VITE_API_URL=http://localhost:8001
```
