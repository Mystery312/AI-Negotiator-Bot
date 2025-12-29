"""
Frontend API Routes for React Resource Management System
Provides REST API endpoints for Employee, Equipment, Inventory, Room, and Booking management
"""
import logging
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models import (
    Employee, Equipment, InventoryItem, Room, Booking,
    EmployeeStatus, EquipmentStatus, RoomStatus, BookingStatus,
    BookingResourceType
)
from app.frontend_storage import FrontendStorage

logger = logging.getLogger(__name__)

# Initialize storage
storage = FrontendStorage()

# Create router with /api prefix to match frontend expectations
router = APIRouter(prefix="/api", tags=["Frontend Resources"])


# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================

@router.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    stats = storage.get_dashboard_stats()
    return stats.dict()


# ============================================================================
# EMPLOYEE ENDPOINTS
# ============================================================================

class EmployeeCreate(BaseModel):
    name: str
    email: str
    department: str
    position: str
    status: EmployeeStatus
    hireDate: str


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    status: Optional[EmployeeStatus] = None
    hireDate: Optional[str] = None


@router.get("/employees")
async def get_employees():
    """Get all employees"""
    employees = storage.get_all_employees()
    return [emp.dict() for emp in employees]


@router.get("/employees/{employee_id}")
async def get_employee(employee_id: str):
    """Get a specific employee"""
    employee = storage.get_employee(employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee.dict()


@router.post("/employees")
async def create_employee(employee_data: EmployeeCreate):
    """Create a new employee"""
    employee = Employee(
        id=str(uuid.uuid4()),
        **employee_data.dict()
    )
    created = storage.create_employee(employee)
    return created.dict()


@router.put("/employees/{employee_id}")
async def update_employee(employee_id: str, employee_data: EmployeeUpdate):
    """Update an existing employee"""
    # Filter out None values
    update_data = {k: v for k, v in employee_data.dict().items() if v is not None}

    updated = storage.update_employee(employee_id, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Employee not found")

    return updated.dict()


@router.delete("/employees/{employee_id}")
async def delete_employee(employee_id: str):
    """Delete an employee"""
    success = storage.delete_employee(employee_id)
    if not success:
        raise HTTPException(status_code=404, detail="Employee not found")

    return {"message": "Employee deleted successfully"}


# ============================================================================
# EQUIPMENT ENDPOINTS
# ============================================================================

class EquipmentCreate(BaseModel):
    name: str
    category: str
    status: EquipmentStatus
    assignedTo: Optional[str] = None
    location: str
    serialNumber: str


class EquipmentUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    status: Optional[EquipmentStatus] = None
    assignedTo: Optional[str] = None
    location: Optional[str] = None
    serialNumber: Optional[str] = None


@router.get("/equipment")
async def get_equipment():
    """Get all equipment"""
    equipment_list = storage.get_all_equipment()
    return [eq.dict() for eq in equipment_list]


@router.get("/equipment/{equipment_id}")
async def get_equipment_item(equipment_id: str):
    """Get a specific equipment item"""
    equipment = storage.get_equipment(equipment_id)
    if not equipment:
        raise HTTPException(status_code=404, detail="Equipment not found")
    return equipment.dict()


@router.post("/equipment")
async def create_equipment(equipment_data: EquipmentCreate):
    """Create a new equipment item"""
    equipment = Equipment(
        id=str(uuid.uuid4()),
        **equipment_data.dict()
    )
    created = storage.create_equipment(equipment)
    return created.dict()


@router.put("/equipment/{equipment_id}")
async def update_equipment(equipment_id: str, equipment_data: EquipmentUpdate):
    """Update an existing equipment item"""
    update_data = {k: v for k, v in equipment_data.dict().items() if v is not None}

    updated = storage.update_equipment(equipment_id, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Equipment not found")

    return updated.dict()


@router.delete("/equipment/{equipment_id}")
async def delete_equipment(equipment_id: str):
    """Delete an equipment item"""
    success = storage.delete_equipment(equipment_id)
    if not success:
        raise HTTPException(status_code=404, detail="Equipment not found")

    return {"message": "Equipment deleted successfully"}


# ============================================================================
# INVENTORY ENDPOINTS
# ============================================================================

class InventoryItemCreate(BaseModel):
    name: str
    category: str
    quantity: int
    reorderLevel: int
    location: str
    unit: str


class InventoryItemUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    quantity: Optional[int] = None
    reorderLevel: Optional[int] = None
    location: Optional[str] = None
    unit: Optional[str] = None


@router.get("/inventory")
async def get_inventory():
    """Get all inventory items"""
    items = storage.get_all_inventory()
    return [item.dict() for item in items]


@router.get("/inventory/{item_id}")
async def get_inventory_item(item_id: str):
    """Get a specific inventory item"""
    item = storage.get_inventory_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Inventory item not found")
    return item.dict()


@router.post("/inventory")
async def create_inventory_item(item_data: InventoryItemCreate):
    """Create a new inventory item"""
    item = InventoryItem(
        id=str(uuid.uuid4()),
        **item_data.dict()
    )
    created = storage.create_inventory_item(item)
    return created.dict()


@router.put("/inventory/{item_id}")
async def update_inventory_item(item_id: str, item_data: InventoryItemUpdate):
    """Update an existing inventory item"""
    update_data = {k: v for k, v in item_data.dict().items() if v is not None}

    updated = storage.update_inventory_item(item_id, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Inventory item not found")

    return updated.dict()


@router.delete("/inventory/{item_id}")
async def delete_inventory_item(item_id: str):
    """Delete an inventory item"""
    success = storage.delete_inventory_item(item_id)
    if not success:
        raise HTTPException(status_code=404, detail="Inventory item not found")

    return {"message": "Inventory item deleted successfully"}


# ============================================================================
# ROOM ENDPOINTS
# ============================================================================

class RoomCreate(BaseModel):
    name: str
    capacity: int
    floor: str
    amenities: list[str]
    status: RoomStatus


class RoomUpdate(BaseModel):
    name: Optional[str] = None
    capacity: Optional[int] = None
    floor: Optional[str] = None
    amenities: Optional[list[str]] = None
    status: Optional[RoomStatus] = None


@router.get("/rooms")
async def get_rooms():
    """Get all rooms"""
    rooms = storage.get_all_rooms()
    return [room.dict() for room in rooms]


@router.get("/rooms/{room_id}")
async def get_room(room_id: str):
    """Get a specific room"""
    room = storage.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room.dict()


@router.post("/rooms")
async def create_room(room_data: RoomCreate):
    """Create a new room"""
    room = Room(
        id=str(uuid.uuid4()),
        **room_data.dict()
    )
    created = storage.create_room(room)
    return created.dict()


@router.put("/rooms/{room_id}")
async def update_room(room_id: str, room_data: RoomUpdate):
    """Update an existing room"""
    update_data = {k: v for k, v in room_data.dict().items() if v is not None}

    updated = storage.update_room(room_id, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Room not found")

    return updated.dict()


@router.delete("/rooms/{room_id}")
async def delete_room(room_id: str):
    """Delete a room"""
    success = storage.delete_room(room_id)
    if not success:
        raise HTTPException(status_code=404, detail="Room not found")

    return {"message": "Room deleted successfully"}


# ============================================================================
# BOOKING ENDPOINTS
# ============================================================================

class BookingCreate(BaseModel):
    resourceType: BookingResourceType
    resourceId: str
    resourceName: str
    bookedBy: str
    startTime: str
    endTime: str
    purpose: str
    status: BookingStatus


class BookingUpdate(BaseModel):
    resourceType: Optional[BookingResourceType] = None
    resourceId: Optional[str] = None
    resourceName: Optional[str] = None
    bookedBy: Optional[str] = None
    startTime: Optional[str] = None
    endTime: Optional[str] = None
    purpose: Optional[str] = None
    status: Optional[BookingStatus] = None


@router.get("/bookings")
async def get_bookings():
    """Get all bookings"""
    bookings = storage.get_all_bookings()
    return [booking.dict() for booking in bookings]


@router.get("/bookings/{booking_id}")
async def get_booking(booking_id: str):
    """Get a specific booking"""
    booking = storage.get_booking(booking_id)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    return booking.dict()


@router.post("/bookings")
async def create_booking(booking_data: BookingCreate):
    """Create a new booking"""
    booking = Booking(
        id=str(uuid.uuid4()),
        **booking_data.dict()
    )
    created = storage.create_booking(booking)
    return created.dict()


@router.put("/bookings/{booking_id}")
async def update_booking(booking_id: str, booking_data: BookingUpdate):
    """Update an existing booking"""
    update_data = {k: v for k, v in booking_data.dict().items() if v is not None}

    updated = storage.update_booking(booking_id, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Booking not found")

    return updated.dict()


@router.delete("/bookings/{booking_id}")
async def delete_booking(booking_id: str):
    """Delete a booking"""
    success = storage.delete_booking(booking_id)
    if not success:
        raise HTTPException(status_code=404, detail="Booking not found")

    return {"message": "Booking deleted successfully"}
