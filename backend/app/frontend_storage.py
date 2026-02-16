"""
Frontend Resource Storage Manager
Manages in-memory storage for Employee, Equipment, Inventory, Room, and Booking data
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, date
from app.models import (
    Employee, Equipment, InventoryItem, Room, Booking,
    EmployeeStatus, EquipmentStatus, RoomStatus, BookingStatus,
    DashboardStats
)

logger = logging.getLogger(__name__)


class FrontendStorage:
    """In-memory storage for frontend resources"""

    def __init__(self):
        self.employees: Dict[str, Employee] = {}
        self.equipment: Dict[str, Equipment] = {}
        self.inventory: Dict[str, InventoryItem] = {}
        self.rooms: Dict[str, Room] = {}
        self.bookings: Dict[str, Booking] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize with sample data
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize with some sample data for testing"""
        # Sample Employees
        sample_employees = [
            Employee(
                id="1",
                name="John Smith",
                email="john.smith@company.com",
                department="Engineering",
                position="Senior Developer",
                status=EmployeeStatus.ACTIVE,
                hireDate="2022-03-15"
            ),
            Employee(
                id="2",
                name="Sarah Johnson",
                email="sarah.j@company.com",
                department="Marketing",
                position="Marketing Manager",
                status=EmployeeStatus.ACTIVE,
                hireDate="2021-06-01"
            ),
            Employee(
                id="3",
                name="Mike Chen",
                email="mike.chen@company.com",
                department="Engineering",
                position="DevOps Engineer",
                status=EmployeeStatus.ACTIVE,
                hireDate="2023-01-10"
            ),
            Employee(
                id="4",
                name="Emily Davis",
                email="emily.d@company.com",
                department="HR",
                position="HR Specialist",
                status=EmployeeStatus.ON_LEAVE,
                hireDate="2020-09-20"
            )
        ]

        # Sample Equipment
        sample_equipment = [
            Equipment(
                id="1",
                name="Dell Laptop XPS 15",
                category="Computer",
                status=EquipmentStatus.IN_USE,
                assignedTo="John Smith",
                location="Floor 2, Desk 42",
                serialNumber="DL-2024-001"
            ),
            Equipment(
                id="2",
                name="MacBook Pro 16\"",
                category="Computer",
                status=EquipmentStatus.AVAILABLE,
                assignedTo=None,
                location="IT Storage Room",
                serialNumber="MB-2024-012"
            ),
            Equipment(
                id="3",
                name="4K Monitor",
                category="Display",
                status=EquipmentStatus.IN_USE,
                assignedTo="Sarah Johnson",
                location="Floor 3, Desk 18",
                serialNumber="MON-2024-045"
            ),
            Equipment(
                id="4",
                name="HP Printer Color LaserJet",
                category="Printer",
                status=EquipmentStatus.MAINTENANCE,
                assignedTo=None,
                location="Floor 2, Print Room",
                serialNumber="PR-2024-003"
            )
        ]

        # Sample Inventory
        sample_inventory = [
            InventoryItem(
                id="1",
                name="A4 Paper",
                category="Office Supplies",
                quantity=150,
                reorderLevel=50,
                location="Storage Room A",
                unit="reams"
            ),
            InventoryItem(
                id="2",
                name="USB-C Cables",
                category="Electronics",
                quantity=25,
                reorderLevel=20,
                location="IT Storage",
                unit="pieces"
            ),
            InventoryItem(
                id="3",
                name="Whiteboard Markers",
                category="Office Supplies",
                quantity=15,
                reorderLevel=30,
                location="Storage Room B",
                unit="boxes"
            ),
            InventoryItem(
                id="4",
                name="HDMI Adapters",
                category="Electronics",
                quantity=8,
                reorderLevel=15,
                location="IT Storage",
                unit="pieces"
            )
        ]

        # Sample Rooms
        sample_rooms = [
            Room(
                id="1",
                name="Conference Room A",
                capacity=10,
                floor="Floor 1",
                amenities=["Projector", "Whiteboard", "Video Conference"],
                status=RoomStatus.AVAILABLE
            ),
            Room(
                id="2",
                name="Meeting Room B",
                capacity=6,
                floor="Floor 2",
                amenities=["TV Screen", "Whiteboard"],
                status=RoomStatus.OCCUPIED
            ),
            Room(
                id="3",
                name="Board Room",
                capacity=20,
                floor="Floor 3",
                amenities=["Projector", "Video Conference", "Phone System"],
                status=RoomStatus.AVAILABLE
            ),
            Room(
                id="4",
                name="Focus Room 1",
                capacity=2,
                floor="Floor 2",
                amenities=["Monitor"],
                status=RoomStatus.AVAILABLE
            )
        ]

        # Sample Bookings
        today = datetime.now().strftime("%Y-%m-%d")
        sample_bookings = [
            Booking(
                id="1",
                resourceType="room",
                resourceId="1",
                resourceName="Conference Room A",
                bookedBy="John Smith",
                startTime=f"{today}T09:00",
                endTime=f"{today}T10:00",
                purpose="Team Standup",
                status=BookingStatus.CONFIRMED
            ),
            Booking(
                id="2",
                resourceType="room",
                resourceId="2",
                resourceName="Meeting Room B",
                bookedBy="Sarah Johnson",
                startTime=f"{today}T14:00",
                endTime=f"{today}T15:30",
                purpose="Client Presentation",
                status=BookingStatus.CONFIRMED
            ),
            Booking(
                id="3",
                resourceType="equipment",
                resourceId="2",
                resourceName="MacBook Pro 16\"",
                bookedBy="Mike Chen",
                startTime=f"{today}T10:00",
                endTime=f"{today}T17:00",
                purpose="Development Testing",
                status=BookingStatus.PENDING
            )
        ]

        # Add to storage
        for emp in sample_employees:
            self.employees[emp.id] = emp
        for eq in sample_equipment:
            self.equipment[eq.id] = eq
        for inv in sample_inventory:
            self.inventory[inv.id] = inv
        for room in sample_rooms:
            self.rooms[room.id] = room
        for booking in sample_bookings:
            self.bookings[booking.id] = booking

        self.logger.info("Initialized sample data successfully")

    # Employee CRUD operations
    def get_all_employees(self) -> List[Employee]:
        return list(self.employees.values())

    def get_employee(self, employee_id: str) -> Optional[Employee]:
        return self.employees.get(employee_id)

    def create_employee(self, employee: Employee) -> Employee:
        self.employees[employee.id] = employee
        self.logger.info(f"Created employee {employee.id}")
        return employee

    def update_employee(self, employee_id: str, employee_data: dict) -> Optional[Employee]:
        if employee_id not in self.employees:
            return None

        employee = self.employees[employee_id]
        for key, value in employee_data.items():
            if hasattr(employee, key):
                setattr(employee, key, value)

        self.logger.info(f"Updated employee {employee_id}")
        return employee

    def delete_employee(self, employee_id: str) -> bool:
        if employee_id in self.employees:
            del self.employees[employee_id]
            self.logger.info(f"Deleted employee {employee_id}")
            return True
        return False

    # Equipment CRUD operations
    def get_all_equipment(self) -> List[Equipment]:
        return list(self.equipment.values())

    def get_equipment(self, equipment_id: str) -> Optional[Equipment]:
        return self.equipment.get(equipment_id)

    def create_equipment(self, equipment: Equipment) -> Equipment:
        self.equipment[equipment.id] = equipment
        self.logger.info(f"Created equipment {equipment.id}")
        return equipment

    def update_equipment(self, equipment_id: str, equipment_data: dict) -> Optional[Equipment]:
        if equipment_id not in self.equipment:
            return None

        equipment = self.equipment[equipment_id]
        for key, value in equipment_data.items():
            if hasattr(equipment, key):
                setattr(equipment, key, value)

        self.logger.info(f"Updated equipment {equipment_id}")
        return equipment

    def delete_equipment(self, equipment_id: str) -> bool:
        if equipment_id in self.equipment:
            del self.equipment[equipment_id]
            self.logger.info(f"Deleted equipment {equipment_id}")
            return True
        return False

    # Inventory CRUD operations
    def get_all_inventory(self) -> List[InventoryItem]:
        return list(self.inventory.values())

    def get_inventory_item(self, item_id: str) -> Optional[InventoryItem]:
        return self.inventory.get(item_id)

    def create_inventory_item(self, item: InventoryItem) -> InventoryItem:
        self.inventory[item.id] = item
        self.logger.info(f"Created inventory item {item.id}")
        return item

    def update_inventory_item(self, item_id: str, item_data: dict) -> Optional[InventoryItem]:
        if item_id not in self.inventory:
            return None

        item = self.inventory[item_id]
        for key, value in item_data.items():
            if hasattr(item, key):
                setattr(item, key, value)

        self.logger.info(f"Updated inventory item {item_id}")
        return item

    def delete_inventory_item(self, item_id: str) -> bool:
        if item_id in self.inventory:
            del self.inventory[item_id]
            self.logger.info(f"Deleted inventory item {item_id}")
            return True
        return False

    # Room CRUD operations
    def get_all_rooms(self) -> List[Room]:
        return list(self.rooms.values())

    def get_room(self, room_id: str) -> Optional[Room]:
        return self.rooms.get(room_id)

    def create_room(self, room: Room) -> Room:
        self.rooms[room.id] = room
        self.logger.info(f"Created room {room.id}")
        return room

    def update_room(self, room_id: str, room_data: dict) -> Optional[Room]:
        if room_id not in self.rooms:
            return None

        room = self.rooms[room_id]
        for key, value in room_data.items():
            if hasattr(room, key):
                setattr(room, key, value)

        self.logger.info(f"Updated room {room_id}")
        return room

    def delete_room(self, room_id: str) -> bool:
        if room_id in self.rooms:
            del self.rooms[room_id]
            self.logger.info(f"Deleted room {room_id}")
            return True
        return False

    # Booking CRUD operations
    def get_all_bookings(self) -> List[Booking]:
        return list(self.bookings.values())

    def get_booking(self, booking_id: str) -> Optional[Booking]:
        return self.bookings.get(booking_id)

    def create_booking(self, booking: Booking) -> Booking:
        self.bookings[booking.id] = booking
        self.logger.info(f"Created booking {booking.id}")
        return booking

    def update_booking(self, booking_id: str, booking_data: dict) -> Optional[Booking]:
        if booking_id not in self.bookings:
            return None

        booking = self.bookings[booking_id]
        for key, value in booking_data.items():
            if hasattr(booking, key):
                setattr(booking, key, value)

        self.logger.info(f"Updated booking {booking_id}")
        return booking

    def delete_booking(self, booking_id: str) -> bool:
        if booking_id in self.bookings:
            del self.bookings[booking_id]
            self.logger.info(f"Deleted booking {booking_id}")
            return True
        return False

    # Dashboard statistics
    def get_dashboard_stats(self) -> DashboardStats:
        """Calculate and return dashboard statistics"""
        total_employees = len(self.employees)
        active_employees = sum(1 for emp in self.employees.values()
                              if emp.status == EmployeeStatus.ACTIVE)

        total_equipment = len(self.equipment)
        available_equipment = sum(1 for eq in self.equipment.values()
                                 if eq.status == EquipmentStatus.AVAILABLE)

        low_stock_items = sum(1 for item in self.inventory.values()
                             if item.quantity < item.reorderLevel)

        total_rooms = len(self.rooms)
        rooms_in_use = sum(1 for room in self.rooms.values()
                          if room.status == RoomStatus.OCCUPIED)

        # Count today's bookings
        today = date.today().isoformat()
        today_bookings = sum(1 for booking in self.bookings.values()
                            if booking.startTime.startswith(today) and
                            booking.status != BookingStatus.CANCELLED)

        return DashboardStats(
            totalEmployees=total_employees,
            activeEmployees=active_employees,
            totalEquipment=total_equipment,
            availableEquipment=available_equipment,
            lowStockItems=low_stock_items,
            totalRooms=total_rooms,
            roomsInUse=rooms_in_use,
            todayBookings=today_bookings
        )
