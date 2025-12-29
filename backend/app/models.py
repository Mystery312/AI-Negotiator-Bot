from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class ResourceType(str, Enum):
    BUDGET = "budget"
    PERSONNEL = "personnel"
    EQUIPMENT = "equipment"
    TIME = "time"

class NegotiationStatus(str, Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    CONSENSUS = "consensus"
    DEADLOCK = "deadlock"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class MessageType(str, Enum):
    RESOURCE_REQUEST = "ResourceRequest"
    PROPOSAL = "Proposal"
    COUNTER_PROPOSAL = "CounterProposal"
    ARGUMENT = "Argument"
    ACCEPTANCE = "Acceptance"
    REJECTION = "Rejection"
    QUESTION = "Question"
    CONCESSION = "Concession"

class DepartmentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

class ChatbotState(str, Enum):
    IDLE = "IDLE"
    ASSESSING = "ASSESSING"
    REQUESTING = "REQUESTING"
    EVALUATING = "EVALUATING"
    NEGOTIATING = "NEGOTIATING"
    COUNTERING = "COUNTERING"
    ACCEPTING = "ACCEPTING"
    REJECTING = "REJECTING"
    FINALIZING = "FINALIZING"

class Project(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    resource_requirements: Dict[str, float] = Field(default_factory=dict)
    priority: int = 0

class DepartmentProfile(BaseModel):
    department_id: str
    department_name: str
    resource_priorities: Dict[str, float] = Field(default_factory=dict)
    current_projects: List[Project] = Field(default_factory=list)
    historical_usage: Dict[str, float] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    strategic_objectives: List[str] = Field(default_factory=list)
    negotiation_style: str = "collaborative"
    min_acceptable_allocation: Dict[str, float] = Field(default_factory=dict)

class NegotiationMessage(BaseModel):
    message_id: str
    sender: str
    receiver: str | List[str]
    message_type: MessageType
    timestamp: datetime = Field(default_factory=datetime.now)
    content: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ResourceRequest(BaseModel):
    request_id: str
    department_id: str
    resource_type: ResourceType
    quantity: float
    justification: str
    priority: int = 0
    min_acceptable: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class Proposal(BaseModel):
    proposal_id: str
    session_id: str
    proposer: str
    allocations: Dict[str, Dict[str, float]]
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    responses: Dict[str, str] = Field(default_factory=dict)

class ResourcePool(BaseModel):
    pool_id: str
    resource_type: ResourceType
    total_available: float
    allocated: Dict[str, float] = Field(default_factory=dict)
    pending_requests: List[ResourceRequest] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def available(self) -> float:
        allocated_total = sum(self.allocated.values())
        return max(0, self.total_available - allocated_total)

class NegotiationSession(BaseModel):
    session_id: str
    participants: List[str]
    resource_pool_id: str
    negotiation_type: str
    status: NegotiationStatus = NegotiationStatus.INITIALIZING
    current_phase: str = "initialization"
    round: int = 0
    messages: List[NegotiationMessage] = Field(default_factory=list)
    proposals: List[Proposal] = Field(default_factory=list)
    final_allocation: Optional[Dict[str, float]] = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    deadline: Optional[datetime] = None
    rules: Dict[str, Any] = Field(default_factory=dict)
    last_update: datetime = Field(default_factory=datetime.now)

class Department(BaseModel):
    department_id: str
    name: str
    status: DepartmentStatus = DepartmentStatus.ACTIVE
    profile: DepartmentProfile
    current_state: Dict[str, Any] = Field(default_factory=dict)
    negotiation_history: List[NegotiationMessage] = Field(default_factory=list)
    chatbot_state: ChatbotState = ChatbotState.IDLE
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class APICredential(BaseModel):
    api_key: str
    organization: str
    email: str
    role: str
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_active: bool = True

# Frontend Resource Models (for the React app integration)
class EmployeeStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ON_LEAVE = "on-leave"

class Employee(BaseModel):
    id: str
    name: str
    email: str
    department: str
    position: str
    status: EmployeeStatus = EmployeeStatus.ACTIVE
    hireDate: str  # Format: 'YYYY-MM-DD'

class EquipmentStatus(str, Enum):
    AVAILABLE = "available"
    IN_USE = "in-use"
    MAINTENANCE = "maintenance"
    RETIRED = "retired"

class Equipment(BaseModel):
    id: str
    name: str
    category: str
    status: EquipmentStatus = EquipmentStatus.AVAILABLE
    assignedTo: Optional[str] = None  # Employee name or null
    location: str
    serialNumber: str

class InventoryItem(BaseModel):
    id: str
    name: str
    category: str
    quantity: int
    reorderLevel: int
    location: str
    unit: str

class RoomStatus(str, Enum):
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    MAINTENANCE = "maintenance"

class Room(BaseModel):
    id: str
    name: str
    capacity: int
    floor: str
    amenities: List[str] = Field(default_factory=list)
    status: RoomStatus = RoomStatus.AVAILABLE

class BookingStatus(str, Enum):
    CONFIRMED = "confirmed"
    PENDING = "pending"
    CANCELLED = "cancelled"

class BookingResourceType(str, Enum):
    ROOM = "room"
    EQUIPMENT = "equipment"

class Booking(BaseModel):
    id: str
    resourceType: BookingResourceType
    resourceId: str
    resourceName: str
    bookedBy: str
    startTime: str  # ISO format: '2024-01-15T09:00'
    endTime: str    # ISO format: '2024-01-15T10:00'
    purpose: str
    status: BookingStatus = BookingStatus.CONFIRMED

class DashboardStats(BaseModel):
    totalEmployees: int
    activeEmployees: int
    totalEquipment: int
    availableEquipment: int
    lowStockItems: int
    totalRooms: int
    roomsInUse: int
    todayBookings: int
