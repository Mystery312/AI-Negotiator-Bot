import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from app.models import (
    Department, DepartmentProfile, ResourcePool, NegotiationSession,
    NegotiationStatus, ResourceType, APICredential
)
from app.api_auth import verify_api_key, require_role
from app.resource_manager import ResourceManager
from app.negotiation_orchestrator import NegotiationOrchestrator
from app.department_chatbot import DepartmentChatbot
from app.base_logic_engine import BaseLogicEngine

logger = logging.getLogger(__name__)

resource_manager = ResourceManager()
base_logic = BaseLogicEngine()
negotiation_orchestrator = NegotiationOrchestrator(resource_manager)

router = APIRouter(prefix="/api/v1", tags=["Resource Management"])

class DepartmentCreate(BaseModel):
    name: str
    resource_priorities: Dict[str, float] = {}
    constraints: Dict[str, Any] = {}
    strategic_objectives: List[str] = []

class ResourcePoolCreate(BaseModel):
    resource_type: str
    total_available: float
    constraints: Dict[str, Any] = {}
    description: Optional[str] = None

class NegotiationCreate(BaseModel):
    participants: List[str]
    resource_pool_id: str
    negotiation_type: str
    deadline: Optional[datetime] = None
    rules: Dict[str, Any] = {}

class InterventionRequest(BaseModel):
    action: str
    proposal: Optional[Dict[str, Any]] = None

class APIKeyRegister(BaseModel):
    organization: str
    email: str
    role: str = "user"

@router.post("/auth/register")
async def register_api_key(
    request: APIKeyRegister
):
    
    from app.api_auth import create_api_credential
    credential = create_api_credential(
        request.organization,
        request.email,
        request.role
    )
    return {
        "api_key": credential.api_key,
        "expires_at": credential.expires_at.isoformat() if credential.expires_at else None,
        "role": credential.role
    }

@router.post("/auth/validate")
async def validate_api_key_endpoint(
    credential: APICredential = Depends(verify_api_key)
):
    return {
        "valid": True,
        "role": credential.role,
        "expires_at": credential.expires_at.isoformat() if credential.expires_at else None
    }

@router.get("/departments")
async def list_departments(
    status: Optional[str] = Query(None),
    credential: APICredential = Depends(verify_api_key)
):
    

    departments = []
    if status:
        departments = [d for d in departments if d.status == status]
    return {"departments": [d.dict() for d in departments]}

@router.get("/departments/{department_id}")
async def get_department(
    department_id: str,
    credential: APICredential = Depends(verify_api_key)
):
    

    raise HTTPException(status_code=404, detail="Department not found")

@router.post("/departments")
async def create_department(
    dept_data: DepartmentCreate,
    credential: APICredential = Depends(require_role("admin"))
):
    
    import uuid
    from app.models import Department, DepartmentStatus

    department = Department(
        department_id=str(uuid.uuid4()),
        name=dept_data.name,
        status=DepartmentStatus.ACTIVE,
        profile=DepartmentProfile(
            department_id=str(uuid.uuid4()),
            department_name=dept_data.name,
            resource_priorities=dept_data.resource_priorities,
            constraints=dept_data.constraints,
            strategic_objectives=dept_data.strategic_objectives
        )
    )

    chatbot = DepartmentChatbot(department, base_logic)
    negotiation_orchestrator.register_chatbot(chatbot)

    return department.dict()

@router.get("/resource-pools")
async def list_resource_pools(
    credential: APICredential = Depends(verify_api_key)
):
    
    pools = resource_manager.list_pools()
    return {
        "pools": [
            {
                "pool_id": p.pool_id,
                "resource_type": p.resource_type,
                "total_available": p.total_available,
                "allocated": p.allocated,
                "available": p.available
            }
            for p in pools
        ]
    }

@router.post("/resource-pools")
async def create_resource_pool(
    pool_data: ResourcePoolCreate,
    credential: APICredential = Depends(require_role("admin"))
):
    
    import uuid

    pool = resource_manager.create_pool(
        pool_id=str(uuid.uuid4()),
        resource_type=ResourceType(pool_data.resource_type),
        total_available=pool_data.total_available,
        constraints=pool_data.constraints,
        description=pool_data.description
    )

    return pool.dict()

@router.post("/negotiations")
async def create_negotiation(
    neg_data: NegotiationCreate,
    credential: APICredential = Depends(verify_api_key)
):
    
    session = negotiation_orchestrator.create_session(
        participants=neg_data.participants,
        resource_pool_id=neg_data.resource_pool_id,
        negotiation_type=neg_data.negotiation_type,
        deadline=neg_data.deadline,
        rules=neg_data.rules
    )

    return {
        "session_id": session.session_id,
        "status": session.status,
        "created_at": session.start_time.isoformat(),
        "participants": session.participants
    }

@router.get("/negotiations")
async def list_negotiations(
    status: Optional[str] = Query(None),
    department_id: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    credential: APICredential = Depends(verify_api_key)
):
    
    status_enum = NegotiationStatus(status) if status else None
    sessions = negotiation_orchestrator.list_sessions(
        status=status_enum,
        department_id=department_id
    )

    total = len(sessions)
    sessions = sessions[offset:offset + limit]

    return {
        "negotiations": [s.dict() for s in sessions],
        "total": total,
        "limit": limit,
        "offset": offset
    }

@router.get("/negotiations/{session_id}")
async def get_negotiation(
    session_id: str,
    credential: APICredential = Depends(verify_api_key)
):
    
    session = negotiation_orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.dict()

@router.get("/negotiations/{session_id}/messages")
async def get_negotiation_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    credential: APICredential = Depends(verify_api_key)
):
    
    session = negotiation_orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = session.messages[offset:offset + limit]
    return {
        "messages": [m.dict() for m in messages],
        "total": len(session.messages),
        "limit": limit,
        "offset": offset
    }

@router.post("/negotiations/{session_id}/intervene")
async def intervene_in_negotiation(
    session_id: str,
    intervention: InterventionRequest,
    credential: APICredential = Depends(require_role("admin"))
):
    
    session = negotiation_orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if intervention.action == "force_consensus":

        pass
    elif intervention.action == "propose" and intervention.proposal:

        pass

    return {"message": "Intervention processed", "session_id": session_id}

@router.post("/negotiations/{session_id}/cancel")
async def cancel_negotiation(
    session_id: str,
    credential: APICredential = Depends(verify_api_key)
):
    
    session = negotiation_orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.status = NegotiationStatus.CANCELLED
    session.end_time = datetime.now()

    return {"message": "Negotiation cancelled"}
