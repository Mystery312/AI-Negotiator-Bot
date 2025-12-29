from app.models import (
    Department, DepartmentProfile, DepartmentStatus, ResourceType
)
from app.resource_manager import ResourceManager
from app.base_logic_engine import BaseLogicEngine
from app.department_chatbot import DepartmentChatbot
from app.negotiation_orchestrator import NegotiationOrchestrator

def example_basic_usage():
    resource_manager = ResourceManager()
    base_logic = BaseLogicEngine()
    orchestrator = NegotiationOrchestrator(resource_manager)
    
    pool = resource_manager.create_pool(
        pool_id="budget-q1-2024",
        resource_type=ResourceType.BUDGET,
        total_available=1000000.0,
        description="Q1 2024 Budget Allocation"
    )
    print(f"Created resource pool: {pool.pool_id}")
    
    engineering_profile = DepartmentProfile(
        department_id="engineering",
        department_name="Engineering",
        resource_priorities={"budget": 0.8, "personnel": 0.9},
        strategic_objectives=["Product development", "Infrastructure"],
        negotiation_style="collaborative"
    )
    
    marketing_profile = DepartmentProfile(
        department_id="marketing",
        department_name="Marketing",
        resource_priorities={"budget": 0.9, "personnel": 0.7},
        strategic_objectives=["Brand awareness", "Lead generation"],
        negotiation_style="aggressive"
    )
    
    engineering_dept = Department(
        department_id="engineering",
        name="Engineering",
        status=DepartmentStatus.ACTIVE,
        profile=engineering_profile
    )
    
    marketing_dept = Department(
        department_id="marketing",
        name="Marketing",
        status=DepartmentStatus.ACTIVE,
        profile=marketing_profile
    )
    
    eng_chatbot = DepartmentChatbot(engineering_dept, base_logic)
    mkt_chatbot = DepartmentChatbot(marketing_dept, base_logic)
    
    orchestrator.register_chatbot(eng_chatbot)
    orchestrator.register_chatbot(mkt_chatbot)
    
    session = orchestrator.create_session(
        participants=["engineering", "marketing"],
        resource_pool_id="budget-q1-2024",
        negotiation_type="budget"
    )
    print(f"Created negotiation session: {session.session_id}")
    
    orchestrator.start_session(session.session_id)
    print(f"Started negotiation. Status: {session.status}")
    
    session = orchestrator.get_session(session.session_id)
    print(f"Session has {len(session.messages)} messages")
    
    return session

if __name__ == "__main__":
    print("Resource Management System - Example Usage")
    print("=" * 50)
    session = example_basic_usage()
    print("\nExample completed successfully!")
