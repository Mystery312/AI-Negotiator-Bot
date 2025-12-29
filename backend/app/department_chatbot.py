import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from app.models import (
    Department, DepartmentProfile, NegotiationMessage, Proposal,
    ResourceRequest, MessageType, ChatbotState, ResourcePool
)
from app.base_logic_engine import BaseLogicEngine

logger = logging.getLogger(__name__)

class DepartmentChatbot:
    def __init__(self, department: Department, base_logic: BaseLogicEngine):
        self.department = department
        self.base_logic = base_logic
        self.logger = logging.getLogger(f"{__name__}.{department.department_id}")
        self.state = ChatbotState.IDLE

    def assess_needs(self, current_projects: List[Any]) -> ResourceRequest:
        self.logger.info(f"Assessing needs for {self.department.name}")
        self.state = ChatbotState.ASSESSING

        historical_data = {
            resource_type: {"average": amount}
            for resource_type, amount in self.department.profile.historical_usage.items()
        }

        request = self.base_logic.assess_needs(
            self.department.profile,
            current_projects,
            historical_data
        )

        self.state = ChatbotState.REQUESTING
        return request

    def generate_resource_request(
        self, resource_pool: ResourcePool
    ) -> NegotiationMessage:
        
        self.logger.info(f"Generating resource request for {self.department.name}")

        current_projects = self.department.profile.current_projects
        resource_request = self.assess_needs(current_projects)

        argument = self.base_logic.generate_argument(
            resource_request,
            self.department.profile,
            resource_pool.constraints
        )

        message = NegotiationMessage(
            message_id=str(uuid.uuid4()),
            sender=self.department.department_id,
            receiver="all",
            message_type=MessageType.RESOURCE_REQUEST,
            content={
                "resource_type": resource_request.resource_type,
                "quantity": resource_request.quantity,
                "priority": resource_request.priority,
                "min_acceptable": resource_request.min_acceptable
            },
            reasoning=argument,
            metadata={
                "request_id": resource_request.request_id,
                "justification": resource_request.justification
            }
        )

        return message

    def evaluate_proposal(
        self, proposal: Proposal, resource_pool: ResourcePool
    ) -> NegotiationMessage:
        
        self.logger.info(f"Evaluating proposal {proposal.proposal_id}")
        self.state = ChatbotState.EVALUATING

        current_projects = self.department.profile.current_projects
        own_needs = self.assess_needs(current_projects)

        decision, reasoning, counter_proposal = self.base_logic.evaluate_proposal(
            proposal,
            own_needs,
            resource_pool,
            self.department.profile
        )

        if decision == "accept":
            self.state = ChatbotState.ACCEPTING
            message_type = MessageType.ACCEPTANCE
            content = {"proposal_id": proposal.proposal_id}
        elif decision == "counter":
            self.state = ChatbotState.COUNTERING
            message_type = MessageType.COUNTER_PROPOSAL
            content = {
                "proposal_id": proposal.proposal_id,
                "counter_proposal": counter_proposal
            }
        else:
            self.state = ChatbotState.REJECTING
            message_type = MessageType.REJECTION
            content = {
                "proposal_id": proposal.proposal_id,
                "reason": reasoning
            }

        message = NegotiationMessage(
            message_id=str(uuid.uuid4()),
            sender=self.department.department_id,
            receiver=proposal.proposer,
            message_type=message_type,
            content=content,
            reasoning=reasoning,
            metadata={"decision": decision}
        )

        return message

    def generate_counter_proposal(
        self, original_proposal: Proposal, resource_pool: ResourcePool
    ) -> Proposal:
        
        self.logger.info(f"Generating counter-proposal for {self.department.name}")
        self.state = ChatbotState.COUNTERING

        current_projects = self.department.profile.current_projects
        own_needs = self.assess_needs(current_projects)

        counter_allocations = original_proposal.allocations.copy()

        own_allocation = counter_allocations.get(
            self.department.department_id, {}
        )
        own_allocation[own_needs.resource_type] = own_needs.quantity * 0.9
        counter_allocations[self.department.department_id] = own_allocation

        counter_proposal = Proposal(
            proposal_id=str(uuid.uuid4()),
            session_id=original_proposal.session_id,
            proposer=self.department.department_id,
            allocations=counter_allocations,
            reasoning=f"Counter-proposal from {self.department.name} based on assessed needs"
        )

        return counter_proposal

    def get_strategic_reasoning(
        self, negotiation_history: List[NegotiationMessage],
        other_departments: List[DepartmentProfile],
        resource_constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        return self.base_logic.strategic_reasoning(
            negotiation_history,
            other_departments,
            resource_constraints
        )

    def update_state(self, new_state: ChatbotState):
        self.state = new_state
        self.department.chatbot_state = new_state

    def get_state(self) -> ChatbotState:
        return self.state
