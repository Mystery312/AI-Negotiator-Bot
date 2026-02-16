import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from app.models import (
    NegotiationSession, NegotiationMessage, Proposal, NegotiationStatus,
    MessageType, ResourcePool
)
from app.department_chatbot import DepartmentChatbot
from app.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

class NegotiationOrchestrator:
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.sessions: Dict[str, NegotiationSession] = {}
        self.chatbots: Dict[str, DepartmentChatbot] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def register_chatbot(self, chatbot: DepartmentChatbot):
        self.chatbots[chatbot.department.department_id] = chatbot
        self.logger.info(f"Registered chatbot for {chatbot.department.name}")

    def create_session(
        self,
        participants: List[str],
        resource_pool_id: str,
        negotiation_type: str,
        deadline: Optional[datetime] = None,
        rules: Optional[Dict[str, Any]] = None
    ) -> NegotiationSession:
        
        session_id = str(uuid.uuid4())

        default_rules = {
            "max_rounds": 10,
            "time_per_turn": 300,
            "consensus_type": "unanimous"
        }
        if rules:
            default_rules.update(rules)

        session = NegotiationSession(
            session_id=session_id,
            participants=participants,
            resource_pool_id=resource_pool_id,
            negotiation_type=negotiation_type,
            deadline=deadline,
            rules=default_rules,
            status=NegotiationStatus.INITIALIZING
        )

        self.sessions[session_id] = session
        self.logger.info(f"Created negotiation session {session_id}")

        return session

    def start_session(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if not session:
            return False

        session.status = NegotiationStatus.ACTIVE
        session.current_phase = "request_submission"
        session.start_time = datetime.now()

        self._request_initial_needs(session)

        self.logger.info(f"Started negotiation session {session_id}")
        return True

    def process_message(
        self, session_id: str, message: NegotiationMessage
    ) -> Optional[NegotiationMessage]:
        
        session = self.sessions.get(session_id)
        if not session:
            return None

        session.messages.append(message)
        session.last_update = datetime.now()

        if message.message_type == MessageType.RESOURCE_REQUEST:
            return self._handle_resource_request(session, message)
        elif message.message_type == MessageType.PROPOSAL:
            return self._handle_proposal(session, message)
        elif message.message_type == MessageType.ACCEPTANCE:
            return self._handle_acceptance(session, message)
        elif message.message_type == MessageType.REJECTION:
            return self._handle_rejection(session, message)
        elif message.message_type == MessageType.COUNTER_PROPOSAL:
            return self._handle_counter_proposal(session, message)

        return None

    def advance_round(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if not session:
            return False

        if session.round >= session.rules.get("max_rounds", 10):
            session.status = NegotiationStatus.DEADLOCK
            return False

        session.round += 1
        self.logger.info(f"Advanced to round {session.round} for session {session_id}")

        if session.current_phase == "evaluation":
            self._generate_proposal(session)

        return True

    def check_consensus(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if not session or not session.proposals:
            return False

        latest_proposal = session.proposals[-1]
        consensus_type = session.rules.get("consensus_type", "unanimous")

        if consensus_type == "unanimous":

            responses = latest_proposal.responses
            all_accepted = all(
                resp == "accept" for resp in responses.values()
            ) and len(responses) == len(session.participants)

            if all_accepted:
                session.status = NegotiationStatus.CONSENSUS
                session.final_allocation = {
                    dept_id: sum(allocs.values())
                    for dept_id, allocs in latest_proposal.allocations.items()
                }
                session.end_time = datetime.now()
                return True

        elif consensus_type == "majority":

            responses = latest_proposal.responses
            acceptances = sum(1 for resp in responses.values() if resp == "accept")
            threshold = len(session.participants) * 2 / 3

            if acceptances >= threshold:
                session.status = NegotiationStatus.CONSENSUS
                session.final_allocation = {
                    dept_id: sum(allocs.values())
                    for dept_id, allocs in latest_proposal.allocations.items()
                }
                session.end_time = datetime.now()
                return True

        return False

    def get_session(self, session_id: str) -> Optional[NegotiationSession]:
        return self.sessions.get(session_id)

    def list_sessions(
        self,
        status: Optional[NegotiationStatus] = None,
        department_id: Optional[str] = None
    ) -> List[NegotiationSession]:
        
        sessions = list(self.sessions.values())

        if status:
            sessions = [s for s in sessions if s.status == status]

        if department_id:
            sessions = [
                s for s in sessions
                if department_id in s.participants
            ]

        return sessions

    def _request_initial_needs(self, session: NegotiationSession):
        resource_pool = self.resource_manager.get_pool(session.resource_pool_id)
        if not resource_pool:
            return

        for dept_id in session.participants:
            chatbot = self.chatbots.get(dept_id)
            if chatbot:
                request_message = chatbot.generate_resource_request(resource_pool)
                session.messages.append(request_message)

    def _handle_resource_request(
        self, session: NegotiationSession, message: NegotiationMessage
    ) -> Optional[NegotiationMessage]:
        

        if len(session.messages) >= len(session.participants):
            session.current_phase = "proposal_generation"
            self._generate_proposal(session)
        return None

    def _handle_proposal(
        self, session: NegotiationSession, message: NegotiationMessage
    ) -> Optional[NegotiationMessage]:
        
        session.current_phase = "evaluation"

        return None

    def _handle_acceptance(
        self, session: NegotiationSession, message: NegotiationMessage
    ) -> Optional[NegotiationMessage]:
        
        if session.proposals:
            latest_proposal = session.proposals[-1]
            latest_proposal.responses[message.sender] = "accept"

            if self.check_consensus(session.session_id):
                self._finalize_allocation(session)
        return None

    def _handle_rejection(
        self, session: NegotiationSession, message: NegotiationMessage
    ) -> Optional[NegotiationMessage]:
        
        if session.proposals:
            latest_proposal = session.proposals[-1]
            latest_proposal.responses[message.sender] = "reject"

            self.advance_round(session.session_id)
        return None

    def _handle_counter_proposal(
        self, session: NegotiationSession, message: NegotiationMessage
    ) -> Optional[NegotiationMessage]:
        

        counter_data = message.content.get("counter_proposal", {})
        if counter_data:
            new_proposal = Proposal(
                proposal_id=str(uuid.uuid4()),
                session_id=session.session_id,
                proposer=message.sender,
                allocations=counter_data,
                reasoning=message.reasoning
            )
            session.proposals.append(new_proposal)
            session.current_phase = "evaluation"
        return None

    def _generate_proposal(self, session: NegotiationSession):
        resource_pool = self.resource_manager.get_pool(session.resource_pool_id)
        if not resource_pool:
            return

        requests = {}
        for message in session.messages:
            if message.message_type == MessageType.RESOURCE_REQUEST:
                dept_id = message.sender
                content = message.content
                requests[dept_id] = {
                    content.get("resource_type", "budget"): content.get("quantity", 0)
                }

        total_requested = sum(
            sum(dept_reqs.values()) for dept_reqs in requests.values()
        )

        if total_requested > resource_pool.total_available:

            scale = resource_pool.total_available / total_requested
        else:
            scale = 1.0

        allocations = {}
        for dept_id, dept_reqs in requests.items():
            allocations[dept_id] = {
                resource_type: amount * scale
                for resource_type, amount in dept_reqs.items()
            }

        proposal = Proposal(
            proposal_id=str(uuid.uuid4()),
            session_id=session.session_id,
            proposer="system",
            allocations=allocations,
            reasoning="Initial proposal based on proportional distribution"
        )

        session.proposals.append(proposal)
        session.current_phase = "evaluation"

    def _finalize_allocation(self, session: NegotiationSession):
        if session.final_allocation:
            self.resource_manager.finalize_allocation(
                session.resource_pool_id,
                session.final_allocation
            )
            session.status = NegotiationStatus.COMPLETED
            self.logger.info(
                f"Finalized allocation for session {session.session_id}"
            )
