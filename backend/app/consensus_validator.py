import logging
from typing import Dict, List, Optional, Tuple
from app.models import NegotiationSession, Proposal, ResourcePool
from app.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

class ConsensusValidator:
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate_proposal(
        self,
        proposal: Proposal,
        resource_pool: ResourcePool
    ) -> Tuple[bool, Optional[str]]:
        

        total_allocated = sum(
            sum(dept_allocs.values())
            for dept_allocs in proposal.allocations.values()
        )

        if total_allocated > resource_pool.total_available:
            return (
                False,
                f"Total allocation {total_allocated} exceeds "
                f"available {resource_pool.total_available}"
            )

        for dept_id, dept_allocs in proposal.allocations.items():
            dept_total = sum(dept_allocs.values())
            if dept_total < 0:
                return (False, f"Negative allocation for {dept_id}")

        for dept_id, dept_allocs in proposal.allocations.items():
            for resource_type, amount in dept_allocs.items():
                if not self._check_allocation_constraints(
                    resource_pool, dept_id, amount
                ):
                    return (
                        False,
                        f"Allocation constraint violation for {dept_id}"
                    )

        return (True, None)

    def validate_consensus(
        self,
        session: NegotiationSession
    ) -> Tuple[bool, Optional[str]]:
        
        if not session.proposals:
            return (False, "No proposals in session")

        latest_proposal = session.proposals[-1]
        resource_pool = self.resource_manager.get_pool(session.resource_pool_id)

        if not resource_pool:
            return (False, "Resource pool not found")

        is_valid, error = self.validate_proposal(latest_proposal, resource_pool)
        if not is_valid:
            return (False, error)

        responses = latest_proposal.responses
        if len(responses) != len(session.participants):
            return (
                False,
                f"Not all participants responded. "
                f"Expected {len(session.participants)}, got {len(responses)}"
            )

        consensus_type = session.rules.get("consensus_type", "unanimous")

        if consensus_type == "unanimous":
            all_accepted = all(
                resp == "accept" for resp in responses.values()
            )
            if not all_accepted:
                return (False, "Not all participants accepted (unanimous required)")

        elif consensus_type == "majority":
            acceptances = sum(1 for resp in responses.values() if resp == "accept")
            threshold = len(session.participants) * 2 / 3
            if acceptances < threshold:
                return (
                    False,
                    f"Not enough acceptances. "
                    f"Got {acceptances}, need {threshold}"
                )

        return (True, None)

    def _check_allocation_constraints(
        self,
        resource_pool: ResourcePool,
        department_id: str,
        amount: float
    ) -> bool:
        
        constraints = resource_pool.constraints

        if "min_allocation" in constraints:
            if amount < constraints["min_allocation"]:
                return False

        if "max_allocation" in constraints:
            if amount > constraints["max_allocation"]:
                return False

        dept_constraints = constraints.get("department_constraints", {})
        if department_id in dept_constraints:
            dept_max = dept_constraints[department_id].get("max", float('inf'))
            if amount > dept_max:
                return False

        return True
