import logging
import copy
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from app.models import (
    ResourceRequest, Proposal, NegotiationMessage,
    MessageType, DepartmentProfile, ResourcePool
)

logger = logging.getLogger(__name__)

class BaseLogicEngine:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def assess_needs(
        self,
        department_profile: DepartmentProfile,
        current_projects: List[Any],
        historical_data: Dict[str, Any]
    ) -> ResourceRequest:
        
        self.logger.info(f"Assessing needs for {department_profile.department_name}")

        current_usage = self._analyze_current_utilization(
            department_profile, current_projects
        )

        upcoming_needs = self._identify_upcoming_requirements(
            department_profile, current_projects
        )

        gaps = self._calculate_gaps(current_usage, upcoming_needs, historical_data)

        prioritized_needs = self._prioritize_needs(
            gaps, department_profile.resource_priorities
        )

        justification = self._generate_justification(
            prioritized_needs, department_profile, current_projects
        )

        request = ResourceRequest(
            request_id=f"req_{datetime.now().timestamp()}",
            department_id=department_profile.department_id,
            resource_type="budget",
            quantity=prioritized_needs.get("total", 0),
            justification=justification,
            priority=self._calculate_priority(prioritized_needs, department_profile)
        )

        return request

    def generate_argument(
        self,
        resource_request: ResourceRequest,
        department_profile: DepartmentProfile,
        constraints: Dict[str, Any]
    ) -> str:
        
        self.logger.info(f"Generating argument for {resource_request.department_id}")

        justifications = {
            "business_impact": self._assess_business_impact(
                resource_request, department_profile
            ),
            "strategic_alignment": self._check_strategic_alignment(
                resource_request, department_profile
            ),
            "urgency": self._assess_urgency(resource_request, department_profile),
            "historical_precedent": self._find_historical_precedent(
                resource_request, department_profile
            ),
            "dependencies": self._identify_dependencies(
                resource_request, department_profile
            )
        }

        argument = self._structure_argument(justifications, resource_request)

        return argument

    def evaluate_proposal(
        self,
        proposal: Proposal,
        own_needs: ResourceRequest,
        resource_pool: ResourcePool,
        department_profile: DepartmentProfile
    ) -> Tuple[str, Optional[str], Optional[Dict[str, float]]]:
        
        self.logger.info(f"Evaluating proposal {proposal.proposal_id}")

        impact = self._assess_impact(proposal, own_needs, department_profile)

        fairness_score = self._evaluate_fairness(proposal, resource_pool)

        tradeoffs = self._identify_tradeoffs(proposal, own_needs)

        threshold = self._determine_acceptance_threshold(
            own_needs, department_profile
        )

        if impact >= threshold and fairness_score >= 0.7:
            return ("accept", "Proposal meets needs and is fair", None)
        elif impact >= threshold * 0.8:
            return ("counter", "Close but needs adjustment",
                   self._generate_counter_proposal(proposal, own_needs))
        else:
            return ("reject", "Proposal does not meet minimum requirements", None)

    def strategic_reasoning(
        self,
        negotiation_history: List[NegotiationMessage],
        other_departments: List[DepartmentProfile],
        resource_constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        self.logger.info("Performing strategic reasoning")

        other_priorities = self._model_other_priorities(
            negotiation_history, other_departments
        )

        predictions = self._predict_responses(
            negotiation_history, other_priorities
        )

        win_win = self._identify_win_win_opportunities(
            negotiation_history, other_departments, resource_constraints
        )

        concession_strategy = self._calculate_concession_strategy(
            negotiation_history, other_priorities, resource_constraints
        )

        next_move = self._determine_next_move(
            negotiation_history, predictions, win_win, concession_strategy
        )

        return {
            "other_priorities": other_priorities,
            "predictions": predictions,
            "win_win_opportunities": win_win,
            "concession_strategy": concession_strategy,
            "recommended_move": next_move
        }

    def _analyze_current_utilization(
        self, profile: DepartmentProfile, projects: List[Any]
    ) -> Dict[str, float]:
        
        utilization = {}
        for project in projects:
            if hasattr(project, 'resource_requirements'):
                for resource_type, amount in project.resource_requirements.items():
                    utilization[resource_type] = utilization.get(resource_type, 0) + amount
        return utilization

    def _identify_upcoming_requirements(
        self, profile: DepartmentProfile, projects: List[Any]
    ) -> Dict[str, float]:
        
        requirements = {}
        for project in projects:
            if hasattr(project, 'resource_requirements'):
                for resource_type, amount in project.resource_requirements.items():
                    requirements[resource_type] = requirements.get(resource_type, 0) + amount
        return requirements

    def _calculate_gaps(
        self, current: Dict[str, float], upcoming: Dict[str, float],
        historical: Dict[str, Any]
    ) -> Dict[str, float]:
        
        gaps = {}
        all_types = set(current.keys()) | set(upcoming.keys())
        for resource_type in all_types:
            current_val = current.get(resource_type, 0)
            upcoming_val = upcoming.get(resource_type, 0)
            historical_avg = historical.get(resource_type, {}).get("average", 0)
            gaps[resource_type] = max(0, upcoming_val - current_val, historical_avg - current_val)
        return gaps

    def _prioritize_needs(
        self, gaps: Dict[str, float], priorities: Dict[str, float]
    ) -> Dict[str, Any]:
        
        prioritized = {}
        total = 0
        for resource_type, gap in gaps.items():
            priority_weight = priorities.get(resource_type, 1.0)
            weighted_gap = gap * priority_weight
            prioritized[resource_type] = {
                "gap": gap,
                "priority": priority_weight,
                "weighted": weighted_gap
            }
            total += weighted_gap
        prioritized["total"] = total
        return prioritized

    def _generate_justification(
        self, needs: Dict[str, Any], profile: DepartmentProfile, projects: List[Any]
    ) -> str:
        
        justification_parts = [
            f"{profile.department_name} requires resources to support",
            f"{len(projects)} active projects."
        ]

        if profile.strategic_objectives:
            justification_parts.append(
                f"These resources align with strategic objectives: "
                f"{', '.join(profile.strategic_objectives[:3])}"
            )

        return " ".join(justification_parts)

    def _calculate_priority(
        self, needs: Dict[str, Any], profile: DepartmentProfile
    ) -> int:
        
        base_priority = int(needs.get("total", 0) / 1000)
        if profile.strategic_objectives:
            base_priority += len(profile.strategic_objectives)
        return min(10, max(1, base_priority))

    def _assess_business_impact(
        self, request: ResourceRequest, profile: DepartmentProfile
    ) -> float:
        

        return len(profile.strategic_objectives) * 0.2

    def _check_strategic_alignment(
        self, request: ResourceRequest, profile: DepartmentProfile
    ) -> bool:
        
        return len(profile.strategic_objectives) > 0

    def _assess_urgency(
        self, request: ResourceRequest, profile: DepartmentProfile
    ) -> str:
        
        return "high" if request.priority >= 7 else "medium" if request.priority >= 4 else "low"

    def _find_historical_precedent(
        self, request: ResourceRequest, profile: DepartmentProfile
    ) -> Optional[str]:
        
        if profile.historical_usage:
            return "Similar allocations have been made in the past"
        return None

    def _identify_dependencies(
        self, request: ResourceRequest, profile: DepartmentProfile
    ) -> List[str]:
        
        return profile.dependencies

    def _structure_argument(
        self, justifications: Dict[str, Any], request: ResourceRequest
    ) -> str:
        
        parts = [f"Request for {request.quantity} units of {request.resource_type}"]

        if justifications.get("business_impact"):
            parts.append(f"Business impact: {justifications['business_impact']:.2f}")

        if justifications.get("strategic_alignment"):
            parts.append("Aligned with strategic objectives")

        if justifications.get("urgency"):
            parts.append(f"Urgency: {justifications['urgency']}")

        return ". ".join(parts) + "."

    def _assess_impact(
        self, proposal: Proposal, needs: ResourceRequest, profile: DepartmentProfile
    ) -> float:
        
        own_allocation = proposal.allocations.get(needs.department_id, {})
        allocated = sum(own_allocation.values())
        requested = needs.quantity
        if requested > 0:
            return allocated / requested
        return 0.0

    def _evaluate_fairness(
        self, proposal: Proposal, resource_pool: ResourcePool
    ) -> float:
        
        total_allocated = sum(
            sum(dept_allocs.values())
            for dept_allocs in proposal.allocations.values()
        )
        if resource_pool.total_available > 0:
            utilization = total_allocated / resource_pool.total_available

            num_depts = len(proposal.allocations)
            if num_depts > 0:
                avg_per_dept = total_allocated / num_depts
                variance = sum(
                    (sum(dept_allocs.values()) - avg_per_dept) ** 2
                    for dept_allocs in proposal.allocations.values()
                ) / num_depts
                fairness = 1.0 / (1.0 + variance / (avg_per_dept ** 2 + 1))
                return min(1.0, fairness)
        return 0.5

    def _identify_tradeoffs(
        self, proposal: Proposal, needs: ResourceRequest
    ) -> List[str]:
        
        tradeoffs = []
        own_allocation = proposal.allocations.get(needs.department_id, {})
        allocated = sum(own_allocation.values())
        if allocated < needs.quantity:
            tradeoffs.append(f"Receiving {allocated} instead of requested {needs.quantity}")
        return tradeoffs

    def _determine_acceptance_threshold(
        self, needs: ResourceRequest, profile: DepartmentProfile
    ) -> float:
        
        min_acceptable = needs.min_acceptable or (needs.quantity * 0.7)
        return min_acceptable / needs.quantity if needs.quantity > 0 else 0.0

    def _generate_counter_proposal(
        self, proposal: Proposal, needs: ResourceRequest
    ) -> Dict[str, float]:
        """Generate a counter-proposal with modified allocations.

        Uses deep copy to avoid mutating the original proposal's allocations.
        """
        # Deep copy to prevent modification of original proposal
        counter = copy.deepcopy(proposal.allocations)

        # Get or create allocation dict for this department
        own_allocation = counter.get(needs.department_id, {})

        # Modify the allocation (90% of requested amount)
        own_allocation[needs.resource_type] = needs.quantity * 0.9
        counter[needs.department_id] = own_allocation

        return counter

    def _model_other_priorities(
        self, history: List[NegotiationMessage], others: List[DepartmentProfile]
    ) -> Dict[str, Dict[str, float]]:
        
        priorities = {}
        for dept in others:
            priorities[dept.department_id] = dept.resource_priorities
        return priorities

    def _predict_responses(
        self, history: List[NegotiationMessage], priorities: Dict[str, Dict[str, float]]
    ) -> Dict[str, str]:
        
        predictions = {}
        for dept_id in priorities.keys():
            predictions[dept_id] = "likely_accept"
        return predictions

    def _identify_win_win_opportunities(
        self, history: List[NegotiationMessage], others: List[DepartmentProfile],
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        
        return []

    def _calculate_concession_strategy(
        self, history: List[NegotiationMessage], priorities: Dict[str, Dict[str, float]],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        return {
            "concession_rate": 0.1,
            "max_concessions": 3,
            "hold_firm_on": []
        }

    def _determine_next_move(
        self, history: List[NegotiationMessage], predictions: Dict[str, str],
        win_win: List[Dict[str, Any]], strategy: Dict[str, Any]
    ) -> str:
        
        if len(history) == 0:
            return "request"
        elif len(history) < 3:
            return "propose"
        else:
            return "counter"
