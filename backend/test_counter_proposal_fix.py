#!/usr/bin/env python3
"""
Test to verify that counter-proposal generation doesn't mutate original proposal.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.base_logic_engine import BaseLogicEngine
from app.models import Proposal, ResourceRequest, DepartmentProfile, ResourcePool

def test_counter_proposal_deep_copy():
    """Test that generating a counter-proposal doesn't mutate the original proposal."""

    # Create a base logic engine
    engine = BaseLogicEngine()

    # Create a sample proposal with nested allocations
    original_proposal = Proposal(
        proposal_id="test-proposal-1",
        session_id="test-session-1",
        proposer="system",
        allocations={
            "dept-1": {"budget": 1000.0, "headcount": 5.0},
            "dept-2": {"budget": 2000.0, "headcount": 10.0}
        },
        reasoning="Initial allocation"
    )

    # Store original values for comparison
    original_dept1_budget = original_proposal.allocations["dept-1"]["budget"]
    original_dept1_headcount = original_proposal.allocations["dept-1"]["headcount"]
    original_dept2_budget = original_proposal.allocations["dept-2"]["budget"]

    # Create a resource request for dept-1
    needs = ResourceRequest(
        request_id="req-1",
        department_id="dept-1",
        resource_type="budget",
        quantity=1500.0,
        justification="Need more budget",
        priority=8
    )

    # Generate counter-proposal
    counter_allocations = engine._generate_counter_proposal(original_proposal, needs)

    # Verify the counter-proposal has the expected value (90% of requested)
    expected_counter_value = 1500.0 * 0.9  # 1350.0
    assert counter_allocations["dept-1"]["budget"] == expected_counter_value, \
        f"Counter-proposal should have {expected_counter_value}, got {counter_allocations['dept-1']['budget']}"

    # CRITICAL: Verify original proposal was NOT mutated
    assert original_proposal.allocations["dept-1"]["budget"] == original_dept1_budget, \
        f"Original proposal dept-1 budget was mutated! Expected {original_dept1_budget}, got {original_proposal.allocations['dept-1']['budget']}"

    assert original_proposal.allocations["dept-1"]["headcount"] == original_dept1_headcount, \
        f"Original proposal dept-1 headcount was mutated! Expected {original_dept1_headcount}, got {original_proposal.allocations['dept-1']['headcount']}"

    assert original_proposal.allocations["dept-2"]["budget"] == original_dept2_budget, \
        f"Original proposal dept-2 budget was mutated! Expected {original_dept2_budget}, got {original_proposal.allocations['dept-2']['budget']}"

    print("✅ Test passed: Original proposal was not mutated")
    print(f"   Original dept-1 budget: {original_proposal.allocations['dept-1']['budget']}")
    print(f"   Counter-proposal dept-1 budget: {counter_allocations['dept-1']['budget']}")

    return True

if __name__ == "__main__":
    try:
        success = test_counter_proposal_deep_copy()
        if success:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
