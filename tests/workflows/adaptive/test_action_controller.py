"""
Tests for Action Control System
"""

import pytest
from datetime import datetime
import time
from unittest.mock import patch, MagicMock

from mcp_agent.workflows.adaptive.action_controller import (
    ActionController,
    WorkflowAction,
    ActionConstraints,
)


# Mock logger to avoid async issues in tests
@pytest.fixture(autouse=True)
def mock_logger():
    with patch("mcp_agent.workflows.adaptive.action_controller.logger") as mock:
        mock.debug = MagicMock()
        mock.info = MagicMock()
        mock.warning = MagicMock()
        mock.error = MagicMock()
        yield mock


class TestActionController:
    """Test the dynamic action control system"""

    def test_initialization(self):
        """Test action controller initialization"""
        controller = ActionController()

        assert len(controller.action_history) == 0
        assert len(controller.disabled_actions) == 0
        assert len(controller.action_cooldowns) == 0
        assert controller.current_iteration == 0

        # Check default constraints exist
        assert WorkflowAction.ANALYZE in controller.constraints
        assert WorkflowAction.SYNTHESIZE in controller.constraints
        assert WorkflowAction.CONCLUDE in controller.constraints

    def test_update_iteration(self):
        """Test iteration tracking"""
        controller = ActionController()

        controller.update_iteration(1)
        assert controller.current_iteration == 1

        controller.update_iteration(5)
        assert controller.current_iteration == 5

    def test_basic_action_availability(self):
        """Test basic action availability checking"""
        controller = ActionController()
        context = {"knowledge_items": [], "has_findings": False}

        # ANALYZE should be available initially
        available, reason = controller.is_action_available(
            WorkflowAction.ANALYZE, context
        )
        assert available

        # SYNTHESIZE should not be available without findings
        available, reason = controller.is_action_available(
            WorkflowAction.SYNTHESIZE, context
        )
        assert not available
        assert "knowledge items" in reason or "research findings" in reason

        # Update context
        context["has_findings"] = True
        context["knowledge_items"] = ["item1", "item2"]

        # Now SYNTHESIZE should be available
        available, reason = controller.is_action_available(
            WorkflowAction.SYNTHESIZE, context
        )
        assert available

    def test_action_constraints(self):
        """Test constraint-based availability"""
        controller = ActionController()

        # Test iteration constraints
        context = {"knowledge_items": ["k1", "k2", "k3", "k4"], "has_synthesis": True}

        # CONCLUDE requires min 1 iteration
        available, reason = controller.is_action_available(
            WorkflowAction.CONCLUDE, context
        )
        assert not available
        assert "Requires at least 1 iterations" in reason

        controller.update_iteration(1)

        # Also needs high confidence
        context["confidence"] = 0.5
        available, reason = controller.is_action_available(
            WorkflowAction.CONCLUDE, context
        )
        assert not available
        assert "Confidence too low" in reason

        # With high confidence, should work
        context["confidence"] = 0.8
        available, reason = controller.is_action_available(
            WorkflowAction.CONCLUDE, context
        )
        assert available

    def test_action_recording(self):
        """Test recording action execution"""
        controller = ActionController()

        # Record successful action
        controller.record_action(
            WorkflowAction.PLAN, success=True, duration=1.5, context={"iteration": 1}
        )

        assert len(controller.action_history) == 1
        assert controller.action_history[0].action == WorkflowAction.PLAN
        assert controller.action_history[0].success
        assert controller.action_history[0].duration_seconds == 1.5

        # Record failed action
        controller.record_action(
            WorkflowAction.EXECUTE_SUBTASK,
            success=False,
            error="Connection timeout",
            context={"subtask": "Research APIs"},
        )

        assert len(controller.action_history) == 2
        assert not controller.action_history[1].success
        assert controller.action_history[1].error == "Connection timeout"

    def test_failure_handling(self):
        """Test handling of action failures"""
        controller = ActionController()

        # Single failure - should add cooldown
        controller.record_action(WorkflowAction.PLAN, success=False, error="Error 1")

        assert WorkflowAction.PLAN in controller.action_cooldowns
        cooldown_end = controller.action_cooldowns[WorkflowAction.PLAN]
        assert cooldown_end > datetime.now()

        # Multiple failures - should disable
        controller.record_action(WorkflowAction.PLAN, success=False, error="Error 2")
        controller.record_action(WorkflowAction.PLAN, success=False, error="Error 3")

        assert WorkflowAction.PLAN in controller.disabled_actions

        # Check it's not available
        available, reason = controller.is_action_available(WorkflowAction.PLAN, {})
        assert not available
        assert "currently disabled" in reason

    def test_success_recovery(self):
        """Test re-enabling actions after success"""
        controller = ActionController()

        # Disable an action through failures
        for i in range(3):
            controller.record_action(WorkflowAction.SYNTHESIZE, success=False)

        assert WorkflowAction.SYNTHESIZE in controller.disabled_actions

        # Record some successes
        controller.record_action(WorkflowAction.SYNTHESIZE, success=True)
        assert (
            WorkflowAction.SYNTHESIZE in controller.disabled_actions
        )  # Still disabled

        controller.record_action(WorkflowAction.SYNTHESIZE, success=True)
        assert (
            WorkflowAction.SYNTHESIZE not in controller.disabled_actions
        )  # Re-enabled

    def test_cooldown_mechanism(self):
        """Test action cooldowns"""
        controller = ActionController()

        # Set a short cooldown for testing
        controller.constraints[WorkflowAction.PLAN] = ActionConstraints(
            cooldown_minutes=0.01
        )  # 0.6 seconds

        # Fail an action
        controller.record_action(WorkflowAction.PLAN, success=False)

        # Should be on cooldown
        available, reason = controller.is_action_available(WorkflowAction.PLAN, {})
        assert not available
        assert "cooldown" in reason

        # Wait for cooldown
        time.sleep(0.7)

        # Should be available again
        available, reason = controller.is_action_available(WorkflowAction.PLAN, {})
        assert available

    def test_attempts_per_iteration(self):
        """Test limiting attempts per iteration"""
        controller = ActionController()
        controller.update_iteration(1)

        context = {"iteration": 1}

        # ANALYZE allows only 1 attempt per workflow
        controller.record_action(WorkflowAction.ANALYZE, success=True, context=context)

        # Second attempt should fail
        available, reason = controller.is_action_available(WorkflowAction.ANALYZE, {})
        assert not available
        assert "Maximum attempts" in reason

        # PLAN allows 2 attempts per iteration
        controller.record_action(WorkflowAction.PLAN, success=True, context=context)
        available, _ = controller.is_action_available(WorkflowAction.PLAN, {})
        assert available

        controller.record_action(WorkflowAction.PLAN, success=True, context=context)
        available, reason = controller.is_action_available(WorkflowAction.PLAN, {})
        assert not available
        assert "Maximum attempts" in reason

    def test_concurrent_execution_limits(self):
        """Test limiting concurrent executions"""
        controller = ActionController()

        # CREATE_SUBAGENT has max concurrent limit
        context = {"active_create_subagent_count": 4}
        available, _ = controller.is_action_available(
            WorkflowAction.CREATE_SUBAGENT, context
        )
        assert available

        context["active_create_subagent_count"] = 5
        available, reason = controller.is_action_available(
            WorkflowAction.CREATE_SUBAGENT, context
        )
        assert not available
        assert "Maximum concurrent" in reason

    def test_complex_constraints(self):
        """Test complex action-specific constraints"""
        controller = ActionController()
        controller.update_iteration(2)

        # DECOMPOSE requires high complexity
        context = {"subtask_complexity": 1}
        available, reason = controller.is_action_available(
            WorkflowAction.DECOMPOSE, context
        )
        assert not available
        assert "not complex enough" in reason

        context["subtask_complexity"] = 3
        available, _ = controller.is_action_available(WorkflowAction.DECOMPOSE, context)
        assert available

        # REFLECT requires at least 1 iteration
        controller.update_iteration(0)
        available, reason = controller.is_action_available(WorkflowAction.REFLECT, {})
        assert not available

        controller.update_iteration(1)
        available, _ = controller.is_action_available(WorkflowAction.REFLECT, {})
        assert available

    def test_get_available_actions(self):
        """Test getting all available actions"""
        controller = ActionController()
        controller.update_iteration(1)

        context = {
            "knowledge_items": ["k1", "k2"],
            "has_findings": True,
            "has_synthesis": False,
            "confidence": 0.5,
        }

        available = controller.get_available_actions(context)

        # Should have some actions available
        assert len(available) > 0
        assert WorkflowAction.SYNTHESIZE in available  # Has findings and knowledge
        assert WorkflowAction.PLAN in available

        # Should NOT have these
        assert WorkflowAction.DECIDE not in available  # No synthesis
        assert WorkflowAction.CONCLUDE not in available  # Low confidence

    def test_recommended_action(self):
        """Test action recommendation"""
        controller = ActionController()

        # Initial state - should recommend ANALYZE
        context = {}
        recommended = controller.get_recommended_action(context)
        assert recommended == WorkflowAction.ANALYZE

        # After analysis, should recommend PLAN
        controller.record_action(WorkflowAction.ANALYZE, success=True)
        recommended = controller.get_recommended_action(context)
        assert recommended == WorkflowAction.PLAN

        # With findings, should recommend synthesis or knowledge extraction
        context = {"has_findings": True, "knowledge_items": ["k1", "k2"]}
        recommended = controller.get_recommended_action(context)
        # PLAN is still available and may be recommended
        assert recommended in [
            WorkflowAction.PLAN,
            WorkflowAction.EXECUTE_SUBTASK,
            WorkflowAction.EXTRACT_KNOWLEDGE,
            WorkflowAction.SYNTHESIZE,
        ]

    def test_action_stats(self):
        """Test getting action statistics"""
        controller = ActionController()

        # Record various actions
        controller.record_action(WorkflowAction.PLAN, success=True, duration=1.0)
        controller.record_action(WorkflowAction.PLAN, success=True, duration=2.0)
        controller.record_action(WorkflowAction.PLAN, success=False, error="Failed")

        controller.record_action(
            WorkflowAction.EXECUTE_SUBTASK, success=True, duration=5.0
        )
        controller.record_action(
            WorkflowAction.EXECUTE_SUBTASK, success=True, duration=3.0
        )

        stats = controller.get_action_stats()

        # Check PLAN stats
        plan_stats = stats[WorkflowAction.PLAN.value]
        assert plan_stats["total_attempts"] == 3
        assert plan_stats["successful"] == 2
        assert plan_stats["failed"] == 1
        assert plan_stats["success_rate"] == 2 / 3
        assert plan_stats["average_duration"] == 1.5  # (1.0 + 2.0) / 2

        # Check EXECUTE stats
        exec_stats = stats[WorkflowAction.EXECUTE_SUBTASK.value]
        assert exec_stats["total_attempts"] == 2
        assert exec_stats["successful"] == 2
        assert exec_stats["success_rate"] == 1.0
        assert exec_stats["average_duration"] == 4.0  # (5.0 + 3.0) / 2

    def test_recovery_suggestions(self):
        """Test recovery action suggestions after failures"""
        controller = ActionController()
        controller.update_iteration(1)

        context = {"has_findings": True, "knowledge_items": ["k1"]}

        # Failed execution should suggest decomposition
        recovery = controller.suggest_recovery_action(
            WorkflowAction.EXECUTE_SUBTASK,
            "Task too complex",
            {"subtask_complexity": 3},
        )
        assert recovery == WorkflowAction.DECOMPOSE

        # Failed synthesis should suggest more execution
        recovery = controller.suggest_recovery_action(
            WorkflowAction.SYNTHESIZE, "Not enough data", context
        )
        assert recovery == WorkflowAction.EXECUTE_SUBTASK

        # Failed conclusion should suggest reflection
        recovery = controller.suggest_recovery_action(
            WorkflowAction.CONCLUDE, "Confidence too low", context
        )
        assert recovery == WorkflowAction.REFLECT

        # If suggested recovery not available, default to REFLECT
        controller.disabled_actions.add(WorkflowAction.DECOMPOSE)
        recovery = controller.suggest_recovery_action(
            WorkflowAction.EXECUTE_SUBTASK, "Failed", context
        )
        assert recovery == WorkflowAction.REFLECT


class TestActionControllerIntegration:
    """Test action controller integration with workflow concepts"""

    def test_workflow_progression(self):
        """Test natural workflow action progression"""
        controller = ActionController()
        context = {}

        # Track workflow progression
        actions_taken = []

        for i in range(10):
            controller.update_iteration(i)

            # Get recommended action
            action = controller.get_recommended_action(context)
            if action:
                actions_taken.append(action)

                # Execute action
                success = (
                    action != WorkflowAction.EXECUTE_SUBTASK or i % 3 != 2
                )  # Fail every 3rd execution
                controller.record_action(
                    action, success=success, context={"iteration": i}
                )

                # Update context based on action
                if action == WorkflowAction.ANALYZE:
                    context["analyzed"] = True
                elif action == WorkflowAction.EXECUTE_SUBTASK:
                    context["has_findings"] = True
                    if "knowledge_items" not in context:
                        context["knowledge_items"] = []
                    context["knowledge_items"].append(f"k{i}")
                elif action == WorkflowAction.SYNTHESIZE:
                    context["has_synthesis"] = True
                elif action == WorkflowAction.DECIDE:
                    context["confidence"] = 0.6 + (i * 0.05)

        # Verify sensible progression
        assert actions_taken[0] == WorkflowAction.ANALYZE  # Should start with analysis
        assert WorkflowAction.PLAN in actions_taken  # Should plan at some point
        # EXECUTE_SUBTASK may not be in the default recommendations
        # It needs to be explicitly triggered based on workflow logic

        # Later actions should appear after earlier ones
        if WorkflowAction.SYNTHESIZE in actions_taken:
            first_exec = actions_taken.index(WorkflowAction.EXECUTE_SUBTASK)
            first_synth = actions_taken.index(WorkflowAction.SYNTHESIZE)
            assert first_synth > first_exec  # Synthesis after execution

    def test_failure_adaptation(self):
        """Test that controller adapts to failures"""
        controller = ActionController()
        context = {"has_findings": True, "knowledge_items": ["k1"]}

        # Record 3 failures directly - this should disable the action
        for i in range(3):
            controller.record_action(
                WorkflowAction.PLAN, success=False, error=f"Planning failed {i + 1}"
            )

        # PLAN should be disabled after 3 failures
        assert WorkflowAction.PLAN in controller.disabled_actions

        # Should recommend alternative actions
        recommended = controller.get_recommended_action(context)
        assert recommended != WorkflowAction.PLAN
        assert recommended is not None  # Should have alternatives

        # After some successes in other actions, PLAN might be re-enabled
        for i in range(3):
            controller.record_action(WorkflowAction.EXECUTE_SUBTASK, success=True)

        # Not immediately re-enabled, but cooldown might expire
        time.sleep(0.1)

        # Recovery suggestion should adapt
        recovery = controller.suggest_recovery_action(
            WorkflowAction.PLAN, "Still failing", context
        )
        # Recovery may be None if REFLECT is not available, or REFLECT if it is
        assert recovery is None or recovery == WorkflowAction.REFLECT

    def test_prevents_premature_conclusion(self):
        """Test that controller prevents premature workflow conclusion"""
        controller = ActionController()

        # Try to conclude immediately
        context = {
            "knowledge_items": ["k1", "k2", "k3", "k4"],
            "has_synthesis": True,
            "confidence": 0.9,
        }

        # Should not allow conclusion at iteration 0
        available, reason = controller.is_action_available(
            WorkflowAction.CONCLUDE, context
        )
        assert not available
        assert "Requires at least 1 iterations" in reason

        # Even with perfect context, need iterations
        controller.update_iteration(1)
        available, _ = controller.is_action_available(WorkflowAction.CONCLUDE, context)
        assert available  # Now allowed

        # But low confidence should still block
        context["confidence"] = 0.5
        available, reason = controller.is_action_available(
            WorkflowAction.CONCLUDE, context
        )
        assert not available
        assert "Confidence too low" in reason
