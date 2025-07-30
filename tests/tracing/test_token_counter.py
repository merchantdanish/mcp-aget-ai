"""Tests for TokenCounter implementation"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from mcp_agent.tracing.token_counter import TokenCounter, TokenUsage, TokenNode
from mcp_agent.workflows.llm.llm_selector import (
    ModelInfo,
    ModelCost,
    ModelMetrics,
    ModelLatency,
    ModelBenchmarks,
)


class TestTokenUsage:
    """Test TokenUsage dataclass"""

    def test_token_usage_initialization(self):
        """Test TokenUsage initialization and auto-calculation of total"""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150
        assert usage.model_name is None
        assert usage.model_info is None
        assert isinstance(usage.timestamp, datetime)

    def test_token_usage_explicit_total(self):
        """Test that explicit total_tokens is preserved"""
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=200)
        assert usage.total_tokens == 200  # Should not be overwritten


class TestTokenNode:
    """Test TokenNode dataclass"""

    def test_token_node_initialization(self):
        """Test TokenNode initialization"""
        node = TokenNode(name="test_node", node_type="agent")
        assert node.name == "test_node"
        assert node.node_type == "agent"
        assert node.parent is None
        assert node.children == []
        assert isinstance(node.usage, TokenUsage)
        assert node.metadata == {}

    def test_add_child(self):
        """Test adding child nodes"""
        parent = TokenNode(name="parent", node_type="app")
        child = TokenNode(name="child", node_type="agent")

        parent.add_child(child)

        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert child.parent == parent

    def test_aggregate_usage_single_node(self):
        """Test aggregate usage for single node"""
        node = TokenNode(name="test", node_type="agent")
        node.usage = TokenUsage(input_tokens=100, output_tokens=50)

        aggregated = node.aggregate_usage()
        assert aggregated.input_tokens == 100
        assert aggregated.output_tokens == 50
        assert aggregated.total_tokens == 150

    def test_aggregate_usage_with_children(self):
        """Test aggregate usage with child nodes"""
        root = TokenNode(name="root", node_type="app")
        root.usage = TokenUsage(input_tokens=100, output_tokens=50)

        child1 = TokenNode(name="child1", node_type="agent")
        child1.usage = TokenUsage(input_tokens=200, output_tokens=100)

        child2 = TokenNode(name="child2", node_type="agent")
        child2.usage = TokenUsage(input_tokens=150, output_tokens=75)

        root.add_child(child1)
        root.add_child(child2)

        aggregated = root.aggregate_usage()
        assert aggregated.input_tokens == 450  # 100 + 200 + 150
        assert aggregated.output_tokens == 225  # 50 + 100 + 75
        assert aggregated.total_tokens == 675

    def test_to_dict(self):
        """Test converting node to dictionary"""
        node = TokenNode(name="test", node_type="agent", metadata={"key": "value"})
        node.usage = TokenUsage(input_tokens=100, output_tokens=50, model_name="gpt-4")

        result = node.to_dict()

        assert result["name"] == "test"
        assert result["type"] == "agent"
        assert result["metadata"] == {"key": "value"}
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50
        assert result["usage"]["total_tokens"] == 150
        assert result["usage"]["model_name"] == "gpt-4"
        assert "timestamp" in result["usage"]
        assert result["children"] == []


class TestTokenCounter:
    """Test TokenCounter class"""

    # Mock logger to avoid async issues in tests
    @pytest.fixture(autouse=True)
    def mock_logger(self):
        with patch("mcp_agent.tracing.token_counter.logger") as mock:
            mock.debug = MagicMock()
            mock.info = MagicMock()
            mock.warning = MagicMock()
            mock.error = MagicMock()
            yield mock

    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing"""
        models = [
            ModelInfo(
                name="gpt-4",
                provider="OpenAI",
                description="GPT-4",
                context_window=8192,
                tool_calling=True,
                structured_outputs=True,
                metrics=ModelMetrics(
                    cost=ModelCost(
                        input_cost_per_1m=10.0,
                        output_cost_per_1m=30.0,
                        blended_cost_per_1m=15.0,
                    ),
                    speed=ModelLatency(
                        time_to_first_token_ms=50.0, tokens_per_second=100.0
                    ),
                    intelligence=ModelBenchmarks(quality_score=0.8),
                ),
            ),
            ModelInfo(
                name="claude-3-opus",
                provider="Anthropic",
                description="Claude 3 Opus",
                context_window=200000,
                tool_calling=True,
                structured_outputs=True,
                metrics=ModelMetrics(
                    cost=ModelCost(
                        input_cost_per_1m=15.0,
                        output_cost_per_1m=75.0,
                        blended_cost_per_1m=30.0,
                    ),
                    speed=ModelLatency(
                        time_to_first_token_ms=40.0, tokens_per_second=120.0
                    ),
                    intelligence=ModelBenchmarks(quality_score=0.9),
                ),
            ),
            ModelInfo(
                name="claude-3-opus",
                provider="AWS Bedrock",
                description="Claude 3 Opus on Bedrock",
                context_window=200000,
                tool_calling=True,
                structured_outputs=True,
                metrics=ModelMetrics(
                    cost=ModelCost(
                        input_cost_per_1m=20.0,
                        output_cost_per_1m=80.0,
                        blended_cost_per_1m=35.0,
                    ),
                    speed=ModelLatency(
                        time_to_first_token_ms=60.0, tokens_per_second=80.0
                    ),
                    intelligence=ModelBenchmarks(quality_score=0.9),
                ),
            ),
        ]
        return models

    @pytest.fixture
    def token_counter(self, mock_models):
        """Create a TokenCounter with mocked model loading"""
        with patch(
            "mcp_agent.tracing.token_counter.load_default_models",
            return_value=mock_models,
        ):
            return TokenCounter()

    def test_initialization(self, token_counter, mock_models):
        """Test TokenCounter initialization"""
        assert token_counter._stack == []
        assert token_counter._root is None
        assert token_counter._current is None
        assert len(token_counter._models) == 3
        assert "gpt-4" in token_counter._model_costs
        assert "claude-3-opus" in token_counter._model_costs

    def test_push_pop_single(self, token_counter):
        """Test push and pop operations"""
        token_counter.push("app", "app")

        assert len(token_counter._stack) == 1
        assert token_counter._current.name == "app"
        assert token_counter._root == token_counter._current

        popped = token_counter.pop()
        assert popped.name == "app"
        assert len(token_counter._stack) == 0
        assert token_counter._current is None

    def test_push_pop_nested(self, token_counter):
        """Test nested push and pop operations"""
        token_counter.push("app", "app")
        token_counter.push("workflow", "workflow")
        token_counter.push("agent", "agent")

        assert len(token_counter._stack) == 3
        assert token_counter.get_current_path() == ["app", "workflow", "agent"]

        # Pop agent
        agent_node = token_counter.pop()
        assert agent_node.name == "agent"
        assert token_counter._current.name == "workflow"

        # Pop workflow
        workflow_node = token_counter.pop()
        assert workflow_node.name == "workflow"
        assert token_counter._current.name == "app"

        # Pop app
        app_node = token_counter.pop()
        assert app_node.name == "app"
        assert token_counter._current is None

    def test_pop_empty_stack(self, token_counter):
        """Test popping from empty stack"""
        result = token_counter.pop()
        assert result is None

    def test_record_usage_no_context(self, token_counter):
        """Test recording usage without context creates root"""
        token_counter.record_usage(
            input_tokens=100, output_tokens=50, model_name="gpt-4", provider="OpenAI"
        )

        assert token_counter._root is not None
        assert token_counter._root.name == "root"
        assert token_counter._root.usage.input_tokens == 100
        assert token_counter._root.usage.output_tokens == 50

    def test_record_usage_with_context(self, token_counter):
        """Test recording usage with context"""
        token_counter.push("test", "agent")

        token_counter.record_usage(
            input_tokens=100, output_tokens=50, model_name="gpt-4", provider="OpenAI"
        )

        assert token_counter._current.usage.input_tokens == 100
        assert token_counter._current.usage.output_tokens == 50
        assert token_counter._current.usage.model_name == "gpt-4"

        # Check global tracking
        assert ("gpt-4", "OpenAI") in token_counter._usage_by_model
        usage = token_counter._usage_by_model[("gpt-4", "OpenAI")]
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_record_usage_multiple_providers(self, token_counter):
        """Test recording usage for same model from different providers"""
        token_counter.push("test", "app")

        # Record usage for Anthropic's Claude
        token_counter.record_usage(
            input_tokens=100,
            output_tokens=50,
            model_name="claude-3-opus",
            provider="Anthropic",
        )

        # Record usage for Bedrock's Claude
        token_counter.record_usage(
            input_tokens=200,
            output_tokens=100,
            model_name="claude-3-opus",
            provider="AWS Bedrock",
        )

        # Check they're tracked separately
        anthropic_usage = token_counter._usage_by_model[("claude-3-opus", "Anthropic")]
        assert anthropic_usage.input_tokens == 100
        assert anthropic_usage.output_tokens == 50

        bedrock_usage = token_counter._usage_by_model[("claude-3-opus", "AWS Bedrock")]
        assert bedrock_usage.input_tokens == 200
        assert bedrock_usage.output_tokens == 100

    def test_find_model_info_exact_match(self, token_counter):
        """Test finding model info by exact match"""
        # Without provider - should return first match
        model = token_counter.find_model_info("gpt-4")
        assert model is not None
        assert model.name == "gpt-4"
        assert model.provider == "OpenAI"

        # With provider - should return exact match
        model = token_counter.find_model_info("claude-3-opus", "AWS Bedrock")
        assert model is not None
        assert model.provider == "AWS Bedrock"

    def test_find_model_info_fuzzy_match(self, token_counter):
        """Test fuzzy matching for model info"""
        # Partial match
        model = token_counter.find_model_info("gpt-4-turbo")  # Not exact
        assert model is not None
        assert model.name == "gpt-4"

        # With provider hint
        model = token_counter.find_model_info("claude-3", "Anthropic")
        assert model is not None
        assert model.name == "claude-3-opus"
        assert model.provider == "Anthropic"

    def test_calculate_cost(self, token_counter):
        """Test cost calculation"""
        # GPT-4 cost calculation
        cost = token_counter.calculate_cost("gpt-4", 1000, 500, "OpenAI")
        expected = (1000 / 1_000_000) * 10.0 + (500 / 1_000_000) * 30.0
        assert cost == pytest.approx(expected)

        # Unknown model - should use default
        cost = token_counter.calculate_cost("unknown-model", 1000, 500)
        expected = (1500 * 0.5) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_get_summary(self, token_counter):
        """Test getting summary of token usage"""
        token_counter.push("app", "app")

        # Record some usage
        token_counter.record_usage(100, 50, "gpt-4", "OpenAI")
        token_counter.record_usage(200, 100, "claude-3-opus", "Anthropic")
        token_counter.record_usage(150, 75, "claude-3-opus", "AWS Bedrock")

        summary = token_counter.get_summary()

        # Check total usage
        assert summary["total_usage"]["input_tokens"] == 450
        assert summary["total_usage"]["output_tokens"] == 225
        assert summary["total_usage"]["total_tokens"] == 675

        # Check by model
        assert "gpt-4 (OpenAI)" in summary["by_model"]
        assert "claude-3-opus (Anthropic)" in summary["by_model"]
        assert "claude-3-opus (AWS Bedrock)" in summary["by_model"]

        # Check costs are calculated
        assert summary["total_cost"] > 0
        assert summary["by_model"]["gpt-4 (OpenAI)"]["cost"] > 0

    def test_get_tree(self, token_counter):
        """Test getting token usage tree"""
        token_counter.push("app", "app", {"version": "1.0"})
        token_counter.push("agent", "agent")
        token_counter.record_usage(100, 50, "gpt-4", "OpenAI")

        tree = token_counter.get_tree()

        assert tree is not None
        assert tree["name"] == "app"
        assert tree["type"] == "app"
        assert tree["metadata"] == {"version": "1.0"}
        assert len(tree["children"]) == 1
        assert tree["children"][0]["name"] == "agent"

    def test_reset(self, token_counter):
        """Test resetting token counter"""
        token_counter.push("app", "app")
        token_counter.record_usage(100, 50, "gpt-4", "OpenAI")

        token_counter.reset()

        assert len(token_counter._stack) == 0
        assert token_counter._root is None
        assert token_counter._current is None
        assert len(token_counter._usage_by_model) == 0

    def test_thread_safety(self, token_counter):
        """Test basic thread safety with concurrent operations"""
        import threading
        import time

        results = []

        def worker(worker_id):
            for i in range(5):
                token_counter.push(f"worker_{worker_id}_{i}", "agent")
                token_counter.record_usage(10, 5, "gpt-4", "OpenAI")
                time.sleep(0.001)  # Small delay to encourage interleaving
                node = token_counter.pop()
                if node:
                    results.append((worker_id, node.usage.total_tokens))

        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All operations should complete without error
        assert len(results) == 15  # 3 workers * 5 iterations

        # Each result should have correct token count
        for _, tokens in results:
            assert tokens == 15  # 10 + 5
