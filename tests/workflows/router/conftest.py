import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np
from typing import List, Dict, Callable, Optional, Any

from mcp_agent.core.context import Context
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.embedding.embedding_base import FloatArray, EmbeddingModel
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM
from mcp_agent.workflows.router.router_base import (
    RouterCategory,
    ServerRouterCategory,
    AgentRouterCategory,
)


@pytest.fixture
def mock_context():
    """
    Returns a mock Context instance for testing.
    """
    mock = MagicMock(spec=Context)
    mock.executor = MagicMock()

    # Setup configuration for different providers
    mock.config = MagicMock()

    # OpenAI config
    mock.config.openai = MagicMock()
    mock.config.openai.api_key = "test_openai_key"
    mock.config.openai.default_model = "gpt-4o"

    # Anthropic config
    mock.config.anthropic = MagicMock()
    mock.config.anthropic.api_key = "test_anthropic_key"
    mock.config.anthropic.default_model = "claude-3-7-sonnet-latest"

    # Cohere config
    mock.config.cohere = MagicMock()
    mock.config.cohere.api_key = "test_cohere_key"

    # Setup server registry
    mock.server_registry = MagicMock()

    # Create a proper server config mock that returns string values
    server_config = MagicMock()
    server_config.name = "test_server"  # Use string value, not a mock
    server_config.description = (
        "A test server for routing"  # Use string value, not a mock
    )
    server_config.embedding = MagicMock()

    mock.server_registry.get_server_config = MagicMock(return_value=server_config)

    return mock


@pytest.fixture
def mock_agent():
    """
    Returns a mock Agent instance for testing.
    """
    mock = MagicMock(spec=Agent)
    mock.name = "test_agent"
    mock.instruction = "This is a test agent instruction"
    mock.server_names = ["test_server"]

    # Make context manager methods work
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)

    return mock


@pytest.fixture
def mock_llm():
    """
    Returns a mock AugmentedLLM instance for testing.
    """
    mock = MagicMock(spec=AugmentedLLM)
    mock.generate = AsyncMock()
    mock.generate_str = AsyncMock()
    mock.generate_structured = AsyncMock()
    return mock


@pytest.fixture
def mock_embedding_model():
    """
    Returns a mock EmbeddingModel instance for testing.
    """
    mock = MagicMock(spec=EmbeddingModel)

    # Generate deterministic but different embeddings for testing
    def embed_side_effect(data: List[str]) -> FloatArray:
        embedding_dim = 1536
        embeddings = np.ones((len(data), embedding_dim), dtype=np.float32)
        for i in range(len(data)):
            # Simple hashing to create different embeddings for different strings
            seed = sum(ord(c) for c in data[i])
            np.random.seed(seed)
            embeddings[i] = np.random.rand(embedding_dim).astype(np.float32)
        return embeddings

    mock.embed = AsyncMock(side_effect=embed_side_effect)
    mock.embedding_dim = 1536

    return mock


@pytest.fixture
def test_function():
    """
    Returns a test function for router testing.
    """

    def test_function(input_text: str) -> str:
        """A test function that echoes the input."""
        return f"Echo: {input_text}"

    return test_function


@pytest.fixture
def test_router_categories(mock_agent, test_function):
    """
    Returns test router categories for testing.
    """
    # Server category
    server_category = ServerRouterCategory(
        name="test_server",
        description="A test server for routing",
        category="test_server",
        tools=[],  # Using empty list for tools to avoid validation issues
    )

    # Agent category
    agent_category = AgentRouterCategory(
        name="test_agent",
        description="A test agent for routing",
        category=mock_agent,
        servers=[server_category],
    )

    # Function category
    function_category = RouterCategory(
        name="test_function",
        description="A test function for routing",
        category=test_function,
    )

    return {
        "server_category": server_category,
        "agent_category": agent_category,
        "function_category": function_category,
    }
