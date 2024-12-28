from typing import Callable, List

from mcp_agent.agents.mcp_agent import Agent
from mcp_agent.mcp_server_registry import ServerRegistry
from mcp_agent.workflows.embedding.embedding_cohere import CohereEmbeddingModel
from mcp_agent.workflows.router.router_embedding import EmbeddingRouter


class CohereEmbeddingRouter(EmbeddingRouter):
    """
    A router that uses Cohere embedding similarity to route requests to appropriate categories.
    This class helps to route an input to a specific MCP server, an Agent (an aggregation of MCP servers),
    or a function (any Callable).
    """

    def __init__(
        self,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        server_registry: ServerRegistry | None = None,
        embedding_model: CohereEmbeddingModel | None = None,
    ):
        embedding_model = embedding_model or CohereEmbeddingModel()

        super().__init__(
            embedding_model=embedding_model,
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            server_registry=server_registry,
        )

    @classmethod
    async def create(
        cls,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        server_registry: ServerRegistry | None = None,
        embedding_model: CohereEmbeddingModel | None = None,
    ) -> "CohereEmbeddingRouter":
        """
        Factory method to create and initialize a router.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            server_registry=server_registry,
            embedding_model=embedding_model,
        )
        await instance.initialize()
        return instance
