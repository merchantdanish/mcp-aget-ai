from typing import Callable, List

from ..agents.mcp_agent import Agent
from ..context import get_current_context
from ..mcp_server_registry import ServerRegistry
from .embedding_openai import OpenAIEmbeddingModel
from .router_embedding import EmbeddingRouter


class OpenAIEmbeddingRouter(EmbeddingRouter):
    """
    A router that uses OpenAI embedding similarity to route requests to appropriate categories.
    This class helps to route an input to a specific MCP server, an Agent (an aggregation of MCP servers),
    or a function (any Callable).
    """

    def __init__(
        self,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        server_registry: ServerRegistry = get_current_context().server_registry,
        embedding_model: OpenAIEmbeddingModel | None = None,
    ):
        embedding_model = embedding_model or OpenAIEmbeddingModel()

        super().__init__(
            embedding_model=embedding_model,
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            server_registry=server_registry,
        )
