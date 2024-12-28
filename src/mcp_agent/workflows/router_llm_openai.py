from typing import Callable, List

from ..agents.mcp_agent import Agent
from ..context import get_current_context
from ..mcp_server_registry import ServerRegistry
from .augmented_llm_openai import OpenAIAugmentedLLM
from .router_llm import LLMRouter

ROUTING_SYSTEM_INSTRUCTION = """
You are a highly accurate request router that directs incoming requests to the most appropriate category.
A category is a specialized destination, such as a Function, an MCP Server (a collection of tools/functions), or an Agent (a collection of servers).
You will be provided with a request and a list of categories to choose from.
You can choose one or more categories, or choose none if no category is appropriate.
"""


class OpenAILLMRouter(LLMRouter):
    """
    An LLM router that uses an OpenAI model to make routing decisions.
    """

    def __init__(
        self,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        server_registry: ServerRegistry = get_current_context().server_registry,
    ):
        openai_llm = OpenAIAugmentedLLM(instruction=ROUTING_SYSTEM_INSTRUCTION)

        super().__init__(
            llm=openai_llm,
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            routing_instruction=routing_instruction,
            server_registry=server_registry,
        )
