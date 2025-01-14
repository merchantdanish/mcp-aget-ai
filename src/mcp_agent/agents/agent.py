from typing import Callable, Dict, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, AnyUrl
from mcp.server.fastmcp.tools import Tool as FastTool
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ListToolsResult,
    TextContent,
    TextResourceContents,
    Tool,
)

from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)

# Define a TypeVar for AugmentedLLM and its subclasses
LLM = TypeVar("LLM", bound=AugmentedLLM)


class AgentResource(EmbeddedResource):
    """
    A resource that returns an agent. Meant for use with tool calls that want to return an Agent for further processing.
    """

    agent: Optional["Agent"] = None


class AgentFunctionResultResource(EmbeddedResource):
    """
    A resource that returns an AgentFunctionResult.
    Meant for use with tool calls that return an AgentFunctionResult for further processing.
    """

    result: "AgentFunctionResult"


class Agent(MCPAggregator):
    """
    An Agent is an entity that has access to a set of MCP servers and can interact with them.
    Each agent should have a purpose defined by its instruction.
    """

    name: str
    instruction: str | Callable[[Dict], str]

    def __init__(
        self,
        name: str,
        instruction: str | Callable[[Dict], str] = "You are a helpful agent.",
        server_names: list[str] = None,
        connection_persistence: bool = True,
    ):
        super().__init__(
            server_names=server_names or [],
            connection_persistence=connection_persistence,
            name=name,
            instruction=instruction,
        )

    async def initialize(self):
        """
        Initialize the agent and connect to the MCP servers.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await (
            self.__aenter__()
        )  # This initializes the connection manager and loads the servers

    async def attach_llm(self, llm_factory: Callable[..., LLM]) -> LLM:
        """
        Create an LLM instance for the agent.

         Args:
            llm_factory: A callable that constructs an AugmentedLLM or its subclass.
                        The factory should accept keyword arguments matching the
                        AugmentedLLM constructor parameters.

        Returns:
            An instance of AugmentedLLM or one of its subclasses.
        """
        return llm_factory(agent=self)

    async def shutdown(self):
        """
        Shutdown the agent and close all MCP server connections.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await super().close()


def create_agent_resource(agent: "Agent") -> AgentResource:
    return AgentResource(
        type="resource",
        agent=agent,
        resource=TextResourceContents(
            text=f"You are now Agent '{agent.name}'. Please review the messages and continue execution",
            uri=AnyUrl("http://this.is.a.fake.url"),
        ),
    )


def create_agent_function_result_resource(
    result: "AgentFunctionResult",
) -> AgentFunctionResultResource:
    return AgentFunctionResultResource(
        type="resource",
        result=result,
        resource=TextResourceContents(
            text=result.value or result.agent.name or "AgentFunctionResult",
            uri=AnyUrl("http://this.is.a.fake.url"),
        ),
    )


async def create_transfer_to_agent_tool(
    agent: "Agent", agent_function: Callable[[], None]
) -> Tool:
    return Tool(
        name="transfer_to_agent",
        description="Transfer control to the agent",
        agent_resource=create_agent_resource(agent),
        agent_function=agent_function,
    )


async def create_agent_function_tool(agent_function: "AgentFunctionCallable") -> Tool:
    return Tool(
        name="agent_function",
        description="Agent function",
        agent_resource=None,
        agent_function=agent_function,
    )


class SwarmAgent(Agent):
    """
    A SwarmAgent is an Agent that can spawn other agents and interactively resolve a task.
    Based on OpenAI Swarm: https://github.com/openai/swarm.

    SwarmAgents have access to tools available on the servers they are connected to, but additionally
    have a list of (possibly local) functions that can be called as tools.
    """

    def __init__(
        self,
        name: str,
        instruction: str | Callable[[Dict], str] = "You are a helpful agent.",
        server_names: list[str] = None,
        functions: List["AgentFunctionCallable"] = None,
        parallel_tool_calls: bool = True,
    ):
        super().__init__(
            name=name,
            instruction=instruction,
            server_names=server_names,
            # TODO: saqadri - figure out if Swarm can maintain connection persistence
            # It's difficult because we don't know when the agent will be done with its task
            connection_persistence=False,
        )
        self.functions = functions
        self.parallel_tool_calls = parallel_tool_calls

        # Map function names to tools
        self._function_tool_map: Dict[str, FastTool] = {}

    async def initialize(self):
        if self.initialized:
            return

        await super().initialize()
        for function in self.functions:
            tool: FastTool = FastTool.from_function(function)
            self._function_tool_map[tool.name] = tool

    async def list_tools(self) -> ListToolsResult:
        if not self.initialized:
            await self.initialize()

        result = await super().list_tools()
        for tool in self._function_tool_map.values():
            result.tools.append(
                Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.parameters,
                )
            )

        return result

    async def call_tool(
        self, name: str, arguments: dict | None = None
    ) -> CallToolResult:
        if not self.initialized:
            await self.initialize()

        if name in self._function_tool_map:
            tool = self._function_tool_map[name]
            result = await tool.run(arguments)

            logger.debug(f"Function tool {name} result:", data=result)

            if isinstance(result, Agent) or isinstance(result, SwarmAgent):
                resource = create_agent_resource(result)
                return CallToolResult(content=[resource])
            elif isinstance(result, AgentFunctionResult):
                resource = create_agent_function_result_resource(result)
                return CallToolResult(content=[resource])
            elif isinstance(result, str):
                # TODO: saqadri - this is likely meant for returning context variables
                return CallToolResult(content=[TextContent(type="text", text=result)])
            elif isinstance(result, dict):
                return CallToolResult(
                    content=[TextContent(type="text", text=str(result))]
                )
            else:
                logger.warning(f"Unknown result type: {result}, returning as text.")
                return CallToolResult(
                    content=[TextContent(type="text", text=str(result))]
                )

        return await super().call_tool(name, arguments)


class AgentFunctionResult(BaseModel):
    """
    Encapsulates the possible return values for a Swarm agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Agent | None = None
    context_variables: dict = {}

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


AgentFunctionReturnType = str | Agent | dict | AgentFunctionResult
"""A type alias for the return type of a Swarm agent function."""

AgentFunctionCallable = Callable[[], AgentFunctionReturnType]
