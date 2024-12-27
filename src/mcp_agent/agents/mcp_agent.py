from typing import Callable, Dict, List

from pydantic import BaseModel, ConfigDict
from mcp.server.fastmcp.tools import Tool as FastTool
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ListToolsResult,
    TextContent,
    TextResourceContents,
    Tool,
)

from ..mcp.mcp_aggregator import MCPAggregator


class AgentResource(EmbeddedResource):
    """
    A resource that returns an agent. Meant for use with tool calls that want to return an agent for further processing.
    """

    agent: "Agent"


class AgentFunctionResultResource(EmbeddedResource):
    """
    A resource that returns an AgentFunctionResult.
    Meant for use with tool calls that return an AgentFunctionResult for further processing.
    """

    result: "AgentFunctionResult"


class Agent(MCPAggregator):
    def __init__(
        self,
        name: str,
        instructions: str | Callable[[], str] = "You are a helpful agent.",
        server_names: list[str] = None,
    ):
        super().__init__(server_names)
        self.name = name
        self.instructions = instructions

    async def initialize(self):
        await super().load_servers()


def create_agent_resource(agent: "Agent") -> AgentResource:
    return AgentResource(agent=agent, resource=TextResourceContents(text=agent.name))


def create_agent_function_result_resource(
    result: "AgentFunctionResult",
) -> AgentFunctionResultResource:
    return AgentFunctionResultResource(
        result=result,
        resource=TextResourceContents(
            text=result.value or result.agent.name or "AgentFunctionResult"
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
        instructions: str | Callable[[], str] = "You are a helpful agent.",
        server_names: list[str] = None,
        functions: List["AgentFunctionCallable"] = None,
        parallel_tool_calls: bool = True,
    ):
        super().__init__(
            name=name, instructions=instructions, server_names=server_names
        )
        self.functions = functions
        self.parallel_tool_calls = parallel_tool_calls

        # Map function names to tools
        self._function_tool_map: Dict[str, FastTool] = {}

    async def initialize(self):
        await super().initialize()
        for function in self.functions:
            tool: FastTool = FastTool.from_function(function)
            self._function_tool_map[tool.name] = tool

    async def list_tools(self) -> ListToolsResult:
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
        if name in self._function_tool_map:
            tool = self._function_tool_map[name]
            result = await tool.run(arguments)

            if isinstance(result, Agent):
                resource = create_agent_resource(result)
                return CallToolResult(content=[resource])
            elif isinstance(result, AgentFunctionResult):
                resource = create_agent_function_result_resource(result)
                return CallToolResult(content=[resource])
            elif isinstance(result, str):
                # TODO: saqadri - this is likely meant for returning context variables
                return CallToolResult(content=[TextContent(text=result)])
            elif isinstance(result, dict):
                return CallToolResult(content=[TextContent(text=str(result))])
            else:
                print(f"Unknown result type: {result}, returning as text.")
                return CallToolResult(content=[TextContent(text=str(result))])

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

    model_config = ConfigDict(extra="allow")


AgentFunctionReturnType = str | Agent | dict | AgentFunctionResult
"""A type alias for the return type of a Swarm agent function."""

AgentFunctionCallable = Callable[[], AgentFunctionReturnType]
