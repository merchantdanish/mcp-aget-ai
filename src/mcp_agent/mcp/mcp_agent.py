from typing import Callable, Dict, List

from mcp.server.fastmcp.tools import Tool as FastTool
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ListToolsResult,
    TextContent,
    TextResourceContents,
    Tool,
)

from .mcp_aggregator import MCPAggregator

AgentFunctionCallable = Callable[[], str | "Agent" | dict]


class AgentResource(EmbeddedResource):
    """
    A resource that returns an agent. Meant for use with tool calls that want to return an agent for further processing.
    """

    resource = TextResourceContents(text="Agent")
    agent: "Agent"


def create_agent_resource(agent: "Agent") -> AgentResource:
    return AgentResource(agent=agent)


async def create_transfer_to_agent_tool(
    agent: "Agent", agent_function: Callable[[], None]
) -> Tool:
    return Tool(
        name="transfer_to_agent",
        description="Transfer control to the agent",
        agent_resource=create_agent_resource(agent),
        agent_function=agent_function,
    )


async def create_agent_function_tool(agent_function: AgentFunctionCallable) -> Tool:
    return Tool(
        name="agent_function",
        description="Agent function",
        agent_resource=None,
        agent_function=agent_function,
    )


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
        functions: List[AgentFunctionCallable] = None,
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
                return CallToolResult(resource=resource)
            elif isinstance(result, str):
                return CallToolResult(content=TextContent(text=result))
            elif isinstance(result, dict):
                return CallToolResult(content=TextContent(text=str(result)))
            else:
                print(f"Unknown result type: {result}, returning as text.")
                return CallToolResult(content=TextContent(text=str(result)))

        return await super().call_tool(name, arguments)
