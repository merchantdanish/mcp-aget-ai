from typing import Dict, Generic
from collections import defaultdict

from mcp.types import CallToolRequest, CallToolResult, TextContent, Tool

from mcp_agent.agents.agent import AgentFunctionResult, AgentResource, SwarmAgent
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
)


class Swarm(AugmentedLLM[MessageParamT, MessageT], Generic[MessageParamT, MessageT]):
    """
    Handles orchestrating agents that can use tools via MCP servers.

    MCP version of the OpenAI Swarm class (https://github.com/openai/swarm.)
    """

    # TODO: saqadri - streaming isn't supported yet because the underlying AugmentedLLM classes don't support it

    def __init__(self, agent: SwarmAgent, context_variables: Dict[str, str] = None):
        """
        Initialize the LLM planner with an agent, which will be used as the
        starting point for the workflow.
        """
        super().__init__(agent=agent)
        self.agent = agent
        self.context_variables = defaultdict(str, context_variables or {})

    async def get_tool(self, tool_name: str) -> Tool | None:
        """Get the schema for a tool by name."""
        result = await self.agent.list_tools()
        for tool in result.tools:
            if tool.name == tool_name:
                return tool

        return None

    async def pre_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest
    ) -> CallToolRequest | bool:
        if not self.agent:
            # If there are no agents, we can't do anything, so we should bail
            return False

        tool = await self.get_tool(request.tool_name)
        if not tool:
            print(
                f"Warning: Tool {request.tool_name} not found in agent tools. Proceeding with original request params."
            )
            return request

        # If the tool has a "context_variables" parameter, we set it to our context variables state
        if "context_variables" in tool.inputSchema:
            request.params.arguments["context_variables"] = self.context_variables

        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ) -> CallToolResult:
        contents = []
        for content in result.content:
            if isinstance(content, AgentResource):
                # Set the new agent as the current agent
                self._set_agent(content.agent)
                contents.append(TextContent(text=content.resource.text))
            elif isinstance(content, AgentFunctionResult):
                self.context_variables.update(content.context_variables)
                if content.agent:
                    # Set the new agent as the current agent
                    self._set_agent(content.agent)

                contents.append(TextContent(text=content.resource.text))
            else:
                content.append(content)

        result.content = contents
        return result

    def _set_agent(
        self,
        agent: SwarmAgent,
    ):
        self.agent = agent
        self.instruction = (
            agent.instruction(self.context_variables)
            if callable(agent.instruction)
            else agent.instruction
        )
