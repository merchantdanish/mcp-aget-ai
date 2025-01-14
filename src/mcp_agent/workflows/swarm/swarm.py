from typing import Callable, Dict, Generic
from collections import defaultdict

from mcp.types import CallToolRequest, CallToolResult, TextContent, Tool

from mcp_agent.agents.agent import AgentFunctionResult, AgentResource, SwarmAgent
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
)
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


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
        self.context_variables = defaultdict(str, context_variables or {})
        self.instruction = (
            agent.instruction(self.context_variables)
            if isinstance(agent.instruction, Callable)
            else agent.instruction
        )
        logger.debug(
            f"Swarm initialized with agent {agent.name}",
            data={
                "context_variables": self.context_variables,
                "instruction": self.instruction,
            },
        )

    async def get_tool(self, tool_name: str) -> Tool | None:
        """Get the schema for a tool by name."""
        result = await self.aggregator.list_tools()
        for tool in result.tools:
            if tool.name == tool_name:
                return tool

        return None

    async def pre_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest
    ) -> CallToolRequest | bool:
        if not self.aggregator:
            # If there are no agents, we can't do anything, so we should bail
            return False

        tool = await self.get_tool(request.params.name)
        if not tool:
            logger.warning(
                f"Warning: Tool '{request.params.name}' not found in agent '{self.aggregator.name}' tools. Proceeding with original request params."
            )
            return request

        # If the tool has a "context_variables" parameter, we set it to our context variables state
        if "context_variables" in tool.inputSchema:
            logger.debug(
                f"Setting context variables on tool_call '{request.params.name}'",
                data=self.context_variables,
            )
            request.params.arguments["context_variables"] = self.context_variables

        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ) -> CallToolResult:
        contents = []
        for content in result.content:
            if isinstance(content, AgentResource):
                # Set the new agent as the current agent
                await self._set_agent(content.agent)
                contents.append(TextContent(type="text", text=content.resource.text))
            elif isinstance(content, AgentFunctionResult):
                logger.info(
                    "Updating context variables with new context variables from agent function result",
                    data=content.context_variables,
                )
                self.context_variables.update(content.context_variables)
                if content.agent:
                    # Set the new agent as the current agent
                    self._set_agent(content.agent)

                contents.append(TextContent(type="text", text=content.resource.text))
            else:
                contents.append(content)

        result.content = contents
        return result

    async def _set_agent(
        self,
        agent: SwarmAgent,
    ):
        logger.info(
            f"Switching from agent '{self.aggregator.name}' -> agent '{agent.name}'"
        )
        if self.aggregator:
            # Close the current agent
            await self.aggregator.shutdown()

        # Initialize the new agent (if it's not None)
        self.aggregator = agent

        if not self.aggregator:
            self.instruction = None
            return

        await self.aggregator.initialize()
        self.instruction = (
            agent.instruction(self.context_variables)
            if callable(agent.instruction)
            else agent.instruction
        )
