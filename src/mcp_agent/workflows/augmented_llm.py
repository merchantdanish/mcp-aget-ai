from typing import Generic, List, Protocol, TypeVar

from mcp.types import (
    CallToolRequest,
    CallToolResult,
    TextContent,
)

from ..agents.mcp_agent import Agent
from ..mcp.mcp_aggregator import MCPAggregator

MessageParamT = TypeVar("MessageParamT")
"""A type representing an input message to an LLM."""

MessageT = TypeVar("MessageT")
"""A type representing an output message from an LLM."""


class Memory(Protocol, Generic[MessageParamT]):
    """
    Simple memory management for storing past interactions in-memory.
    """

    # TODO: saqadri - add checkpointing and other advanced memory capabilities

    def __init__(self): ...

    def extend(self, messages: List[MessageParamT]) -> None: ...

    def set(self, messages: List[MessageParamT]) -> None: ...

    def append(self, message: MessageParamT) -> None: ...

    def get(self) -> List[MessageParamT]: ...

    def clear(self) -> None: ...


class SimpleMemory(Memory, Generic[MessageParamT]):
    """
    Simple memory management for storing past interactions in-memory.
    """

    def __init__(self):
        self.history: List[MessageParamT] = []

    def extend(self, messages: List[MessageParamT]):
        self.history.extend(messages)

    def set(self, messages: List[MessageParamT]):
        self.history = messages.copy()

    def append(self, message: MessageParamT):
        self.history.append(message)

    def get(self) -> List[MessageParamT]:
        return self.history

    def clear(self):
        self.history = []


class AugmentedLLM(Protocol, Generic[MessageParamT, MessageT]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    # TODO: saqadri - add streaming support (e.g. generate_stream)
    # TODO: saqadri - consider adding middleware patterns for pre/post processing of messages, for now we have pre/post_tool_call

    def __init__(
        self,
        server_names: List[str] | None,
        instruction: str | None,
        name: str | None,
        agent: Agent | None,
    ):
        """
        Initialize the LLM with a list of server names and an instruction.
        If a name is provided, it will be used to identify the LLM.
        If an agent is provided, all other properties are optional
        """
        self.aggregator = agent if agent is not None else MCPAggregator(server_names)
        self.name = name or (agent.name if agent else None)
        self.instruction = instruction or (
            agent.instructions if agent and isinstance(agent.instruction, str) else None
        )
        self.history: Memory[MessageParamT] = SimpleMemory[MessageParamT]()

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "",
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> List[MessageT]:
        """Request an LLM generation, which may run multiple iterations, and return the result"""

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "",
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> str:
        """Request an LLM generation and return the string representation of the result"""

    async def pre_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest
    ) -> CallToolRequest | bool:
        """Called before a tool is executed. Return False to prevent execution."""
        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ) -> CallToolResult:
        """Called after a tool execution. Can modify the result before it's returned."""
        return result

    async def call_tool(
        self,
        request: CallToolRequest,
        tool_call_id: str | None = None,
    ) -> CallToolResult:
        """Call a tool with the given parameters and optional ID"""

        try:
            preprocess = await self.pre_tool_call(
                tool_call_id=tool_call_id,
                request=request,
            )

            if isinstance(preprocess, bool):
                if not preprocess:
                    return CallToolResult(
                        isError=True,
                        content=TextContent(
                            text=f"Error: Tool '{request.params.name}' was not allowed to run."
                        ),
                    )
            else:
                request = preprocess

            tool_name = request.params.name
            tool_args = request.params.arguments
            result = await self.aggregator.call_tool(tool_name, tool_args)

            postprocess = await self.post_tool_call(
                tool_call_id=tool_call_id, request=request, result=result
            )

            if isinstance(postprocess, CallToolResult):
                result = postprocess

        except Exception as e:
            return CallToolResult(
                isError=True,
                content=TextContent(
                    text=f"Error executing tool '{request.name}': {str(e)}"
                ),
            )
