from typing import List

from anthropic import Anthropic
from anthropic.types import (
    Message,
    MessageParam,
    ToolParam,
    ToolResultBlockParam,
)
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
)

from .augmented_llm import AugmentedLLM
from ..context import get_current_config


class AnthropicAugmentedLLM(AugmentedLLM[MessageParam, Message]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    async def generate(
        self,
        message,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "claude-3-5-sonnet-20241022",
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = False,
    ):
        """
        Process a query using an LLM and available tools.
        The default implementation uses Claude as the LLM.
        Override this method to use a different LLM.
        """
        config = get_current_config()
        anthropic = Anthropic(api_key=config.anthropic_api_key)

        messages: List[MessageParam] = []

        if use_history:
            messages.extend(self.history.get())

        if isinstance(message, str):
            messages.append({"role": "user", "content": message})
        elif isinstance(message, list):
            messages.extend(message)
        else:
            messages.append(message)

        response = await self.aggregator.list_tools()
        available_tools: List[ToolParam] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        responses: List[Message] = []

        for _ in range(max_iterations):
            response = anthropic.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                system=self.instruction or None,
                stop_sequences=stop_sequences,
                tools=available_tools,
            )

            responses.append(response)

            if response.stop_reason == "end_turn":
                break
            elif response.stop_reason == "stop_sequence":
                # We have reached a stop sequence
                break
            elif response.stop_reason == "max_tokens":
                # We have reached the max tokens limit
                # TODO: saqadri - would be useful to return the reason for stopping to the caller
                break
            else:  # response.stop_reason == "tool_use":
                for content in response.content:
                    if content.type == "text":
                        messages.append({"role": "assistant", "content": content})
                    elif content.type == "tool_use":
                        tool_name = content.name
                        tool_args = content.input
                        tool_use_id = content.id

                        tool_call_request = CallToolRequest(
                            method="tools/call",
                            params=CallToolRequestParams(
                                name=tool_name, arguments=tool_args
                            ),
                        )

                        result = await self.call_tool(
                            request=tool_call_request, tool_call_id=tool_use_id
                        )

                        # Continue conversation with tool results
                        if hasattr(content, "text") and content.text:
                            messages.append(
                                {"role": "assistant", "content": content.text}
                            )

                        messages.append(
                            ToolResultBlockParam(
                                type="tool_result",
                                tool_use_id=tool_use_id,
                                content=result.content,
                                is_error=result.isError,
                            )
                        )

        if use_history:
            self.history.set(messages)

        return responses

    async def generate_str(
        self,
        message,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "claude-3-5-sonnet-20241022",
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = False,
    ) -> str:
        """
        Process a query using an LLM and available tools.
        The default implementation uses Claude as the LLM.
        Override this method to use a different LLM.
        """
        responses = await self.generate(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        final_text: List[str] = []

        for response in responses:
            for content in response.content:
                if content.type == "text":
                    final_text.append(content.text)
                elif content.type == "tool_use":
                    final_text.append(
                        f"[Calling tool {content.name} with args {content.input}]"
                    )

        return "\n".join(final_text)

    def message_param_str(self, message: MessageParam) -> str:
        """Convert an input message to a string representation."""

        if message.get("content"):
            content = message["content"]
            if isinstance(content, str):
                return content
            else:
                final_text: List[str] = []
                for block in content:
                    if block.text:
                        final_text.append(str(block.text))
                    else:
                        final_text.append(str(block))

                return "\n".join(final_text)

        return str(message)

    def message_str(self, message: Message) -> str:
        """Convert an output message to a string representation."""
        content = message.content

        if content:
            if isinstance(content, list):
                final_text: List[str] = []
                for block in content:
                    if block.text:
                        final_text.append(str(block.text))
                    else:
                        final_text.append(str(block))

                return "\n".join(final_text)
            else:
                return str(content)

        return str(message)
