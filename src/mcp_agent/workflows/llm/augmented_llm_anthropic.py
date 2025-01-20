from typing import List, Type

import instructor
from anthropic import Anthropic
from anthropic.types import (
    Message,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
)

from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, ModelT
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class AnthropicAugmentedLLM(AugmentedLLM[MessageParam, Message]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    @classmethod
    async def convert_message_to_message_param(
        cls, message: Message, **kwargs
    ) -> MessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        content = []

        for content_block in message.content:
            if content_block.type == "text":
                content.append(TextBlockParam(type="text", text=content_block.text))
            elif content_block.type == "tool_use":
                content.append(
                    ToolUseBlockParam(
                        type="tool_use",
                        name=content_block.name,
                        input=content_block.input,
                        id=content_block.id,
                    )
                )

        return MessageParam(role="assistant", content=content, **kwargs)

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
        config = self.context.config
        anthropic = Anthropic(api_key=config.anthropic.api_key)

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

        for i in range(max_iterations):
            arguments = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
                "system": self.instruction or None,
                "stop_sequences": stop_sequences,
                "tools": available_tools,
            }
            logger.debug(
                f"Iteration {i}: Calling {model} with messages:",
                data=messages,
            )

            executor_result = await self.executor.execute(
                anthropic.messages.create, **arguments
            )

            response = executor_result[0]

            logger.debug(
                f"Iteration {i}: {model} response:",
                data=response,
            )

            response_as_message = await self.convert_message_to_message_param(response)
            messages.append(response_as_message)
            responses.append(response)

            if response.stop_reason == "end_turn":
                logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'end_turn'"
                )
                break
            elif response.stop_reason == "stop_sequence":
                # We have reached a stop sequence
                logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'stop_sequence'"
                )
                break
            elif response.stop_reason == "max_tokens":
                # We have reached the max tokens limit
                logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'max_tokens'"
                )
                # TODO: saqadri - would be useful to return the reason for stopping to the caller
                break
            else:  # response.stop_reason == "tool_use":
                for content in response.content:
                    if content.type == "tool_use":
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

                        messages.append(
                            MessageParam(
                                role="user",
                                content=[
                                    ToolResultBlockParam(
                                        type="tool_result",
                                        tool_use_id=tool_use_id,
                                        content=result.content,
                                        is_error=result.isError,
                                    )
                                ],
                            )
                        )

        if use_history:
            self.history.set(messages)

        logger.debug("Final response:", data=responses)

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
        responses: List[Message] = await self.generate(
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

    async def generate_structured(
        self,
        message,
        response_model: Type[ModelT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "claude-3-5-sonnet-20240620",
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> ModelT:
        # First we invoke the LLM to generate a string response
        # We need to do this in a two-step process because Instructor doesn't
        # know how to invoke MCP tools via call_tool, so we'll handle all the
        # processing first and then pass the final response through Instructor
        response = await self.generate_str(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Next we pass the text through instructor to extract structured data
        client = instructor.from_anthropic(
            Anthropic(api_key=self.context.config.anthropic.api_key),
        )

        # Extract structured data from natural language
        structured_response = client.chat.completions.create(
            model=model,
            response_model=response_model,
            messages=[{"role": "user", "content": response}],
            max_tokens=max_tokens,
        )

        return structured_response

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
