import json
from typing import List, Type

import instructor
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from mcp.types import (
    CallToolRequestParams,
    CallToolRequest,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)

from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, ModelT
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class OpenAIAugmentedLLM(
    AugmentedLLM[ChatCompletionMessageParam, ChatCompletionMessage]
):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    This implementation uses OpenAI's ChatCompletion as the LLM.
    """

    @classmethod
    async def convert_message_to_message_param(
        cls, message: ChatCompletionMessage, **kwargs
    ) -> ChatCompletionMessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            content=message.content,
            tool_calls=message.tool_calls,
            audio=message.audio,
            refusal=message.refusal,
            **kwargs,
        )

    async def generate(
        self,
        message,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "gpt-4o",
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ):
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """
        config = self.context.config
        openai_client = OpenAI(api_key=config.openai.api_key)
        messages: List[ChatCompletionMessageParam] = []

        if self.instruction:
            messages.append(
                ChatCompletionSystemMessageParam(
                    role="system", content=self.instruction
                )
            )

        if use_history:
            messages.extend(self.history.get())

        if isinstance(message, str):
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=message)
            )
        elif isinstance(message, list):
            messages.extend(message)
        else:
            messages.append(message)

        response = await self.aggregator.list_tools()
        available_tools: List[ChatCompletionToolParam] = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                    # TODO: saqadri - determine if we should specify "strict" to True by default
                },
            )
            for tool in response.tools
        ]
        if not available_tools:
            available_tools = None

        responses: List[ChatCompletionMessage] = []

        for i in range(max_iterations):
            arguments = {
                "model": model,
                "messages": messages,
                "stop": stop_sequences,
                "tools": available_tools,
                "max_tokens": max_tokens,
            }

            if available_tools:
                arguments["tools"] = available_tools
                arguments["parallel_tool_calls"] = parallel_tool_calls

            logger.debug(
                f"Iteration {i}: Calling OpenAI ChatCompletion with messages:",
                data=messages,
            )

            executor_result = await self.executor.execute(
                openai_client.chat.completions.create, **arguments
            )

            response = executor_result[0]

            logger.debug(
                f"Iteration {i}: OpenAI ChatCompletion response:",
                data=response,
            )

            if not response.choices or len(response.choices) == 0:
                # No response from the model, we're done
                break

            # TODO: saqadri - handle multiple choices for more complex interactions.
            # Keeping it simple for now because multiple choices will also complicate memory management
            choice = response.choices[0]
            messages.append(choice.message)
            responses.append(choice.message)

            if choice.finish_reason == "stop":
                # We have reached the end of the conversation
                logger.debug(f"Iteration {i}: Stopping because finish_reason is 'stop'")
                break
            elif choice.finish_reason == "length":
                # We have reached the max tokens limit
                logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'length'"
                )
                # TODO: saqadri - would be useful to return the reason for stopping to the caller
                break
            elif choice.finish_reason == "content_filter":
                # The response was filtered by the content filter
                logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'content_filter'"
                )
                # TODO: saqadri - would be useful to return the reason for stopping to the caller
                break
            else:  #  choice.finish_reason in ["tool_calls", "function_call"]
                message = choice.message

                if message.content:
                    messages.append(
                        self.convert_message_to_message_param(message, name=self.name)
                    )

                if message.tool_calls:
                    # Execute all tool calls in parallel
                    tool_tasks = [
                        self.execute_tool_call(tool_call)
                        for tool_call in message.tool_calls
                    ]

                    # Wait for all tool calls to complete
                    tool_results = await self.executor.execute(*tool_tasks)
                    logger.debug(
                        f"Iteration {i}: Tool call results: {str(tool_results) if tool_results else 'None'}"
                    )

                    # Add non-None results to messages
                    for result in tool_results:
                        if isinstance(result, BaseException):
                            # Handle any unexpected exceptions during parallel execution
                            logger.error(
                                f"Warning: Unexpected error during tool execution: {result}. Continuing..."
                            )
                            continue
                        if result is not None:
                            messages.append(result)
        if use_history:
            self.history.set(messages)

        return responses

    async def generate_str(
        self,
        message,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "gpt-4o",
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ):
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
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
            content = response.content
            if not content:
                continue

            if isinstance(content, str):
                final_text.append(content)
                continue

        return "\n".join(final_text)

    async def generate_structured(
        self,
        message,
        response_model: Type[ModelT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
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
            model=model or "gpt-4o",
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Next we pass the text through instructor to extract structured data
        client = instructor.from_openai(
            OpenAI(api_key=self.context.config.openai.api_key),
            mode=instructor.Mode.TOOLS_STRICT,
        )

        # Extract structured data from natural language
        structured_response = client.chat.completions.create(
            model=model or "gpt-4o",
            response_model=response_model,
            messages=[
                {"role": "user", "content": response},
            ],
        )

        return structured_response

    async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest):
        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ):
        return result

    async def execute_tool_call(
        self,
        tool_call: ChatCompletionToolParam,
    ) -> ChatCompletionToolMessageParam | None:
        """
        Execute a single tool call and return the result message.
        Returns None if there's no content to add to messages.
        """
        tool_name = tool_call.function.name
        tool_args_str = tool_call.function.arguments
        tool_call_id = tool_call.id
        tool_args = {}

        try:
            if tool_args_str:
                tool_args = json.loads(tool_args_str)
        except json.JSONDecodeError as e:
            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call_id,
                content=f"Invalid JSON provided in tool call arguments for '{tool_name}'. Failed to load JSON: {str(e)}",
            )

        tool_call_request = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=tool_name, arguments=tool_args),
        )

        result = await self.call_tool(
            request=tool_call_request, tool_call_id=tool_call_id
        )

        if result.content:
            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call_id,
                content=[mcp_content_to_openai_content(c) for c in result.content],
            )

        return None

    def message_param_str(self, message: ChatCompletionMessageParam) -> str:
        """Convert an input message to a string representation."""
        if message.get("content"):
            content = message["content"]
            if isinstance(content, str):
                return content
            else:  # content is a list
                final_text: List[str] = []
                for part in content:
                    text_part = part.get("text")
                    if text_part:
                        final_text.append(str(text_part))
                    else:
                        final_text.append(str(part))

                return "\n".join(final_text)

        return str(message)

    def message_str(self, message: ChatCompletionMessage) -> str:
        """Convert an output message to a string representation."""
        content = message.content
        if content:
            return content

        return str(message)


def mcp_content_to_openai_content(
    content: TextContent | ImageContent | EmbeddedResource,
) -> ChatCompletionContentPartTextParam:
    if isinstance(content, TextContent):
        return ChatCompletionContentPartTextParam(type="text", text=content.text)
    elif isinstance(content, ImageContent):
        # Best effort to convert an image to text
        return ChatCompletionContentPartTextParam(
            type="text", text=f"{content.mimeType}:{content.data}"
        )
    elif isinstance(content, EmbeddedResource):
        if isinstance(content.resource, TextResourceContents):
            return ChatCompletionContentPartTextParam(
                type="text", text=content.resource.text
            )
        else:  # BlobResourceContents
            return ChatCompletionContentPartTextParam(
                type="text", text=f"{content.resource.mimeType}:{content.resource.blob}"
            )
    else:
        # Last effort to convert the content to a string
        return ChatCompletionContentPartTextParam(type="text", text=str(content))
