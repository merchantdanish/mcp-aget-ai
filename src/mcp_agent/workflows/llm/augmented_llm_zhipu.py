import json
from typing import Type, Dict
import asyncio

from zhipuai import ZhipuAI
from zhipuai.types.chat.chat_completion import (
    CompletionMessage,
)
from mcp.types import (
    CallToolResult,
    ModelPreferences,
    TextContent,
)

from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    ModelT,
    MCPMessageParam,
    MCPMessageResult,
    ProviderToMCPConverter,
    RequestParams,
    image_url_to_mime_and_base64,
)
from mcp_agent.logging.logger import get_logger


class ZhipuTypeConverter(ProviderToMCPConverter[Dict, CompletionMessage]):
    """Converts between Zhipu AI and MCP types"""

    @classmethod
    def from_mcp_message_result(cls, result: MCPMessageResult) -> CompletionMessage:
        """Convert an MCP message result to a Zhipu AI message result."""
        # This is a simplified implementation - for a complete implementation,
        # we would need to map all MCP message properties to Zhipu AI properties
        content = ""
        for part in result.message.content:
            if part.type == "text":
                content += part.content

        return CompletionMessage(
            role="assistant",
            content=content,
        )

    @classmethod
    def to_mcp_message_result(cls, result: CompletionMessage) -> MCPMessageResult:
        """Convert a Zhipu AI message result to an MCP message result."""
        # This is a simplified implementation - for a complete implementation,
        # we would need to map all Zhipu AI properties to MCP properties

        # Extract content from result based on type
        content = []
        if isinstance(result.content, str):
            content.append(TextContent(type="text", content=result.content))
        elif isinstance(result.content, list):
            # Handle list of content parts
            for part in result.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    content.append(
                        TextContent(type="text", content=part.get("text", ""))
                    )
                # Handle other content types as needed

        # Create a simplified MCPMessageResult
        return MCPMessageResult(
            message={
                "role": "assistant",
                "content": content,
            }
        )

    @classmethod
    def from_mcp_message_param(cls, param: MCPMessageParam) -> Dict:
        """Convert an MCP message parameter to a Zhipu AI message parameter dict."""
        role = param.role

        # Handle different content types
        if param.content and isinstance(param.content, list):
            # Multi-modal content processing
            content_parts = []
            has_non_text = False

            for part in param.content:
                if part.type == "text":
                    content_parts.append({"type": "text", "text": part.content})
                    has_non_text = True
                elif part.type == "image":
                    # Convert image to Zhipu AI format
                    mime_type, base64_data = image_url_to_mime_and_base64(part.url)
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_data}"
                            },
                        }
                    )
                    has_non_text = True

            if has_non_text:
                return {"role": role, "content": content_parts}

        # Text-only content
        content = ""
        if param.content and isinstance(param.content, list):
            for part in param.content:
                if part.type == "text":
                    content += part.content

        return {"role": role, "content": content}

    @classmethod
    def to_mcp_message_param(cls, param: Dict) -> MCPMessageParam:
        """Convert a Zhipu AI message parameter to an MCP message parameter."""
        content_list = []

        # Process content based on type
        if isinstance(param.get("content"), str):
            content_list.append({"type": "text", "content": param.get("content")})
        elif isinstance(param.get("content"), list):
            for item in param.get("content"):
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        content_list.append(
                            {"type": "text", "content": item.get("text", "")}
                        )
                    # Handle other content types as needed

        return MCPMessageParam(role=param.get("role"), content=content_list)

    @classmethod
    def from_mcp_tool_result(cls, result: CallToolResult, tool_use_id: str) -> Dict:
        """Convert an MCP tool result to a Zhipu AI tool message parameter dict."""
        # Ensure result can be serialized as JSON string
        try:
            # Try to extract output content
            if hasattr(result, "output") and result.output is not None:
                content = json.dumps(result.output, ensure_ascii=False)
            elif hasattr(result, "content"):
                # Try to process content field
                if isinstance(result.content, list):
                    # If content is a list, extract all text content
                    text_parts = []
                    for part in result.content:
                        if hasattr(part, "text"):
                            text_parts.append(part.text)
                        elif isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                    content = json.dumps(
                        {"result": " ".join(text_parts)}, ensure_ascii=False
                    )
                else:
                    content = json.dumps(
                        {"result": str(result.content)}, ensure_ascii=False
                    )
            else:
                # Fallback: convert entire result object to string
                content = json.dumps({"result": str(result)}, ensure_ascii=False)
        except Exception as e:
            # If serialization fails, provide a default message
            content = json.dumps(
                {"error": f"Unable to serialize tool result: {str(e)}"},
                ensure_ascii=False,
            )

        return {
            "role": "tool",
            "tool_call_id": tool_use_id,
            "content": content,
        }


class ZhipuAugmentedLLM(AugmentedLLM[Dict, CompletionMessage]):
    """
    An implementation of AugmentedLLM that uses Zhipu AI's models with support for
    function calling and multi-modal inputs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=ZhipuTypeConverter, **kwargs)

        self.provider = "ZhipuAI"
        # Initialize logger with name if available
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

        self.model_preferences = self.model_preferences or ModelPreferences(
            costPriority=0.3,
            speedPriority=0.4,
            intelligencePriority=0.3,
        )

        # Initialize tool result history records
        self.tool_results_history = []

        # Get default model from config if available
        chosen_model = "glm-4-flashx-250414"  # Fallback default

        if (
            self.context
            and self.context.config
            and hasattr(self.context.config, "zhipu")
        ):
            if hasattr(self.context.config.zhipu, "default_model"):
                chosen_model = self.context.config.zhipu.default_model

        self.default_request_params = self.default_request_params or RequestParams(
            model=chosen_model,
            modelPreferences=self.model_preferences,
            maxTokens=4096,
            systemPrompt=self.instruction,
            parallel_tool_calls=False,
            max_iterations=10,
            use_history=True,
        )

    @classmethod
    def convert_message_to_message_param(
        cls, message: CompletionMessage, **kwargs
    ) -> Dict:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        assistant_message_params = {
            "role": "assistant",
            **kwargs,
        }
        if message.content is not None:
            assistant_message_params["content"] = message.content

        # Fix tool call serialization issues
        if hasattr(message, "tool_calls") and message.tool_calls is not None:
            # Convert tool_calls to serializable dict list
            tool_calls_serializable = []
            for tc in message.tool_calls:
                tool_call_dict = {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                tool_calls_serializable.append(tool_call_dict)

            assistant_message_params["tool_calls"] = tool_calls_serializable

        return assistant_message_params

    async def generate(
        self,
        message,
        request_params: RequestParams | None = None,
        force_tools: bool = False,
    ):
        """
        Process a query using an LLM and available tools.
        This implementation uses Zhipu AI's ChatCompletion as the LLM.

        Args:
            message: The input message or messages
            request_params: Configuration parameters for the request
            force_tools: If True, modify the prompt to force tool usage
        """
        config = self.context.config
        zhipu_client = ZhipuAI(api_key=config.zhipu.api_key)

        messages = []
        params = self.get_request_params(request_params)

        if params.use_history:
            messages.extend(self.history.get())

        system_prompt = self.instruction or params.systemPrompt

        if system_prompt and len(messages) == 0:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        # Prepare tools for function calling if available
        response = await self.aggregator.list_tools()
        available_tools = []
        if response.tools:
            available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in response.tools
            ]

        responses = []
        model = await self.select_model(params)

        for i in range(params.max_iterations):
            arguments = {
                "model": model,
                "messages": messages,
                "max_tokens": params.maxTokens,
            }

            # Add tools if available
            if available_tools:
                arguments["tools"] = available_tools

                if force_tools:
                    arguments["tool_choice"] = "auto"

                    if "temperature" not in arguments and params.temperature is None:
                        arguments["temperature"] = 0.1
                    elif params.temperature is not None:
                        arguments["temperature"] = params.temperature

            # Add stop sequences if available
            if params.stopSequences:
                arguments["stop"] = params.stopSequences

            # Add extra parameters if available in metadata
            if params.metadata:
                arguments["extra_body"] = params.metadata

            self.logger.debug(f"Zhipu AI request arguments: {arguments}")
            self._log_chat_progress(chat_turn=len(messages) // 2, model=model)

            try:
                # Call in synchronous mode
                completion_response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: zhipu_client.chat.completions.create(**arguments)
                )

                self.logger.debug(
                    "Zhipu AI Completion response:",
                    data=completion_response,
                )

                if isinstance(completion_response, BaseException):
                    self.logger.error(f"Error: {completion_response}")
                    break

                if (
                    not completion_response.choices
                    or len(completion_response.choices) == 0
                ):
                    # No response from the model, we're done
                    break

                choice = completion_response.choices[0]
                message = choice.message
                responses.append(message)

                # Format message for next iteration or tool calls
                converted_message = self.convert_message_to_message_param(message)
                messages.append(converted_message)

                # If tools are being used and tool_calls are present
                if hasattr(message, "tool_calls") and message.tool_calls:
                    # Execute all tool calls
                    tool_tasks = [
                        self.execute_tool_call(tool_call)
                        for tool_call in message.tool_calls
                    ]
                    # Wait for all tool calls to complete
                    tool_results = await self.executor.execute(*tool_tasks)

                    self.logger.debug(
                        f"Iteration {i}: Tool call results: {str(tool_results) if tool_results else 'None'}"
                    )

                    # Add non-None results to messages
                    for result in tool_results:
                        if isinstance(result, BaseException):
                            self.logger.error(
                                f"Warning: Unexpected error during tool execution: {result}. Continuing..."
                            )
                            continue
                        if result is not None:
                            messages.append(result)
                elif hasattr(message, "function_call") and message.function_call:
                    function_call = message.function_call
                    tool_call = type(
                        "ToolCall",
                        (),
                        {
                            "id": str(i),
                            "function": type(
                                "Function",
                                (),
                                {
                                    "name": function_call.get("name", ""),
                                    "arguments": function_call.get("arguments", "{}"),
                                },
                            ),
                        },
                    )

                    result = await self.execute_tool_call(tool_call)
                    if result is not None:
                        messages.append(result)
            except Exception as e:
                self.logger.error(f"Error during Zhipu AI request: {e}")
                break

        self._log_chat_finished(model=model)
        return responses

    async def generate_str(
        self,
        message,
        request_params: RequestParams | None = None,
        force_tools: bool = True,  # Enable forced tool calls by default
    ):
        """Request an LLM generation and return the string representation of the result"""
        responses = await self.generate(
            message, request_params, force_tools=force_tools
        )
        if not responses:
            return ""

        # Extract text content from the final response
        return self.message_str(responses[-1])

    async def generate_structured(
        self,
        message,
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
        force_tools: bool = True,  # Enable forced tool calls by default
    ) -> ModelT:
        """Generate a structured response conforming to the specified model."""
        # First generate a string response
        response_str = await self.generate_str(
            message, request_params, force_tools=force_tools
        )

        # Then convert to the structured model using pydantic parsing
        try:
            # Simple parsing approach - in a real implementation, more
            # sophisticated parsing might be needed
            return response_model.parse_raw(response_str)
        except Exception as e:
            self.logger.error(f"Error parsing structured response: {e}")
            # If parsing fails, try to create an empty instance
            return response_model()

    async def execute_tool_call(
        self,
        tool_call,
    ) -> Dict | None:
        """Execute a tool call from the LLM and return the result."""
        if not tool_call:
            return None

        tool_call_id = tool_call.id
        function_name = tool_call.function.name

        # Parse function arguments from the tool call
        function_args = {}
        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing function arguments: {e}")
            return None

        self.logger.info(
            f"Executing tool call: {function_name}", extra={"data": function_args}
        )

        # Get available tools
        tools_response = await self.aggregator.list_tools()
        available_tools = (
            {tool.name: tool for tool in tools_response.tools}
            if tools_response.tools
            else {}
        )

        # Tool name processing logic - check if server prefix needs to be added
        actual_tool_name = function_name
        if function_name not in available_tools:
            # Check if it's missing a server prefix
            server_prefixed_names = [
                name
                for name in available_tools.keys()
                if name.endswith(f"_{function_name}")
            ]
            if server_prefixed_names:
                # Use the first matching tool found
                actual_tool_name = server_prefixed_names[0]
                self.logger.info(
                    f"Tool name '{function_name}' mapped to '{actual_tool_name}'"
                )

        # Call the tool
        try:
            # Use the processed tool name for calling
            self.logger.info(
                f"Calling tool '{actual_tool_name}' with arguments: {function_args}"
            )
            result = await self.aggregator.call_tool(
                name=actual_tool_name, arguments=function_args
            )

            if result:
                # Create tool response message
                tool_result = self.type_converter.from_mcp_tool_result(
                    result, tool_call_id
                )

                # Save tool call results to history
                if tool_result:
                    self.tool_results_history.append(tool_result)

                return tool_result
            else:
                self.logger.warning(f"Tool '{actual_tool_name}' returned no result")
        except Exception as e:
            self.logger.error(f"Error executing tool call '{actual_tool_name}': {e}")
        return None

    def message_param_str(self, message: Dict) -> str:
        """Convert a message parameter to a string representation."""
        if isinstance(message.get("content"), str):
            return message.get("content", "")
        elif isinstance(message.get("content"), list):
            # Extract text content from multi-modal content
            text_parts = []
            for part in message.get("content", []):
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            return " ".join(text_parts)
        return ""

    def message_str(self, message: CompletionMessage) -> str:
        """Convert a message to a string representation."""
        # First check if the message content is valid
        if message.content:
            if isinstance(message.content, str):
                return message.content
            elif isinstance(message.content, list):
                # Extract text content from multi-modal content
                text_parts = []
                for part in message.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                return " ".join(text_parts)

        # If message content is empty but has tool call history, try to process and format the most recent tool result
        if hasattr(self, "tool_results_history") and self.tool_results_history:
            # Get the last tool result
            last_tool_result = self.tool_results_history[-1]

            # Process based on tool call result type
            if isinstance(last_tool_result, dict) and "content" in last_tool_result:
                try:
                    return self._format_tool_result(last_tool_result["content"])
                except Exception as e:
                    self.logger.error(f"Error formatting tool result: {e}")
                    # Return original content in case of formatting failure
                    return f"Tool result: {last_tool_result['content']}"

        return ""

    def _format_tool_result(self, result_content: str) -> str:
        """Format tool result content based on its structure and type."""
        try:
            # Try to parse JSON content
            parsed = json.loads(result_content)

            # If it's a standard "result" wrapper format
            if isinstance(parsed, dict) and "result" in parsed:
                inner_content = parsed["result"]

                # Try to further parse nested JSON
                try:
                    inner_json = json.loads(inner_content)

                    # Generic structured result handling - simple JSON formatting
                    return json.dumps(inner_json, indent=2, ensure_ascii=False)

                except (json.JSONDecodeError, TypeError):
                    # If inner content is not JSON, return directly
                    return inner_content

            # For other JSON structures, return formatted JSON
            if isinstance(parsed, dict) or isinstance(parsed, list):
                return json.dumps(parsed, indent=2, ensure_ascii=False)

            return str(parsed)

        except (json.JSONDecodeError, TypeError):
            # If not JSON format, return original content
            return result_content
