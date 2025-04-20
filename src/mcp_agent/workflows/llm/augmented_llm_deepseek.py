from typing import List
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM,mcp_content_to_openai_content
import json
from loguru import logger
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionToolParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from mcp.types import (
    CallToolRequestParams,
    CallToolRequest,
    CallToolResult,
    TextContent,
)


from mcp_agent.workflows.llm.augmented_llm import (
    RequestParams,
)




class DeepSeekAugmentedLLM(OpenAIAugmentedLLM):
    def get_messages(self, message, params: RequestParams | None = None):
        messages: List[ChatCompletionMessageParam] = []
        system_prompt = self.instruction or params.systemPrompt
        if system_prompt:
            messages.append(
                ChatCompletionSystemMessageParam(role="system", content=system_prompt)
            )

        if params.use_history:
            messages.extend(self.history.get())

        if isinstance(message, str):
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=message)
            )
        elif isinstance(message, list):
            messages.extend(message)
        else:
            messages.append(message)
        return messages
    
    async def generate(self, message, request_params: RequestParams | None = None):
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """
        config = self.context.config
        params = self.get_request_params(request_params)
        openai_client = OpenAI(
            api_key=config.openai.api_key, base_url=config.openai.base_url
        )
       
        i = 0
        try:
            while i< request_params.max_iterations:
                try:
                    if not hasattr(self,'tool_response'):
                        tool_response = await self.aggregator.list_tools()
                        self.tool_response = tool_response
                        break
                except Exception as e:
                    logger.error(f"Error: {e}")
                    i+=1
            available_tools: List[ChatCompletionToolParam] = [
                ChatCompletionToolParam(
                    type="function",
                    function={
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                        "strict": True,
                    },
                )
                for tool in self.tool_response.tools
            ]
            if not available_tools:
                available_tools = None

            responses: List[ChatCompletionMessage] = []
            model = await self.select_model(params)
            base_messages = self.get_messages(message, params)
            messages = base_messages
            for i in range(params.max_iterations):
                arguments = {
                    "model": model,
                    "messages": messages,
                    "stop": params.stopSequences,
                    "tools": available_tools,
                }
                if self._reasoning:
                    arguments = {
                        **arguments,
                        "max_completion_tokens": params.maxTokens,
                        "reasoning_effort": self._reasoning_effort,
                    }
                else:
                    arguments = {**arguments, "max_tokens": params.maxTokens}
                    # if available_tools:
                    #     arguments["parallel_tool_calls"] = params.parallel_tool_calls

                if params.metadata:
                    arguments = {**arguments, **params.metadata}

                self.logger.debug(f"{arguments}")
                self._log_chat_progress(chat_turn=len(messages) // 2, model=model)

                executor_result = await self.executor.execute(
                    openai_client.chat.completions.create, **arguments
                )
                
                response = executor_result[0]

                self.logger.debug(
                    "OpenAI ChatCompletion response:",
                    data=response,
                )

                if isinstance(response, BaseException):
                    self.logger.error(f"Error: {response}")
                    break

                if not response.choices or len(response.choices) == 0:
                    # No response from the model, we're done
                    break

                # TODO: saqadri - handle multiple choices for more complex interactions.
                # Keeping it simple for now because multiple choices will also complicate memory management
                choice = response.choices[0]
                message = choice.message
                responses.append(message)

                converted_message = self.convert_message_to_message_param(
                    message, name=self.name
                )
                messages.append(converted_message)

                if (
                    choice.finish_reason in ["tool_calls", "function_call"]
                    and message.tool_calls
                ):
                    # Execute all tool calls in parallel.
                    tool_tasks = [
                        self.execute_tool_call(tool_call)
                        for tool_call in message.tool_calls
                    ]
                    # Wait for all tool calls to complete.
                    tool_results = await self.executor.execute(*tool_tasks, return_exceptions=True)
                    self.logger.debug(
                        f"Iteration {i}: Tool call results: {str(tool_results) if tool_results else 'None'}"
                    )
                    # Add results (success or failure) to messages.
                    # Use zip to correlate results back to original tool calls for proper error reporting
                    for tool_call, result in zip(message.tool_calls, tool_results):
                        tool_call_id = tool_call.id
                        if isinstance(result, BaseException):
                            error_message = f"Error executing tool {tool_call.function.name}: {str(result)}"
                            self.logger.error(
                                f"Warning: Error during tool execution for call {tool_call_id}: {result}. Appending error message to history."
                            )
                            # Append error message to messages
                            messages.append(
                                ChatCompletionToolMessageParam(
                                    role="tool",
                                    tool_call_id=tool_call_id,
                                    content=error_message,
                                )
                            )
                        elif result is not None:
                            # Append successful tool result to messages
                            messages.append(result)
                        # If result is None, do nothing (tool produced no message content)
                elif choice.finish_reason == "length":
                    # We have reached the max tokens limit
                    self.logger.debug(
                        f"Iteration {i}: Stopping because finish_reason is 'length'"
                    )
                    # TODO: saqadri - would be useful to return the reason for stopping to the caller
                    break
                elif choice.finish_reason == "content_filter":
                    # The response was filtered by the content filter
                    self.logger.debug(
                        f"Iteration {i}: Stopping because finish_reason is 'content_filter'"
                    )
                    # TODO: saqadri - would be useful to return the reason for stopping to the caller
                    break
                elif choice.finish_reason == "stop":
                    self.logger.debug(
                        f"Iteration {i}: Stopping because finish_reason is 'stop'"
                    )
                    break

            if params.use_history:
                self.history.set(messages)

            self._log_chat_finished(model=model)
        finally:
            try:
                openai_client.close()
            except Exception as e:
                logger.error(f"Error closing OpenAI client: {e}")
        return responses


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
            raise Exception(f"Error parsing tool call arguments for '{tool_name}': {str(e)}")

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
                content='\n'.join([mcp_content_to_openai_content(c)['text'] for c in result.content]),
            )

        return None
    
    
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
                        content=[
                            TextContent(
                                text=f"Error: Tool '{request.params.name}' was not allowed to run."
                            )
                        ],
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

            return result
        except Exception as e:
            raise Exception(f"Error executing tool '{request.params.name}': {str(e)}")


