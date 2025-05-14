import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.ai.inference.models import (
    ChatResponseMessage,
    UserMessage,
    ToolMessage,
    ChatCompletionsToolCall,
    FunctionCall,
    TextContentItem,
)
from pydantic import BaseModel

from mcp.types import (
    TextContent,
    SamplingMessage,
)

from mcp_agent.workflows.llm.augmented_llm_azure import (
    AzureAugmentedLLM,
    RequestParams,
    MCPAzureTypeConverter,
)


class TestAzureAugmentedLLM:
    """
    Tests for the AzureAugmentedLLM class.
    """

    @pytest.fixture
    def mock_llm(self, mock_context, mock_aggregator):
        """
        Creates a mock Azure LLM instance with common mocks set up.
        """
        # Setup Azure-specific context attributes
        mock_context.config.azure = MagicMock()
        mock_context.config.azure.api_key = "test_key"
        mock_context.config.azure.endpoint = "https://test-endpoint.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini"
        mock_context.config.azure.default_model = "gpt-4o-mini"
        mock_context.config.azure.api_version = "2025-01-01-preview"
        mock_context.config.azure.credential_scopes = [
            "https://cognitiveservices.azure.com/.default"
        ]

        # Create LLM instance
        llm = AzureAugmentedLLM(name="test", context=mock_context)

        # Apply common mocks
        llm.aggregator = mock_aggregator
        llm.history = MagicMock()
        llm.history.get = MagicMock(return_value=[])
        llm.history.set = MagicMock()
        llm.select_model = AsyncMock(return_value="gpt-4o-mini")
        llm._log_chat_progress = MagicMock()
        llm._log_chat_finished = MagicMock()

        # Mock the Azure client
        llm.azure_client = MagicMock()
        llm.azure_client.complete = AsyncMock()

        return llm

    @pytest.fixture
    def default_usage(self):
        """
        Returns a default usage object for testing.
        """
        return {
            "completion_tokens": 100,
            "prompt_tokens": 150,
            "total_tokens": 250,
        }

    @staticmethod
    def create_text_response(text, finish_reason="stop", usage=None):
        """
        Creates a text response for testing.
        """
        message = ChatResponseMessage(
            role="assistant",
            content=text,
        )

        response = MagicMock()
        response.choices = [
            MagicMock(message=message, finish_reason=finish_reason, index=0)
        ]
        response.id = "chatcmpl-123"
        response.created = 1677858242
        response.model = "gpt-4o-mini"
        response.usage = usage

        return response

    @staticmethod
    def create_tool_use_response(
        tool_name, tool_args, tool_id, finish_reason="tool_calls", usage=None
    ):
        """
        Creates a tool use response for testing.
        """
        function_call = FunctionCall(
            name=tool_name,
            arguments=json.dumps(tool_args),
        )

        tool_call = ChatCompletionsToolCall(
            id=tool_id,
            type="function",
            function=function_call,
        )

        message = ChatResponseMessage(
            role="assistant",
            content=None,
            tool_calls=[tool_call],
        )

        response = MagicMock()
        response.choices = [
            MagicMock(message=message, finish_reason=finish_reason, index=0)
        ]
        response.id = "chatcmpl-123"
        response.created = 1677858242
        response.model = "gpt-4o-mini"
        response.usage = usage

        return response

    # Test 1: Basic Text Generation
    @pytest.mark.asyncio
    async def test_basic_text_generation(
        self, mock_llm: AzureAugmentedLLM, default_usage
    ):
        """
        Tests basic text generation without tools.
        """
        # Setup mock executor
        mock_llm.executor.execute = AsyncMock(
            return_value=[
                self.create_text_response(
                    "This is a test response", usage=default_usage
                )
            ]
        )

        # Call LLM with default parameters
        responses = await mock_llm.generate("Test query")

        # Assertions
        assert len(responses) == 1
        assert responses[0].content == "This is a test response"
        assert mock_llm.executor.execute.call_count == 1

        # Check the first call arguments passed to execute
        first_call_args = mock_llm.executor.execute.call_args_list[0][1]
        assert first_call_args["model"] == "gpt-4o-mini"
        assert isinstance(first_call_args["messages"][0], UserMessage)
        assert first_call_args["messages"][0].content == "Test query"

    # Test 2: Generate String
    @pytest.mark.asyncio
    async def test_generate_str(self, mock_llm: AzureAugmentedLLM, default_usage):
        """
        Tests the generate_str method which returns string output.
        """
        # Setup mock executor
        mock_llm.executor.execute = AsyncMock(
            return_value=[
                self.create_text_response(
                    "This is a test response", usage=default_usage
                )
            ]
        )

        # Call LLM with default parameters
        response_text = await mock_llm.generate_str("Test query")

        # Assertions
        assert response_text == "This is a test response"
        assert mock_llm.executor.execute.call_count == 1

    # Test 3: Generate Structured Output
    @pytest.mark.asyncio
    async def test_generate_structured(
        self, mock_llm: AzureAugmentedLLM, default_usage
    ):
        """
        Tests structured output generation using Azure's JsonSchemaFormat.
        """

        # Define a simple response model
        class TestResponseModel(BaseModel):
            name: str
            value: int

        # Set up the mock for text generation
        mock_llm.executor.execute = AsyncMock(
            return_value=[
                self.create_text_response(
                    '{"name": "Test", "value": 42}', usage=default_usage
                )
            ]
        )

        # Call the method
        result = await mock_llm.generate_structured("Test query", TestResponseModel)

        # Assertions
        assert isinstance(result, TestResponseModel)
        assert result.name == "Test"
        assert result.value == 42

        # Verify metadata was set correctly
        call_args = mock_llm.executor.execute.call_args_list[0][1]
        assert "response_format" in call_args
        assert call_args["response_format"].name == "TestResponseModel"

    # Test 4: With History
    @pytest.mark.asyncio
    async def test_with_history(self, mock_llm: AzureAugmentedLLM, default_usage):
        """
        Tests generation with message history.
        """
        # Setup history
        history_message = UserMessage(content="Previous message")
        mock_llm.history.get = MagicMock(return_value=[history_message])

        # Setup mock executor
        mock_llm.executor.execute = AsyncMock(
            return_value=[
                self.create_text_response("Response with history", usage=default_usage)
            ]
        )

        # Call LLM with history enabled
        responses = await mock_llm.generate(
            "Follow-up query", RequestParams(use_history=True)
        )

        # Assertions
        assert len(responses) == 1

        # Verify history was included in the request
        first_call_args = mock_llm.executor.execute.call_args_list[0][1]
        assert len(first_call_args["messages"]) >= 2
        assert first_call_args["messages"][0] == history_message
        assert isinstance(first_call_args["messages"][1], UserMessage)
        assert first_call_args["messages"][1].content == "Follow-up query"

    # Test 5: Without History
    @pytest.mark.asyncio
    async def test_without_history(self, mock_llm: AzureAugmentedLLM, default_usage):
        """
        Tests generation without message history.
        """
        # Mock the history method to track if it gets called
        mock_history = MagicMock(return_value=[UserMessage(content="Ignored history")])
        mock_llm.history.get = mock_history

        # Setup mock executor
        mock_llm.executor.execute = AsyncMock(
            return_value=[
                self.create_text_response(
                    "Response without history", usage=default_usage
                )
            ]
        )

        # Call LLM with history disabled
        await mock_llm.generate("New query", RequestParams(use_history=False))

        # Assertions
        # Verify history.get() was not called since use_history=False
        mock_history.assert_not_called()

        # Check arguments passed to execute
        call_args = mock_llm.executor.execute.call_args[1]
        # Verify only the user message and  was included
        assert len(call_args["messages"]) == 2
        assert call_args["messages"][0]["content"] == "New query"
        assert call_args["messages"][1]["content"] == "Response without history"

    # Test 6: Tool Usage
    @pytest.mark.asyncio
    async def test_tool_usage(self, mock_llm, default_usage):
        """
        Tests tool usage in the LLM.
        """
        # Create a custom side effect function for execute
        call_count = 0

        async def custom_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # First call is for the regular execute
            if call_count == 1:
                return [
                    self.create_tool_use_response(
                        "test_tool",
                        {"query": "test query"},
                        "tool_123",
                        usage=default_usage,
                    )
                ]
            # Second call is for tool call execution
            elif call_count == 2:
                # This is the tool result passed back to the LLM
                tool_message = ToolMessage(
                    tool_call_id="tool_123",
                    content="Tool result",
                )
                return [tool_message]
            # Third call is for the final response
            else:
                return [
                    self.create_text_response(
                        "Final response after tool use", usage=default_usage
                    )
                ]

        # Setup mocks
        mock_llm.executor.execute = AsyncMock(side_effect=custom_side_effect)

        # Call LLM
        responses = await mock_llm.generate("Test query with tool")

        # Assertions
        assert len(responses) == 3
        assert hasattr(responses[0], "tool_calls")
        assert responses[0].tool_calls is not None
        assert responses[0].tool_calls[0].function.name == "test_tool"
        assert responses[1].tool_call_id == "tool_123"
        assert responses[2].content == "Final response after tool use"

    # Test 7: Tool Error Handling
    @pytest.mark.asyncio
    async def test_tool_error_handling(self, mock_llm, default_usage):
        """
        Tests handling of errors from tool calls.
        """
        # Create a custom side effect function for execute
        call_count = 0

        async def custom_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # First call is for the regular execute
            if call_count == 1:
                return [
                    self.create_tool_use_response(
                        "test_tool",
                        {"query": "test query"},
                        "tool_123",
                        usage=default_usage,
                    )
                ]
            # Second call is for tool call execution - returns the tool execution result
            elif call_count == 2:
                # This is an error tool result passed back to the LLM
                tool_message = ToolMessage(
                    tool_call_id="tool_123",
                    content="Tool execution failed with error",
                )
                return [tool_message]
            # Third call is for the final response
            else:
                return [
                    self.create_text_response(
                        "Response after tool error", usage=default_usage
                    )
                ]

        # Setup mocks
        mock_llm.executor.execute = AsyncMock(side_effect=custom_side_effect)

        # Call LLM
        responses = await mock_llm.generate("Test query with tool error")

        # Assertions
        assert len(responses) == 3
        assert responses[-1].content == "Response after tool error"

    # Test 8: API Error Handling
    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_llm):
        """
        Tests handling of API errors.
        """
        # Setup mock executor to raise an exception
        mock_llm.executor.execute = AsyncMock(return_value=[Exception("API Error")])

        # Call LLM
        responses = await mock_llm.generate("Test query with API error")

        # Assertions
        assert len(responses) == 0  # Should return empty list on error
        assert mock_llm.executor.execute.call_count == 1

    # Test 9: Model Selection
    @pytest.mark.asyncio
    async def test_model_selection(self, mock_llm, default_usage):
        """
        Tests model selection logic.
        """
        # Reset the mock to verify it's called
        mock_llm.select_model = AsyncMock(return_value="gpt-4-turbo")

        # Setup mock executor
        mock_llm.executor.execute = AsyncMock(
            return_value=[
                self.create_text_response("Model selection test", usage=default_usage)
            ]
        )

        # Call LLM with a specific model in request_params
        request_params = RequestParams(model="gpt-4-custom")
        await mock_llm.generate("Test query", request_params)

        # Assertions
        assert mock_llm.select_model.call_count == 1
        # Verify the model parameter was passed
        assert mock_llm.select_model.call_args[0][0].model == "gpt-4-custom"

    # Test 10: Request Parameters Merging
    @pytest.mark.asyncio
    async def test_request_params_merging(self, mock_llm, default_usage):
        """
        Tests merging of request parameters with defaults.
        """
        # Setup mock executor
        mock_llm.executor.execute = AsyncMock(
            return_value=[self.create_text_response("Params test", usage=default_usage)]
        )

        # Create custom request params that override some defaults
        request_params = RequestParams(
            maxTokens=2000, temperature=0.8, max_iterations=5
        )

        # Call LLM with custom params
        await mock_llm.generate("Test query", request_params)

        # Get the merged params that were passed
        merged_params = mock_llm.get_request_params(request_params)

        # Assertions
        assert merged_params.maxTokens == 2000  # Our override
        assert merged_params.temperature == 0.8  # Our override
        assert merged_params.max_iterations == 5  # Our override
        # Should still have default model
        assert merged_params.model == mock_llm.default_request_params.model

    # Test 11: Type Conversion
    def test_type_conversion(self):
        """
        Tests the MCPAzureTypeConverter for converting between Azure and MCP types.
        """
        # Test conversion from Azure message to MCP result
        azure_message = ChatResponseMessage(role="assistant", content="Test content")
        mcp_result = MCPAzureTypeConverter.to_mcp_message_result(azure_message)
        assert mcp_result.role == "assistant"
        assert mcp_result.content.text == "Test content"

        # Test conversion from MCP message param to Azure message param
        mcp_message = SamplingMessage(
            role="user", content=TextContent(type="text", text="Test MCP content")
        )
        azure_param = MCPAzureTypeConverter.from_mcp_message_param(mcp_message)
        assert azure_param.role == "user"

        # Test content conversion
        if isinstance(azure_param.content, str):
            assert azure_param.content == "Test MCP content"
        else:
            assert isinstance(azure_param.content, list)
            assert len(azure_param.content) == 1
            assert isinstance(azure_param.content[0], TextContentItem)
            assert azure_param.content[0].text == "Test MCP content"

    # Test 12: Content Type Handling
    def test_content_type_handling(self):
        """
        Tests handling of different content types in messages.
        """
        # Test text content
        text_content = "Hello world"
        azure_message = ChatResponseMessage(role="assistant", content=text_content)
        converted = MCPAzureTypeConverter.to_mcp_message_result(azure_message)
        assert converted.content.text == text_content

        # Test content items list
        content_items = [
            TextContentItem(text="Hello"),
            TextContentItem(text="World"),
        ]
        message_with_items = UserMessage(content=content_items)
        message_str = AzureAugmentedLLM.message_param_str(None, message_with_items)
        assert "Hello" in message_str
        assert "World" in message_str
