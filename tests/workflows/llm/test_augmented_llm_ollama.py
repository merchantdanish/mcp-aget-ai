from unittest.mock import AsyncMock, MagicMock, patch

import instructor
import pytest
from pydantic import BaseModel

from mcp_agent.workflows.llm.augmented_llm_ollama import (
    OllamaAugmentedLLM,
)


class TestOllamaAugmentedLLM:
    """
    Tests for the OllamaAugmentedLLM class.
    Focuses only on Ollama-specific functionality since OllamaAugmentedLLM
    inherits from OpenAIAugmentedLLM, which has its own test suite.
    """

    @pytest.fixture
    def mock_llm(self, mock_context, mock_aggregator):
        """
        Creates a mock Ollama LLM instance with common mocks set up.
        """
        # Setup OpenAI/Ollama-specific context attributes
        mock_context.config.openai = MagicMock()
        mock_context.config.openai.api_key = "test_api_key"
        mock_context.config.openai.base_url = "http://localhost:11434/v1"
        mock_context.config.openai.default_model = "llama3.2:3b"

        # Create LLM instance
        llm = OllamaAugmentedLLM(name="test", context=mock_context)

        # Apply common mocks
        llm.aggregator = mock_aggregator
        llm.select_model = AsyncMock(return_value="llama3.2:3b")

        return llm

    @pytest.fixture
    def mock_context_factory(self):
        def factory():
            mock_context = MagicMock()
            mock_context.config = MagicMock()
            # mock_context.config.openai will be set by tests as needed
            return mock_context

        return factory

    # Test 1: Initialization
    def test_initialization(
        self, mock_context_factory
    ):  # Use a factory for clean contexts
        """
        Tests that the OllamaAugmentedLLM constructor properly sets up defaults
        when the OpenAI parent class does NOT find a default model in its own config.
        """

        # --- Test with Ollama's internal default model ("llama3.2:3b") ---
        # Create a context where config.openai does not have 'default_model'
        context_no_openai_default = mock_context_factory()
        # Define what attributes config.openai is expected to have by the parent, excluding 'default_model'
        # Add any other attributes OpenAIAugmentedLLM might access from config.openai during init
        openai_spec = [
            "api_key",
            "base_url",
            "reasoning_effort",
        ]  # Adjust as per OpenAIAugmentedLLM

        # Ensure the basic structure required by OpenAIAugmentedLLM for config.openai
        # but specifically make 'default_model' not present for hasattr checks.
        # We also need to provide other attributes OpenAIAugmentedLLM might access.
        mock_openai_config = MagicMock(spec=openai_spec)
        mock_openai_config.api_key = "test_api_key"

        context_no_openai_default.config.openai = mock_openai_config

        llm_default = OllamaAugmentedLLM(
            name="test_ollama_default", context=context_no_openai_default
        )

        assert llm_default.provider == "Ollama"
        # The default_model from OllamaAugmentedLLM's init ("llama3.2:3b") should be used
        # because context_no_openai_default.config.openai.default_model is not found by hasattr.
        assert llm_default.default_request_params.model == "llama3.2:3b"

        # --- Test with custom default_model passed to OllamaAugmentedLLM ---
        # Re-use or create a similar clean context
        context_no_openai_default_for_custom = mock_context_factory()
        mock_openai_config_for_custom = MagicMock(spec=openai_spec)
        mock_openai_config_for_custom.api_key = "test_api_key"
        # mock_openai_config_for_custom.reasoning_effort = "medium"
        context_no_openai_default_for_custom.config.openai = (
            mock_openai_config_for_custom
        )

        llm_custom = OllamaAugmentedLLM(
            name="test_ollama_custom",
            context=context_no_openai_default_for_custom,
            default_model="mistral:7b",
        )
        assert llm_custom.provider == "Ollama"
        # The custom default_model ("mistral:7b") passed to OllamaAugmentedLLM should be used.
        assert llm_custom.default_request_params.model == "mistral:7b"

        # --- Test scenario: OpenAI context config *does* have a default model ---
        # This would test that the OpenAI parent's config takes precedence.
        context_with_openai_default = mock_context_factory()
        # Now, config.openai *will* have a default_model.
        # No need for spec here if we want MagicMock's default behavior of creating attrs on access.
        context_with_openai_default.config.openai = MagicMock()
        context_with_openai_default.config.openai.api_key = "test_api_key"
        context_with_openai_default.config.openai.default_model = (
            "openai-parent-default:v1"  # Must be a string
        )

        llm_parent_override = OllamaAugmentedLLM(
            name="test_parent_override", context=context_with_openai_default
        )
        assert llm_parent_override.provider == "Ollama"
        # The default_model from context_with_openai_default.config.openai should take precedence.
        assert (
            llm_parent_override.default_request_params.model
            == "openai-parent-default:v1"
        )

    # Test 2: Generate Structured Method - JSON Mode
    @pytest.mark.asyncio
    async def test_generate_structured_json_mode(self, mock_llm):
        """
        Tests that the generate_structured method uses JSON mode for Instructor.
        """

        # Define a simple response model
        class TestResponseModel(BaseModel):
            name: str
            value: int

        # Mock the generate_str method
        mock_llm.generate_str = AsyncMock(return_value="name: Test, value: 42")

        # Mock OpenAI client creation and instructor
        with (
            patch("openai.OpenAI"),
            patch("instructor.from_openai") as mock_instructor,
        ):
            # Setup instructor mock
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = TestResponseModel(
                name="Test", value=42
            )
            mock_instructor.return_value = mock_client

            # Call the method
            await mock_llm.generate_structured("Test query", TestResponseModel)

            # Verify instructor was called with mode=JSON
            # This is different from OpenAIAugmentedLLM which doesn't specify mode
            mock_instructor.assert_called_once()
            assert mock_instructor.call_args[1]["mode"] == instructor.Mode.JSON

    # Test 3: OpenAI Client Initialization
    @pytest.mark.asyncio
    async def test_openai_client_initialization(
        self, mock_context_factory
    ):  # Use factory
        """
        Tests that the OpenAI client used by instructor is initialized with the correct
        api_key and base_url for connecting to Ollama's API.
        """
        # Create a context and ensure config.openai.default_model is a string
        # because OpenAIAugmentedLLM's __init__ will access it.
        context = mock_context_factory()
        context.config.openai = MagicMock()  # Allow attributes to be created on access
        context.config.openai.api_key = (
            "test_key_for_instructor"  # This key is for instructor's OpenAI client
        )
        context.config.openai.base_url = (
            "http://localhost:11434/v1"  # For Ollama's instructor client
        )
        context.config.openai.default_model = "some-valid-string-model"

        with (
            patch("openai.OpenAI") as mock_openai_client_constructor,
            patch("instructor.from_openai") as mock_instructor_from_openai,
        ):
            mock_instructor_instance = MagicMock()
            mock_instructor_instance.chat.completions.create.return_value = (
                MagicMock()
            )  # Mock the response
            mock_instructor_from_openai.return_value = mock_instructor_instance

            # Create LLM. Its __init__ will use context.config.openai.default_model
            llm = OllamaAugmentedLLM(name="test_instructor_client", context=context)

            # Mock generate_str as it's called by generate_structured
            llm.generate_str = AsyncMock(return_value="text response from llm")
            # Mock select_model as it's called by generate_structured to determine model for instructor
            llm.select_model = AsyncMock(return_value="selected-model-for-instructor")

            class TestResponseModel(BaseModel):
                name: str

            await llm.generate_structured("query for structured", TestResponseModel)

            # Verify OpenAI client for instructor was initialized with correct args
            mock_openai_client_constructor.assert_called_once()
            constructor_args, constructor_kwargs = (
                mock_openai_client_constructor.call_args
            )
            assert constructor_kwargs["api_key"] == "test_key_for_instructor"
            assert constructor_kwargs["base_url"] == "http://localhost:11434/v1"

            # Verify instructor.from_openai was called with the client
            mock_instructor_from_openai.assert_called_once_with(
                mock_openai_client_constructor.return_value,  # The client instance
                mode=instructor.Mode.JSON,
            )

            # Verify instructor's client.chat.completions.create was called correctly
            mock_instructor_instance.chat.completions.create.assert_called_once()
            instructor_call_kwargs = (
                mock_instructor_instance.chat.completions.create.call_args.kwargs
            )
            assert instructor_call_kwargs["model"] == "selected-model-for-instructor"
            assert instructor_call_kwargs["response_model"] == TestResponseModel
            assert instructor_call_kwargs["messages"] == [
                {"role": "user", "content": "text response from llm"}
            ]
