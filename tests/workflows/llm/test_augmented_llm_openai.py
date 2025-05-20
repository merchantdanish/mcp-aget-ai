from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field
import asyncio

from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams


class TestOpenAIAugmentedLLM:
    """
    Tests for the OpenAIAugmentedLLM class.
    """

    @pytest.fixture
    def mock_llm(self):
        """
        Creates a mock LLM instance with common mocks set up.
        """
        # Setup mock objects
        mock_context = MagicMock()
        mock_context.config.openai = MagicMock()
        mock_context.config.openai.default_model = "gpt-4o"
        mock_context.config.openai.api_key = "test_key"
        mock_context.config.openai.base_url = "https://api.openai.com/v1"
        mock_context.config.openai.http_client = None

        # Create LLM instance
        llm = OpenAIAugmentedLLM(name="test", context=mock_context)

        # Setup common mocks
        llm.aggregator = MagicMock()
        llm.aggregator.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        llm.history = MagicMock()
        llm.history.get = MagicMock(return_value=[])
        llm.history.set = MagicMock()
        llm.select_model = AsyncMock(return_value="gpt-4o")
        llm._log_chat_progress = MagicMock()
        llm._log_chat_finished = MagicMock()

        return llm

    @pytest.mark.asyncio
    async def test_generate_structured(self, mock_llm):
        """
        Test that generate_structured method works correctly.
        """

        # Define our test model
        class MovieRecommendation(BaseModel):
            title: str = Field(description="The title of the movie")
            year: int = Field(description="The year the movie was released")
            genres: list[str] = Field(description="List of genres for the movie")
            why_recommended: str = Field(
                description="Explanation for why this movie is recommended"
            )

        # Mock the generate_str method which is called by generate_structured
        mock_llm.generate_str = AsyncMock(
            return_value="I recommend watching Blade Runner from 1982"
        )

        # Mock the instructor client
        with patch("instructor.from_openai") as mock_instructor:
            # Setup the instructor mock
            mock_async_client = MagicMock()
            mock_async_client.chat.completions.create = AsyncMock(
                return_value=MovieRecommendation(
                    title="Blade Runner",
                    year=1982,
                    genres=["Sci-Fi", "Thriller"],
                    why_recommended="Classic sci-fi noir with amazing visuals",
                )
            )
            mock_instructor.return_value = mock_async_client

            # Call generate_structured
            result = await mock_llm.generate_structured(
                message="Recommend a sci-fi movie from the 1980s",
                response_model=MovieRecommendation,
                request_params=RequestParams(model="gpt-4o"),
            )

            # Verify the result
            assert isinstance(result, MovieRecommendation)
            assert result.title == "Blade Runner"
            assert result.year == 1982
            assert "Sci-Fi" in result.genres
            assert result.why_recommended is not None

            # Verify methods were called correctly
            mock_llm.generate_str.assert_called_once()
            mock_instructor.assert_called_once()
            mock_async_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_structured_retry(self, mock_llm):
        """
        Test that generate_structured retries with JSON mode on failure.
        """

        # Define our test model
        class MovieRecommendation(BaseModel):
            title: str = Field(description="The title of the movie")
            year: int = Field(description="The year the movie was released")
            genres: list[str] = Field(description="List of genres for the movie")
            why_recommended: str = Field(
                description="Explanation for why this movie is recommended"
            )

        # Mock the generate_str method
        mock_llm.generate_str = AsyncMock(
            return_value="I recommend watching The Terminator from 1984"
        )

        # Create success response
        successful_response = MovieRecommendation(
            title="The Terminator",
            year=1984,
            genres=["Sci-Fi", "Action"],
            why_recommended="Groundbreaking sci-fi action film",
        )

        # For this test we'll use a different approach without unused variables

        # For this test, we'll simply patch the method we want to test
        # and verify it behaves correctly when a retry is needed

        # First, mock the generate_str method to return a simple string
        mock_llm.generate_str = AsyncMock(return_value="I recommend Aliens (1986)")

        # Track whether retry logic was triggered
        retry_triggered = [False]

        # Create a patched version of from_openai that simulates a retry

        # Create two clients - one that will fail, and one that will succeed
        initial_client = MagicMock()
        retry_client = MagicMock()

        # Configure the initial client to fail
        async def initial_create(*args, **kwargs):
            # Simulate the failure that should trigger retry
            retry_triggered[0] = True
            # The actual class will catch InstructorRetryException
            # We use RuntimeError as it's standard and doesn't need special init
            raise RuntimeError("Simulated parse error that should trigger retry")

        initial_client.chat = MagicMock()
        initial_client.chat.completions = MagicMock()
        initial_client.chat.completions.create = AsyncMock(side_effect=initial_create)

        # Configure the retry client to succeed
        retry_client.chat = MagicMock()
        retry_client.chat.completions = MagicMock()
        retry_client.chat.completions.create = AsyncMock(
            return_value=successful_response
        )

        # Mock instructor.from_openai to return different clients
        def mock_from_openai(client, mode=None):
            # Return retry client if JSON mode is specified (indicating retry)
            if mode and str(mode).endswith("JSON"):
                return retry_client
            return initial_client

        # Apply the patches
        with patch("instructor.from_openai", side_effect=mock_from_openai):
            # Patch the exception handling by replacing the exception class
            with patch("instructor.exceptions.InstructorRetryException", RuntimeError):
                # Call generate_structured
                result = await mock_llm.generate_structured(
                    message="Recommend a sci-fi movie from the 1980s",
                    response_model=MovieRecommendation,
                    request_params=RequestParams(model="gpt-4o"),
                )

            # Verify the result
            assert isinstance(result, MovieRecommendation)
            assert result.title == "The Terminator"
            assert result.year == 1984
            assert "Sci-Fi" in result.genres
            assert result.why_recommended is not None

            # Verify retry was triggered
            assert retry_triggered[0], "Retry should have been triggered"

            # Verify methods were called correctly
            mock_llm.generate_str.assert_called_once()
            # Verify that initial_client.chat.completions.create was called
            initial_client.chat.completions.create.assert_called_once()
            # Verify that retry_client.chat.completions.create was called
            retry_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_structured_non_blocking(self, mock_llm):
        """
        Test that generate_structured doesn't block other async tasks.
        """

        # Define our test model
        class MovieRecommendation(BaseModel):
            title: str = Field(description="The title of the movie")
            year: int = Field(description="The year the movie was released")
            genres: list[str] = Field(description="List of genres for the movie")
            why_recommended: str = Field(
                description="Explanation for why this movie is recommended"
            )

        # Create a slow generate_str method
        async def slow_generate_str(*args, **kwargs):
            await asyncio.sleep(0.2)  # Add delay
            return "I recommend watching Aliens from 1986"

        mock_llm.generate_str = AsyncMock(side_effect=slow_generate_str)

        # Create a slow instructor client
        with patch("instructor.from_openai") as mock_instructor:
            # Setup a slow instructor client
            async def slow_create(*args, **kwargs):
                await asyncio.sleep(0.2)  # Add delay
                return MovieRecommendation(
                    title="Aliens",
                    year=1986,
                    genres=["Sci-Fi", "Action", "Horror"],
                    why_recommended="One of the best sci-fi action sequels",
                )

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=slow_create)
            mock_instructor.return_value = mock_client

            # Create a faster task
            async def fast_async_func():
                await asyncio.sleep(0.1)
                return "Fast task completed"

            # Run both tasks concurrently
            structured_task = asyncio.create_task(
                mock_llm.generate_structured(
                    message="Recommend a sci-fi movie from the 1980s",
                    response_model=MovieRecommendation,
                    request_params=RequestParams(model="gpt-4o"),
                )
            )
            fast_task = asyncio.create_task(fast_async_func())

            # Wait for both tasks
            done, pending = await asyncio.wait(
                [structured_task, fast_task], return_when=asyncio.ALL_COMPLETED
            )

            # Get results
            structured_result = structured_task.result()
            fast_result = fast_task.result()

            # Verify results
            assert isinstance(structured_result, MovieRecommendation)
            assert structured_result.title == "Aliens"
            assert structured_result.year == 1986
            assert "Sci-Fi" in structured_result.genres

            assert fast_result == "Fast task completed"
