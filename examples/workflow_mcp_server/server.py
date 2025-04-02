import asyncio
import os
from typing import Dict, Any, Optional
from mcp.server.helpers.stdio import stdio_server

from mcp_agent.app import MCPApp
from mcp_agent.app_server import create_mcp_server_for_app
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.executor.workflow import Workflow, WorkflowResult


class DataProcessorWorkflow(Workflow[str]):
    """
    A workflow that processes data using multiple agents, each specialized for a different task.
    This workflow demonstrates how to use multiple agents to process data in a sequence.
    """

    async def initialize(self):
        await super().initialize()
        self.state.status = "ready"

        # Create agents for different steps of the workflow
        self.finder_agent = Agent(
            name="finder",
            instruction="You are specialized in finding and retrieving information from files or URLs.",
            server_names=["fetch", "filesystem"],
        )

        self.analyzer_agent = Agent(
            name="analyzer",
            instruction="You are specialized in analyzing text data and extracting key insights.",
            server_names=["fetch"],
        )

        self.formatter_agent = Agent(
            name="formatter",
            instruction="You are specialized in formatting data into structured outputs.",
            server_names=[],
        )

        # Initialize the agents
        await self.finder_agent.initialize()
        await self.analyzer_agent.initialize()
        await self.formatter_agent.initialize()

        # Attach LLMs to the agents
        self.finder_llm = await self.finder_agent.attach_llm(OpenAIAugmentedLLM)
        self.analyzer_llm = await self.analyzer_agent.attach_llm(OpenAIAugmentedLLM)
        self.formatter_llm = await self.formatter_agent.attach_llm(OpenAIAugmentedLLM)

    async def cleanup(self):
        # Clean up resources
        await self.finder_agent.cleanup()
        await self.analyzer_agent.cleanup()
        await self.formatter_agent.cleanup()
        await super().cleanup()

    async def run(
        self,
        source: str,
        analysis_prompt: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> WorkflowResult[str]:
        """
        Run the data processing workflow.

        Args:
            source: The source to process. Can be a file path or URL.
            analysis_prompt: Optional specific instructions for the analysis step.
            output_format: Optional format for the output (e.g., "json", "markdown", "summary").

        Returns:
            WorkflowResult containing the processed data.
        """
        self.state.status = "running"
        self._logger.info(f"Starting data processing workflow for source: {source}")

        # Step 1: Find and retrieve the data
        self._logger.info("Step 1: Finding and retrieving data")
        self.state.metadata["current_step"] = "retrieval"

        retrieval_prompt = f"Retrieve the content from {source} and return it verbatim."
        raw_data = await self.finder_llm.generate_str(retrieval_prompt)

        self.state.metadata["retrieval_completed"] = True
        self.state.metadata["content_length"] = len(raw_data)

        # Step 2: Analyze the data
        self._logger.info("Step 2: Analyzing data")
        self.state.metadata["current_step"] = "analysis"

        analysis_instruction = (
            analysis_prompt
            or "Analyze this content and extract the key points, main themes, and most important information."
        )
        analysis = await self.analyzer_llm.generate_str(
            f"{analysis_instruction}\n\nHere is the content to analyze:\n\n{raw_data[:5000]}"  # Limit to 5000 chars for safety
        )

        self.state.metadata["analysis_completed"] = True

        # Step 3: Format the result
        self._logger.info("Step 3: Formatting output")
        self.state.metadata["current_step"] = "formatting"

        format_instruction = output_format or "markdown"
        format_prompt = f"Format the following analysis into {format_instruction} format, highlighting the most important points:\n\n{analysis}"

        formatted_result = await self.formatter_llm.generate_str(format_prompt)

        self.state.metadata["formatting_completed"] = True
        self.state.status = "completed"

        # Create and return the final result
        result = WorkflowResult[str](
            value=formatted_result,
            metadata={
                "source": source,
                "content_length": len(raw_data),
                "analysis_prompt": analysis_prompt,
                "output_format": format_instruction,
                "workflow_completed": True,
            },
            start_time=self.state.metadata.get("start_time"),
            end_time=self.state.updated_at,
        )

        return result


class SummarizationWorkflow(Workflow[Dict[str, Any]]):
    """
    A workflow that summarizes text content with customizable parameters.
    This workflow demonstrates how to create a simple summarization pipeline.
    """

    async def initialize(self):
        await super().initialize()

        # Create an agent for summarization
        self.summarizer_agent = Agent(
            name="summarizer",
            instruction="You are specialized in summarizing content clearly and concisely.",
            server_names=["fetch", "filesystem"],
        )

        # Initialize the agent
        await self.summarizer_agent.initialize()

        # Attach LLM to the agent
        self.summarizer_llm = await self.summarizer_agent.attach_llm(OpenAIAugmentedLLM)

    async def cleanup(self):
        await self.summarizer_agent.cleanup()
        await super().cleanup()

    async def run(
        self,
        content: str,
        max_length: int = 500,
        style: str = "concise",
        key_points: int = 3,
    ) -> WorkflowResult[Dict[str, Any]]:
        """
        Summarize the provided content.

        Args:
            content: The text content to summarize.
            max_length: Maximum length of the summary in characters.
            style: Style of summarization (concise, detailed, technical, simple).
            key_points: Number of key points to include.

        Returns:
            WorkflowResult containing the summary and metadata.
        """
        self.state.status = "running"
        self._logger.info(
            f"Starting summarization workflow (style: {style}, key_points: {key_points})"
        )

        # Record the start time
        start_time = self.state.updated_at

        # Build the summarization prompt
        prompt = f"""
        Summarize the following content in a {style} style. 
        Include {key_points} key points.
        Keep the summary under {max_length} characters.
        
        Content to summarize:
        ---
        {content[:10000]}  # Limit content to 10,000 chars for safety
        ---
        """

        summary = await self.summarizer_llm.generate_str(prompt)

        # Extract key points using a follow-up prompt
        key_points_prompt = f"Based on the content I just summarized, list exactly {key_points} key points in bullet point format."
        key_points_list = await self.summarizer_llm.generate_str(key_points_prompt)

        self.state.status = "completed"

        # Create the structured result
        result = WorkflowResult[Dict[str, Any]](
            value={
                "summary": summary,
                "key_points": key_points_list,
                "style": style,
                "length": len(summary),
                "requested_max_length": max_length,
            },
            metadata={
                "workflow_name": self.name,
                "content_length": len(content),
                "completion_status": "success",
            },
            start_time=start_time,
            end_time=self.state.updated_at,
        )

        return result


# Initialize the app
app = MCPApp(name="workflow_mcp_server")


# Register workflows with the app
@app.workflow
class DataProcessorWorkflowRegistered(DataProcessorWorkflow):
    pass


@app.workflow
class SummarizationWorkflowRegistered(SummarizationWorkflow):
    pass


async def main():
    # Initialize the app
    await app.initialize()

    # Add the current directory to the filesystem server's args if needed
    context = app.context
    context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

    # Create the MCP server
    mcp_server = create_mcp_server_for_app(app)

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(read_stream, write_stream)


if __name__ == "__main__":
    asyncio.run(main())
