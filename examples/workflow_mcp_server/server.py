"""
Workflow MCP Server Example

This example demonstrates three approaches to creating agents and workflows:
1. Traditional workflow-based approach with manual agent creation
2. Programmatic agent configuration using AgentConfig
3. Declarative agent configuration using FastMCPApp decorators
"""

import asyncio
import os
import logging
from typing import Dict, Any, Optional

from mcp_agent.fast_app import FastMCPApp
from mcp_agent.app_server import create_mcp_server_for_app
from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_config import (
    AgentConfig,
    AugmentedLLMConfig,
    ParallelWorkflowConfig,
)
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.executor.workflow import Workflow, WorkflowResult
from mcp_agent.executor.executor import Executor

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessorWorkflow(Workflow[str]):
    """
    A workflow that processes data using multiple agents, each specialized for a different task.
    This workflow demonstrates how to use multiple agents to process data in a sequence.
    """

    @classmethod
    async def create(
        cls, executor: Executor, name: str | None = None, **kwargs: Any
    ) -> "DataProcessorWorkflow":
        """
        Factory method to create and initialize the DataProcessorWorkflow.
        Demonstrates how to override the default create method for specialized initialization.

        Args:
            executor: The executor to use
            name: Optional workflow name
            **kwargs: Additional parameters for customization

        Returns:
            An initialized DataProcessorWorkflow instance
        """
        # Create the workflow instance
        workflow = cls(executor=executor, name=name or "data_processor", **kwargs)

        # Initialize it (which will set up agents, etc.)
        await workflow.initialize()

        return workflow

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
        await self.finder_agent.shutdown()
        await self.analyzer_agent.shutdown()
        await self.formatter_agent.shutdown()
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
            start_time=self.state.metadata.get(
                "start_time"
            ),  # TODO: saqadri (MAC) - fix
            end_time=self.state.updated_at,
        )

        return result


class SummarizationWorkflow(Workflow[Dict[str, Any]]):
    """
    A workflow that summarizes text content with customizable parameters.
    This workflow demonstrates how to create a simple summarization pipeline.

    This workflow uses the default create() implementation from the base Workflow class,
    showing that it's not necessary to override create() in every workflow.
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
        await self.summarizer_agent.shutdown()
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


# Create a single FastMCPApp instance (which extends MCPApp)
app = FastMCPApp(name="workflow_mcp_server")

# -------------------------------------------------------------------------
# Approach 1: Traditional workflow registration with @app.workflow decorator
# -------------------------------------------------------------------------


# Register workflows with the app
@app.workflow
class DataProcessorWorkflowRegistered(DataProcessorWorkflow):
    """Data processing workflow registered with the app."""

    pass


@app.workflow
class SummarizationWorkflowRegistered(SummarizationWorkflow):
    """Summarization workflow registered with the app."""

    pass


# -------------------------------------------------------------------------
# Approach 2: Programmatic agent configuration with AgentConfig
# -------------------------------------------------------------------------

# Create a basic agent configuration
research_agent_config = AgentConfig(
    name="researcher",
    instruction="You are a helpful research assistant that finds information and presents it clearly.",
    server_names=["fetch", "filesystem"],
    llm_config=AugmentedLLMConfig(
        factory=OpenAIAugmentedLLM,
        model="gpt-4o",
        temperature=0.7,
        provider_params={"max_tokens": 2000},
    ),
)

# Create component agents for a parallel workflow
programmatic_summarizer_config = AgentConfig(
    name="programmatic_summarizer",
    instruction="You are specialized in summarizing information clearly and concisely.",
    server_names=["fetch"],
    llm_config=AugmentedLLMConfig(
        factory=AnthropicAugmentedLLM, model="claude-3-sonnet-20240229"
    ),
)

programmatic_fact_checker_config = AgentConfig(
    name="programmatic_fact_checker",
    instruction="You verify facts and identify potential inaccuracies in information.",
    server_names=["fetch", "filesystem"],
    llm_config=AugmentedLLMConfig(factory=OpenAIAugmentedLLM, model="gpt-4o"),
)

programmatic_editor_config = AgentConfig(
    name="programmatic_editor",
    instruction="You refine and improve text, focusing on clarity and readability.",
    server_names=[],
    llm_config=AugmentedLLMConfig(factory=OpenAIAugmentedLLM, model="gpt-4o"),
)

# Create a parallel workflow configuration
programmatic_research_team_config = AgentConfig(
    name="programmatic_research_team",
    instruction="You are a research team that produces high-quality, accurate content.",
    server_names=["fetch", "filesystem"],
    llm_config=AugmentedLLMConfig(
        factory=AnthropicAugmentedLLM, model="claude-3-opus-20240229"
    ),
    parallel_config=ParallelWorkflowConfig(
        fan_in_agent="programmatic_editor",
        fan_out_agents=["programmatic_summarizer", "programmatic_fact_checker"],
        concurrent=True,
    ),
)

# Register the configurations with the app using programmatic method
app.register_agent_config(research_agent_config)
app.register_agent_config(programmatic_summarizer_config)
app.register_agent_config(programmatic_fact_checker_config)
app.register_agent_config(programmatic_editor_config)
app.register_agent_config(programmatic_research_team_config)

# -------------------------------------------------------------------------
# Approach 3: Declarative agent configuration with FastMCPApp decorators
# -------------------------------------------------------------------------


# Basic agent with OpenAI LLM
@app.agent(
    "assistant",
    "You are a helpful assistant that answers questions concisely.",
    server_names=["calculator"],
)
def assistant_config(config):
    # Configure the LLM to use
    config.llm_config = AugmentedLLMConfig(
        factory=OpenAIAugmentedLLM, model="gpt-4o", temperature=0.7
    )
    return config


# Component agents for router workflow
@app.agent(
    "mathematician",
    "You solve mathematical problems with precision.",
    server_names=["calculator"],
)
def mathematician_config(config):
    config.llm_config = AugmentedLLMConfig(factory=OpenAIAugmentedLLM, model="gpt-4o")
    return config


@app.agent(
    "programmer",
    "You write and debug code in various programming languages.",
    server_names=["filesystem"],
)
def programmer_config(config):
    config.llm_config = AugmentedLLMConfig(factory=OpenAIAugmentedLLM, model="gpt-4o")
    return config


@app.agent("writer", "You write creative and engaging content.", server_names=[])
def writer_config(config):
    config.llm_config = AugmentedLLMConfig(
        factory=AnthropicAugmentedLLM, model="claude-3-sonnet-20240229"
    )
    return config


# Router workflow using the decorator syntax
@app.router(
    "specialist_router",
    "You route requests to the appropriate specialist.",
    agent_names=["mathematician", "programmer", "writer"],
)
def router_config(config):
    config.llm_config = AugmentedLLMConfig(factory=OpenAIAugmentedLLM, model="gpt-4o")
    # Configure top_k for the router
    config.router_config.top_k = 1
    return config


async def main():
    # Initialize the app
    await app.initialize()

    # Add the current directory to the filesystem server's args if needed
    context = app.context
    if "filesystem" in context.config.mcp.servers:
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

    # Log registered workflows and agent configurations
    logger.info(f"Creating MCP server for {app.name}")

    logger.info("Registered workflows:")
    for workflow_id in app.workflows:
        logger.info(f"  - {workflow_id}")

    logger.info("Registered agent configurations:")
    for name, config in app.agent_configs.items():
        workflow_type = config.get_workflow_type() or "basic"
        logger.info(f"  - {name} ({workflow_type})")

    # Create the MCP server that exposes both workflows and agent configurations
    mcp_server = create_mcp_server_for_app(app)

    # Run the server
    await mcp_server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
