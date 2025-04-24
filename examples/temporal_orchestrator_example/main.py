"""
Example of using Temporal as the execution engine for MCP Agent workflows.
This example demonstrates how to create a workflow using the app.workflow and app.workflow_run
decorators, and how to run it using the Temporal executor.
"""

import asyncio
import logging
import os

from mcp_agent.agents.agent import Agent
from mcp_agent.executor.temporal import TemporalExecutor
from mcp_agent.app import MCPApp
from mcp_agent.executor.workflow import Workflow, WorkflowResult
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator

# Create the app with Temporal as the execution engine
app = MCPApp(name="temporal_workflow_example")

finder_agent = Agent(
    name="finder",
    instruction="""You are an agent with access to the filesystem, 
    as well as the ability to fetch URLs. Your job is to identify 
    the closest match to a user's request, make the appropriate tool calls, 
    and return the URI and CONTENTS of the closest match.""",
    server_names=["fetch", "filesystem"],
)

writer_agent = Agent(
    name="writer",
    instruction="""You are an agent that can write to the filesystem.
    You are tasked with taking the user's input, addressing it, and 
    writing the result to disk in the appropriate location.""",
    server_names=["filesystem"],
)

proofreader = Agent(
    name="proofreader",
    instruction=""""Review the short story for grammar, spelling, and punctuation errors.
    Identify any awkward phrasing or structural issues that could improve clarity. 
    Provide detailed feedback on corrections.""",
    server_names=["fetch"],
)

fact_checker = Agent(
    name="fact_checker",
    instruction="""Verify the factual consistency within the story. Identify any contradictions,
    logical inconsistencies, or inaccuracies in the plot, character actions, or setting. 
    Highlight potential issues with reasoning or coherence.""",
    server_names=["fetch"],
)

style_enforcer = Agent(
    name="style_enforcer",
    instruction="""Analyze the story for adherence to style guidelines.
    Evaluate the narrative flow, clarity of expression, and tone. Suggest improvements to 
    enhance storytelling, readability, and engagement.""",
    server_names=["fetch"],
)

orchestrator = Orchestrator(
    llm_factory=OpenAIAugmentedLLM,
    available_agents=[
        finder_agent,
        # writer_agent,
        # proofreader,
        # fact_checker,
        # style_enforcer,
    ],
    # We will let the orchestrator iteratively plan the task at every step
    plan_type="full",
)


@app.workflow
class SimpleWorkflow(Workflow[str]):
    """
    A simple workflow that demonstrates the basic structure of a Temporal workflow.
    """

    @app.workflow_run
    async def run(self, input: str) -> WorkflowResult[str]:
        """
        Run the workflow, processing the input data.

        Args:
            input_data: The data to process

        Returns:
            A WorkflowResult containing the processed data
        """

        result = await orchestrator.generate_str(
            message=input, request_params=RequestParams(model="gpt-4o-mini")
        )

        return WorkflowResult(value=result)


async def main():
    async with app.run() as orchestrator_app:
        context = orchestrator_app.context
        logger = orchestrator_app.logger

        logger.info("Current config:", data=context.config.model_dump())

        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        executor: TemporalExecutor = orchestrator_app.executor

        handle = await executor.start_workflow(
            "SimpleWorkflow",
            """Print the first 2 paragraphs of https://modelcontextprotocol.io/introduction""",
        )
        a = await handle.result()
        print(a)


if __name__ == "__main__":
    asyncio.run(main())
