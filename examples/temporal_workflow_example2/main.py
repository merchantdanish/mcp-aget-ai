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

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the app with Temporal as the execution engine
app = MCPApp(name="temporal_workflow_example")

finder_agent = Agent(
    name="finder",
    instruction="""You are a helpful assistant.""",
    server_names=["fetch", "filesystem"],
)


@app.workflow
class SimpleWorkflow(Workflow[str]):
    """
    A simple workflow that demonstrates the basic structure of a Temporal workflow.
    """

    @app.workflow_run
    async def run(self, input_data: str) -> WorkflowResult[str]:
        """
        Run the workflow, processing the input data.

        Args:
            input_data: The data to process

        Returns:
            A WorkflowResult containing the processed data
        """
        context = finder_agent.context
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        # async with finder_agent:
        initialized = await finder_agent.initialize_activity()
        print(f"Finder agent initialized: {initialized}")

        if initialized:
            tools = await finder_agent.list_tools_activity()
            return WorkflowResult(value=tools)
        else:
            return WorkflowResult(value="Finder agent initialization failed.")

        # finder_llm = finder_agent.attach_llm(OpenAIAugmentedLLM)

        # result = await finder_llm.generate_str(
        #     message="Print the first 2 paragraphs of https://modelcontextprotocol.io/introduction",
        # )

        # logger.info(f"Running SimpleWorkflow with input: {input_data}")

        # # Execute the workflow task as a Temporal activity
        # result = input_data.upper()

        # return WorkflowResult(value=result)


async def main():
    async with app.run() as agent_app:
        executor: TemporalExecutor = agent_app.executor
        handle = await executor.start_workflow("SimpleWorkflow", "Hello, World!")
        a = await handle.result()
        print(a)


if __name__ == "__main__":
    asyncio.run(main())
