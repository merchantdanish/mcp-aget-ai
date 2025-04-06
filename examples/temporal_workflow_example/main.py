"""
Example of using Temporal as the execution engine for MCP Agent workflows.
This example demonstrates how to create a workflow using the app.workflow and app.workflow_run
decorators, and how to run it using the Temporal executor.
"""

import asyncio
import logging

from mcp_agent.executor.temporal import TemporalExecutor
from mcp_agent.app import MCPApp
from mcp_agent.executor.workflow import Workflow, WorkflowResult

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the app with Temporal as the execution engine
app = MCPApp(name="temporal_workflow_example")


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
        logger.info(f"Running SimpleWorkflow with input: {input_data}")

        # Execute the workflow task as a Temporal activity
        result = input_data.upper()

        return WorkflowResult(value=result)


async def main():
    async with app.run() as agent_app:
        executor: TemporalExecutor = agent_app.executor
        handle = await executor.start_workflow("SimpleWorkflow", "Hello, World!")
        a = await handle.result()
        print(a)


if __name__ == "__main__":
    asyncio.run(main())
