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
from pydantic import BaseModel

from mcp_agent.app import MCPApp
from mcp_agent.executor.workflow_signal import Signal
from mcp_agent.server.app_server import create_mcp_server_for_app
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.executor.workflow import Workflow, WorkflowResult

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a single FastMCPApp instance (which extends MCPApp)
app = MCPApp(name="basic_agent_server", description="Basic agent server example")


# class RunParams(BaseModel):
#     input: str


@app.workflow
class BasicAgentWorkflow(Workflow[str]):
    """
    A basic workflow that demonstrates how to create a simple agent.
    This workflow is used as an example of a basic agent configuration.
    """

    @app.workflow_signal
    def resume(self, value: str | None = None) -> None:
        state = app.context.signal_registry.get_state("resume")
        state["completed"] = True
        state["value"] = value

    @app.workflow_run
    async def run(self) -> WorkflowResult[str]:
        """
        Run the basic agent workflow.

        Args:
            input: The input string to prompt the agent.

        Returns:
            WorkflowResult containing the processed data.
        """
        await app.context.executor.signal_bus.wait_for_signal(Signal(name="resume"))
        # finder_agent = Agent(
        #     name="finder",
        #     instruction="""You are a helpful assistant.""",
        #     server_names=["fetch", "filesystem"],
        # )

        # context = finder_agent.context
        # context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        # async with finder_agent:
        #     finder_llm = await finder_agent.attach_llm(OpenAIAugmentedLLM)

        #     result = await finder_llm.generate_str(
        #         message=params.input,
        #     )
        return WorkflowResult(value="Hello world!")


async def main():
    async with app.run() as agent_app:
        # Log registered workflows and agent configurations
        logger.info(f"Creating MCP server for {agent_app.name}")

        logger.info("Registered workflows:")
        for workflow_id in agent_app.workflows:
            logger.info(f"  - {workflow_id}")

        logger.info("Registered agent configurations:")
        for name, config in agent_app.agent_configs.items():
            workflow_type = config.get_agent_type() or "basic"
            logger.info(f"  - {name} ({workflow_type})")

        # Create the MCP server that exposes both workflows and agent configurations
        mcp_server = create_mcp_server_for_app(agent_app)

        # Run the server
        await mcp_server.run_sse_async()


if __name__ == "__main__":
    asyncio.run(main())
