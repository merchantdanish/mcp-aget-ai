"""
Worker script for the Temporal workflow example.
This script starts a Temporal worker that can execute workflows and activities.
Run this script in a separate terminal window before running the main.py script.

This leverages the TemporalExecutor's start_worker method to handle the worker setup.
"""

import asyncio
import logging


from mcp_agent.agents.agent import AgentTasks
from mcp_agent.executor.temporal import create_temporal_worker_for_app

# from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicCompletionTasks
# from mcp_agent.workflows.llm.augmented_llm_azure import AzureCompletionTasks
# from mcp_agent.workflows.llm.augmented_llm_bedrock import BedrockCompletionTasks
# from mcp_agent.workflows.llm.augmented_llm_google import GoogleCompletionTasks
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAICompletionTasks

from main import app

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """
    Start a Temporal worker for the example workflows using the app's executor.
    """
    # Initialize the app to set up the context and executor
    async with app.run() as running_app:
        agent_tasks = AgentTasks(context=running_app.context)
        app.workflow_task()(agent_tasks.call_tool_task)
        app.workflow_task()(agent_tasks.get_capabilities_task)
        app.workflow_task()(agent_tasks.get_prompt_task)
        app.workflow_task()(agent_tasks.initialize_aggregator_task)
        app.workflow_task()(agent_tasks.list_prompts_task)
        app.workflow_task()(agent_tasks.list_tools_task)
        app.workflow_task()(agent_tasks.shutdown_aggregator_task)

        # app.workflow_task()(AnthropicCompletionTasks.request_completion_task)
        # app.workflow_task()(AnthropicCompletionTasks.request_structured_completion_task)

        # app.workflow_task()(AzureCompletionTasks.request_completion_task)

        # app.workflow_task()(BedrockCompletionTasks.request_completion_task)
        # app.workflow_task()(BedrockCompletionTasks.request_structured_completion_task)

        # app.workflow_task()(GoogleCompletionTasks.request_completion_task)
        # app.workflow_task()(GoogleCompletionTasks.request_structured_completion_task)

        app.workflow_task()(OpenAICompletionTasks.request_completion_task)
        app.workflow_task()(OpenAICompletionTasks.request_structured_completion_task)

        async with create_temporal_worker_for_app(running_app) as worker:
            await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
