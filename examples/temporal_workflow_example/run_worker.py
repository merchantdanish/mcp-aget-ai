"""
Worker script for the Temporal workflow example.
This script starts a Temporal worker that can execute workflows and activities.
Run this script in a separate terminal window before running the main.py script.

This leverages the TemporalExecutor's start_worker method to handle the worker setup.
"""

import asyncio
import logging
import os

# Import Temporal libraries
from mcp_agent.agents.agent import Agent
from mcp_agent.executor.temporal import TemporalExecutor
from main import app, finder_agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """
    Start a Temporal worker for the example workflows using the app's executor.
    """
    # Initialize the app to set up the context and executor
    async with app.run() as running_app:
        context = finder_agent.context
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        async with finder_agent:
            finder_llm = finder_agent.attach_llm(OpenAIAugmentedLLM)
            executor: TemporalExecutor = running_app.executor
            await executor.start_worker(agents=[finder_llm])


if __name__ == "__main__":
    asyncio.run(main())
