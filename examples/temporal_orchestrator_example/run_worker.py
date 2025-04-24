"""
Worker script for the Temporal workflow example.
This script starts a Temporal worker that can execute workflows and activities.
Run this script in a separate terminal window before running the main.py script.

This leverages the TemporalExecutor's start_worker method to handle the worker setup.
"""

import asyncio
import contextlib
import logging
import os

# Import Temporal libraries
from mcp_agent.executor.temporal import TemporalExecutor
from main import app, orchestrator

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """
    Start a Temporal worker for the example workflows using the app's executor.
    """
    # Initialize the app to set up the context and executor
    async with app.run() as orchestrator_app:
        logger = orchestrator_app.logger

        context = orchestrator_app.context
        logger.info("Current config:", data=context.config.model_dump())

        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        async with contextlib.AsyncExitStack() as stack:
            context_agents = []

            context_agents.append(orchestrator.planner)
            context_agents.append(orchestrator.sythesizer)

            for agent in orchestrator.agents.values():
                context_agent = await stack.enter_async_context(agent)
                llm = context_agent.attach_llm(orchestrator.llm_factory)
                context_agents.append(llm)

            executor: TemporalExecutor = orchestrator_app.executor
            await executor.start_worker(agents=context_agents)


if __name__ == "__main__":
    asyncio.run(main())
