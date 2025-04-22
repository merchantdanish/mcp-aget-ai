"""
Worker script for the Temporal workflow example.
This script starts a Temporal worker that can execute workflows and activities.
Run this script in a separate terminal window before running the main.py script.

This leverages the TemporalExecutor's start_worker method to handle the worker setup.
"""

import asyncio
import logging

# Import Temporal libraries
from mcp_agent.executor.temporal import create_temporal_worker_for_app
from main import app
from mcp_agent.agents.agent_activities import AgentTasks

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """
    Start a Temporal worker for the example workflows using the app's executor.
    """
    # Initialize the app to set up the context and executor
    async with app.run() as running_app:
        agent_activities = AgentTasks(context=running_app.context, app=running_app)

        app.workflow_task()(agent_activities.initialize_agent)
        app.workflow_task(name="shutdown_agent")(agent_activities.shutdown_agent)
        app.workflow_task(name="shutdown_all_agents")(
            agent_activities.shutdown_all_agents
        )
        app.workflow_task(name="list_tools")(agent_activities.list_tools)
        async with create_temporal_worker_for_app(running_app) as worker:
            await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
