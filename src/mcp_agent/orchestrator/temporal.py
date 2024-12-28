"""
Temporal based orchestrator for the MCP Agent.
Temporal provides durable execution and robust workflow orchestration,
as well as dynamic control flow, making it a good choice for an AI agent orchestrator.
Read more: https://docs.temporal.io/develop/python/core-application
"""

from temporalio import workflow
from temporalio.client import Client

from mcp_agent.config import settings


def activity_call(activity_func, **activity_opts):
    # decorator that returns a function
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Instead of running logic locally, we capture arguments and call execute_activity
            return await workflow.execute_activity(
                activity_func, *args, **activity_opts, **kwargs
            )

        return wrapper

    return decorator


async def get_temporal_client():
    temporal_client = await Client.connect(
        target_host=settings.temporal.host, namespace=settings.temporal.namespace
    )

    return temporal_client
