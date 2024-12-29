"""
Temporal based orchestrator for the MCP Agent.
Temporal provides durable execution and robust workflow orchestration,
as well as dynamic control flow, making it a good choice for an AI agent orchestrator.
Read more: https://docs.temporal.io/develop/python/core-application
"""

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Coroutine, List

from temporalio import activity, workflow
from temporalio.client import Client

from mcp_agent.config import settings
from mcp_agent.executor.executor import Executor, R


@dataclass
class ActivityInfo:
    """Metadata for an activity"""

    func: Callable
    name: str


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


class TemporalExecutor(Executor):
    """Executor that runs tasks as Temporal activities"""

    def __init__(self, task_queue: str = "temporal-executor-queue"):
        self.task_queue = task_queue

    @staticmethod
    def wrap_as_activity(
        func: Callable[..., R] | Coroutine[Any, Any, R],
    ) -> ActivityInfo:
        """
        Convert a function into a Temporal activity and return its info.
        """
        # Generate unique name to avoid collisions
        # TODO: saqadri - validate if this is the correct way to generate activity name
        activity_name = f"activity_{id(func)}"

        @activity.defn(name=activity_name)
        async def wrapped_activity(*args, **kwargs):
            if asyncio.iscoroutine(func):
                return await func
            elif asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        return ActivityInfo(func=wrapped_activity, name=activity_name)

    async def execute(
        self,
        *tasks: Callable[..., R] | Coroutine[Any, Any, R],
        **kwargs: Any,
    ) -> List[R | BaseException]:
        # Must be called from within a workflow
        if not workflow._Runtime.current():
            raise RuntimeError(
                "TemporalExecutor.execute must be called from within a workflow"
            )

        # Convert tasks to activity names
        activities = [self.wrap_as_activity(task) for task in tasks]

        # Execute activities in parallel using Temporal's workflow API
        futures = []
        for activity_info in activities:
            futures.append(
                # TODO: saqadri - validate where the positional arguments are being passed
                workflow.execute_activity(
                    activity_info.func, task_queue=self.task_queue, **kwargs
                )
            )

        return await workflow.wait(futures)

    async def execute_streaming(
        self,
        *tasks: Callable[..., R] | Coroutine[Any, Any, R],
        **kwargs: Any,
    ) -> AsyncIterator[R | BaseException]:
        if not workflow._Runtime.current():
            raise RuntimeError(
                "TemporalExecutor.execute_streaming must be called from within a workflow"
            )

        activity_names = [self.wrap_as_activity(task) for task in tasks]
        futures = []

        for name in activity_names:
            activity_ref = getattr(workflow, name)
            futures.append(
                # TODO: saqadri - validate where the positional arguments are being passed
                workflow.execute_activity(
                    activity_ref, task_queue=self.task_queue, **kwargs
                )
            )

        pending = set(futures)

        while pending:
            done, pending = await workflow.wait(
                futures, return_when=asyncio.FIRST_COMPLETED
            )

            for future in done:
                yield await future
