"""
Temporal based orchestrator for the MCP Agent.
Temporal provides durable execution and robust workflow orchestration,
as well as dynamic control flow, making it a good choice for an AI agent orchestrator.
Read more: https://docs.temporal.io/develop/python/core-application
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
import functools
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
)
import inspect
import uuid
import warnings

from pydantic import ConfigDict
from temporalio import activity, workflow, exceptions
from temporalio.client import Client as TemporalClient, WorkflowHandle
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.common import WorkflowIDReusePolicy
from temporalio.worker import Worker

from mcp_agent.config import TemporalSettings
from mcp_agent.executor.executor import Executor, ExecutorConfig, R
from mcp_agent.executor.temporal.workflow_signal import TemporalSignalHandler
from mcp_agent.executor.workflow_signal import SignalHandler
from mcp_agent.logging.logger import get_logger
from mcp_agent.utils.common import unwrap

if TYPE_CHECKING:
    from random import Random
    from uuid import UUID

    from mcp_agent.app import MCPApp
    from mcp_agent.core.context import Context

logger = get_logger(__name__)


class TemporalExecutorConfig(ExecutorConfig, TemporalSettings):
    """Configuration for Temporal executors."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class TemporalExecutor(Executor):
    """Executor that runs @workflows as Temporal workflows, with @workflow_tasks as Temporal activities"""

    def __init__(
        self,
        config: TemporalExecutorConfig | None = None,
        signal_bus: SignalHandler | None = None,
        client: TemporalClient | None = None,
        context: Optional["Context"] = None,
        **kwargs,
    ):
        signal_bus = signal_bus or TemporalSignalHandler(executor=self)
        super().__init__(
            engine="temporal",
            config=config,
            signal_bus=signal_bus,
            context=context,
            **kwargs,
        )
        self.config: TemporalExecutorConfig = (
            config or self.context.config.temporal or TemporalExecutorConfig()
        )
        self.client = client
        self._worker = None
        self._activity_semaphore = None

        if config.max_concurrent_activities is not None:
            self._activity_semaphore = asyncio.Semaphore(
                self.config.max_concurrent_activities
            )

    @staticmethod
    def wrap_as_activity(
        activity_name: str,
        func: Callable[..., R] | Coroutine[Any, Any, R],
        **kwargs: Any,
    ) -> Coroutine[Any, Any, R]:
        """
        Convert a function into a Temporal activity and return its info.
        """

        @activity.defn(name=activity_name)
        async def wrapped_activity(*args, **local_kwargs):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(**args[0])
                elif asyncio.iscoroutine(func):
                    return await func
                else:
                    return func(**args[0])
            except Exception as e:
                # Handle exceptions gracefully
                raise e

        return wrapped_activity

    async def _execute_task_as_async(
        self, task: Callable[..., R] | Coroutine[Any, Any, R], *args, **kwargs
    ) -> R | BaseException:
        async def run_task(task: Callable[..., R] | Coroutine[Any, Any, R]) -> R:
            try:
                if asyncio.iscoroutine(task):
                    return await task
                elif asyncio.iscoroutinefunction(task):
                    return await task(*args, **kwargs)
                else:
                    # Check if we're in a Temporal workflow context
                    if workflow._Runtime.current():
                        wrapped_task = functools.partial(task, *args, **kwargs)
                        result = wrapped_task()
                    else:
                        # Outside a workflow, use standard asyncio executor
                        loop = asyncio.get_running_loop()
                        wrapped_task = functools.partial(task, *args, **kwargs)
                        result = await loop.run_in_executor(None, wrapped_task)

                    # Handle case where the sync function returns a coroutine
                    if asyncio.iscoroutine(result):
                        return await result

                    return result
            except Exception as e:
                # TODO: saqadri - set up logger
                # logger.error(f"Error executing task: {e}")
                return e

        if self._activity_semaphore:
            async with self._activity_semaphore:
                return await run_task(task)
        else:
            return await run_task(task)

    async def _execute_task(
        self, task: Callable[..., R] | Coroutine[Any, Any, R], *args, **kwargs
    ) -> R | BaseException:
        func = task.func if isinstance(task, functools.partial) else task
        func = unwrap(func)

        is_workflow_task = getattr(func, "is_workflow_task", False)
        execution_metadata: Dict[str, Any] = getattr(func, "execution_metadata", {})
        activity_name: str | None = execution_metadata.get("activity_name", None)

        if not is_workflow_task or not activity_name:
            return await self._execute_task_as_async(task, *args, **kwargs)

        activity_registry = self.context.task_registry
        activity_task = activity_registry.get_activity(activity_name)

        schedule_to_close = execution_metadata.get(
            "schedule_to_close_timeout", self.config.timeout_seconds
        )

        if schedule_to_close is not None and not isinstance(
            schedule_to_close, timedelta
        ):
            # Convert to timedelta if it's not already
            schedule_to_close = timedelta(seconds=schedule_to_close)

        retry_policy = execution_metadata.get("retry_policy", None)

        try:
            result = await workflow.execute_activity(
                activity_task,
                args=args,
                task_queue=self.config.task_queue,
                schedule_to_close_timeout=schedule_to_close,
                retry_policy=retry_policy,
            )
            return result
        except Exception as e:
            # Properly propagate activity errors
            if isinstance(e, exceptions.ActivityError):
                raise e.cause if e.cause else e
            raise

    async def execute(
        self,
        task: Callable[..., R] | Coroutine[Any, Any, R],
        *args,
        **kwargs,
    ) -> R | BaseException:
        """Execute multiple tasks (activities) in parallel."""

        # Must be called from within a workflow
        if not workflow._Runtime.current():
            raise RuntimeError(
                "TemporalExecutor.execute must be called from within a workflow"
            )

        # TODO: saqadri - validate if async with self.execution_context() is needed here
        async with self.execution_context():
            return await self._execute_task(task, *args, **kwargs)

    async def execute_many(
        self,
        tasks: List[Callable[..., R] | Coroutine[Any, Any, R]],
        *args,
        **kwargs,
    ) -> List[R | BaseException]:
        """Execute multiple tasks (activities) in parallel."""

        # Must be called from within a workflow
        if not workflow._Runtime.current():
            raise RuntimeError(
                "TemporalExecutor.execute must be called from within a workflow"
            )

        # TODO: saqadri - validate if async with self.execution_context() is needed here
        async with self.execution_context():
            return await asyncio.gather(
                *[self._execute_task(task, *args, **kwargs) for task in tasks],
                return_exceptions=True,
            )

    async def execute_streaming(
        self,
        tasks: List[Callable[..., R] | Coroutine[Any, Any, R]],
        *args,
        **kwargs,
    ) -> AsyncIterator[R | BaseException]:
        if not workflow._Runtime.current():
            raise RuntimeError(
                "TemporalExecutor.execute_streaming must be called from within a workflow"
            )

        # TODO: saqadri - validate if async with self.execution_context() is needed here
        async with self.execution_context():
            # Create futures for all tasks
            futures = [self._execute_task(task, *args, **kwargs) for task in tasks]
            pending = set(futures)

            while pending:
                done, pending = await workflow.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for future in done:
                    try:
                        result = await future
                        yield result
                    except Exception as e:
                        yield e

    async def ensure_client(self):
        """Ensure we have a connected Temporal client."""
        if self.client is None:
            self.client = await TemporalClient.connect(
                target_host=self.config.host,
                namespace=self.config.namespace,
                api_key=self.config.api_key,
                tls=self.config.tls,
                data_converter=pydantic_data_converter,
            )

        return self.client

    async def start_workflow(
        self,
        workflow_class_name_or_id: str,
        *args: Any,
        workflow_id: str = None,
        wait_for_result: bool = False,
        **kwargs: Any,
    ) -> WorkflowHandle:
        """
        Starts a workflow instance of the specified workflow class.

        The workflow class is looked up from the registered workflows using the provided
        class name. A unique workflow instance ID is either provided explicitly or 
        auto-generated to identify this specific execution.

        Args:
            workflow_class_name_or_id (str): Name of the registered workflow class to start.
                This is used to lookup the workflow class from app.workflows registry.
                For backward compatibility: if workflow_id is None, this also serves as 
                the basis for auto-generating the instance ID (deprecated pattern).
            *args: Positional arguments to pass to the workflow's run() method.
            workflow_id (str, optional): Unique identifier for this workflow instance.
                If not provided, auto-generated as '{workflow_class_name}-{uuid4()}'.
                Used for querying, signaling, and distinguishing this execution.
            wait_for_result (bool): Whether to wait for the workflow to complete and return the result.
            **kwargs: Keyword arguments to pass to the workflow's run() method.

        Returns:
            WorkflowHandle: Handle for the started workflow instance.
            If wait_for_result is True, returns the workflow result instead.

        Example:
            # New recommended pattern
            handle = await executor.start_workflow(
                "MyWorkflowClass", 
                arg1, arg2,
                workflow_id="my-unique-id-123"
            )
            
            # Old pattern (deprecated, issues warning)
            handle = await executor.start_workflow("MyWorkflowClass", arg1, arg2)
        """
        await self.ensure_client()

        # Handle backward compatibility
        if workflow_id is None:
            # Old calling pattern: start_workflow("MyWorkflowClass", ...)
            # where first arg was used as both class lookup and instance ID
            warnings.warn(
                "Using the first parameter as both workflow class name and instance ID is deprecated. "
                "Please use start_workflow(workflow_class_name, ..., workflow_id='unique-id') instead. "
                "A workflow ID will be auto-generated, but you should consider providing your own "
                "for better traceability and to enable workflow queries/signals by ID.",
                DeprecationWarning,
                stacklevel=2
            )
            workflow_class_name = workflow_class_name_or_id
            workflow_id = f"{workflow_class_name_or_id}-{uuid.uuid4()}"
        else:
            # New calling pattern: start_workflow("MyWorkflowClass", ..., workflow_id="unique-id")
            workflow_class_name = workflow_class_name_or_id

        # Lookup the workflow class
        wf = self.context.app.workflows.get(workflow_class_name)
        if not inspect.isclass(wf):
            wf = wf.__class__

        # Inspect the `run(self, …)` signature
        sig = inspect.signature(wf.run)
        params = [p for p in sig.parameters.values() if p.name != "self"]

        # Bind args in declaration order
        bound_args = []
        for idx, param in enumerate(params):
            if idx < len(args):
                bound_args.append(args[idx])
            elif param.name in kwargs:
                bound_args.append(kwargs[param.name])
            elif param.default is not inspect.Parameter.empty:
                # optional param, skip if not provided
                continue
            else:
                raise ValueError(f"Missing required workflow argument '{param.name}'")

        # Too many positionals?
        if len(args) > len(params):
            raise ValueError(
                f"Got {len(args)} positional args but run() only takes {len(params)}"
            )

        # Determine what to pass to the start_workflow function
        input_arg = None
        if not bound_args:
            # zero-arg workflow
            pass
        elif len(bound_args) == 1:
            # single-arg workflow
            input_arg = bound_args[0]
        else:
            # multi-arg workflow - pack into a sequence
            input_arg = bound_args

        # Start the workflow
        if input_arg is not None:
            handle: WorkflowHandle = await self.client.start_workflow(
                wf,
                input_arg,
                id=workflow_id,
                task_queue=self.config.task_queue,
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
            )
        else:
            handle: WorkflowHandle = await self.client.start_workflow(
                wf,
                id=workflow_id,
                task_queue=self.config.task_queue,
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
            )

        # Wait for the result if requested
        if wait_for_result:
            return await handle.result()

        return handle

    async def execute_workflow(
        self,
        workflow_id: str,
        *workflow_args: Any,
        **workflow_kwargs: Any,
    ) -> Any:
        """
        Execute a workflow and wait for its result.

        This is a convenience wrapper around start_workflow with wait_for_result=True.
        """
        return await self.start_workflow(
            workflow_id, *workflow_args, wait_for_result=True, **workflow_kwargs
        )

    async def terminate_workflow(
        self,
        workflow_id: str,
        run_id: str | None = None,
        reason: str | None = "Cancellation",
    ) -> None:
        """
        Terminate a workflow execution.

        Args:
            workflow_id (str): Identifier of the workflow to terminate.
            run_id (Optional[str]): If provided, terminates the specific run.
                                Otherwise terminates the latest run.
            reason (Optional[str]): A reason for the termination.
        """
        await self.ensure_client()
        workflow_handle = self.client.get_workflow_handle(
            workflow_id=workflow_id, run_id=run_id
        )
        await workflow_handle.terminate(reason=reason)

    def uuid(self) -> "UUID":
        """
        Generate a UUID using Temporal's deterministic UUID generator.
        """
        try:
            return workflow.uuid4()
        except exceptions.TemporalError:
            return super().uuid()

    def random(self) -> "Random":
        """
        Get an instance of Temporal's deterministic pseudo-random number generator.

        Note, this random number generator is not cryptographically safe and should
        not be used for security purposes.

        Returns:
            The deterministically-seeded pseudo-random number generator.
        """
        try:
            return workflow.random()
        except exceptions.TemporalError:
            return super().random()


@asynccontextmanager
async def create_temporal_worker_for_app(app: "MCPApp"):
    """
    Create a Temporal worker for the given app.
    """
    activities = []

    # Initialize the app to set up the context and executor
    async with app.run() as running_app:
        if not isinstance(running_app.executor, TemporalExecutor):
            raise ValueError("App executor is not a TemporalExecutor.")

        await running_app.executor.ensure_client()

        from mcp_agent.agents.agent import AgentTasks

        agent_tasks = AgentTasks(context=running_app.context)
        app.workflow_task()(agent_tasks.call_tool_task)
        app.workflow_task()(agent_tasks.get_capabilities_task)
        app.workflow_task()(agent_tasks.get_prompt_task)
        app.workflow_task()(agent_tasks.initialize_aggregator_task)
        app.workflow_task()(agent_tasks.list_prompts_task)
        app.workflow_task()(agent_tasks.list_tools_task)
        app.workflow_task()(agent_tasks.shutdown_aggregator_task)

        # Collect activities from the global registry
        activity_registry = running_app.context.task_registry

        for name in activity_registry.list_activities():
            activities.append(activity_registry.get_activity(name))

        # Collect workflows from the registered workflows
        workflows = running_app.context.app.workflows.values()

        worker = Worker(
            client=running_app.executor.client,
            task_queue=running_app.executor.config.task_queue,
            activities=activities,
            workflows=workflows,
        )

        try:
            # Yield the worker to allow the caller to use it
            yield worker
        finally:
            # No explicit cleanup needed here as the app context will handle it
            # when the async with block exits
            pass
