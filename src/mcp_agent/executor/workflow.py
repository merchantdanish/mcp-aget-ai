import asyncio

# import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, ConfigDict, Field

from mcp_agent.context import get_current_config
from mcp_agent.executor.executor import Executor
from mcp_agent.executor.decorator_registry import get_decorator_registry
from mcp_agent.executor.task_registry import get_activity_registry

R = TypeVar("R")
T = TypeVar("T")


class WorkflowState(BaseModel):
    """
    Simple container for persistent workflow state.
    This can hold fields that should persist across tasks.
    """

    status: str = "initialized"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    updated_at: float | None = None
    error: Dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def record_error(self, error: Exception) -> None:
        self.error = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.utcnow().timestamp(),
        }


class WorkflowResult(BaseModel, Generic[T]):
    value: Union[T, None] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None

    # def complete(self) -> "WorkflowResult[T]":
    #     import asyncio

    #     if self.start_time is None:
    #         self.start_time = asyncio.get_event_loop().time()
    #     self.end_time = asyncio.get_event_loop().time()
    #     self.metadata["duration"] = self.end_time - self.start_time
    #     return self


class Workflow(ABC, Generic[T]):
    """
    Base class for user-defined workflows.
    Handles execution and state management.
    Some key notes:
        - To enable the executor engine to recognize and orchestrate the workflow,
            - the class MUST be decorated with @workflow.
            - the main entrypoint method MUST be decorated with @workflow_run.
            - any task methods MUST be decorated with @workflow_task.

        - Persistent state: Provides a simple `state` object for storing data across tasks.
    """

    def __init__(
        self,
        executor: Executor,
        name: str | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.executor = executor
        self.name = name or self.__class__.__name__
        self.init_kwargs = kwargs
        # TODO: handle logging
        # self._logger = logging.getLogger(self.name)

        # A simple workflow state object
        # If under Temporal, storing it as a field on this class
        # means it can be replayed automatically
        self.state = WorkflowState(name=name, metadata=metadata or {})

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> "WorkflowResult[T]":
        """
        Main workflow implementation. Myst be overridden by subclasses.
        """

    async def update_state(self, **kwargs):
        """Syntactic sugar to update workflow state."""
        for key, value in kwargs.items():
            self.state[key] = value
            setattr(self.state, key, value)

        self.state.updated_at = datetime.utcnow().timestamp()

    async def wait_for_input(self, description: str = "Provide input") -> str:
        """
        Convenience method for human input. Uses `human_input` signal
        so we can unify local (console input) and Temporal signals.
        """
        return await self.executor.wait_for_signal(
            "human_input", description=description
        )


#####################
# Workflow Decorators
#####################


def get_execution_engine() -> str:
    """Get the current execution engine (asyncio, Temporal, etc)."""
    config = get_current_config()
    return config.execution_engine or "asyncio"


def workflow(cls: Type, *args, **kwargs) -> Type:
    """
    Decorator for a workflow class. By default it's a no-op,
    but different executors can use this to customize behavior
    for workflow registration.

    Example:
        If Temporal is available & we use a TemporalExecutor,
        this decorator will wrap with temporal_workflow.defn.
    """
    decorator_registry = get_decorator_registry()
    execution_engine = get_execution_engine()
    workflow_defn_decorator = decorator_registry.get_workflow_defn_decorator(
        execution_engine
    )

    if workflow_defn_decorator:
        return workflow_defn_decorator(cls, *args, **kwargs)

    # Default no-op
    return cls


def workflow_run(fn: Callable[..., R]) -> Callable[..., R]:
    """
    Decorator for a workflow's main 'run' method.
    Different executors can use this to customize behavior for workflow execution.

    Example:
        If Temporal is in use, this gets converted to @workflow.run.
    """

    decorator_registry = get_decorator_registry()
    execution_engine = get_execution_engine()
    workflow_run_decorator = decorator_registry.get_workflow_run_decorator(
        execution_engine
    )

    if workflow_run_decorator:
        return workflow_run_decorator(fn)

    # Default no-op
    def wrapper(*args, **kwargs):
        # no-op wrapper
        return fn(*args, **kwargs)

    return wrapper


def workflow_task(
    name: str | None = None,
    schedule_to_close_timeout: timedelta | None = None,
    retry_policy: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to mark a function as a workflow task,
    automatically registering it in the global activity registry.

    Args:
        name: Optional custom name for the activity
        schedule_to_close_timeout: Maximum time the task can take to complete
        retry_policy: Retry policy configuration
        **kwargs: Additional metadata passed to the activity registration

    Returns:
        Decorated function that preserves async and typing information

    Raises:
        TypeError: If the decorated function is not async
        ValueError: If the retry policy or timeout is invalid
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(f"Function {func.__name__} must be async.")

        actual_name = name or f"{func.__module__}.{func.__qualname__}"
        timeout = schedule_to_close_timeout or timedelta(minutes=10)
        metadata = {
            "activity_name": actual_name,
            "schedule_to_close_timeout": timeout,
            "retry_policy": retry_policy or {},
            **kwargs,
        }
        activity_registry = get_activity_registry()
        activity_registry.register(actual_name, func, metadata)

        setattr(func, "is_workflow_task", True)
        setattr(func, "execution_metadata", metadata)

        # TODO: saqadri - determine if we need this
        # Preserve metadata through partial application
        # @functools.wraps(func)
        # async def wrapper(*args: Any, **kwargs: Any) -> R:
        #     result = await func(*args, **kwargs)
        #     return cast(R, result)  # Ensure type checking works

        # # Add metadata that survives partial application
        # wrapper.is_workflow_task = True  # type: ignore
        # wrapper.execution_metadata = metadata  # type: ignore

        # # Make metadata accessible through partial
        # def __getattr__(name: str) -> Any:
        #     if name == "is_workflow_task":
        #         return True
        #     if name == "execution_metadata":
        #         return metadata
        #     raise AttributeError(f"'{func.__name__}' has no attribute '{name}'")

        # wrapper.__getattr__ = __getattr__  # type: ignore

        # return wrapper

        return func

    return decorator


def is_workflow_task(func: Callable[..., Any]) -> bool:
    """
    Check if a function is marked as a workflow task.
    This gets set for functions that are decorated with @workflow_task."""
    return bool(getattr(func, "is_workflow_task", False))


# ############################
# # Example: DocumentWorkflow
# ############################


# @workflow_defn  # <-- This becomes @temporal_workflow.defn if in Temporal mode, else no-op
# class DocumentWorkflow(Workflow[List[Dict[str, Any]]]):
#     """
#     Example workflow with persistent state.
#     If run locally, `self.state` is ephemeral.
#     If run in Temporal mode, `self.state` is replayed automatically.
#     """

#     @workflow_task(
#         schedule_to_close_timeout=timedelta(minutes=10),
#         retry_policy={"initial_interval": 1, "max_attempts": 3},
#     )
#     async def process_document(self, doc_id: str) -> Dict[str, Any]:
#         """Activity that simulates document processing."""
#         await asyncio.sleep(1)
#         # Optionally mutate workflow state
#         self.state.metadata.setdefault("processed_docs", []).append(doc_id)
#         return {
#             "doc_id": doc_id,
#             "status": "processed",
#             "timestamp": datetime.utcnow().isoformat(),
#         }

#     @workflow_run  # <-- This becomes @temporal_workflow.run(...) if Temporal is used
#     async def _run_impl(
#         self, documents: List[str], batch_size: int = 2
#     ) -> List[Dict[str, Any]]:
#         """Main workflow logic, which becomes the official 'run' in Temporal mode."""
#         self._logger.info("Workflow starting, state=%s", self.state)
#         self.state.update_status("running")

#         all_results = []
#         for i in range(0, len(documents), batch_size):
#             batch = documents[i : i + batch_size]
#             tasks = [self.process_document(doc) for doc in batch]
#             results = await self.executor.execute(*tasks)

#             for res in results:
#                 if isinstance(res.value, Exception):
#                     self._logger.error(
#                         f"Error processing document: {res.metadata.get('error')}"
#                     )
#                 else:
#                     all_results.append(res.value)

#         self.state.update_status("completed")
#         return all_results


# ########################
# # 12. Example Local Usage
# ########################


# async def run_example_local():
#     from . import AsyncIOExecutor, DocumentWorkflow  # if in a package

#     executor = AsyncIOExecutor()
#     wf = DocumentWorkflow(executor)

#     documents = ["doc1", "doc2", "doc3", "doc4"]
#     result = await wf.run(documents, batch_size=2)

#     print("Local results:", result.value)
#     print("Local workflow final state:", wf.state)
#     # Notice `wf.state.metadata['processed_docs']` has the processed doc IDs.


# ########################
# # Example Temporal Usage
# ########################


# async def run_example_temporal():
#     from . import TemporalExecutor, DocumentWorkflow  # if in a package

#     # 1) Create a TemporalExecutor (client side)
#     executor = TemporalExecutor(task_queue="my_task_queue")
#     await executor.ensure_client()

#     # 2) Start a worker in the same process (or do so in a separate process)
#     asyncio.create_task(executor.start_worker())
#     await asyncio.sleep(2)  # Wait for worker to be up

#     # 3) Now we can run the workflow by normal means if we like,
#     #    or rely on the Worker picking it up. Typically, you'd do:
#     #    handle = await executor._client.start_workflow(...)
#     #    but let's keep it simple and show conceptually
#     #    that 'DocumentWorkflow' is now recognized as a real Temporal workflow
#     print(
#         "Temporal environment is running. Use the Worker logs or CLI to start 'DocumentWorkflow'."
#     )
