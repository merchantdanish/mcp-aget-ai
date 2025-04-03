from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    TypeVar,
    Union,
    List,
    TYPE_CHECKING,
)

import uuid

from pydantic import BaseModel, ConfigDict, Field

from mcp_agent.context_dependent import ContextDependent
from mcp_agent.executor.workflow_signal import Signal
from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp_agent.context import Context
    import asyncio

T = TypeVar("T")


class WorkflowRegistry:
    """
    Registry for tracking workflow instances.
    Provides a central place to register, look up, and manage workflow instances.

    TODO: saqadri (MAC) - How does this work with proper workflow orchestration?
    For example, when using Temporal, this registry should interface with the
    workflow service to manage workflow instances.
    """

    def __init__(self):
        self._workflows: Dict[str, "Workflow"] = {}
        self._tasks: Dict[str, "asyncio.Task"] = {}
        self._logger = get_logger("workflow.registry")

    def register(
        self,
        workflow_id: str,
        workflow: "Workflow",
        task: Optional["asyncio.Task"] = None,
    ) -> None:
        """
        Register a workflow instance and its associated task.

        Args:
            workflow_id: The unique ID for the workflow
            workflow: The workflow instance
            task: The asyncio task running the workflow
        """
        self._workflows[workflow_id] = workflow
        if task:
            self._tasks[workflow_id] = task

    def unregister(self, workflow_id: str) -> None:
        """
        Remove a workflow instance from the registry.

        Args:
            workflow_id: The unique ID for the workflow
        """
        self._workflows.pop(workflow_id, None)
        self._tasks.pop(workflow_id, None)

    def get_workflow(self, workflow_id: str) -> Optional["Workflow"]:
        """
        Get a workflow instance by ID.

        Args:
            workflow_id: The unique ID for the workflow

        Returns:
            The workflow instance, or None if not found
        """
        return self._workflows.get(workflow_id)

    async def resume_workflow(
        self,
        workflow_id: str,
        signal_name: str | None = "resume",
        payload: str | None = None,
    ) -> bool:
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            self._logger.error(
                f"Cannot resume workflow {workflow_id}: workflow not found"
            )
            return False

        return await workflow.resume(signal_name, payload)

    async def cancel_workflow(self, workflow_id: str) -> bool:
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            self._logger.error(
                f"Cannot cancel workflow {workflow_id}: workflow not found"
            )
            return False

        return await workflow.cancel()

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a workflow by ID.

        Args:
            workflow_id: The unique ID of the workflow to check

        Returns:
            The workflow status if found, None otherwise
        """
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return None

        return workflow.get_status()

    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        List all registered workflow instances with their status.

        Returns:
            A list of dictionaries with workflow information
        """
        result = []
        for workflow_id, workflow in self._workflows.items():
            # Get the workflow status directly to have consistent behavior
            status = workflow.get_status()
            result.append(status)

        return result


class WorkflowState(BaseModel):
    """
    Simple container for persistent workflow state.
    This can hold fields that should persist across tasks.
    """

    # TODO: saqadri - (MAC) - This should be a proper status enum
    status: str = "initialized"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    updated_at: float | None = None
    error: Dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def record_error(self, error: Exception) -> None:
        self.error = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now(timezone.utc).timestamp(),
        }


class WorkflowResult(BaseModel, Generic[T]):
    value: Union[T, None] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None


class Workflow(ABC, Generic[T], ContextDependent):
    """
    Base class for user-defined workflows.
    Handles execution and state management.

    Workflows represent user-defined application logic modules that can use Agents and AugmentedLLMs.
    Typically, workflows are registered with an MCPApp and can be exposed as MCP tools via app_server.py.

    Some key notes:
        - The class MUST be decorated with @app.workflow.
        - Persistent state: Provides a simple `state` object for storing data across tasks.
        - Lifecycle management: Provides run_async, pause, resume, cancel, and get_status methods.
    """

    def __init__(
        self,
        name: str | None = None,
        metadata: Dict[str, Any] | None = None,
        context: Optional["Context"] = None,
        **kwargs: Any,
    ):
        # Initialize the ContextDependent mixin
        ContextDependent.__init__(self, context=context)

        self.name = name or self.__class__.__name__
        self._logger = get_logger(f"workflow.{self.name}")
        self._initialized = False
        self._workflow_id = None
        self._run_task = None

        # A simple workflow state object
        # If under Temporal, storing it as a field on this class
        # means it can be replayed automatically
        self.state = WorkflowState(metadata=metadata or {})

    @property
    def executor(self):
        """Get the workflow executor from the context."""
        executor = self.context.executor
        if executor is None:
            raise ValueError("No executor available in context")
        return executor

    @property
    def id(self) -> str | None:
        """
        Get the workflow ID if it has been assigned.
        NOTE: The run() method will assign a new workflow ID on every run.
        """
        return self._workflow_id

    @classmethod
    async def create(
        cls, name: str | None = None, context: Optional["Context"] = None, **kwargs: Any
    ) -> "Workflow":
        """
        Factory method to create and initialize a workflow instance.

        This default implementation creates a workflow instance and calls initialize().
        Subclasses can override this method for custom initialization logic.

        Args:
            name: Optional name for the workflow (defaults to class name)
            context: Optional context to use (falls back to global context if not provided)
            **kwargs: Additional parameters to pass to the workflow constructor

        Returns:
            An initialized workflow instance
        """
        workflow = cls(name=name, context=context, **kwargs)
        await workflow.initialize()
        return workflow

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> "WorkflowResult[T]":
        """
        Main workflow implementation. Must be overridden by subclasses.

        This is where the user-defined application logic goes. Typically, this involves:
        1. Setting up Agents and attaching LLMs to them
        2. Executing operations using the Agents and their LLMs
        3. Processing results and returning them

        Returns:
            WorkflowResult containing the output of the workflow
        """

    async def _cancel_task(self):
        """
        Wait for a cancel signal and cancel the workflow task.
        """
        signal = await self.executor.wait_for_signal(
            "cancel",
            workflow_id=self._workflow_id,
            signal_description="Waiting for cancel signal",
        )

        self._logger.info(f"Cancel signal received for workflow {self._workflow_id}")
        self.update_status("cancelling")

        # The run task will be cancelled in the run_async method
        return signal

    async def run_async(self, *args: Any, **kwargs: Any) -> str:
        """
        Run the workflow asynchronously and return a workflow ID.

        This creates an async task that will be executed through the executor
        and returns immediately with a workflow ID that can be used to
        check status, resume, or cancel.

        TODO: saqadri - (MAC) - This needs to be updated to use
        the executor for proper workflow orchestration. For example, asyncio vs. Temporal.
        Current implementation only works with asyncio.

        Args:
            *args: Positional arguments to pass to the run method
            **kwargs: Keyword arguments to pass to the run method

        Returns:
            str: A unique workflow ID that can be used to reference this workflow instance
        """

        import asyncio
        from concurrent.futures import CancelledError

        # Generate a unique ID for this workflow instance
        if not self._workflow_id:
            self._workflow_id = str(uuid.uuid4())

        self.update_status("scheduled")

        # Define the workflow execution function
        async def _execute_workflow():
            try:
                # Run the workflow through the executor with pause/cancel monitoring
                self.update_status("running")

                run_task = asyncio.create_task(self.run(*args, **kwargs))
                cancel_task = asyncio.create_task(self._cancel_task())

                # Simply wait for either the run task or cancel task to complete
                try:
                    # Wait for either task to complete, whichever happens first
                    done, _ = await asyncio.wait(
                        [run_task, cancel_task], return_when=asyncio.FIRST_COMPLETED
                    )

                    # Check which task completed
                    if cancel_task in done:
                        # Cancel signal received, cancel the run task
                        run_task.cancel()
                        self.update_status("cancelled")
                        raise CancelledError("Workflow was cancelled")
                    elif run_task in done:
                        # Run task completed, cancel the cancel task
                        cancel_task.cancel()
                        # Get the result (or propagate any exception)
                        result = await run_task
                        self.update_status("completed")
                        return result

                except Exception as e:
                    self._logger.error(f"Error waiting for tasks: {e}")
                    raise

            except CancelledError:
                # Handle cancellation gracefully
                self._logger.info(
                    f"Workflow {self.name} (ID: {self._workflow_id}) was cancelled"
                )
                self.update_status("cancelled")
                raise
            except Exception as e:
                # Log and propagate exceptions
                self._logger.error(
                    f"Error in workflow {self.name} (ID: {self._workflow_id}): {str(e)}"
                )
                self.update_status("error")
                self.state.record_error(e)
                raise
            finally:
                try:
                    # Always attempt to clean up the workflow
                    await self.cleanup()
                except Exception as cleanup_error:
                    # Log but don't fail if cleanup fails
                    self._logger.error(
                        f"Error cleaning up workflow {self.name} (ID: {self._workflow_id}): {str(cleanup_error)}"
                    )

                # Unregister from the workflow registry (if available)
                if self.context and self.context.workflow_registry:
                    self.context.workflow_registry.unregister(self._workflow_id)

        # TODO: saqadri (MAC) - figure out how to do this for different executors.
        # For Temporal, we would replace this with workflow.start() which also doesn't block
        self._run_task = asyncio.create_task(_execute_workflow())

        # Register this workflow with the registry
        if self.context and self.context.workflow_registry:
            self.context.workflow_registry.register(
                self._workflow_id, self, self._run_task
            )

        return self._workflow_id

    async def resume(
        self, signal_name: str | None = "resume", payload: str | None = None
    ) -> bool:
        """
        Send a resume signal to the workflow.

        Args:
            signal_name: The name of the signal to send (default: "resume")
            payload: Optional data to provide to the workflow upon resuming

        Returns:
            bool: True if the resume signal was sent successfully, False otherwise
        """
        if not self._workflow_id:
            self._logger.error("Cannot resume workflow with no ID")
            return False

        try:
            signal = Signal(
                name=signal_name, workflow_id=self._workflow_id, payload=payload
            )
            await self.executor.signal_bus.signal(signal)
            self._logger.info(
                f"{signal_name} signal sent to workflow {self._workflow_id}"
            )
            self.update_status("running")
            return True
        except Exception as e:
            self._logger.error(
                f"Error sending resume signal to workflow {self._workflow_id}: {e}"
            )
            return False

    async def cancel(self) -> bool:
        """
        Cancel the workflow by sending a cancel signal and cancelling its task.

        Returns:
            bool: True if the workflow was cancelled successfully, False otherwise
        """
        if not self._workflow_id:
            self._logger.error("Cannot cancel workflow with no ID")
            return False

        try:
            # First signal the workflow to cancel - this allows for graceful cancellation
            # when the workflow checks for cancellation
            self._logger.info(f"Sending cancel signal to workflow {self._workflow_id}")
            await self.executor.signal("cancel", workflow_id=self._workflow_id)
            return True
        except Exception as e:
            self._logger.error(f"Error cancelling workflow {self._workflow_id}: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the workflow.

        Returns:
            Dict[str, Any]: A dictionary with workflow status information
        """
        status = {
            "id": self._workflow_id,
            "name": self.name,
            "running": self._run_task is not None and not self._run_task.done()
            if self._run_task
            else False,
            "state": self.state.model_dump()
            if hasattr(self.state, "model_dump")
            else self.state.__dict__,
        }

        # Add result/error information if the task is done
        if self._run_task and self._run_task.done():
            try:
                result = self._run_task.result()

                # Convert result to a useful format
                if hasattr(result, "model_dump"):
                    result_data = result.model_dump()
                elif hasattr(result, "__dict__"):
                    result_data = result.__dict__
                else:
                    result_data = str(result)

                status["result"] = result_data
                status["completed"] = True
                status["error"] = None
            except Exception as e:
                status["result"] = None
                status["completed"] = False
                status["error"] = str(e)
                status["exception_type"] = type(e).__name__

        return status

    def update_status(self, status: str) -> None:
        """
        Update the workflow status.

        Args:
            status: The new status to set
        """
        self.state.status = status
        self.state.updated_at = datetime.now(timezone.utc).timestamp()

    # Static registry methods have been moved to the WorkflowRegistry class

    async def update_state(self, **kwargs):
        """Syntactic sugar to update workflow state."""
        for key, value in kwargs.items():
            if hasattr(self.state, "__getitem__"):
                self.state[key] = value
            setattr(self.state, key, value)

        self.state.updated_at = datetime.now(timezone.utc).timestamp()

    async def initialize(self):
        """
        Initialization method that will be called before run.
        Override this to set up any resources needed by the workflow.

        This checks the _initialized flag to prevent double initialization.
        """
        if self._initialized:
            self._logger.debug(f"Workflow {self.name} already initialized, skipping")
            return

        self.state.status = "initializing"
        self._logger.debug(f"Initializing workflow {self.name}")
        self._initialized = True
        self.state.updated_at = datetime.now(timezone.utc).timestamp()

    async def cleanup(self):
        """
        Cleanup method that will be called after run.
        Override this to clean up any resources used by the workflow.

        This checks the _initialized flag to ensure cleanup is only done on initialized workflows.
        """
        if not self._initialized:
            self._logger.debug(
                f"Workflow {self.name} not initialized, skipping cleanup"
            )
            return

        self._logger.debug(f"Cleaning up workflow {self.name}")
        self._initialized = False
        self.state.status = "cleaned_up"
        self.state.updated_at = datetime.now(timezone.utc).timestamp()

    async def __aenter__(self):
        """Support for async context manager pattern."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support for async context manager pattern."""
        await self.cleanup()
