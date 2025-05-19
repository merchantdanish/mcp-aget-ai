import asyncio

from typing import (
    Any,
    Dict,
    Optional,
    List,
    TYPE_CHECKING,
)

from mcp_agent.logging.logger import get_logger
from mcp_agent.executor.workflow_registry import WorkflowRegistry

if TYPE_CHECKING:
    from mcp_agent.executor.temporal import TemporalExecutor
    from mcp_agent.executor.workflow import Workflow

logger = get_logger(__name__)


class TemporalWorkflowRegistry(WorkflowRegistry):
    """
    Registry for tracking workflow instances in Temporal.
    This implementation queries Temporal for workflow status and manages workflows.
    """

    def __init__(self, executor: "TemporalExecutor"):
        super().__init__()
        self._executor = executor
        # We still keep a local cache for fast lookups, but the source of truth is Temporal
        self._local_workflows: Dict[str, "Workflow"] = {}  # run_id -> workflow
        self._workflow_ids: Dict[str, List[str]] = {}  # workflow_id -> list of run_ids

    async def register(
        self,
        workflow: "Workflow",
        run_id: str | None = None,
        workflow_id: str | None = None,
        task: Optional["asyncio.Task"] = None,
    ) -> None:
        self._local_workflows[run_id] = workflow

        # Add run_id to the list for this workflow_id
        if workflow_id not in self._workflow_ids:
            self._workflow_ids[workflow_id] = []
        self._workflow_ids[workflow_id].append(run_id)

    async def unregister(self, run_id: str, workflow_id: str | None = None) -> None:
        if run_id in self._local_workflows:
            workflow = self._local_workflows[run_id]
            workflow_id = workflow.workflow_id

            # Remove from workflow_ids mapping
            if workflow_id in self._workflow_ids:
                if run_id in self._workflow_ids[workflow_id]:
                    self._workflow_ids[workflow_id].remove(run_id)
                if not self._workflow_ids[workflow_id]:
                    del self._workflow_ids[workflow_id]

            # Remove workflow from local cache
            self._local_workflows.pop(run_id, None)

    async def get_workflow(
        self, run_id: str, workflow_id: str | None = None
    ) -> Optional["Workflow"]:
        return self._local_workflows.get(run_id)

    async def resume_workflow(
        self,
        run_id: str,
        workflow_id: str | None = None,
        signal_name: str | None = "resume",
        payload: Any | None = None,
    ) -> bool:
        # Ensure the Temporal client is connected
        await self._executor.ensure_client()

        try:
            # Get the workflow handle directly from Temporal
            # In Temporal, we need both workflow_id and run_id to target a specific run
            workflow = self.get_workflow(run_id)
            if not workflow:
                logger.error(
                    f"Workflow with run_id {run_id} not found in local registry"
                )
                return False

            workflow_id = workflow.workflow_id

            # Get the handle and send the signal
            handle = self._executor.client.get_workflow_handle(
                workflow_id=workflow_id, run_id=run_id
            )
            await handle.signal(signal_name, payload)
            logger.info(
                f"Sent signal {signal_name} to workflow {workflow_id} run {run_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Error signaling workflow {run_id}: {e}")
            return False

    async def cancel_workflow(
        self, run_id: str, workflow_id: str | None = None
    ) -> bool:
        # Ensure the Temporal client is connected
        await self._executor.ensure_client()

        try:
            # Get the workflow from local registry
            workflow = self.get_workflow(run_id)
            if not workflow:
                logger.error(
                    f"Workflow with run_id {run_id} not found in local registry"
                )
                return False

            workflow_id = workflow.workflow_id

            # Get the handle and cancel the workflow
            handle = self._executor.client.get_workflow_handle(
                workflow_id=workflow_id, run_id=run_id
            )
            await handle.cancel()
            logger.info(f"Cancelled workflow {workflow_id} run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling workflow {run_id}: {e}")
            return False

    async def _get_temporal_workflow_status(
        self, workflow_id: str, run_id: str
    ) -> Dict[str, Any]:
        """
        Get the status of a workflow directly from Temporal.

        Args:
            workflow_id: The workflow ID
            run_id: The run ID

        Returns:
            A dictionary with workflow status information from Temporal
        """
        # Ensure the Temporal client is connected
        await self._executor.ensure_client()

        try:
            # Get the workflow handle and describe the workflow
            handle = self._executor.client.get_workflow_handle(
                workflow_id=workflow_id, run_id=run_id
            )

            # Get the workflow description
            describe = await handle.describe()

            # Convert to a dictionary with our standard format
            status = {
                "id": workflow_id,
                "run_id": run_id,
                "name": describe.workflow_type.name,
                "status": describe.status.name,
                "start_time": describe.start_time.timestamp()
                if describe.start_time
                else None,
                "execution_time": describe.execution_time.timestamp()
                if describe.execution_time
                else None,
                "close_time": describe.close_time.timestamp()
                if describe.close_time
                else None,
                "history_length": describe.history_length,
                "parent_namespace": describe.parent_namespace,
                "parent_workflow_id": describe.parent_workflow_id,
                "parent_run_id": describe.parent_run_id,
                "search_attributes": describe.search_attributes,
                "memo": describe.memo,
            }

            return status
        except Exception as e:
            logger.error(f"Error getting temporal workflow status: {e}")
            # Return basic status with error information
            return {
                "id": workflow_id,
                "run_id": run_id,
                "status": "ERROR",
                "error": str(e),
            }

    async def get_workflow_status_2(
        self, run_id: str, workflow_id: str | None = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the status of a workflow by run ID.

        Args:
            run_id: The unique ID of the workflow run to check

        Returns:
            The workflow status if found, None otherwise
        """
        workflow = self.get_workflow(run_id)
        if not workflow:
            return None

        # For Temporal, we need to query the service for the latest status
        # This is done asynchronously, so we need to create a task
        import asyncio

        loop = asyncio.get_event_loop()
        workflow_id = workflow.workflow_id

        # If we're in an async context, use coroutine, otherwise run the coroutine in the loop
        if loop.is_running():
            # Create task and wait for it to complete
            asyncio.create_task(self._get_temporal_workflow_status(workflow_id, run_id))
            # We can't await here because we're in a synchronous method
            # Return the local status, and clients should use the async version if possible
            return workflow.get_status()
        else:
            # We're not in an async context, run the coroutine in the loop
            return loop.run_until_complete(
                self._get_temporal_workflow_status(workflow_id, run_id)
            )

    async def get_workflow_status(
        self, run_id: str, workflow_id: str | None = None
    ) -> Optional[Dict[str, Any]]:
        workflow = self.get_workflow(run_id)
        if not workflow:
            return None

        workflow_id = workflow.workflow_id
        return await self._get_temporal_workflow_status(workflow_id, run_id)

    async def list_workflow_statuses(self) -> List[Dict[str, Any]]:
        result = []
        for run_id, workflow in self._local_workflows.items():
            workflow_id = workflow.workflow_id
            status = await self._get_temporal_workflow_status(workflow_id, run_id)
            result.append(status)

        return result

    def list_workflow_statuses2(self) -> List[Dict[str, Any]]:
        """
        List all registered workflow instances with their status.
        For Temporal, this uses local status information.

        Returns:
            A list of dictionaries with workflow information
        """
        result = []
        for run_id, workflow in self._local_workflows.items():
            # Get the workflow status from the local instance
            status = workflow.get_status()
            result.append(status)

        return result

    async def list_workflows(self) -> List["Workflow"]:
        """
        List all registered workflow instances.

        Returns:
            A list of workflow instances
        """
        return list(self._local_workflows.values())
