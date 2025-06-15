import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import warnings
from mcp_agent.executor.temporal import TemporalExecutor, TemporalExecutorConfig


@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def mock_context():
    context = MagicMock()
    context.config.temporal = TemporalExecutorConfig(
        host="localhost:7233",
        namespace="test-namespace",
        task_queue="test-queue",
        timeout_seconds=10,
    )
    context.task_registry = MagicMock()
    context.app = MagicMock()
    context.app.workflows = MagicMock()
    return context


@pytest.fixture
def executor(mock_client, mock_context):
    config = TemporalExecutorConfig(
        host="localhost:7233",
        namespace="test-namespace",
        task_queue="test-queue",
        timeout_seconds=10,
    )
    return TemporalExecutor(config=config, client=mock_client, context=mock_context)


@pytest.mark.asyncio
async def test_ensure_client(executor):
    # Should not reconnect if client is already set
    client = await executor.ensure_client()
    assert client is executor.client


def test_wrap_as_activity(executor):
    def test_func(x=1, y=2):
        return x + y

    wrapped = executor.wrap_as_activity("test_activity", test_func)
    assert hasattr(wrapped, "__temporal_activity_definition")


@pytest.mark.asyncio
@patch("temporalio.workflow._Runtime.current", return_value=None)
async def test_execute_task_as_async_sync(mock_runtime, executor):
    def sync_func(x, y):
        return x + y

    result = await executor._execute_task_as_async(sync_func, 2, 3)
    assert result == 5


@pytest.mark.asyncio
async def test_execute_task_as_async_async(executor):
    async def async_func(x, y):
        return x * y

    result = await executor._execute_task_as_async(async_func, 2, 4)
    assert result == 8


@pytest.mark.asyncio
@patch("temporalio.workflow._Runtime.current", return_value=None)
async def test_execute_task_outside_workflow(mock_runtime, executor):
    def test_func():
        return 42

    result = await executor._execute_task(test_func)
    assert result == 42


@pytest.mark.asyncio
async def test_start_workflow_with_explicit_workflow_id_uses_provided_id(executor, mock_context):
    """Test that providing explicit workflow_id parameter uses that exact ID"""
    class DummyWorkflow:
        @staticmethod
        async def run(arg1):
            return "ok"

    mock_workflow = DummyWorkflow
    mock_context.app.workflows.get.return_value = mock_workflow
    executor.client.start_workflow = AsyncMock(return_value=AsyncMock())

    await executor.start_workflow("MyWorkflowClass", "arg1", workflow_id="custom-id-123", wait_for_result=False)

    # Verify workflow class lookup and start_workflow call
    mock_context.app.workflows.get.assert_called_once_with("MyWorkflowClass")
    executor.client.start_workflow.assert_called_once()

    # Verify the workflow_id used is the custom one
    call_kwargs = executor.client.start_workflow.call_args[1]
    assert call_kwargs['id'] == "custom-id-123"


@pytest.mark.asyncio
async def test_start_workflow_without_workflow_id_warns_and_autogenerates_id(executor, mock_context):
    """Test that omitting workflow_id triggers deprecation warning and auto-generates ID"""
    class DummyWorkflow:
        @staticmethod
        async def run(arg1):
            return "ok"

    mock_workflow = DummyWorkflow
    mock_context.app.workflows.get.return_value = mock_workflow
    executor.client.start_workflow = AsyncMock(return_value=AsyncMock())

    # Test deprecated pattern should issue a warning
    with pytest.warns(DeprecationWarning, match="Using the first parameter as both workflow class name and instance ID is deprecated"):
        await executor.start_workflow("MyWorkflowClass", "arg1", wait_for_result=False)

    # Verify workflow class lookup
    mock_context.app.workflows.get.assert_called_once_with("MyWorkflowClass")
    executor.client.start_workflow.assert_called_once()

    # Verify auto-generated workflow_id format
    call_kwargs = executor.client.start_workflow.call_args[1]
    assert call_kwargs['id'].startswith("MyWorkflowClass-")
    assert len(call_kwargs['id']) == len("MyWorkflowClass-") + 36  # UUID4 length


@pytest.mark.asyncio
async def test_start_workflow_with_zero_arguments_passes_no_input(executor, mock_context):
    """Test workflow with no run() arguments passes no input to temporal client"""
    class ZeroArgWorkflow:
        @staticmethod
        async def run():
            return "ok"

    mock_workflow = ZeroArgWorkflow
    mock_context.app.workflows.get.return_value = mock_workflow
    executor.client.start_workflow = AsyncMock(return_value=AsyncMock())

    await executor.start_workflow("ZeroArgWorkflow", workflow_id="test-id")

    # Should call start_workflow without input argument
    executor.client.start_workflow.assert_called_once()
    call_args = executor.client.start_workflow.call_args[0]
    call_kwargs = executor.client.start_workflow.call_args[1]

    assert len(call_args) == 1  # Only workflow class, no input arg
    assert call_kwargs['id'] == "test-id"


@pytest.mark.asyncio
async def test_start_workflow_with_multiple_arguments_packs_into_sequence(executor, mock_context):
    """Test workflow with multiple run() arguments packs them into a sequence"""
    class MultiArgWorkflow:
        @staticmethod
        async def run(arg1, arg2, arg3):
            return "ok"

    mock_workflow = MultiArgWorkflow
    mock_context.app.workflows.get.return_value = mock_workflow
    executor.client.start_workflow = AsyncMock(return_value=AsyncMock())

    await executor.start_workflow("MultiArgWorkflow", "val1", "val2", "val3", workflow_id="test-id")

    executor.client.start_workflow.assert_called_once()
    call_args = executor.client.start_workflow.call_args[0]
    call_kwargs = executor.client.start_workflow.call_args[1]

    # Should pack multiple args into a sequence
    assert len(call_args) == 2  # workflow class + input sequence
    assert call_args[1] == ["val1", "val2", "val3"]
    assert call_kwargs['id'] == "test-id"


@pytest.mark.asyncio
async def test_start_workflow_with_keyword_arguments_binds_correctly(executor, mock_context):
    """Test workflow with keyword arguments binds them in correct order"""
    class KwargsWorkflow:
        @staticmethod
        async def run(arg1, optional_arg="default"):
            return "ok"

    mock_workflow = KwargsWorkflow
    mock_context.app.workflows.get.return_value = mock_workflow
    executor.client.start_workflow = AsyncMock(return_value=AsyncMock())

    await executor.start_workflow("KwargsWorkflow", "val1", workflow_id="test-id", optional_arg="custom")

    executor.client.start_workflow.assert_called_once()
    call_args = executor.client.start_workflow.call_args[0]

    # Should bind kwargs properly
    assert call_args[1] == ["val1", "custom"]


@pytest.mark.asyncio
async def test_start_workflow_wait_for_result_returns_workflow_result(executor, mock_context):
    """Test wait_for_result=True waits for and returns the workflow result"""
    class DummyWorkflow:
        @staticmethod
        async def run():
            return "workflow_result"

    mock_workflow = DummyWorkflow
    mock_context.app.workflows.get.return_value = mock_workflow

    mock_handle = AsyncMock()
    mock_handle.result = AsyncMock(return_value="workflow_result")
    executor.client.start_workflow = AsyncMock(return_value=mock_handle)

    result = await executor.start_workflow("DummyWorkflow", workflow_id="test-id", wait_for_result=True)

    # Should wait for result and return it
    mock_handle.result.assert_awaited_once()
    assert result == "workflow_result"


@pytest.mark.asyncio
async def test_start_workflow_autogenerated_ids_are_unique_across_calls(executor, mock_context):
    """Test that auto-generated workflow IDs are unique across multiple calls"""
    class DummyWorkflow:
        @staticmethod
        async def run():
            return "ok"

    mock_workflow = DummyWorkflow
    mock_context.app.workflows.get.return_value = mock_workflow
    executor.client.start_workflow = AsyncMock(return_value=AsyncMock())

    # Call multiple times with deprecated pattern
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        await executor.start_workflow("DummyWorkflow")
        first_call_id = executor.client.start_workflow.call_args[1]['id']

        executor.client.start_workflow.reset_mock()
        await executor.start_workflow("DummyWorkflow")
        second_call_id = executor.client.start_workflow.call_args[1]['id']

    # IDs should be different
    assert first_call_id != second_call_id
    assert first_call_id.startswith("DummyWorkflow-")
    assert second_call_id.startswith("DummyWorkflow-")


@pytest.mark.asyncio
async def test_terminate_workflow(executor):
    mock_handle = AsyncMock()
    executor.client.get_workflow_handle = MagicMock(return_value=mock_handle)
    await executor.terminate_workflow("workflow-id", "run-id", "Termination reason")
    executor.client.get_workflow_handle.assert_called_once_with(
        workflow_id="workflow-id", run_id="run-id"
    )
    mock_handle.terminate.assert_awaited_once_with(reason="Termination reason")
