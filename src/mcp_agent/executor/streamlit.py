import asyncio
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    TYPE_CHECKING,
)

import streamlit as st

from pydantic import ConfigDict, ValidationError

from mcp_agent.config import StreamlitSettings
from mcp_agent.executor.executor import (
    AsyncioExecutor,
    ExecutorConfig,
    ExecutionState,
)
from mcp_agent.executor.workflow_signal import (
    BaseSignalHandler,
    Signal,
    SignalHandler,
    SignalRegistration,
    SignalValueT,
    PendingSignal,
)

if TYPE_CHECKING:
    from mcp_agent.context import Context


class StreamlitExecutorConfig(ExecutorConfig, StreamlitSettings):
    """Configuration for Streamlit executor."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class StreamlitSignalHandler(BaseSignalHandler[SignalValueT]):
    """
    Signal handler for Streamlit which uses st.session_state to persist pending signals and registered handlers.
    """

    def __init__(self):
        super().__init__()
        # Initialize session-state keys if they are not already present.
        if "pending_signals" not in st.session_state:
            st.session_state["pending_signals"] = {}  # type: Dict[str, List[PendingSignal]]
        if "signal_handlers" not in st.session_state:
            st.session_state["signal_handlers"] = {}  # type: Dict[str, List[tuple[str, Callable]]]

    async def validate_signal(self, signal: Signal[SignalValueT]) -> None:
        # Run the base validation (which, for example, ensures the signal name is present).
        super().validate_signal(signal)

    async def wait_for_signal(
        self, signal: Signal[SignalValueT], timeout_seconds: int | None = None
    ) -> SignalValueT:
        """
        Wait for a signal to be emitted.
        We create a new PendingSignal with an asyncio.Event and store it in st.session_state.
        """
        await self.validate_signal(signal)
        event = asyncio.Event()
        unique_name = str(uuid.uuid4())
        registration = SignalRegistration(
            signal_name=signal.name,
            unique_name=unique_name,
            workflow_id=signal.workflow_id,
        )
        pending_signal = PendingSignal(registration=registration, event=event)

        pending = st.session_state["pending_signals"].setdefault(signal.name, [])
        pending.append(pending_signal)

        try:
            if timeout_seconds is not None:
                await asyncio.wait_for(event.wait(), timeout_seconds)
            else:
                await event.wait()
            return pending_signal.value
        except asyncio.TimeoutError as e:
            # Remove the pending signal in case of a timeout.
            st.session_state["pending_signals"][signal.name] = [
                ps
                for ps in st.session_state["pending_signals"][signal.name]
                if ps.registration.unique_name != unique_name
            ]
            raise TimeoutError(f"Timeout waiting for signal {signal.name}") from e

    def on_signal(self, signal_name: str) -> Callable:
        """
        Decorator to register a handler for a signal. The handler is stored in st.session_state so that
        it persists across reruns.

        Example:
            @signal_handler.on_signal("approval_needed")
            async def handle_approval(signal: Signal):
                print(f"Approval signal received with payload: {signal.payload}")
        """

        def decorator(func: Callable) -> Callable:
            unique_name = f"{signal_name}_{uuid.uuid4()}"

            async def wrapped(signal: Signal[SignalValueT]):
                try:
                    if asyncio.iscoroutinefunction(func):
                        await func(signal)
                    else:
                        func(signal)
                except Exception as e:
                    print(f"Error in signal handler {signal_name}: {e}")

            handlers = st.session_state["signal_handlers"].setdefault(signal_name, [])
            handlers.append((unique_name, wrapped))
            return wrapped

        return decorator

    async def signal(self, signal: Signal[SignalValueT]) -> None:
        """
        Emit a signal: notify any pending waiters and run all registered handler callbacks.
        """
        await self.validate_signal(signal)
        # Notify waiting coroutines.
        pending = st.session_state["pending_signals"].get(signal.name, [])
        for ps in pending:
            ps.value = signal.payload
            ps.event.set()
        # Clear out the pending signals list for this signal.
        st.session_state["pending_signals"][signal.name] = []

        # Call any registered signal handlers.
        handlers = st.session_state["signal_handlers"].get(signal.name, [])
        tasks = []
        for _, handler in handlers:
            tasks.append(handler(signal))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class StreamlitExecutor(AsyncioExecutor):
    """
    Executor that uses st.session_state to store durable workflow state across Streamlit reruns.
    """

    def __init__(
        self,
        config: ExecutorConfig | None = None,
        signal_bus: SignalHandler | None = None,
        context: Optional["Context"] = None,
        **kwargs,
    ):
        signal_bus = signal_bus or StreamlitSignalHandler()
        super().__init__(
            engine="streamlit",
            config=config,
            signal_bus=signal_bus,
            context=context,
            **kwargs,
        )
        self.config: StreamlitExecutorConfig = (
            config or self.context.config.streamlit or StreamlitExecutorConfig()
        )
        # Ensure we have a key in session_state to hold workflow states.
        if "workflow_states" not in st.session_state:
            st.session_state["workflow_states"] = {}  # type: Dict[str, ExecutionState]

    async def get_state(self, workflow_id: str, key: Optional[str]) -> ExecutionState:
        """
        Retrieve the workflow state from st.session_state.
        If no state exists for the given workflow_id, a new ExecutionState is created.
        If key is provided, then the corresponding attribute is returned (or an error is raised).
        """
        if not workflow_id:
            raise ValueError("get_state request requires a workflow_id.")
        workflow_states: Dict[str, ExecutionState] = st.session_state["workflow_states"]
        if workflow_id not in workflow_states:
            workflow_states[workflow_id] = ExecutionState()
        state = workflow_states[workflow_id]
        if not key:
            return state
        try:
            value = getattr(state, key)
            if not isinstance(value, ExecutionState):
                raise ValueError(
                    f"Object retrieved from state field '{key}' is not an ExecutionState type"
                )
            return value
        except AttributeError:
            raise KeyError(f"Field '{key}' not found in ExecutionState")

    async def set_state(self, workflow_id: str, key: str, state_value: Any) -> bool:
        """
        Save the state (or a part thereof) for the given workflow_id in st.session_state.
        """
        if not workflow_id:
            raise ValueError("set_state request requires a workflow_id.")
        workflow_states: Dict[str, ExecutionState] = st.session_state["workflow_states"]
        if workflow_id not in workflow_states:
            workflow_states[workflow_id] = ExecutionState()
        try:
            setattr(workflow_states[workflow_id], key, state_value)
            return True
        except ValidationError as e:
            raise ValueError(f"Invalid state value for field '{key}': {e}")
