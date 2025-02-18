"""
Streamlit based executor for the MCP Agent.
Streamlit reruns the entire application on state changes,
and therefore needs careful state management to cache execution state.
This class provides that.
"""

from typing import (
    Optional,
    TYPE_CHECKING,
)

from pydantic import ConfigDict

from mcp_agent.config import StreamlitSettings
from mcp_agent.executor.executor import (
    AsyncioExecutor,
    ExecutorConfig,
)
from mcp_agent.executor.workflow_signal import (
    BaseSignalHandler,
    SignalHandler,
    SignalValueT,
)

if TYPE_CHECKING:
    from mcp_agent.context import Context


class StreamlitExecutorConfig(ExecutorConfig, StreamlitSettings):
    """Configuration for Streamlit executor."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class StreamlitSignalHandler(BaseSignalHandler[SignalValueT]):
    """Temporal-based signal handling using workflow signals"""

    async def wait_for_signal(self, signal, timeout_seconds=None):
        raise NotImplementedError("...")

    async def on_signal(self, signal_name):
        raise NotImplementedError("...")

    async def signal(self, signal):
        raise NotImplementedError("...")

    async def validate_signal(self, signal):
        raise NotImplementedError("...")


class StreamlitExecutor(AsyncioExecutor):
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

    async def get_state(self, workflow_id, key):
        raise NotImplementedError("...")

    async def set_state(self, workflow_id, key, state):
        raise NotImplementedError("...")
