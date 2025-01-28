import asyncio
from dataclasses import dataclass
from typing import Any

import streamlit as st

from mcp_agent.executor.workflow_signal import SignalHandler, Signal


@dataclass
class SignalState:
    """State for a pending signal that persists across reruns"""

    value: Any | None = None
    completed: bool = False
    timestamp: float | None = None


class StreamlitSignalHandler(SignalHandler):
    """Signal handler that works with Streamlit's rerun model"""

    def __init__(self):
        # Initialize session state
        if "pending_signals" not in st.session_state:
            st.session_state.pending_signals = {}

    async def signal(self, signal: Signal):
        """Store signal in session state and mark as completed"""
        signal_state = SignalState(
            value=signal.payload,
            completed=True,
            timestamp=asyncio.get_event_loop().time(),
        )
        st.session_state.pending_signals[signal.name] = signal_state

        # Force a rerun to update any waiting components
        st.rerun()

    async def wait_for_signal(self, signal: Signal, timeout_seconds: int | None = None):
        """
        Wait for signal using polling that's safe with Streamlit reruns.
        """
        # Register this wait if not already registered
        if signal.name not in st.session_state.pending_signals:
            st.session_state.pending_signals[signal.name] = SignalState()

        start_time = asyncio.get_event_loop().time()

        while True:
            # Check if signal completed
            signal_state = st.session_state.pending_signals.get(signal.name)
            if signal_state and signal_state.completed:
                # Clean up and return value
                value = signal_state.value
                del st.session_state.pending_signals[signal.name]
                return value

            # Check timeout
            if timeout_seconds is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout_seconds:
                    del st.session_state.pending_signals[signal.name]
                    raise TimeoutError(f"Timeout waiting for signal {signal.name}")

            # Small wait before checking again
            # This prevents tight loop that could block Streamlit
            await asyncio.sleep(0.1)

    def on_signal(self, signal_name: str):
        """Register a handler for a signal"""

        def decorator(func):
            if "signal_handlers" not in st.session_state:
                st.session_state.signal_handlers = {}
            st.session_state.signal_handlers[signal_name] = func
            return func

        return decorator
