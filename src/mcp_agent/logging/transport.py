"""
Transports for the Logger module for MCP Agent, including:
- Local + optional remote event transport
- Async event bus
"""

import asyncio

from typing import Dict, Protocol

from opentelemetry import trace

from mcp_agent.logging.events import Event
from mcp_agent.logging.listeners import EventListener, LifecycleAwareListener


class EventTransport(Protocol):
    """
    Pluggable interface for sending events to a remote or external system
    (Kafka, RabbitMQ, REST, etc.).
    """

    async def send_event(self, event: Event):
        """
        Send an event to the external system.
        Args:
            event: Event to send.
        """
        ...


class NoOpTransport(EventTransport):
    """Default transport that does nothing (purely local)."""

    async def send_event(self, event):
        """Do nothing."""
        pass


class RemoteTransport(EventTransport):
    """
    Stub for "distributed" transport.
    Override this to implement actual networking/publishing.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def send_event(self, event: Event):
        print(
            f"[RemoteTransport] Sending event to {self.endpoint} => {event.name} {event.type}"
        )


class AsyncEventBus:
    """
    Async event bus with local in-process listeners + optional remote transport.
    Also injects distributed tracing (trace_id, span_id) if there's a current span.
    """

    _instance = None

    def __init__(self, transport: EventTransport | None = None):
        self.transport: EventTransport = transport or NoOpTransport()

        self.listeners: Dict[str, EventListener] = {}
        self._queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    @classmethod
    def get(cls, transport: EventTransport | None = None) -> "AsyncEventBus":
        """
        Get the singleton instance of the event bus.
        """
        if cls._instance is None:
            cls._instance = cls(transport=transport)
        return cls._instance

    async def start(self):
        """Start the event bus and all lifecycle-aware listeners."""
        # Start each lifecycle-aware listener
        for listener in self.listeners.values():
            if isinstance(listener, LifecycleAwareListener):
                await listener.start()

        if not self._task:
            self._task = asyncio.create_task(self._process_events())

    async def stop(self):
        """Stop the event bus and all lifecycle-aware listeners."""
        # Cancel background processing
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Stop each lifecycle-aware listener
        for listener in self.listeners.values():
            if isinstance(listener, LifecycleAwareListener):
                await listener.stop()

    async def emit(self, event: Event):
        # Inject current tracing info if available
        span = trace.get_current_span()
        if span.is_recording():
            ctx = span.get_span_context()
            event.trace_id = f"{ctx.trace_id:032x}"
            event.span_id = f"{ctx.span_id:016x}"

        # Enqueue for local in-process listeners
        await self._queue.put(event)
        # Also forward to remote transport
        await self.transport.send_event(event)

    def add_listener(self, name: str, listener: EventListener):
        """Add a listener to the event bus."""
        self.listeners[name] = listener

    def remove_listener(self, name: str):
        """Remove a listener from the event bus."""
        self.listeners.pop(name, None)

    async def _process_events(self):
        while True:
            event = await self._queue.get()
            tasks = []
            for listener in self.listeners.values():
                tasks.append(listener.handle_event(event))
            await asyncio.gather(*tasks, return_exceptions=True)
            self._queue.task_done()


def configure_distributed(
    transport: RemoteTransport | None = None, endpoint: str | None = None
):
    """
    Enable a "distributed" transport so events also go to a remote system.
    """
    if not transport and not endpoint:
        raise ValueError("Must provide either a transport or an endpoint")

    transport = transport or RemoteTransport(endpoint=endpoint)
    bus = AsyncEventBus.get(transport=transport)
    return bus
