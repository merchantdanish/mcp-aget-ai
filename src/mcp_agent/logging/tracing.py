"""
Telemetry manager that defines distributed tracing decorators for OpenTelemetry traces/spans
for the Logger module for MCP Agent
"""

import asyncio
import functools
from typing import Any, Dict, Callable

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from mcp_agent.context import get_current_context


class TelemetryManager:
    """
    Simple manager for creating OpenTelemetry spans automatically.
    Decorator usage: @telemetry.traced("SomeSpanName")
    """

    def __init__(self):
        # If needed, configure resources, exporters, etc.
        # E.g.: from opentelemetry.sdk.trace import TracerProvider
        # trace.set_tracer_provider(TracerProvider(...))
        context = get_current_context()
        self.tracer = context.tracer or trace.get_tracer("mcp_agent")

    def traced(
        self,
        name: str | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Dict[str, Any] = None,
    ) -> Callable:
        """
        Decorator that automatically creates and manages a span for a function.
        Works for both async and sync functions.
        """

        def decorator(func):
            span_name = name or f"{func.__module__}.{func.__qualname__}"

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(span_name, kind=kind) as span:
                    if attributes:
                        for k, v in attributes.items():
                            span.set_attribute(k, v)
                    # Record simple args
                    self._record_args(span, args, kwargs)
                    try:
                        res = await func(*args, **kwargs)
                        return res
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR))
                        raise

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(span_name, kind=kind) as span:
                    if attributes:
                        for k, v in attributes.items():
                            span.set_attribute(k, v)
                    # Record simple args
                    self._record_args(span, args, kwargs)
                    try:
                        res = func(*args, **kwargs)
                        return res
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR))
                        raise

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def _record_args(self, span, args, kwargs):
        """Optionally record primitive args as span attributes."""
        for i, arg in enumerate(args):
            if isinstance(arg, (str, int, float, bool)):
                span.set_attribute(f"arg_{i}", str(arg))
        for k, v in kwargs.items():
            if isinstance(v, (str, int, float, bool)):
                span.set_attribute(k, str(v))


telemetry = TelemetryManager()
