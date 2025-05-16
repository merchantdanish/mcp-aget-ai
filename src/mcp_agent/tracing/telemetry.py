"""
Telemetry manager that defines distributed tracing decorators for OpenTelemetry traces/spans
for the Logger module for MCP Agent
"""

import asyncio
from collections.abc import Sequence
import functools
import inspect
from typing import Any, Dict, Callable, Optional, Tuple, TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.context import Context as OtelContext
from opentelemetry.propagate import extract as otel_extract
from opentelemetry.trace import set_span_in_context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from opentelemetry.trace import SpanKind, Status, StatusCode

from mcp_agent.core.context_dependent import ContextDependent

if TYPE_CHECKING:
    from mcp_agent.core.context import Context


class TelemetryManager(ContextDependent):
    """
    Simple manager for creating OpenTelemetry spans automatically.
    Decorator usage: @telemetry.traced("SomeSpanName")
    """

    def __init__(self, context: Optional["Context"] = None, **kwargs):
        super().__init__(context=context, **kwargs)

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
                tracer = self.context.tracer or trace.get_tracer("mcp-agent")
                with tracer.start_as_current_span(span_name, kind=kind) as span:
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
                tracer = self.context.tracer or trace.get_tracer("mcp-agent")
                with tracer.start_as_current_span(span_name, kind=kind) as span:
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
        """Optionally record primitive args and function/coroutine metadata as span attributes."""
        for i, arg in enumerate(args):
            self._record_attribute(span, f"arg_{i}", arg)

        for k, v in kwargs.items():
            self._record_attribute(span, k, v)

    def _record_attribute(self, span, key, value):
        """Record a single value intelligently on the span."""
        if is_otel_serializable(value):
            span.set_attribute(key, str(value))
        elif isinstance(value, Callable):
            span.set_attribute(
                f"{key}_callable_name", getattr(value, "__qualname__", str(value))
            )
            span.set_attribute(
                f"{key}_callable_module", getattr(value, "__module__", "unknown")
            )
            span.set_attribute(
                f"{key}_is_coroutine", asyncio.iscoroutinefunction(value)
            )
        elif inspect.iscoroutine(value):
            span.set_attribute(f"{key}_coroutine", str(value))
            span.set_attribute(f"{key}_is_coroutine", True)


class MCPRequestTrace:
    """Helper class for trace context propagation in MCP"""

    @staticmethod
    def start_span_from_mcp_request(
        method: str, params: Dict[str, Any]
    ) -> Tuple[trace.Span, OtelContext]:
        """Extract trace context from incoming MCP request and start a new span"""
        # Extract trace context from _meta if present
        carrier = {}
        _meta = params.get("_meta", {})
        if "traceparent" in _meta:
            carrier["traceparent"] = _meta["traceparent"]
        if "tracestate" in _meta:
            carrier["tracestate"] = _meta["tracestate"]

        # Extract context and start span
        ctx = otel_extract(carrier, context=OtelContext())
        tracer = trace.get_tracer(__name__)
        span = tracer.start_span(method, context=ctx, kind=SpanKind.SERVER)
        return span, set_span_in_context(span)

    @staticmethod
    def inject_trace_context(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Inject current trace context into outgoing MCP request arguments"""
        carrier = {}
        TraceContextTextMapPropagator().inject(carrier)

        # Create or update _meta with trace context
        _meta = arguments.get("_meta", {})
        if "traceparent" in carrier:
            _meta["traceparent"] = carrier["traceparent"]
        if "tracestate" in carrier:
            _meta["tracestate"] = carrier["tracestate"]
        arguments["_meta"] = _meta

        return arguments


def is_otel_serializable(value: Any) -> bool:
    """
    Check if a value is serializable by OpenTelemetry
    """
    allowed_types = (bool, str, bytes, int, float)
    if isinstance(value, allowed_types):
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(isinstance(item, allowed_types) for item in value)
    return False


telemetry = TelemetryManager()
