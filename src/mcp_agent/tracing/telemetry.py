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
            span_name = name or f"{func.__qualname__}"

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer(self.context)
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
                tracer = get_tracer(self.context)
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
            record_attribute(span, f"arg_{i}", arg)

        record_attributes(span, kwargs)


def serialize_attribute(key: str, value: Any) -> Dict[str, Any]:
    """Serialize a single attribute value into a flat dict of OpenTelemetry-compatible values."""
    serialized = {}

    if is_otel_serializable(value):
        serialized[key] = value

    elif isinstance(value, dict):
        for sub_key, sub_value in value.items():
            serialized.update(serialize_attribute(f"{key}.{sub_key}", sub_value))

    elif isinstance(value, Callable):
        serialized[f"{key}_callable_name"] = getattr(value, "__qualname__", str(value))
        serialized[f"{key}_callable_module"] = getattr(value, "__module__", "unknown")
        serialized[f"{key}_is_coroutine"] = asyncio.iscoroutinefunction(value)

    elif inspect.iscoroutine(value):
        serialized[f"{key}_coroutine"] = str(value)
        serialized[f"{key}_is_coroutine"] = True

    return serialized


def serialize_attributes(
    attributes: Dict[str, Any], prefix: str = ""
) -> Dict[str, Any]:
    """Serialize a dict of attributes into a flat OpenTelemetry-compatible dict."""
    serialized = {}
    prefix = f"{prefix}." if prefix else ""

    for key, value in attributes.items():
        full_key = f"{prefix}{key}"
        serialized.update(serialize_attribute(full_key, value))

    return serialized


def record_attribute(span, key, value):
    """Record a single serializable value on the span."""
    if is_otel_serializable(value):
        span.set_attribute(key, value)
    else:
        serialized = serialize_attribute(key, value)
        for attr_key, attr_value in serialized.items():
            span.set_attribute(attr_key, attr_value)


def record_attributes(span, attributes: Dict[str, Any], prefix: str = ""):
    """Record a dict of attributes on the span after serialization."""
    serialized = serialize_attributes(attributes, prefix)
    for attr_key, attr_value in serialized.items():
        span.set_attribute(attr_key, attr_value)


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


def get_tracer(context: "Context") -> trace.Tracer:
    """
    Get the OpenTelemetry tracer for the context.
    """
    return context.tracer or trace.get_tracer("mcp-agent")


telemetry = TelemetryManager()
