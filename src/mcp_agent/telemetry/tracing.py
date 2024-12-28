"""
Set up OpenTelemetry tracing for the MCP Agent application
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import set_global_textmap, extract as otel_extract
from opentelemetry.trace import set_span_in_context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from mcp_agent.config import settings

# Set global propagator
set_global_textmap(TraceContextTextMapPropagator())


def setup_tracing(
    service_name="my-orchestrator",
    service_instance_id="instance-id-1",
    service_version="1.0.0",
):
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.instance.id": service_instance_id,
            "service.version": service_version,
        }
    )
    tracer_provider = TracerProvider(resource=resource)
    otlp_endpoint = settings.otlp_endpoint
    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        # Default to console exporter in development
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(tracer_provider)


# def start_span_from_mcp_request(request):
#     tracer = get_tracer(__name__)
#     carrier = request.get("params", {}).get("_meta", {})
#     ctx = TraceContextTextMapPropagator().extract(carrier)
#     with tracer.start_as_current_span(
#         request.get("method", "unknown"), context=ctx
#     ) as span:
#         return span
def start_span_from_mcp_request(request):
    # Extract traceparent from request["params"]["_meta"] if present
    carrier = {}
    _meta = request.get("params", {}).get("_meta", {})
    if "traceparent" in _meta:
        carrier["traceparent"] = _meta["traceparent"]
    if "tracestate" in _meta:
        carrier["tracestate"] = _meta["tracestate"]
    ctx = otel_extract(carrier, context=Context())
    tracer = trace.get_tracer(__name__)
    span = tracer.start_span(request["method"], context=ctx)
    return span, set_span_in_context(span)


def inject_trace_context(arguments):
    """Inject current span context into arguments['_meta']"""
    carrier = {}
    TraceContextTextMapPropagator().inject(carrier)
    _meta = arguments.get("_meta", {})
    if "traceparent" in carrier:
        _meta["traceparent"] = carrier["traceparent"]
    if "tracestate" in carrier:
        _meta["tracestate"] = carrier["tracestate"]
    arguments["_meta"] = _meta
