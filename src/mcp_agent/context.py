"""
A central context object to store global state that is shared across the application.
"""

import asyncio

from pydantic import BaseModel, ConfigDict
from mcp import ServerSession
from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from mcp_agent.config import Settings, settings
from mcp_agent.logging.events import EventFilter
from mcp_agent.logging.logger import LoggingConfig
from mcp_agent.logging.transport import create_transport
from mcp_agent.mcp_server_registry import ServerRegistry


class Context(BaseModel):
    """
    Context that is passed around through the application.
    This is a global context that is shared across the application.
    """

    config: Settings | None = None
    upstream_session: ServerSession | None = None  # TODO: saqadri - figure this out
    server_registry: ServerRegistry | None = None
    tracer: trace.Tracer | None = None

    model_config = ConfigDict(extra="allow")


async def configure_otel(config: Settings):
    """
    Configure OpenTelemetry based on the application config.
    """
    if not config.otel.enabled:
        return

    # Set up global textmap propagator first
    set_global_textmap(TraceContextTextMapPropagator())

    service_name = config.otel.service_name
    service_instance_id = config.otel.service_instance_id
    service_version = config.otel.service_version

    # Create resource identifying this service
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.instance.id": service_instance_id,
            "service.version": service_version,
        }
    )

    # Create provider with resource
    tracer_provider = TracerProvider(resource=resource)

    # Add exporters based on config
    otlp_endpoint = config.otel.otlp_endpoint
    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

        if config.otel.console_debug:
            tracer_provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )
    else:
        # Default to console exporter in development
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Set as global tracer provider
    trace.set_tracer_provider(tracer_provider)


async def configure_logger(config: Settings):
    """
    Configure logging and tracing based on the application config.
    """
    event_filter: EventFilter = EventFilter(min_level=config.logger.level)
    transport = create_transport(config.logger)
    await LoggingConfig.configure(
        event_filter=event_filter,
        transport=transport,
        batch_size=config.logger.batch_size,
        flush_interval=config.logger.flush_interval,
    )


async def configure_usage_telemetry(_config: Settings):
    """
    Configure usage telemetry based on the application config.
    TODO: saqadri - implement usage tracking
    """
    pass


async def initialize_context(config: Settings | None = None):
    """
    Initialize the global application context.
    """
    if config is None:
        config = settings

    context = Context()
    context.config = config
    context.server_registry = ServerRegistry(config.config_yaml)

    # Configure logging and telemetry
    await configure_otel(config)
    await configure_logger(config)
    await configure_usage_telemetry(config)

    # Store the tracer in context if needed
    context.tracer = trace.get_tracer(config.otel.service_name)

    return context


async def cleanup_context():
    """
    Cleanup the global application context.
    """

    # Shutdown logging and telemetry
    await LoggingConfig.shutdown()


global_context = asyncio.run(initialize_context())


def get_current_context():
    """
    Get the current application context.
    """
    return global_context


def get_current_config():
    """
    Get the current application config.
    """
    return get_current_context().config or settings
