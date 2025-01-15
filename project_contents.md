# Project: mcp-agent

## Directory Structure

```
📁 mcp-agent
├── 📁 .git
├── 📄 .gitignore
├── 📄 .pre-commit-config.yaml
├── 📄 .python-version
├── 📁 .ruff_cache
├── 📁 .venv
├── 📁 .vscode
├── 📄 CONTRIBUTING.md
├── 📄 LICENSE
├── 📄 README.md
├── 📁 dist
│   ├── 📄 .gitignore
│   └── 📄 mcp_agent-0.0.1.tar.gz
├── 📁 examples
├── 📄 project_contents.md
├── 📄 pyproject.toml
├── 📁 schema
│   └── 📄 mcp-agent.config.schema.json
├── 📁 scripts
└── 📁 src
    └── 📁 mcp_agent
        ├── 📄 __init__.py
        ├── 📁 __pycache__
        ├── 📁 agents
        │   ├── 📄 __init__.py
        │   ├── 📁 __pycache__
        │   └── 📄 agent.py
        ├── 📁 cli
        │   ├── 📄 __init__.py
        │   ├── 📄 __main__.py
        │   ├── 📁 __pycache__
        │   ├── 📁 commands
        │   │   ├── 📁 __pycache__
        │   │   └── 📄 config.py
        │   ├── 📄 main.py
        │   └── 📄 terminal.py
        ├── 📄 config.py
        ├── 📄 context.py
        ├── 📁 eval
        │   └── 📄 __init__.py
        ├── 📁 executor
        │   ├── 📄 __init__.py
        │   ├── 📁 __pycache__
        │   ├── 📄 decorator_registry.py
        │   ├── 📄 executor.py
        │   ├── 📄 task_registry.py
        │   ├── 📄 temporal.py
        │   ├── 📄 workflow.py
        │   └── 📄 workflow_signal.py
        ├── 📁 logging
        ├── 📁 mcp
        │   ├── 📄 __init__.py
        │   ├── 📁 __pycache__
        │   ├── 📄 gen_client.py
        │   ├── 📄 mcp_activity.py
        │   ├── 📄 mcp_agent_client_session.py
        │   ├── 📄 mcp_agent_server.py
        │   ├── 📄 mcp_aggregator.py
        │   └── 📄 mcp_connection_manager.py
        ├── 📄 mcp_server_registry.py
        ├── 📄 py.typed
        ├── 📁 telemetry
        │   ├── 📄 __init__.py
        │   └── 📄 usage_tracking.py
        └── 📁 workflows
            ├── 📄 __init__.py
            ├── 📁 __pycache__
            ├── 📁 embedding
            │   ├── 📄 __init__.py
            │   ├── 📄 embedding_base.py
            │   ├── 📄 embedding_cohere.py
            │   └── 📄 embedding_openai.py
            ├── 📁 evaluator_optimizer
            │   ├── 📄 __init__.py
            │   ├── 📁 __pycache__
            │   └── 📄 evaluator_optimizer.py
            ├── 📁 intent_classifier
            │   ├── 📄 __init__.py
            │   ├── 📄 intent_classifier_base.py
            │   ├── 📄 intent_classifier_embedding.py
            │   ├── 📄 intent_classifier_embedding_cohere.py
            │   ├── 📄 intent_classifier_embedding_openai.py
            │   ├── 📄 intent_classifier_llm.py
            │   ├── 📄 intent_classifier_llm_anthropic.py
            │   └── 📄 intent_classifier_llm_openai.py
            ├── 📁 llm
            │   ├── 📄 __init__.py
            │   ├── 📁 __pycache__
            │   ├── 📄 augmented_llm.py
            │   ├── 📄 augmented_llm_anthropic.py
            │   └── 📄 augmented_llm_openai.py
            ├── 📁 orchestrator
            │   ├── 📄 __init__.py
            │   ├── 📁 __pycache__
            │   ├── 📄 orchestrator.py
            │   ├── 📄 orchestrator_models.py
            │   └── 📄 orchestrator_prompts.py
            ├── 📁 parallel
            │   ├── 📄 __init__.py
            │   ├── 📁 __pycache__
            │   ├── 📄 fan_in.py
            │   ├── 📄 fan_out.py
            │   └── 📄 parallel_llm.py
            ├── 📁 router
            │   ├── 📄 __init__.py
            │   ├── 📁 __pycache__
            │   ├── 📄 router_base.py
            │   ├── 📄 router_embedding.py
            │   ├── 📄 router_embedding_cohere.py
            │   ├── 📄 router_embedding_openai.py
            │   ├── 📄 router_llm.py
            │   ├── 📄 router_llm_anthropic.py
            │   └── 📄 router_llm_openai.py
            └── 📁 swarm
                ├── 📄 __init__.py
                ├── 📁 __pycache__
                ├── 📄 swarm.py
                ├── 📄 swarm_anthropic.py
                └── 📄 swarm_openai.py
```

### src/mcp_agent/**init**.py

```py

```

### src/mcp_agent/agents/**init**.py

```py

```

### src/mcp_agent/agents/agent.py

```py
from typing import Callable, Dict, TypeVar

from mcp_agent.mcp.mcp_aggregator import MCPAggregator
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)

# Define a TypeVar for AugmentedLLM and its subclasses
LLM = TypeVar("LLM", bound=AugmentedLLM)


class Agent(MCPAggregator):
    """
    An Agent is an entity that has access to a set of MCP servers and can interact with them.
    Each agent should have a purpose defined by its instruction.
    """

    name: str
    instruction: str | Callable[[Dict], str]

    def __init__(
        self,
        name: str,
        instruction: str | Callable[[Dict], str] = "You are a helpful agent.",
        server_names: list[str] = None,
        connection_persistence: bool = True,
    ):
        super().__init__(
            server_names=server_names or [],
            connection_persistence=connection_persistence,
            name=name,
            instruction=instruction,
        )

    async def initialize(self):
        """
        Initialize the agent and connect to the MCP servers.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await (
            self.__aenter__()
        )  # This initializes the connection manager and loads the servers

    async def attach_llm(self, llm_factory: Callable[..., LLM]) -> LLM:
        """
        Create an LLM instance for the agent.

         Args:
            llm_factory: A callable that constructs an AugmentedLLM or its subclass.
                        The factory should accept keyword arguments matching the
                        AugmentedLLM constructor parameters.

        Returns:
            An instance of AugmentedLLM or one of its subclasses.
        """
        return llm_factory(agent=self)

    async def shutdown(self):
        """
        Shutdown the agent and close all MCP server connections.
        NOTE: This method is called automatically when the agent is used as an async context manager.
        """
        await super().close()

```

### src/mcp_agent/cli/**init**.py

```py

```

### src/mcp_agent/cli/**main**.py

```py
from mcp_agent.cli.main import app

if __name__ == "__main__":
    app()

```

### src/mcp_agent/cli/commands/config.py

```py
import typer

app = typer.Typer()


@app.command()
def show():
    """Show the configuration."""
    print("NotImplemented")

```

### src/mcp_agent/cli/main.py

```py
import typer
from mcp_agent.cli.terminal import Application
from mcp_agent.cli.commands import config

app = typer.Typer()

# Subcommands
app.add_typer(config.app, name="config")

# Shared application context
application = Application()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Disable output"),
    color: bool = typer.Option(
        True, "--color/--no-color", help="Enable/disable color output"
    ),
):
    """Main entry point for the MCP Agent CLI."""
    application.verbosity = 1 if verbose else 0 if not quiet else -1
    application.console = application.console if color else None

```

### src/mcp_agent/cli/terminal.py

```py
from rich.console import Console


class Application:
    def __init__(self, verbosity: int = 0, enable_color: bool = True):
        self.verbosity = verbosity
        self.console = Console(color_system="auto" if enable_color else None)

    def log(self, message: str, level: str = "info"):
        if level == "info" or (level == "debug" and self.verbosity > 0):
            self.console.print(f"[{level.upper()}] {message}")

    def status(self, message: str):
        return self.console.status(f"[bold cyan]{message}[/bold cyan]")

```

### src/mcp_agent/config.py

```py
"""
Reading settings from environment variables and providing a settings object
for the application configuration.
"""

from pathlib import Path
from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerAuthSettings(BaseModel):
    """Represents authentication configuration for a server."""

    api_key: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class MCPServerSettings(BaseModel):
    """
    Represents the configuration for an individual server.
    """

    # TODO: saqadri - server name should be something a server can provide itself during initialization
    name: str | None = None
    """The name of the server."""

    # TODO: saqadri - server description should be something a server can provide itself during initialization
    description: str | None = None
    """The description of the server."""

    transport: Literal["stdio", "sse"] = "stdio"
    """The transport mechanism."""

    command: str | None = None
    """The command to execute the server (e.g. npx)."""

    args: List[str] | None = None
    """The arguments for the server command."""

    read_timeout_seconds: int | None = None
    """The timeout in seconds for the server connection."""

    url: str | None = None
    """The URL for the server (e.g. for SSE transport)."""

    auth: MCPServerAuthSettings | None = None
    """The authentication configuration for the server."""


class MCPSettings(BaseModel):
    """Configuration for all MCP servers."""

    servers: Dict[str, MCPServerSettings] = {}
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class AnthropicSettings(BaseModel):
    """
    Settings for using Anthropic models in the MCP Agent application.
    """

    api_key: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class CohereSettings(BaseModel):
    """
    Settings for using Cohere models in the MCP Agent application.
    """

    api_key: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class OpenAISettings(BaseModel):
    """
    Settings for using OpenAI models in the MCP Agent application.
    """

    api_key: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class TemporalSettings(BaseModel):
    """
    Temporal settings for the MCP Agent application.
    """

    host: str
    namespace: str = "default"
    task_queue: str
    api_key: str | None = None


class UsageTelemetrySettings(BaseModel):
    """
    Settings for usage telemetry in the MCP Agent application.
    Anonymized usage metrics are sent to a telemetry server to help improve the product.
    """

    enabled: bool = True
    """Enable usage telemetry in the MCP Agent application."""

    enable_detailed_telemetry: bool = False
    """If enabled, detailed telemetry data, including prompts and agents, will be sent to the telemetry server."""


class OpenTelemetrySettings(BaseModel):
    """
    OTEL settings for the MCP Agent application.
    """

    enabled: bool = True

    service_name: str = "mcp-agent"
    service_instance_id: str | None = None
    service_version: str | None = None

    otlp_endpoint: str | None = None
    """OTLP endpoint for OpenTelemetry tracing"""

    console_debug: bool = False
    """Log spans to console"""

    sample_rate: float = 1.0
    """Sample rate for tracing (1.0 = sample everything)"""


class LoggerSettings(BaseModel):
    """
    Logger settings for the MCP Agent application.
    """

    type: Literal["none", "console", "http"] = "console"

    level: Literal["debug", "info", "warning", "error"] = "info"
    """Minimum logging level"""

    batch_size: int = 100
    """Number of events to accumulate before processing"""

    flush_interval: float = 2.0
    """How often to flush events in seconds"""

    max_queue_size: int = 2048
    """Maximum queue size for event processing"""

    # HTTP transport settings
    http_endpoint: str | None = None
    """HTTP endpoint for event transport"""

    http_headers: dict[str, str] | None = None
    """HTTP headers for event transport"""

    http_timeout: float = 5.0
    """HTTP timeout seconds for event transport"""


class Settings(BaseSettings):
    """
    Settings class for the MCP Agent application.
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        nested_model_default_partial_update=True,
    )  # Customize the behavior of settings here

    mcp: MCPSettings | None = MCPSettings()
    """MCP config, such as MCP servers"""

    execution_engine: Literal["asyncio", "temporal"] = "asyncio"
    """Execution engine for the MCP Agent application"""

    temporal: TemporalSettings | None = None
    """Settings for Temporal workflow orchestration"""

    anthropic: AnthropicSettings | None = None
    """Settings for using Anthropic models in the MCP Agent application"""

    cohere: CohereSettings | None = None
    """Settings for using Cohere models in the MCP Agent application"""

    openai: OpenAISettings | None = None
    """Settings for using OpenAI models in the MCP Agent application"""

    otel: OpenTelemetrySettings | None = OpenTelemetrySettings()
    """OpenTelemetry logging settings for the MCP Agent application"""

    logger: LoggerSettings | None = LoggerSettings()
    """Logger settings for the MCP Agent application"""

    usage_telemetry: UsageTelemetrySettings | None = UsageTelemetrySettings()
    """Usage tracking settings for the MCP Agent application"""

    @classmethod
    def find_config(cls) -> Path | None:
        """Find the config file in the current directory or parent directories."""
        current_dir = Path.cwd()

        # Check current directory and parent directories
        while current_dir != current_dir.parent:
            for filename in ["mcp-agent.config.yaml", "mcp_agent.config.yaml"]:
                config_path = current_dir / filename
                if config_path.exists():
                    return config_path
            current_dir = current_dir.parent

        return None


# Global settings object
_settings: Settings | None = None


def get_settings(config_path: str | None = None) -> Settings:
    """Get settings instance, automatically loading from config file if available."""

    def deep_merge(base: dict, update: dict) -> dict:
        """Recursively merge two dictionaries, preserving nested structures."""
        merged = base.copy()
        for key, value in update.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    global _settings
    if _settings:
        return _settings

    config_file = config_path or Settings.find_config()
    merged_settings = {}

    if config_file:
        if not config_file.exists():
            pass
        else:
            import yaml  # pylint: disable=C0415

            # Load main config
            with open(config_file, "r", encoding="utf-8") as f:
                yaml_settings = yaml.safe_load(f) or {}
                merged_settings = yaml_settings

            # Look for secrets file in the same directory
            secrets_file = config_file.parent / "mcp_agent.secrets.yaml"
            if secrets_file.exists():
                with open(secrets_file, "r", encoding="utf-8") as f:
                    yaml_secrets = yaml.safe_load(f) or {}
                    merged_settings = deep_merge(merged_settings, yaml_secrets)

            _settings = Settings(**merged_settings)
            return _settings
    else:
        pass

    _settings = Settings()
    return _settings

```

### src/mcp_agent/context.py

```py
"""
A central context object to store global state that is shared across the application.
"""

import asyncio
import concurrent.futures

from pydantic import BaseModel, ConfigDict
from mcp import ServerSession
from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from mcp_agent.config import Settings, get_settings
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

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


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
        attributes={
            key: value
            for key, value in {
                "service.name": service_name,
                "service.instance.id": service_instance_id,
                "service.version": service_version,
            }.items()
            if value is not None
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
        config = get_settings()

    context = Context()
    context.config = config
    context.server_registry = ServerRegistry(config=config)

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


_global_context: Context | None = None


def get_current_context() -> Context:
    """
    Synchronous initializer/getter for global application context.
    For async usage, use aget_current_context instead.
    """
    global _global_context
    if _global_context is None:
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new loop in a separate thread
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    return new_loop.run_until_complete(initialize_context())

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    _global_context = pool.submit(run_async).result()
            else:
                _global_context = loop.run_until_complete(initialize_context())
        except RuntimeError:
            _global_context = asyncio.run(initialize_context())
    return _global_context


async def aget_current_context() -> Context:
    """
    Get the current application context, initializing if necessary.
    """
    global _global_context
    if _global_context is None:
        _global_context = await initialize_context()
    return _global_context


def get_current_config():
    """
    Get the current application config.
    """
    return get_current_context().config or get_settings()


async def aget_current_config():
    """
    Async verion of get_current_config, to get the current application config.
    """
    context = await aget_current_context()
    return context.config or get_settings()

```

### src/mcp_agent/eval/**init**.py

```py

```

### src/mcp_agent/executor/**init**.py

```py

```

### src/mcp_agent/executor/decorator_registry.py

```py
"""
Keep track of all workflow decorator overloads indexed by executor backend.
Different executors may have different ways of configuring workflows.
"""

from typing import Callable, Dict, Type, TypeVar

R = TypeVar("R")


class DecoratorRegistry:
    """Centralized decorator management with validation and metadata."""

    def __init__(self):
        self._workflow_defn_decorators: Dict[str, Callable[[Type], Type]] = {}
        self._workflow_run_decorators: Dict[
            str, Callable[[Callable[..., R]], Callable[..., R]]
        ] = {}

    def register_workflow_defn_decorator(
        self,
        executor_name: str,
        decorator: Callable[[Type], Type],
    ):
        """
        Registers a workflow definition decorator for a given executor.

        :param executor_name: Unique name of the executor.
        :param decorator: The decorator to register.
        """
        if executor_name in self._workflow_defn_decorators:
            print(
                "Workflow definition decorator already registered for '%s'. Overwriting.",
                executor_name,
            )
        self._workflow_defn_decorators[executor_name] = decorator

    def get_workflow_defn_decorator(self, executor_name: str) -> Callable[[Type], Type]:
        """
        Retrieves a workflow definition decorator for a given executor.

        :param executor_name: Unique name of the executor.
        :return: The decorator function.
        """
        return self._workflow_defn_decorators.get(executor_name)

    def register_workflow_run_decorator(
        self,
        executor_name: str,
        decorator: Callable[[Callable[..., R]], Callable[..., R]],
    ):
        """
        Registers a workflow run decorator for a given executor.

        :param executor_name: Unique name of the executor.
        :param decorator: The decorator to register.
        """
        if executor_name in self._workflow_run_decorators:
            print(
                "Workflow run decorator already registered for '%s'. Overwriting.",
                executor_name,
            )
        self._workflow_run_decorators[executor_name] = decorator

    def get_workflow_run_decorator(
        self, executor_name: str
    ) -> Callable[[Callable[..., R]], Callable[..., R]]:
        """
        Retrieves a workflow run decorator for a given executor.

        :param executor_name: Unique name of the executor.
        :return: The decorator function.
        """
        return self._workflow_run_decorators.get(executor_name)


# Global decorator registry
_global_decorator_registry: DecoratorRegistry | None = None


def default_workflow_defn(cls: Type, *args, **kwargs) -> Type:
    """Default no-op workflow definition decorator."""
    return cls


def default_workflow_run(fn: Callable[..., R]) -> Callable[..., R]:
    """Default no-op workflow run decorator."""

    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def register_asyncio_decorators(decorator_registry: DecoratorRegistry):
    """Registers default asyncio decorators."""
    executor_name = "asyncio"
    decorator_registry.register_workflow_defn_decorator(
        executor_name, default_workflow_defn
    )
    decorator_registry.register_workflow_run_decorator(
        executor_name, default_workflow_run
    )


def register_temporal_decorators(decorator_registry: DecoratorRegistry):
    """Registers Temporal decorators if Temporal SDK is available."""
    try:
        import temporalio.workflow as temporal_workflow

        TEMPORAL_AVAILABLE = True
    except ImportError:
        TEMPORAL_AVAILABLE = False

    if not TEMPORAL_AVAILABLE:
        print(
            "Temporal SDK is not available. Skipping Temporal decorator registration."
        )
        return

    executor_name = "temporal"
    decorator_registry.register_workflow_defn_decorator(
        executor_name, temporal_workflow.defn
    )
    decorator_registry.register_workflow_run_decorator(
        executor_name, temporal_workflow.run
    )


def get_decorator_registry() -> DecoratorRegistry:
    """
    Retrieves or initializes the global decorator registry.

    :return: The global instance of DecoratorRegistry.
    """
    global _global_decorator_registry
    if _global_decorator_registry is None:
        _global_decorator_registry = DecoratorRegistry()
        register_asyncio_decorators(_global_decorator_registry)
        register_temporal_decorators(_global_decorator_registry)

    return _global_decorator_registry

```

### src/mcp_agent/executor/executor.py

```py
import asyncio
import functools
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Type, TypeVar

from pydantic import BaseModel, ConfigDict

from mcp_agent.executor.workflow_signal import (
    AsyncioSignalHandler,
    Signal,
    SignalHandler,
    SignalValueT,
)
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)

# Type variable for the return type of tasks
R = TypeVar("R")


class ExecutorConfig(BaseModel):
    """Configuration for executors."""

    max_concurrent_activities: int | None = None  # Unbounded by default
    timeout_seconds: timedelta | None = None  # No timeout by default
    retry_policy: Dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class Executor(ABC):
    """Abstract base class for different execution backends"""

    def __init__(
        self,
        engine: str,
        config: ExecutorConfig | None = None,
        signal_bus: SignalHandler = None,
    ):
        self.execution_engine = engine

        if config:
            self.config = config
        else:
            # TODO: saqadri - executor config should be loaded from settings
            # ctx = get_current_context()
            self.config = ExecutorConfig()

        self.signal_bus = signal_bus

    @asynccontextmanager
    async def execution_context(self):
        """Context manager for execution setup/teardown."""
        try:
            yield
        except Exception as e:
            # TODO: saqadri - add logging or other error handling here
            raise e

    @abstractmethod
    async def execute(
        self,
        *tasks: Callable[..., R] | Coroutine[Any, Any, R],
        **kwargs: Any,
    ) -> List[R | BaseException]:
        """Execute a list of tasks and return their results"""

    @abstractmethod
    async def execute_streaming(
        self,
        *tasks: List[Callable[..., R] | Coroutine[Any, Any, R]],
        **kwargs: Any,
    ) -> AsyncIterator[R | BaseException]:
        """Execute tasks and yield results as they complete"""

    async def map(
        self,
        func: Callable[..., R],
        inputs: List[Any],
        **kwargs: Any,
    ) -> List[R | BaseException]:
        """
        Run `func(item)` for each item in `inputs` with concurrency limit.
        """
        results: List[R, BaseException] = []

        async def run(item):
            if self.config.max_concurrent_activities:
                semaphore = asyncio.Semaphore(self.config.max_concurrent_activities)
                async with semaphore:
                    return await self.execute(functools.partial(func, item), **kwargs)
            else:
                return await self.execute(functools.partial(func, item), **kwargs)

        coros = [run(x) for x in inputs]
        # gather all, each returns a single-element list
        list_of_lists = await asyncio.gather(*coros, return_exceptions=True)

        # Flatten results
        for entry in list_of_lists:
            if isinstance(entry, list):
                results.extend(entry)
            else:
                # Means we got an exception at the gather level
                results.append(entry)

        return results

    async def validate_task(
        self, task: Callable[..., R] | Coroutine[Any, Any, R]
    ) -> None:
        """Validate a task before execution."""
        if not (asyncio.iscoroutine(task) or asyncio.iscoroutinefunction(task)):
            raise TypeError(f"Task must be async: {task}")

    async def signal(
        self,
        signal_name: str,
        payload: SignalValueT = None,
        signal_description: str | None = None,
    ) -> None:
        """
        Emit a signal.
        """
        signal = Signal[SignalValueT](
            name=signal_name, payload=payload, description=signal_description
        )
        await self.signal_bus.signal(signal)

    async def wait_for_signal(
        self,
        signal_name: str,
        signal_description: str | None = None,
        timeout_seconds: int | None = None,
        signal_type: Type[SignalValueT] = str,
    ) -> SignalValueT:
        """
        Wait until a signal with signal_name is emitted (or timeout).
        Return the signal's payload when triggered, or raise on timeout.
        """
        signal = Signal[signal_type](name=signal_name, description=signal_description)
        return await self.signal_bus.wait_for_signal(signal)


class AsyncioExecutor(Executor):
    """Default executor using asyncio"""

    def __init__(
        self,
        config: ExecutorConfig | None = None,
        signal_bus: SignalHandler | None = None,
    ):
        signal_bus = signal_bus or AsyncioSignalHandler()
        super().__init__(engine="asyncio", config=config, signal_bus=signal_bus)

        self._activity_semaphore: asyncio.Semaphore | None = None
        if self.config.max_concurrent_activities is not None:
            self._activity_semaphore = asyncio.Semaphore(
                self.config.max_concurrent_activities
            )

    async def _execute_task(
        self, task: Callable[..., R] | Coroutine[Any, Any, R], **kwargs: Any
    ) -> R | BaseException:
        async def run_task(task: Callable[..., R] | Coroutine[Any, Any, R]) -> R:
            try:
                if asyncio.iscoroutine(task):
                    return await task
                elif asyncio.iscoroutinefunction(task):
                    return await task(**kwargs)
                else:
                    # Execute the callable and await if it returns a coroutine
                    loop = asyncio.get_running_loop()

                    # If kwargs are provided, wrap the function with partial
                    if kwargs:
                        wrapped_task = functools.partial(task, **kwargs)
                        result = await loop.run_in_executor(None, wrapped_task)
                    else:
                        result = await loop.run_in_executor(None, task)

                    # Handle case where the sync function returns a coroutine
                    if asyncio.iscoroutine(result):
                        return await result

                    return result
            except Exception as e:
                # TODO: saqadri - adding logging or other error handling here
                return e

        if self._activity_semaphore:
            async with self._activity_semaphore:
                return await run_task(task)
        else:
            return await run_task(task)

    async def execute(
        self,
        *tasks: Callable[..., R] | Coroutine[Any, Any, R],
        **kwargs: Any,
    ) -> List[R | BaseException]:
        # TODO: saqadri - validate if async with self.execution_context() is needed here
        async with self.execution_context():
            return await asyncio.gather(
                *(self._execute_task(task, **kwargs) for task in tasks),
                return_exceptions=True,
            )

    async def execute_streaming(
        self,
        *tasks: List[Callable[..., R] | Coroutine[Any, Any, R]],
        **kwargs: Any,
    ) -> AsyncIterator[R | BaseException]:
        # TODO: saqadri - validate if async with self.execution_context() is needed here
        async with self.execution_context():
            # Create futures for all tasks
            futures = [
                asyncio.create_task(self._execute_task(task, **kwargs))
                for task in tasks
            ]
            pending = set(futures)

            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for future in done:
                    yield await future

    async def signal(
        self,
        signal_name: str,
        payload: SignalValueT = None,
        signal_description: str | None = None,
    ) -> None:
        await super().signal(signal_name, payload, signal_description)

    async def wait_for_signal(
        self,
        signal_name: str,
        signal_description: str | None = None,
        timeout_seconds: int | None = None,
        signal_type: Type[SignalValueT] = str,
    ) -> SignalValueT:
        return await super().wait_for_signal(
            signal_name, signal_description, timeout_seconds, signal_type
        )

```

### src/mcp_agent/executor/task_registry.py

```py
"""
Keep track of all activities/tasks that the executor needs to run.
This is used by the workflow engine to dynamically orchestrate a workflow graph.
The user just writes standard functions annotated with @workflow_task, but behind the scenes a workflow graph is built.
"""

from typing import Any, Callable, Dict, List


class ActivityRegistry:
    """Centralized task/activity management with validation and metadata."""

    def __init__(self):
        self._activities: Dict[str, Callable] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self, name: str, func: Callable, metadata: Dict[str, Any] | None = None
    ):
        if name in self._activities:
            raise ValueError(f"Activity '{name}' is already registered.")
        self._activities[name] = func
        self._metadata[name] = metadata or {}

    def get_activity(self, name: str) -> Callable:
        if name not in self._activities:
            raise KeyError(f"Activity '{name}' not found.")
        return self._activities[name]

    def get_metadata(self, name: str) -> Dict[str, Any]:
        return self._metadata.get(name, {})

    def list_activities(self) -> List[str]:
        return list(self._activities.keys())


global_task_registry = ActivityRegistry()


def get_activity_registry():
    """
    Get the current activity/task registry.
    """
    return global_task_registry

```

### src/mcp_agent/executor/temporal.py

```py
"""
Temporal based orchestrator for the MCP Agent.
Temporal provides durable execution and robust workflow orchestration,
as well as dynamic control flow, making it a good choice for an AI agent orchestrator.
Read more: https://docs.temporal.io/develop/python/core-application
"""

import asyncio
import functools
import uuid
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List

from pydantic import ConfigDict
from temporalio import activity, workflow, exceptions
from temporalio.client import Client as TemporalClient
from temporalio.worker import Worker

from mcp_agent.config import TemporalSettings
from mcp_agent.context import get_current_config
from mcp_agent.executor.executor import Executor, ExecutorConfig, R
from mcp_agent.executor.task_registry import get_activity_registry
from mcp_agent.executor.workflow_signal import (
    BaseSignalHandler,
    Signal,
    SignalHandler,
    SignalRegistration,
    SignalValueT,
)


class TemporalSignalHandler(BaseSignalHandler[SignalValueT]):
    """Temporal-based signal handling using workflow signals"""

    async def wait_for_signal(self, signal, timeout_seconds=None) -> SignalValueT:
        if not workflow._Runtime.current():
            raise RuntimeError(
                "TemporalSignalHandler.wait_for_signal must be called from within a workflow"
            )

        unique_signal_name = f"{signal.name}_{uuid.uuid4()}"
        registration = SignalRegistration(
            signal_name=signal.name,
            unique_name=unique_signal_name,
            workflow_id=workflow.info().workflow_id,
        )

        # Container for signal value
        container = {"value": None, "completed": False}

        # Define the signal handler for this specific registration
        @workflow.signal(name=unique_signal_name)
        def signal_handler(value: SignalValueT):
            container["value"] = value
            container["completed"] = True

        async with self._lock:
            # Register both the signal registration and handler atomically
            self._pending_signals.setdefault(signal.name, []).append(registration)
            self._handlers.setdefault(signal.name, []).append(
                (unique_signal_name, signal_handler)
            )

        try:
            # Wait for signal with optional timeout
            await workflow.wait_condition(
                lambda: container["completed"], timeout=timeout_seconds
            )

            return container["value"]
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"Timeout waiting for signal {signal.name}") from exc
        finally:
            async with self._lock:
                # Remove ourselves from _pending_signals
                if signal.name in self._pending_signals:
                    self._pending_signals[signal.name] = [
                        sr
                        for sr in self._pending_signals[signal.name]
                        if sr.unique_name != unique_signal_name
                    ]
                    if not self._pending_signals[signal.name]:
                        del self._pending_signals[signal.name]

                # Remove ourselves from _handlers
                if signal.name in self._handlers:
                    self._handlers[signal.name] = [
                        h
                        for h in self._handlers[signal.name]
                        if h[0] != unique_signal_name
                    ]
                    if not self._handlers[signal.name]:
                        del self._handlers[signal.name]

    def on_signal(self, signal_name):
        """Decorator to register a signal handler."""

        def decorator(func: Callable) -> Callable:
            # Create unique signal name for this handler
            unique_signal_name = f"{signal_name}_{uuid.uuid4()}"

            # Create the actual handler that will be registered with Temporal
            @workflow.signal(name=unique_signal_name)
            async def wrapped(signal_value: SignalValueT):
                # Create a signal object to pass to the handler
                signal = Signal(
                    name=signal_name,
                    payload=signal_value,
                    workflow_id=workflow.info().workflow_id,
                )
                if asyncio.iscoroutinefunction(func):
                    await func(signal)
                else:
                    func(signal)

            # Register the handler under the original signal name
            self._handlers.setdefault(signal_name, []).append(
                (unique_signal_name, wrapped)
            )
            return func

        return decorator

    async def signal(self, signal):
        self.validate_signal(signal)

        workflow_handle = workflow.get_external_workflow_handle(
            workflow_id=signal.workflow_id
        )

        # Send the signal to all registrations of this signal
        async with self._lock:
            signal_tasks = []

            if signal.name in self._pending_signals:
                for pending_signal in self._pending_signals[signal.name]:
                    registration = pending_signal.registration
                    if registration.workflow_id == signal.workflow_id:
                        # Only signal for registrations of that workflow
                        signal_tasks.append(
                            workflow_handle.signal(
                                registration.unique_name, signal.payload
                            )
                        )
                    else:
                        continue

            # Notify any registered handler functions
            if signal.name in self._handlers:
                for unique_name, _ in self._handlers[signal.name]:
                    signal_tasks.append(
                        workflow_handle.signal(unique_name, signal.payload)
                    )

        await asyncio.gather(*signal_tasks, return_exceptions=True)

    def validate_signal(self, signal):
        super().validate_signal(signal)
        # Add TemporalSignalHandler-specific validation
        if signal.workflow_id is None:
            raise ValueError(
                "No workflow_id provided on Signal. That is required for Temporal signals"
            )


class TemporalExecutorConfig(ExecutorConfig, TemporalSettings):
    """Configuration for Temporal executors."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class TemporalExecutor(Executor):
    """Executor that runs @workflows as Temporal workflows, with @workflow_tasks as Temporal activities"""

    def __init__(
        self,
        config: TemporalExecutorConfig | None = None,
        signal_bus: SignalHandler | None = None,
        client: TemporalClient | None = None,
    ):
        signal_bus = signal_bus or TemporalSignalHandler()
        super().__init__(engine="temporal", config=config, signal_bus=signal_bus)
        self.config: TemporalExecutorConfig = config or self._get_config()
        self.client = client
        self._worker = None
        self._activity_semaphore = None

        if config.max_concurrent_activities is not None:
            self._activity_semaphore = asyncio.Semaphore(
                self.config.max_concurrent_activities
            )

    @classmethod
    def _get_config(cls) -> TemporalExecutorConfig:
        config = get_current_config()
        if config.temporal:
            return TemporalExecutorConfig(**config.temporal.model_dump())

        return TemporalExecutorConfig()

    @staticmethod
    def wrap_as_activity(
        activity_name: str,
        func: Callable[..., R] | Coroutine[Any, Any, R],
        **kwargs: Any,
    ) -> Coroutine[Any, Any, R]:
        """
        Convert a function into a Temporal activity and return its info.
        """

        @activity.defn(name=activity_name)
        async def wrapped_activity(*args, **local_kwargs):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **local_kwargs)
                elif asyncio.iscoroutine(func):
                    return await func
                else:
                    return func(*args, **local_kwargs)
            except Exception as e:
                # Handle exceptions gracefully
                raise e

        return wrapped_activity

    async def _execute_task_as_async(
        self, task: Callable[..., R] | Coroutine[Any, Any, R], **kwargs: Any
    ) -> R | BaseException:
        async def run_task(task: Callable[..., R] | Coroutine[Any, Any, R]) -> R:
            try:
                if asyncio.iscoroutine(task):
                    return await task
                elif asyncio.iscoroutinefunction(task):
                    return await task(**kwargs)
                else:
                    # Execute the callable and await if it returns a coroutine
                    loop = asyncio.get_running_loop()

                    # If kwargs are provided, wrap the function with partial
                    if kwargs:
                        wrapped_task = functools.partial(task, **kwargs)
                        result = await loop.run_in_executor(None, wrapped_task)
                    else:
                        result = await loop.run_in_executor(None, task)

                    # Handle case where the sync function returns a coroutine
                    if asyncio.iscoroutine(result):
                        return await result

                    return result
            except Exception as e:
                # TODO: saqadri - adding logging or other error handling here
                return e

        if self._activity_semaphore:
            async with self._activity_semaphore:
                return await run_task(task)
        else:
            return await run_task(task)

    async def _execute_task(
        self, task: Callable[..., R] | Coroutine[Any, Any, R], **kwargs: Any
    ) -> R | BaseException:
        func = task.func if isinstance(task, functools.partial) else task
        is_workflow_task = getattr(func, "is_workflow_task", False)
        if not is_workflow_task:
            return await asyncio.create_task(
                self._execute_task_as_async(task, **kwargs)
            )

        execution_metadata: Dict[str, Any] = getattr(func, "execution_metadata", {})

        # Derive stable activity name, e.g. module + qualname
        activity_name = execution_metadata.get("activity_name")
        if not activity_name:
            activity_name = f"{func.__module__}.{func.__qualname__}"

        schedule_to_close = execution_metadata.get(
            "schedule_to_close_timeout", self.config.timeout_seconds
        )

        retry_policy = execution_metadata.get("retry_policy", None)

        _task_activity = self.wrap_as_activity(activity_name=activity_name, func=task)

        # # For partials, we pass the partial's arguments
        # args = task.args if isinstance(task, functools.partial) else ()
        try:
            result = await workflow.execute_activity(
                activity_name,
                args=kwargs.get("args", ()),
                task_queue=self.config.task_queue,
                schedule_to_close_timeout=schedule_to_close,
                retry_policy=retry_policy,
                **kwargs,
            )
            return result
        except Exception as e:
            # Properly propagate activity errors
            if isinstance(e, exceptions.ActivityError):
                raise e.cause if e.cause else e
            raise

    async def execute(
        self,
        *tasks: Callable[..., R] | Coroutine[Any, Any, R],
        **kwargs: Any,
    ) -> List[R | BaseException]:
        # Must be called from within a workflow
        if not workflow._Runtime.current():
            raise RuntimeError(
                "TemporalExecutor.execute must be called from within a workflow"
            )

        # TODO: saqadri - validate if async with self.execution_context() is needed here
        async with self.execution_context():
            return await asyncio.gather(
                *(self._execute_task(task, **kwargs) for task in tasks),
                return_exceptions=True,
            )

    async def execute_streaming(
        self,
        *tasks: Callable[..., R] | Coroutine[Any, Any, R],
        **kwargs: Any,
    ) -> AsyncIterator[R | BaseException]:
        if not workflow._Runtime.current():
            raise RuntimeError(
                "TemporalExecutor.execute_streaming must be called from within a workflow"
            )

        # TODO: saqadri - validate if async with self.execution_context() is needed here
        async with self.execution_context():
            # Create futures for all tasks
            futures = [self._execute_task(task, **kwargs) for task in tasks]
            pending = set(futures)

            while pending:
                done, pending = await workflow.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for future in done:
                    try:
                        result = await future
                        yield result
                    except Exception as e:
                        yield e

    async def ensure_client(self):
        """Ensure we have a connected Temporal client."""
        if self.client is None:
            self.client = await TemporalClient.connect(
                target_host=self.config.host,
                namespace=self.config.namespace,
                api_key=self.config.api_key,
            )

        return self.client

    async def start_worker(self):
        """
        Start a worker in this process, auto-registering all tasks
        from the global registry. Also picks up any classes decorated
        with @workflow_defn as recognized workflows.
        """
        await self.ensure_client()

        if self._worker is None:
            # We'll collect the activities from the global registry
            # and optionally wrap them with `activity.defn` if we want
            # (Not strictly required if your code calls `execute_activity("name")` by name)
            activity_registry = get_activity_registry()
            activities = []
            for name in activity_registry.list_activities():
                activities.append(activity_registry.get_activity(name))

            # Now we attempt to discover any classes that are recognized as workflows
            # But in this simple example, we rely on the user specifying them or
            # we might do a dynamic scan.
            # For demonstration, we'll just assume the user is only using
            # the workflow classes they decorate with `@workflow_defn`.
            # We'll rely on them passing the classes or scanning your code.

            self._worker = Worker(
                client=self.client,
                task_queue=self.config.task_queue,
                activities=activities,
                workflows=[],  # We'll auto-load by Python scanning or let the user specify
            )
            print(
                f"Starting Temporal Worker on task queue '{self.config.task_queue}' with {len(activities)} activities."
            )

        await self._worker.run()

```

### src/mcp_agent/executor/workflow.py

```py
import asyncio

# import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, ConfigDict, Field

from mcp_agent.context import get_current_config
from mcp_agent.executor.executor import Executor
from mcp_agent.executor.decorator_registry import get_decorator_registry
from mcp_agent.executor.task_registry import get_activity_registry

R = TypeVar("R")
T = TypeVar("T")


class WorkflowState(BaseModel):
    """
    Simple container for persistent workflow state.
    This can hold fields that should persist across tasks.
    """

    status: str = "initialized"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    updated_at: float | None = None
    error: Dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def record_error(self, error: Exception) -> None:
        self.error = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.utcnow().timestamp(),
        }


class WorkflowResult(BaseModel, Generic[T]):
    value: Union[T, None] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    start_time: float | None = None
    end_time: float | None = None

    # def complete(self) -> "WorkflowResult[T]":
    #     import asyncio

    #     if self.start_time is None:
    #         self.start_time = asyncio.get_event_loop().time()
    #     self.end_time = asyncio.get_event_loop().time()
    #     self.metadata["duration"] = self.end_time - self.start_time
    #     return self


class Workflow(ABC, Generic[T]):
    """
    Base class for user-defined workflows.
    Handles execution and state management.
    Some key notes:
        - To enable the executor engine to recognize and orchestrate the workflow,
            - the class MUST be decorated with @workflow.
            - the main entrypoint method MUST be decorated with @workflow_run.
            - any task methods MUST be decorated with @workflow_task.

        - Persistent state: Provides a simple `state` object for storing data across tasks.
    """

    def __init__(
        self,
        executor: Executor,
        name: str | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.executor = executor
        self.name = name or self.__class__.__name__
        self.init_kwargs = kwargs
        # TODO: handle logging
        # self._logger = logging.getLogger(self.name)

        # A simple workflow state object
        # If under Temporal, storing it as a field on this class
        # means it can be replayed automatically
        self.state = WorkflowState(name=name, metadata=metadata or {})

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> "WorkflowResult[T]":
        """
        Main workflow implementation. Myst be overridden by subclasses.
        """

    async def update_state(self, **kwargs):
        """Syntactic sugar to update workflow state."""
        for key, value in kwargs.items():
            self.state[key] = value
            setattr(self.state, key, value)

        self.state.updated_at = datetime.utcnow().timestamp()

    async def wait_for_input(self, description: str = "Provide input") -> str:
        """
        Convenience method for human input. Uses `human_input` signal
        so we can unify local (console input) and Temporal signals.
        """
        return await self.executor.wait_for_signal(
            "human_input", description=description
        )


#####################
# Workflow Decorators
#####################


def get_execution_engine() -> str:
    """Get the current execution engine (asyncio, Temporal, etc)."""
    config = get_current_config()
    return config.execution_engine or "asyncio"


def workflow(cls: Type, *args, **kwargs) -> Type:
    """
    Decorator for a workflow class. By default it's a no-op,
    but different executors can use this to customize behavior
    for workflow registration.

    Example:
        If Temporal is available & we use a TemporalExecutor,
        this decorator will wrap with temporal_workflow.defn.
    """
    decorator_registry = get_decorator_registry()
    execution_engine = get_execution_engine()
    workflow_defn_decorator = decorator_registry.get_workflow_defn_decorator(
        execution_engine
    )

    if workflow_defn_decorator:
        return workflow_defn_decorator(cls, *args, **kwargs)

    # Default no-op
    return cls


def workflow_run(fn: Callable[..., R]) -> Callable[..., R]:
    """
    Decorator for a workflow's main 'run' method.
    Different executors can use this to customize behavior for workflow execution.

    Example:
        If Temporal is in use, this gets converted to @workflow.run.
    """

    decorator_registry = get_decorator_registry()
    execution_engine = get_execution_engine()
    workflow_run_decorator = decorator_registry.get_workflow_run_decorator(
        execution_engine
    )

    if workflow_run_decorator:
        return workflow_run_decorator(fn)

    # Default no-op
    def wrapper(*args, **kwargs):
        # no-op wrapper
        return fn(*args, **kwargs)

    return wrapper


def workflow_task(
    name: str | None = None,
    schedule_to_close_timeout: timedelta | None = None,
    retry_policy: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to mark a function as a workflow task,
    automatically registering it in the global activity registry.

    Args:
        name: Optional custom name for the activity
        schedule_to_close_timeout: Maximum time the task can take to complete
        retry_policy: Retry policy configuration
        **kwargs: Additional metadata passed to the activity registration

    Returns:
        Decorated function that preserves async and typing information

    Raises:
        TypeError: If the decorated function is not async
        ValueError: If the retry policy or timeout is invalid
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(f"Function {func.__name__} must be async.")

        actual_name = name or f"{func.__module__}.{func.__qualname__}"
        timeout = schedule_to_close_timeout or timedelta(minutes=10)
        metadata = {
            "activity_name": actual_name,
            "schedule_to_close_timeout": timeout,
            "retry_policy": retry_policy or {},
            **kwargs,
        }
        activity_registry = get_activity_registry()
        activity_registry.register(actual_name, func, metadata)

        setattr(func, "is_workflow_task", True)
        setattr(func, "execution_metadata", metadata)

        # TODO: saqadri - determine if we need this
        # Preserve metadata through partial application
        # @functools.wraps(func)
        # async def wrapper(*args: Any, **kwargs: Any) -> R:
        #     result = await func(*args, **kwargs)
        #     return cast(R, result)  # Ensure type checking works

        # # Add metadata that survives partial application
        # wrapper.is_workflow_task = True  # type: ignore
        # wrapper.execution_metadata = metadata  # type: ignore

        # # Make metadata accessible through partial
        # def __getattr__(name: str) -> Any:
        #     if name == "is_workflow_task":
        #         return True
        #     if name == "execution_metadata":
        #         return metadata
        #     raise AttributeError(f"'{func.__name__}' has no attribute '{name}'")

        # wrapper.__getattr__ = __getattr__  # type: ignore

        # return wrapper

        return func

    return decorator


def is_workflow_task(func: Callable[..., Any]) -> bool:
    """
    Check if a function is marked as a workflow task.
    This gets set for functions that are decorated with @workflow_task."""
    return bool(getattr(func, "is_workflow_task", False))


# ############################
# # Example: DocumentWorkflow
# ############################


# @workflow_defn  # <-- This becomes @temporal_workflow.defn if in Temporal mode, else no-op
# class DocumentWorkflow(Workflow[List[Dict[str, Any]]]):
#     """
#     Example workflow with persistent state.
#     If run locally, `self.state` is ephemeral.
#     If run in Temporal mode, `self.state` is replayed automatically.
#     """

#     @workflow_task(
#         schedule_to_close_timeout=timedelta(minutes=10),
#         retry_policy={"initial_interval": 1, "max_attempts": 3},
#     )
#     async def process_document(self, doc_id: str) -> Dict[str, Any]:
#         """Activity that simulates document processing."""
#         await asyncio.sleep(1)
#         # Optionally mutate workflow state
#         self.state.metadata.setdefault("processed_docs", []).append(doc_id)
#         return {
#             "doc_id": doc_id,
#             "status": "processed",
#             "timestamp": datetime.utcnow().isoformat(),
#         }

#     @workflow_run  # <-- This becomes @temporal_workflow.run(...) if Temporal is used
#     async def _run_impl(
#         self, documents: List[str], batch_size: int = 2
#     ) -> List[Dict[str, Any]]:
#         """Main workflow logic, which becomes the official 'run' in Temporal mode."""
#         self._logger.info("Workflow starting, state=%s", self.state)
#         self.state.update_status("running")

#         all_results = []
#         for i in range(0, len(documents), batch_size):
#             batch = documents[i : i + batch_size]
#             tasks = [self.process_document(doc) for doc in batch]
#             results = await self.executor.execute(*tasks)

#             for res in results:
#                 if isinstance(res.value, Exception):
#                     self._logger.error(
#                         f"Error processing document: {res.metadata.get('error')}"
#                     )
#                 else:
#                     all_results.append(res.value)

#         self.state.update_status("completed")
#         return all_results


# ########################
# # 12. Example Local Usage
# ########################


# async def run_example_local():
#     from . import AsyncIOExecutor, DocumentWorkflow  # if in a package

#     executor = AsyncIOExecutor()
#     wf = DocumentWorkflow(executor)

#     documents = ["doc1", "doc2", "doc3", "doc4"]
#     result = await wf.run(documents, batch_size=2)

#     print("Local results:", result.value)
#     print("Local workflow final state:", wf.state)
#     # Notice `wf.state.metadata['processed_docs']` has the processed doc IDs.


# ########################
# # Example Temporal Usage
# ########################


# async def run_example_temporal():
#     from . import TemporalExecutor, DocumentWorkflow  # if in a package

#     # 1) Create a TemporalExecutor (client side)
#     executor = TemporalExecutor(task_queue="my_task_queue")
#     await executor.ensure_client()

#     # 2) Start a worker in the same process (or do so in a separate process)
#     asyncio.create_task(executor.start_worker())
#     await asyncio.sleep(2)  # Wait for worker to be up

#     # 3) Now we can run the workflow by normal means if we like,
#     #    or rely on the Worker picking it up. Typically, you'd do:
#     #    handle = await executor._client.start_workflow(...)
#     #    but let's keep it simple and show conceptually
#     #    that 'DocumentWorkflow' is now recognized as a real Temporal workflow
#     print(
#         "Temporal environment is running. Use the Worker logs or CLI to start 'DocumentWorkflow'."
#     )

```

### src/mcp_agent/executor/workflow_signal.py

```py
import asyncio
import uuid
from abc import abstractmethod, ABC
from typing import Any, Callable, Dict, Generic, List, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

SignalValueT = TypeVar("SignalValueT")

# TODO: saqadri - handle signals properly that works with other execution backends like Temporal as well


class Signal(BaseModel, Generic[SignalValueT]):
    """Represents a signal that can be sent to a workflow."""

    name: str
    description: str = "Workflow Signal"
    payload: SignalValueT | None = None
    metadata: Dict[str, Any] | None = None
    workflow_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SignalRegistration(BaseModel):
    """Tracks registration of a signal handler."""

    signal_name: str
    unique_name: str
    workflow_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SignalHandler(Protocol, Generic[SignalValueT]):
    """Protocol for handling signals."""

    @abstractmethod
    async def signal(self, signal: Signal[SignalValueT]) -> None:
        """Emit a signal to all waiting handlers and registered callbacks."""

    @abstractmethod
    async def wait_for_signal(
        self,
        signal: Signal[SignalValueT],
        timeout_seconds: int | None = None,
    ) -> SignalValueT:
        """Wait for a signal to be emitted."""

    def on_signal(self, signal_name: str) -> Callable:
        """
        Decorator to register a handler for a signal.

        Example:
            @signal_handler.on_signal("approval_needed")
            async def handle_approval(value: str):
                print(f"Got approval signal with value: {value}")
        """


class PendingSignal(BaseModel):
    """Tracks a waiting signal handler and its event."""

    registration: SignalRegistration
    event: asyncio.Event | None = None
    value: SignalValueT | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseSignalHandler(ABC, Generic[SignalValueT]):
    """Base class implementing common signal handling functionality."""

    def __init__(self):
        # Map signal_name -> list of PendingSignal objects
        self._pending_signals: Dict[str, List[PendingSignal]] = {}
        # Map signal_name -> list of (unique_name, handler) tuples
        self._handlers: Dict[str, List[tuple[str, Callable]]] = {}
        self._lock = asyncio.Lock()

    async def cleanup(self, signal_name: str | None = None):
        """Clean up handlers and registrations for a signal or all signals."""
        async with self._lock:
            if signal_name:
                if signal_name in self._handlers:
                    del self._handlers[signal_name]
                if signal_name in self._pending_signals:
                    del self._pending_signals[signal_name]
            else:
                self._handlers.clear()
                self._pending_signals.clear()

    def validate_signal(self, signal: Signal[SignalValueT]):
        """Validate signal properties."""
        if not signal.name:
            raise ValueError("Signal name is required")
        # Subclasses can override to add more validation

    def on_signal(self, signal_name: str) -> Callable:
        """Register a handler for a signal."""

        def decorator(func: Callable) -> Callable:
            unique_name = f"{signal_name}_{uuid.uuid4()}"

            async def wrapped(value: SignalValueT):
                try:
                    if asyncio.iscoroutinefunction(func):
                        await func(value)
                    else:
                        func(value)
                except Exception as e:
                    # Log the error but don't fail the entire signal handling
                    print(f"Error in signal handler {signal_name}: {str(e)}")

            self._handlers.setdefault(signal_name, []).append((unique_name, wrapped))
            return wrapped

        return decorator

    @abstractmethod
    async def signal(self, signal: Signal[SignalValueT]) -> None:
        """Emit a signal to all waiting handlers and registered callbacks."""

    @abstractmethod
    async def wait_for_signal(
        self,
        signal: Signal[SignalValueT],
        timeout_seconds: int | None = None,
    ) -> SignalValueT:
        """Wait for a signal to be emitted."""


class ConsoleSignalHandler(SignalHandler[str]):
    """Simple console-based signal handling (blocks on input)."""

    def __init__(self):
        self._pending_signals: Dict[str, List[PendingSignal]] = {}
        self._handlers: Dict[str, List[Callable]] = {}

    async def wait_for_signal(self, signal, timeout_seconds=None):
        """Block and wait for console input."""
        print(f"\n[SIGNAL: {signal.name}] {signal.description}")
        if timeout_seconds:
            print(f"(Timeout in {timeout_seconds} seconds)")

        # Use asyncio.get_event_loop().run_in_executor to make input non-blocking
        loop = asyncio.get_event_loop()
        if timeout_seconds is not None:
            try:
                value = await asyncio.wait_for(
                    loop.run_in_executor(None, input, "Enter value: "), timeout_seconds
                )
            except asyncio.TimeoutError:
                print("\nTimeout waiting for input")
                raise
        else:
            value = await loop.run_in_executor(None, input, "Enter value: ")

        return value

        # value = input(f"[SIGNAL: {signal.name}] {signal.description}: ")
        # return value

    def on_signal(self, signal_name):
        def decorator(func):
            async def wrapped(value: SignalValueT):
                if asyncio.iscoroutinefunction(func):
                    await func(value)
                else:
                    func(value)

            self._handlers.setdefault(signal_name, []).append(wrapped)
            return wrapped

        return decorator

    async def signal(self, signal):
        print(f"[SIGNAL SENT: {signal.name}] Value: {signal.payload}")

        handlers = self._handlers.get(signal.name, [])
        await asyncio.gather(
            *(handler(signal) for handler in handlers), return_exceptions=True
        )

        # Notify any waiting coroutines
        if signal.name in self._pending_signals:
            for ps in self._pending_signals[signal.name]:
                ps.value = signal.payload
                ps.event.set()


class AsyncioSignalHandler(BaseSignalHandler[SignalValueT]):
    """
    Asyncio-based signal handling using an internal dictionary of asyncio Events.
    """

    async def wait_for_signal(
        self, signal, timeout_seconds: int | None = None
    ) -> SignalValueT:
        event = asyncio.Event()
        unique_name = str(uuid.uuid4())

        registration = SignalRegistration(
            signal_name=signal.name,
            unique_name=unique_name,
            workflow_id=signal.workflow_id,
        )

        pending_signal = PendingSignal(registration=registration, event=event)

        async with self._lock:
            # Add to pending signals
            self._pending_signals.setdefault(signal.name, []).append(pending_signal)

        try:
            # Wait for signal
            if timeout_seconds is not None:
                await asyncio.wait_for(event.wait(), timeout_seconds)
            else:
                await event.wait()

            return pending_signal.value
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"Timeout waiting for signal {signal.name}") from e
        finally:
            async with self._lock:
                # Remove from pending signals
                if signal.name in self._pending_signals:
                    self._pending_signals[signal.name] = [
                        ps
                        for ps in self._pending_signals[signal.name]
                        if ps.registration.unique_name != unique_name
                    ]
                    if not self._pending_signals[signal.name]:
                        del self._pending_signals[signal.name]

    def on_signal(self, signal_name):
        def decorator(func):
            async def wrapped(value: SignalValueT):
                if asyncio.iscoroutinefunction(func):
                    await func(value)
                else:
                    func(value)

            self._handlers.setdefault(signal_name, []).append(wrapped)
            return wrapped

        return decorator

    async def signal(self, signal):
        async with self._lock:
            # Notify any waiting coroutines
            if signal.name in self._pending_signals:
                pending = self._pending_signals[signal.name]
                for ps in pending:
                    ps.value = signal.payload
                    ps.event.set()

        # Notify any registered handler functions
        tasks = []
        handlers = self._handlers.get(signal.name, [])
        for _, handler in handlers:
            tasks.append(handler(signal))

        await asyncio.gather(*tasks, return_exceptions=True)


# TODO: saqadri - check if we need to do anything to combine this and AsyncioSignalHandler
class LocalSignalStore:
    """
    Simple in-memory structure that allows coroutines to wait for a signal
    and triggers them when a signal is emitted.
    """

    def __init__(self):
        # For each signal_name, store a list of futures that are waiting for it
        self._waiters: Dict[str, List[asyncio.Future]] = {}

    async def emit(self, signal_name: str, payload: Any):
        # If we have waiting futures, set their result
        if signal_name in self._waiters:
            for future in self._waiters[signal_name]:
                if not future.done():
                    future.set_result(payload)
            self._waiters[signal_name].clear()

    async def wait_for(
        self, signal_name: str, timeout_seconds: int | None = None
    ) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        self._waiters.setdefault(signal_name, []).append(future)

        if timeout_seconds is not None:
            try:
                return await asyncio.wait_for(future, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                # remove the fut from list
                if not future.done():
                    self._waiters[signal_name].remove(future)
                raise
        else:
            return await future

```

### src/mcp_agent/mcp/**init**.py

```py

```

### src/mcp_agent/mcp/gen_client.py

```py
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncGenerator, Callable

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession

from mcp_agent.context import get_current_context
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp_server_registry import ServerRegistry
from mcp_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

logger = get_logger(__name__)


@asynccontextmanager
async def gen_client(
    server_name: str,
    client_session_factory: Callable[
        [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
        ClientSession,
    ] = MCPAgentClientSession,
    server_registry: ServerRegistry | None = None,
) -> AsyncGenerator[ClientSession, None]:
    """
    Create a client session to the specified server.
    Handles server startup, initialization, and message receive loop setup.
    If required, callers can specify their own message receive loop and ClientSession class constructor to customize further.
    For persistent connections, use connect() or MCPConnectionManager instead.
    """
    ctx = get_current_context()
    server_registry = server_registry or ctx.server_registry

    if not server_registry:
        raise ValueError(
            "Server registry not found in the context. Please specify one either on this method, or in the context."
        )

    async with server_registry.initialize_server(
        server_name=server_name,
        client_session_factory=client_session_factory,
    ) as session:
        yield session


async def connect(
    server_name: str,
    client_session_factory: Callable[
        [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
        ClientSession,
    ] = MCPAgentClientSession,
    server_registry: ServerRegistry | None = None,
) -> ClientSession:
    """
    Create a persistent client session to the specified server.
    Handles server startup, initialization, and message receive loop setup.
    If required, callers can specify their own message receive loop and ClientSession class constructor to customize further.
    """
    ctx = get_current_context()
    server_registry = server_registry or ctx.server_registry

    if not server_registry:
        raise ValueError(
            "Server registry not found in the context. Please specify one either on this method, or in the context."
        )

    server_connection = await server_registry.connection_manager.get_server(
        server_name=server_name,
        client_session_factory=client_session_factory,
    )

    return server_connection.session


async def disconnect(
    server_name: str | None,
    server_registry: ServerRegistry | None = None,
) -> None:
    """
    Disconnect from the specified server. If server_name is None, disconnect from all servers.
    """
    ctx = get_current_context()
    server_registry = server_registry or ctx.server_registry

    if not server_registry:
        raise ValueError(
            "Server registry not found in the context. Please specify one either on this method, or in the context."
        )

    if server_name:
        await server_registry.connection_manager.disconnect_server(
            server_name=server_name
        )
    else:
        await server_registry.connection_manager.disconnect_all_servers()

```

### src/mcp_agent/mcp/mcp_activity.py

```py
# import functools
# from temporalio import activity
# from typing import Dict, Any, List, Callable, Awaitable
# from .gen_client import gen_client


# def mcp_activity(server_name: str, mcp_call: Callable):
#     def decorator(func):
#         @activity.defn
#         @functools.wraps(func)
#         async def wrapper(*activity_args, **activity_kwargs):
#             params = await func(*activity_args, **activity_kwargs)
#             async with gen_client(server_name) as client:
#                 return await mcp_call(client, params)

#         return wrapper

#     return decorator

```

### src/mcp_agent/mcp/mcp_agent_client_session.py

```py
"""
A derived client session for the MCP Agent framework.
It adds logging and supports sampling requests.
"""

from mcp import ClientSession
from mcp.shared.session import (
    RequestResponder,
    ReceiveResultT,
    ReceiveNotificationT,
    RequestId,
    SendNotificationT,
    SendRequestT,
    SendResultT,
)
from mcp.types import (
    ClientResult,
    CreateMessageRequest,
    CreateMessageResult,
    ErrorData,
    JSONRPCNotification,
    JSONRPCRequest,
    ServerRequest,
    TextContent,
)

from mcp_agent.context import get_current_context, get_current_config
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class MCPAgentClientSession(ClientSession):
    """
    MCP Agent framework acts as a client to the servers providing tools/resources/prompts for the agent workloads.
    This is a simple client session for those server connections, and supports
        - handling sampling requests
        - notifications

    Developers can extend this class to add more custom functionality as needed
    """

    async def initialize(self) -> None:
        logger.debug("initialize...")
        try:
            await super().initialize()
            logger.debug("initialized")
        except Exception as e:
            logger.error(f"initialize failed: {e}")
            raise

    async def __aenter__(self):
        # logger.debug(
        #     f"__aenter__ {str(self)}: current_task={anyio.get_current_task()}, id={id(anyio.get_current_task())}"
        # )
        return await super().__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # logger.debug(
        #     f"__aexit__ {str(self)}: current_task={anyio.get_current_task()}, id={id(anyio.get_current_task())}"
        # )
        return await super().__aexit__(exc_type, exc_val, exc_tb)

    async def _received_request(
        self, responder: RequestResponder[ServerRequest, ClientResult]
    ) -> None:
        logger.debug("Received request:", data=responder.request.model_dump())
        request = responder.request.root

        if isinstance(request, CreateMessageRequest):
            return await self.handle_sampling_request(request, responder)

        # Handle other requests as usual
        await super()._received_request(responder)

    async def send_request(
        self,
        request: SendRequestT,
        result_type: type[ReceiveResultT],
    ) -> ReceiveResultT:
        logger.debug("send_request: request=", data=request.model_dump())
        try:
            result = await super().send_request(request, result_type)
            logger.debug("send_request: response=", data=result.model_dump())
            return result
        except Exception as e:
            logger.error(f"send_request failed: {e}")
            raise

    async def send_notification(self, notification: SendNotificationT) -> None:
        logger.debug("send_notification:", data=notification.model_dump())
        try:
            return await super().send_notification(notification)
        except Exception as e:
            logger.error("send_notification failed", data=e)
            raise

    async def _send_response(
        self, request_id: RequestId, response: SendResultT | ErrorData
    ) -> None:
        logger.debug(
            f"send_response: request_id={request_id}, response=",
            data=response.model_dump(),
        )
        return await super()._send_response(request_id, response)

    async def _received_notification(self, notification: ReceiveNotificationT) -> None:
        """
        Can be overridden by subclasses to handle a notification without needing
        to listen on the message stream.
        """
        logger.info(
            "_received_notification: notification=",
            data=notification.model_dump(),
        )
        return await super()._received_notification(notification)

    async def send_progress_notification(
        self, progress_token: str | int, progress: float, total: float | None = None
    ) -> None:
        """
        Sends a progress notification for a request that is currently being
        processed.
        """
        logger.debug(
            "send_progress_notification: progress_token={progress_token}, progress={progress}, total={total}"
        )
        return await super().send_progress_notification(
            progress_token=progress_token, progress=progress, total=total
        )

    async def _receive_loop(self) -> None:
        async with (
            self._read_stream,
            self._write_stream,
            self._incoming_message_stream_writer,
        ):
            async for message in self._read_stream:
                if isinstance(message, Exception):
                    await self._incoming_message_stream_writer.send(message)
                elif isinstance(message.root, JSONRPCRequest):
                    validated_request = self._receive_request_type.model_validate(
                        message.root.model_dump(
                            by_alias=True, mode="json", exclude_none=True
                        )
                    )
                    responder = RequestResponder(
                        request_id=message.root.id,
                        request_meta=validated_request.root.params.meta
                        if validated_request.root.params
                        else None,
                        request=validated_request,
                        session=self,
                    )

                    await self._received_request(responder)
                    if not responder._responded:
                        await self._incoming_message_stream_writer.send(responder)
                elif isinstance(message.root, JSONRPCNotification):
                    notification = self._receive_notification_type.model_validate(
                        message.root.model_dump(
                            by_alias=True, mode="json", exclude_none=True
                        )
                    )

                    await self._received_notification(notification)
                    await self._incoming_message_stream_writer.send(notification)
                else:  # Response or error
                    stream = self._response_streams.pop(message.root.id, None)
                    if stream:
                        await stream.send(message.root)
                    else:
                        await self._incoming_message_stream_writer.send(
                            RuntimeError(
                                "Received response with an unknown "
                                f"request ID: {message}"
                            )
                        )

    async def handle_sampling_request(
        self,
        request: CreateMessageRequest,
        responder: RequestResponder[ServerRequest, ClientResult],
    ):
        logger.info("Handling sampling request: %s", request)
        ctx = get_current_context()
        config = get_current_config()
        session = ctx.upstream_session
        if session is None:
            # TODO: saqadri - consider whether we should be handling the sampling request here as a client
            print(
                f"Error: No upstream client available for sampling requests. Request: {request}"
            )
            try:
                from anthropic import AsyncAnthropic

                client = AsyncAnthropic(api_key=config.anthropic.api_key)

                params = request.params
                response = await client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=params.maxTokens,
                    messages=[
                        {
                            "role": m.role,
                            "content": m.content.text
                            if hasattr(m.content, "text")
                            else m.content.data,
                        }
                        for m in params.messages
                    ],
                    system=getattr(params, "systemPrompt", None),
                    temperature=getattr(params, "temperature", 0.7),
                    stop_sequences=getattr(params, "stopSequences", None),
                )

                await responder.respond(
                    CreateMessageResult(
                        model="claude-3-sonnet-20240229",
                        role="assistant",
                        content=TextContent(type="text", text=response.content[0].text),
                    )
                )
            except Exception as e:
                logger.error(f"Error handling sampling request: {e}")
                await responder.respond(ErrorData(code=-32603, message=str(e)))
        else:
            try:
                # If a session is available, we'll pass-through the sampling request to the upstream client
                result = await session.send_request(
                    request=ServerRequest(request), result_type=CreateMessageResult
                )

                # Pass the result from the upstream client back to the server. We just act as a pass-through client here.
                await responder.send_result(result)
            except Exception as e:
                await responder.send_error(code=-32603, message=str(e))

```

### src/mcp_agent/mcp/mcp_agent_server.py

```py
import asyncio
from mcp.server import NotificationOptions
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from mcp_agent.executor.temporal import get_temporal_client
from mcp_agent.telemetry.tracing import setup_tracing

app = FastMCP("mcp-agent-server")

setup_tracing("mcp-agent-server")


async def run():
    async with stdio_server() as (read_stream, write_stream):
        await app._mcp_server.run(
            read_stream,
            write_stream,
            app._mcp_server.create_initialization_options(
                notification_options=NotificationOptions(
                    tools_changed=True, resources_changed=True
                )
            ),
        )


@app.tool
async def run_workflow(query: str):
    """Run the workflow given its name or id"""
    pass


@app.tool
async def pause_workflow(workflow_id: str):
    """Pause a running workflow."""
    temporal_client = await get_temporal_client()
    handle = temporal_client.get_workflow_handle(workflow_id)
    await handle.signal("pause")


@app.tool
async def resume_workflow(workflow_id: str):
    """Resume a paused workflow."""
    temporal_client = await get_temporal_client()
    handle = temporal_client.get_workflow_handle(workflow_id)
    await handle.signal("resume")


async def provide_user_input(workflow_id: str, input_data: str):
    """Provide user/human input to a waiting workflow step."""
    temporal_client = await get_temporal_client()
    handle = temporal_client.get_workflow_handle(workflow_id)
    await handle.signal("human_input", input_data)


if __name__ == "__main__":
    asyncio.run(run())

```

### src/mcp_agent/mcp/mcp_aggregator.py

```py
from asyncio import Lock, gather
from typing import List, Dict

from pydantic import BaseModel, ConfigDict
from mcp.client.session import ClientSession
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    Tool,
)

from mcp_agent.context import get_current_context
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.gen_client import gen_client

from mcp_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager


logger = get_logger(__name__)

SEP = "-"


class NamespacedTool(BaseModel):
    """
    A tool that is namespaced by server name.
    """

    tool: Tool
    server_name: str
    namespaced_tool_name: str


class MCPAggregator(BaseModel):
    """
    Aggregates multiple MCP servers. When a developer calls, e.g. call_tool(...),
    the aggregator searches all servers in its list for a server that provides that tool.
    """

    initialized: bool = False
    """Whether the aggregator has been initialized with tools and resources from all servers."""

    connection_persistence: bool = False
    """Whether to maintain a persistent connection to the server."""

    server_names: List[str]
    """A list of server names to connect to."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    async def __aenter__(self):
        if self.initialized:
            return self

        # Keep a connection manager to manage persistent connections for this aggregator
        if self.connection_persistence:
            ctx = get_current_context()
            self._persistent_connection_manager = MCPConnectionManager(
                ctx.server_registry
            )
            await self._persistent_connection_manager.__aenter__()

        await self.load_servers()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __init__(
        self, server_names: List[str], connection_persistence: bool = False, **kwargs
    ):
        """
        :param server_names: A list of server names to connect to.
        Note: The server names must be resolvable by the gen_client function, and specified in the server registry.
        """
        super().__init__(
            server_names=server_names,
            connection_persistence=connection_persistence,
            **kwargs,
        )

        self._persistent_connection_manager: MCPConnectionManager = None

        # Maps namespaced_tool_name -> namespaced tool info
        self._namespaced_tool_map: Dict[str, NamespacedTool] = {}
        # Maps server_name -> list of tools
        self._server_to_tool_map: Dict[str, List[NamespacedTool]] = {}
        self._tool_map_lock = Lock()

        # TODO: saqadri - add resources and prompt maps as well

    async def close(self):
        """
        Close all persistent connections when the aggregator is deleted.
        """
        if self.connection_persistence and self._persistent_connection_manager:
            try:
                await self._persistent_connection_manager.disconnect_all()
                self.initialized = False
            finally:
                await self._persistent_connection_manager.__aexit__(None, None, None)

    @classmethod
    async def create(
        cls,
        server_names: List[str],
        connection_persistence: bool = False,
    ) -> "MCPAggregator":
        """
        Factory method to create and initialize an MCPAggregator.
        Use this instead of constructor since we need async initialization.
        If connection_persistence is True, the aggregator will maintain a
        persistent connection to the servers for as long as this aggregator is around.
        By default we do not maintain a persistent connection.
        """

        logger.info(f"Creating MCPAggregator with servers: {server_names}")

        instance = cls(
            server_names=server_names,
            connection_persistence=connection_persistence,
        )

        try:
            await instance.__aenter__()

            logger.debug("Loading servers...")
            await instance.load_servers()

            logger.debug("MCPAggregator created and initialized.")
            return instance
        except Exception as e:
            logger.error(f"Error creating MCPAggregator: {e}")
            await instance.__aexit__(None, None, None)

    async def load_servers(self):
        """
        Discover tools from each server in parallel and build an index of namespaced tool names.
        """
        if self.initialized:
            logger.debug("MCPAggregator already initialized.")
            return

        async with self._tool_map_lock:
            self._namespaced_tool_map.clear()
            self._server_to_tool_map.clear()

        for server_name in self.server_names:
            if self.connection_persistence:
                await self._persistent_connection_manager.get_server(
                    server_name, client_session_factory=MCPAgentClientSession
                )

        async def fetch_tools(client: ClientSession):
            try:
                result: ListToolsResult = await client.list_tools()
                return result.tools or []
            except Exception as e:
                print(f"Error loading tools from server '{server_name}': {e}")
                return []

        async def load_server_tools(server_name: str):
            tools: List[Tool] = []
            if self.connection_persistence:
                server_connection = (
                    await self._persistent_connection_manager.get_server(
                        server_name, client_session_factory=MCPAgentClientSession
                    )
                )
                tools = await fetch_tools(server_connection.session)
            else:
                async with gen_client(server_name) as client:
                    tools = await fetch_tools(client)

            return server_name, tools

        # Gather tools from all servers concurrently
        results = await gather(
            *(load_server_tools(server_name) for server_name in self.server_names),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, BaseException):
                continue
            server_name, tools = result

            self._server_to_tool_map[server_name] = []
            for tool in tools:
                namespaced_tool_name = f"{server_name}{SEP}{tool.name}"
                namespaced_tool = NamespacedTool(
                    tool=tool,
                    server_name=server_name,
                    namespaced_tool_name=namespaced_tool_name,
                )

                self._namespaced_tool_map[namespaced_tool_name] = namespaced_tool
                self._server_to_tool_map[server_name].append(namespaced_tool)

        self.initialized = True

    async def list_servers(self) -> List[str]:
        """Return the list of server names aggregated by this agent."""
        if not self.initialized:
            await self.load_servers()

        return self.server_names

    async def list_tools(self) -> ListToolsResult:
        """
        :return: Tools from all servers aggregated, and renamed to be dot-namespaced by server name.
        """
        if not self.initialized:
            await self.load_servers()

        return ListToolsResult(
            tools=[
                namespaced_tool.tool.model_copy(update={"name": namespaced_tool_name})
                for namespaced_tool_name, namespaced_tool in self._namespaced_tool_map.items()
            ]
        )

    async def call_tool(
        self, name: str, arguments: dict | None = None
    ) -> CallToolResult:
        """
        Call a namespaced tool, e.g., 'server_name.tool_name'.
        """
        if not self.initialized:
            await self.load_servers()

        server_name: str = None
        local_tool_name: str = None

        if SEP in name:  # Namespaced tool name
            server_name, local_tool_name = name.split(SEP, 1)
        else:
            # Assume un-namespaced, loop through all servers to find the tool. First match wins.
            for _, tools in self._server_to_tool_map.items():
                for namespaced_tool in tools:
                    if namespaced_tool.tool.name == name:
                        server_name = namespaced_tool.server_name
                        local_tool_name = name
                        break

            if server_name is None or local_tool_name is None:
                logger.error(f"Error: Tool '{name}' not found")
                return CallToolResult(isError=True, message=f"Tool '{name}' not found")

        logger.info(
            f"MCPServerAggregator: Requesting tool call '{name}'. Calling tool '{local_tool_name}' on server '{server_name}'"
        )

        async def try_call_tool(client: ClientSession):
            try:
                return await client.call_tool(name=local_tool_name, arguments=arguments)
            except Exception as e:
                return CallToolResult(
                    isError=True,
                    message=f"Failed to call tool '{local_tool_name}' on server '{server_name}': {e}",
                )

        if self.connection_persistence:
            server_connection = await self._persistent_connection_manager.get_server(
                server_name, client_session_factory=MCPAgentClientSession
            )
            return await try_call_tool(server_connection.session)
        else:
            async with gen_client(server_name) as client:
                return await try_call_tool(client)


class MCPCompoundServer(Server):
    """
    A compound server (server-of-servers) that aggregates multiple MCP servers and is itself an MCP server
    """

    def __init__(self, server_names: List[str], name: str = "MCPCompoundServer"):
        super().__init__(name)
        self.aggregator = MCPAggregator(server_names)

        # Register handlers
        # TODO: saqadri - once we support resources and prompts, add handlers for those as well
        self.list_tools()(self._list_tools)
        self.call_tool()(self._call_tool)

    async def _list_tools(self) -> List[Tool]:
        """List all tools aggregated from connected MCP servers."""
        tools_result = await self.aggregator.list_tools()
        return tools_result.tools

    async def _call_tool(
        self, name: str, arguments: dict | None = None
    ) -> CallToolResult:
        """Call a specific tool from the aggregated servers."""
        try:
            result = await self.aggregator.call_tool(name=name, arguments=arguments)
            return result.content
        except Exception as e:
            return CallToolResult(isError=True, message=f"Error calling tool: {e}")

    async def run_stdio_async(self) -> None:
        """Run the server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.run(
                read_stream=read_stream,
                write_stream=write_stream,
                initialization_options=self.create_initialization_options(),
            )

```

### src/mcp_agent/mcp/mcp_connection_manager.py

```py
"""
Manages the lifecycle of multiple MCP server connections.
"""

from datetime import timedelta
from typing import (
    AsyncGenerator,
    Callable,
    Dict,
    Optional,
    TYPE_CHECKING,
)

from anyio import Event, create_task_group, Lock
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.types import JSONRPCMessage

from mcp_agent.config import MCPServerSettings
from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp_agent.mcp_server_registry import InitHookCallable, ServerRegistry

logger = get_logger(__name__)


class ServerConnection:
    """
    Represents a long-lived MCP server connection, including:
    - The ClientSession to the server
    - The transport streams (via stdio/sse, etc.)
    """

    def __init__(
        self,
        server_name: str,
        server_config: MCPServerSettings,
        transport_context_factory: Callable[
            [],
            AsyncGenerator[
                tuple[
                    MemoryObjectReceiveStream[JSONRPCMessage | Exception],
                    MemoryObjectSendStream[JSONRPCMessage],
                ],
                None,
            ],
        ],
        client_session_factory: Callable[
            [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
            ClientSession,
        ],
        init_hook: Optional["InitHookCallable"] = None,
    ):
        self.server_name = server_name
        self.server_config = server_config
        self.session: ClientSession | None = None
        self._client_session_factory = client_session_factory
        self._init_hook = init_hook
        self._transport_context_factory = transport_context_factory

        # Signal that session is fully up and initialized
        self._initialized_event = Event()

        # Signal we want to shut down
        self._shutdown_event = Event()

    def request_shutdown(self) -> None:
        """
        Request the server to shut down. Signals the server lifecycle task to exit.
        """
        self._shutdown_event.set()

    async def wait_for_shutdown_request(self) -> None:
        """
        Wait until the shutdown event is set.
        """
        await self._shutdown_event.wait()

    async def initialize_session(self) -> None:
        """
        Initializes the server connection and session.
        Must be called within an async context.
        """
        logger.info(f"{self.server_name}: Initializing server session...")
        await self.session.initialize()
        logger.info(f"{self.server_name}: Session initialized.")

        # If there's an init hook, run it
        if self._init_hook:
            logger.info(f"{self.server_name}: Executing init hook.")
            self._init_hook(self.session, self.server_config.auth)

        # Now the session is ready for use
        self._initialized_event.set()

    async def wait_for_initialized(self) -> None:
        """
        Wait until the session is fully initialized.
        """
        await self._initialized_event.wait()

    def create_session(
        self,
        read_stream: MemoryObjectReceiveStream,
        send_stream: MemoryObjectSendStream,
    ) -> ClientSession:
        """
        Create a new session instance for this server connection.
        """

        read_timeout = (
            timedelta(seconds=self.server_config.read_timeout_seconds)
            if self.server_config.read_timeout_seconds
            else None
        )

        session = self._client_session_factory(read_stream, send_stream, read_timeout)

        self.session = session

        return session


async def _server_lifecycle_task(server_conn: ServerConnection) -> None:
    """
    Manage the lifecycle of a single server connection.
    Runs inside the MCPConnectionManager's shared TaskGroup.
    """
    server_name = server_conn.server_name
    try:
        transport_context = server_conn._transport_context_factory()

        async with transport_context as (read_stream, write_stream):
            # Build a session
            server_conn.create_session(read_stream, write_stream)

            async with server_conn.session:
                # Initialize the session
                await server_conn.initialize_session()

                # Wait until we’re asked to shut down
                await server_conn.wait_for_shutdown_request()

    except Exception as exc:
        logger.error(
            f"{server_name}: Lifecycle task encountered an error: {exc}", exc_info=True
        )
        # If there's an error, we should also set the event so that
        # 'get_server' won't hang
        server_conn._initialized_event.set()
        raise
    finally:
        logger.debug(f"{server_name}: _lifecycle_task is exiting.")


class MCPConnectionManager:
    """
    Manages the lifecycle of multiple MCP server connections.
    """

    def __init__(self, server_registry: "ServerRegistry"):
        self.server_registry = server_registry
        self.running_servers: Dict[str, ServerConnection] = {}
        self._lock = Lock()
        self._tg: TaskGroup | None = None

    async def __aenter__(self):
        # We create a task group to manage all server lifecycle tasks
        self._tg = create_task_group()
        await self._tg.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.debug("MCPConnectionManager: shutting down all server tasks...")
        if self._tg:
            await self._tg.__aexit__(exc_type, exc_val, exc_tb)
        self._tg = None

    async def launch_server(
        self,
        server_name: str,
        client_session_factory: Callable[
            [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
            ClientSession,
        ],
        init_hook: Optional["InitHookCallable"] = None,
    ) -> ServerConnection:
        """
        Connect to a server and return a RunningServer instance that will persist
        until explicitly disconnected.
        """
        if not self._tg:
            raise RuntimeError(
                "MCPConnectionManager must be used inside an async context (i.e. 'async with' or after __aenter__)."
            )

        config = self.server_registry.registry.get(server_name)
        if not config:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        logger.debug(
            f"{server_name}: Found server configuration=", data=config.model_dump()
        )

        def transport_context_factory():
            if config.transport == "stdio":
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                )
                return stdio_client(server_params)
            elif config.transport == "sse":
                return sse_client(config.url)
            else:
                raise ValueError(f"Unsupported transport: {config.transport}")

        server_conn = ServerConnection(
            server_name=server_name,
            server_config=config,
            transport_context_factory=transport_context_factory,
            client_session_factory=client_session_factory,
            init_hook=init_hook or self.server_registry.init_hooks.get(server_name),
        )

        async with self._lock:
            # Check if already running
            if server_name in self.running_servers:
                return self.running_servers[server_name]

            self.running_servers[server_name] = server_conn
            self._tg.start_soon(_server_lifecycle_task, server_conn)

        logger.info(f"{server_name}: Up and running with a persistent connection!")
        return server_conn

    async def get_server(
        self,
        server_name: str,
        client_session_factory: Callable,
        init_hook: Optional["InitHookCallable"] = None,
    ) -> ServerConnection:
        """
        Get a running server instance, launching it if needed.
        """
        # Get the server connection if it's already running
        async with self._lock:
            server_conn = self.running_servers.get(server_name)
            if server_conn:
                return server_conn

        # Launch the connection
        server_conn = await self.launch_server(
            server_name=server_name,
            client_session_factory=client_session_factory,
            init_hook=init_hook,
        )

        # Wait until it's fully initialized, or an error occurs
        await server_conn.wait_for_initialized()

        # If the session is still None, it means the lifecycle task crashed
        if not server_conn or not server_conn.session:
            raise RuntimeError(
                f"{server_name}: Failed to initialize server; check logs for errors."
            )
        return server_conn

    async def disconnect_server(self, server_name: str) -> None:
        """
        Disconnect a specific server if it's running under this connection manager.
        """
        logger.info(f"{server_name}: Disconnecting persistent connection to server...")

        async with self._lock:
            server_conn = self.running_servers.pop(server_name, None)
        if server_conn:
            server_conn.request_shutdown()
            logger.info(
                f"{server_name}: Shutdown signal sent (lifecycle task will exit)."
            )
        else:
            logger.info(
                f"{server_name}: No persistent connection found. Skipping server shutdown"
            )

    async def disconnect_all(self) -> None:
        """
        Disconnect all servers that are running under this connection manager.
        """
        logger.info("Disconnecting all persistent server connections...")
        async with self._lock:
            for conn in self.running_servers.values():
                conn.request_shutdown()
            self.running_servers.clear()
        logger.info("All persistent server connections signaled to disconnect.")

```

### src/mcp_agent/mcp_server_registry.py

```py
"""
This module defines a `ServerRegistry` class for managing MCP server configurations
and initialization logic.

The class loads server configurations from a YAML file,
supports dynamic registration of initialization hooks, and provides methods for
server initialization.
"""

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Callable, Dict, AsyncGenerator

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client

from mcp_agent.config import (
    get_settings,
    MCPServerAuthSettings,
    MCPServerSettings,
    Settings,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager

logger = get_logger(__name__)

InitHookCallable = Callable[[ClientSession | None, MCPServerAuthSettings | None], bool]
"""
A type alias for an initialization hook function that is invoked after MCP server initialization.

Args:
    session (ClientSession | None): The client session for the server connection.
    auth (MCPServerAuthSettings | None): The authentication configuration for the server.

Returns:
    bool: Result of the post-init hook (false indicates failure).
"""


class ServerRegistry:
    """
    A registry for managing server configurations and initialization logic.

    The `ServerRegistry` class is responsible for loading server configurations
    from a YAML file, registering initialization hooks, initializing servers,
    and executing post-initialization hooks dynamically.

    Attributes:
        config_path (str): Path to the YAML configuration file.
        registry (Dict[str, MCPServerSettings]): Loaded server configurations.
        init_hooks (Dict[str, InitHookCallable]): Registered initialization hooks.
    """

    def __init__(self, config: Settings | None = None, config_path: str | None = None):
        """
        Initialize the ServerRegistry with a configuration file.

        Args:
            config (Settings): The Settings object containing the server configurations.
            config_path (str): Path to the YAML configuration file.
        """
        self.registry = (
            self.load_registry_from_file(config_path)
            if config is None
            else config.mcp.servers
        )
        self.init_hooks: Dict[str, InitHookCallable] = {}
        self.connection_manager = MCPConnectionManager(self)

    def load_registry_from_file(
        self, config_path: str | None = None
    ) -> Dict[str, MCPServerSettings]:
        """
        Load the YAML configuration file and validate it.

        Returns:
            Dict[str, MCPServerSettings]: A dictionary of server configurations.

        Raises:
            ValueError: If the configuration is invalid.
        """

        servers = get_settings(config_path).mcp.servers or {}
        return servers

    @asynccontextmanager
    async def start_server(
        self,
        server_name: str,
        client_session_factory: Callable[
            [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
            ClientSession,
        ] = ClientSession,
    ) -> AsyncGenerator[ClientSession, None]:
        """
        Starts the server process based on its configuration. To initialize, call initialize_server

        Args:
            server_name (str): The name of the server to initialize.

        Returns:
            StdioServerParameters: The server parameters for stdio transport.

        Raises:
            ValueError: If the server is not found or has an unsupported transport.
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        config = self.registry[server_name]

        read_timeout_seconds = (
            timedelta(config.read_timeout_seconds)
            if config.read_timeout_seconds
            else None
        )

        if config.transport == "stdio":
            if not config.command or not config.args:
                raise ValueError(
                    f"Command and args are required for stdio transport: {server_name}"
                )

            server_params = StdioServerParameters(
                command=config.command, args=config.args
            )

            async with stdio_client(server_params) as (read_stream, write_stream):
                session = client_session_factory(
                    read_stream,
                    write_stream,
                    read_timeout_seconds,
                )
                async with session:
                    logger.info(
                        f"{server_name}: Connected to server using stdio transport."
                    )
                    try:
                        yield session
                    finally:
                        logger.debug(f"{server_name}: Closed session to server")

        elif config.transport == "sse":
            if not config.url:
                raise ValueError(f"URL is required for SSE transport: {server_name}")

            # Use sse_client to get the read and write streams
            async with sse_client(config.url) as (read_stream, write_stream):
                session = client_session_factory(
                    read_stream,
                    write_stream,
                    read_timeout_seconds,
                )
                async with session:
                    logger.info(
                        f"{server_name}: Connected to server using SSE transport."
                    )
                    try:
                        yield session
                    finally:
                        logger.debug(f"{server_name}: Closed session to server")

        # Unsupported transport
        else:
            raise ValueError(f"Unsupported transport: {config.transport}")

    @asynccontextmanager
    async def initialize_server(
        self,
        server_name: str,
        client_session_factory: Callable[
            [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
            ClientSession,
        ] = ClientSession,
        init_hook: InitHookCallable = None,
    ) -> AsyncGenerator[ClientSession, None]:
        """
        Initialize a server based on its configuration.
        After initialization, also calls any registered or provided initialization hook for the server.

        Args:
            server_name (str): The name of the server to initialize.
            init_hook (InitHookCallable): Optional initialization hook function to call after initialization.

        Returns:
            StdioServerParameters: The server parameters for stdio transport.

        Raises:
            ValueError: If the server is not found or has an unsupported transport.
        """

        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        config = self.registry[server_name]

        async with self.start_server(
            server_name, client_session_factory=client_session_factory
        ) as session:
            try:
                logger.info(f"{server_name}: Initializing server...")
                await session.initialize()
                logger.info(f"{server_name}: Initialized.")

                intialization_callback = (
                    init_hook
                    if init_hook is not None
                    else self.init_hooks.get(server_name)
                )

                if intialization_callback:
                    logger.info(f"{server_name}: Executing init hook")
                    intialization_callback(session, config.auth)

                logger.info(f"{server_name}: Up and running!")
                yield session
            finally:
                logger.info(f"{server_name}: Ending server session.")

    def register_init_hook(self, server_name: str, hook: InitHookCallable) -> None:
        """
        Register an initialization hook for a specific server. This will get called
        after the server is initialized.

        Args:
            server_name (str): The name of the server.
            hook (callable): The initialization function to register.
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        self.init_hooks[server_name] = hook

    def execute_init_hook(self, server_name: str, session=None) -> bool:
        """
        Execute the initialization hook for a specific server.

        Args:
            server_name (str): The name of the server.
            session: The session object to pass to the initialization hook.
        """
        if server_name in self.init_hooks:
            hook = self.init_hooks[server_name]
            config = self.registry[server_name]
            logger.info(f"Executing init hook for '{server_name}'")
            return hook(session, config.auth)
        else:
            logger.info(f"No init hook registered for '{server_name}'")

    def get_server_config(self, server_name: str) -> MCPServerSettings | None:
        """
        Get the configuration for a specific server.

        Args:
            server_name (str): The name of the server.

        Returns:
            MCPServerSettings: The server configuration.
        """
        server_config = self.registry.get(server_name)
        if server_config is None:
            logger.warning(f"Server '{server_name}' not found in registry.")
            return None
        elif server_config.name is None:
            server_config.name = server_name
        return server_config

```

### src/mcp_agent/py.typed

```typed

```

### src/mcp_agent/telemetry/**init**.py

```py

```

### src/mcp_agent/telemetry/usage_tracking.py

```py
from mcp_agent.config import get_settings


def send_usage_data():
    config = get_settings()
    if not config.usage_telemetry.enabled:
        print("Usage tracking disabled")
        return

    # TODO: saqadri - implement usage tracking
    # data = {"installation_id": str(uuid.uuid4()), "version": "0.1.0"}
    # try:
    #     requests.post("https://telemetry.example.com/usage", json=data, timeout=2)
    # except:
    #     pass

```

### src/mcp_agent/workflows/**init**.py

```py

```

### src/mcp_agent/workflows/embedding/**init**.py

```py

```

### src/mcp_agent/workflows/embedding/embedding_base.py

```py
from abc import ABC, abstractmethod
from typing import Dict, List

from numpy import float32
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity


FloatArray = NDArray[float32]


class EmbeddingModel(ABC):
    """Abstract interface for embedding models"""

    @abstractmethod
    async def embed(self, data: List[str]) -> FloatArray:
        """
        Generate embeddings for a list of messages

        Args:
            data: List of text strings to embed

        Returns:
            Array of embeddings, shape (len(texts), embedding_dim)
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings"""


def compute_similarity_scores(
    embedding_a: FloatArray, embedding_b: FloatArray
) -> Dict[str, float]:
    """
    Compute different similarity metrics between embeddings
    """
    # Reshape for sklearn's cosine_similarity
    a_emb = embedding_a.reshape(1, -1)
    b_emb = embedding_b.reshape(1, -1)

    cosine_sim = float(cosine_similarity(a_emb, b_emb)[0, 0])

    # Could add other similarity metrics here
    return {
        "cosine": cosine_sim,
        # "euclidean": float(euclidean_similarity),
        # "dot_product": float(dot_product)
    }


def compute_confidence(similarity_scores: Dict[str, float]) -> float:
    """
    Compute overall confidence score from individual similarity metrics
    """
    # For now, just use cosine similarity as confidence
    # Could implement more sophisticated combination of scores
    return similarity_scores["cosine"]

```

### src/mcp_agent/workflows/embedding/embedding_cohere.py

```py
from typing import List

from cohere import Client
from numpy import array, float32

from mcp_agent.context import get_current_config
from mcp_agent.workflows.embedding.embedding_base import EmbeddingModel, FloatArray


class CohereEmbeddingModel(EmbeddingModel):
    """Cohere embedding model implementation"""

    def __init__(self, model: str = "embed-multilingual-v3.0"):
        self.client = Client(api_key=get_current_config().cohere.api_key)
        self.model = model
        # Cache the dimension since it's fixed per model
        # https://docs.cohere.com/v2/docs/cohere-embed
        self._embedding_dim = {
            "embed-english-v2.0": 4096,
            "embed-english-light-v2.0": 1024,
            "embed-english-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-v2.0": 768,
            "embed-multilingual-v3.0": 1024,
            "embed-multilingual-light-v3.0": 384,
        }[model]

    async def embed(self, data: List[str]) -> FloatArray:
        response = self.client.embed(
            texts=data,
            model=self.model,
            input_type="classification",
            embedding_types=["float"],
        )

        embeddings = array(response.embeddings, dtype=float32)
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

```

### src/mcp_agent/workflows/embedding/embedding_openai.py

```py
from typing import List

from numpy import array, float32, stack
from openai import OpenAI

from mcp_agent.context import get_current_config
from mcp_agent.workflows.embedding.embedding_base import EmbeddingModel, FloatArray


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model implementation"""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=get_current_config().openai.api_key)
        self.model = model
        # Cache the dimension since it's fixed per model
        self._embedding_dim = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }[model]

    async def embed(self, data: List[str]) -> FloatArray:
        response = self.client.embeddings.create(
            model=self.model, input=data, encoding_format="float"
        )

        # Sort the embeddings by their index to ensure correct order
        sorted_embeddings = sorted(response.data, key=lambda x: x["index"])

        # Stack all embeddings into a single array
        embeddings = stack(
            [
                array(embedding["embedding"], dtype=float32)
                for embedding in sorted_embeddings
            ]
        )
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

```

### src/mcp_agent/workflows/evaluator_optimizer/**init**.py

```py

```

### src/mcp_agent/workflows/evaluator_optimizer/evaluator_optimizer.py

```py
import contextlib
from enum import Enum
from typing import Callable, List, Type
from pydantic import BaseModel, Field

from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
)
from mcp_agent.agents.agent import Agent
from mcp_agent.executor.executor import Executor
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class QualityRating(str, Enum):
    """Enum for evaluation quality ratings"""

    POOR = 0  # Major improvements needed
    FAIR = 1  # Several improvements needed
    GOOD = 2  # Minor improvements possible
    EXCELLENT = 3  # No improvements needed


class EvaluationResult(BaseModel):
    """Model representing the evaluation result from the evaluator LLM"""

    rating: QualityRating = Field(description="Quality rating of the response")
    feedback: str = Field(
        description="Specific feedback and suggestions for improvement"
    )
    needs_improvement: bool = Field(
        description="Whether the output needs further improvement"
    )
    focus_areas: List[str] = Field(
        default_factory=list, description="Specific areas to focus on in next iteration"
    )


class EvaluatorOptimizerLLM(AugmentedLLM[MessageParamT, MessageT]):
    """
    Implementation of the evaluator-optimizer workflow where one LLM generates responses
    while another provides evaluation and feedback in a refinement loop.

    This can be used either:
    1. As a standalone workflow with its own optimizer agent
    2. As a wrapper around another workflow (Orchestrator, Router, ParallelLLM) to add
       evaluation and refinement capabilities

    When to use this workflow:
    - When you have clear evaluation criteria and iterative refinement provides value
    - When LLM responses improve with articulated feedback
    - When the task benefits from focused iteration on specific aspects

    Examples:
    - Literary translation with "expert" refinement
    - Complex search tasks needing multiple rounds
    - Document writing requiring multiple revisions
    """

    def __init__(
        self,
        optimizer: Agent | AugmentedLLM,
        evaluator: str | Agent | AugmentedLLM,
        min_rating: QualityRating = QualityRating.GOOD,
        max_refinements: int = 3,
        llm_factory: Callable[[Agent], AugmentedLLM] | None = None,
        executor: Executor | None = None,
    ):
        """
        Initialize the evaluator-optimizer workflow.

        Args:
            optimizer: The agent/LLM/workflow that generates responses. Can be:
                     - An Agent that will be converted to an AugmentedLLM
                     - An AugmentedLLM instance
                     - An Orchestrator/Router/ParallelLLM workflow
            evaluator_agent: The agent/LLM that evaluates responses
            evaluation_criteria: Criteria for the evaluator to assess responses
            min_rating: Minimum acceptable quality rating
            max_refinements: Maximum refinement iterations
            llm_factory: Optional factory to create LLMs from agents
            executor: Optional executor for parallel operations
        """
        super().__init__(executor=executor)

        # Set up the optimizer
        self.name = optimizer.name
        self.llm_factory = llm_factory
        self.optimizer = optimizer
        self.evaluator = evaluator

        if isinstance(optimizer, Agent):
            if not llm_factory:
                raise ValueError("llm_factory is required when using an Agent")

            self.optimizer_llm = llm_factory(agent=optimizer)
            self.aggregator = optimizer
            self.instruction = (
                optimizer.instruction
                if isinstance(optimizer.instruction, str)
                else None
            )

        elif isinstance(optimizer, AugmentedLLM):
            self.optimizer_llm = optimizer
            self.aggregator = optimizer.aggregator
            self.instruction = optimizer.instruction

        else:
            raise ValueError(f"Unsupported optimizer type: {type(optimizer)}")

        self.history = self.optimizer_llm.history

        # Set up the evaluator
        if isinstance(evaluator, AugmentedLLM):
            self.evaluator_llm = evaluator
        elif isinstance(evaluator, Agent):
            if not llm_factory:
                raise ValueError(
                    "llm_factory is required when using an Agent evaluator"
                )

            self.evaluator_llm = llm_factory(agent=evaluator)
        elif isinstance(evaluator, str):
            # If a string is passed as the evaluator, we use it as the evaluation criteria
            # and create an evaluator agent with that instruction
            if not llm_factory:
                raise ValueError(
                    "llm_factory is required when using a string evaluator"
                )

            self.evaluator_llm = llm_factory(
                agent=Agent(name="Evaluator", instruction=evaluator)
            )
        else:
            raise ValueError(f"Unsupported evaluator type: {type(evaluator)}")

        self.min_rating = min_rating
        self.max_refinements = max_refinements

        # Track iteration history
        self.refinement_history = []

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> List[MessageT]:
        """Generate an optimized response through evaluation-guided refinement"""
        refinement_count = 0
        response = None
        best_response = None
        best_rating = QualityRating.POOR
        self.refinement_history = []

        # Initial generation
        async with contextlib.AsyncExitStack() as stack:
            if isinstance(self.optimizer, Agent):
                await stack.enter_async_context(self.optimizer)
            response = await self.optimizer_llm.generate(
                message=message,
                use_history=use_history,
                max_iterations=max_iterations,
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )

        best_response = response

        while refinement_count < self.max_refinements:
            logger.debug("Optimizer result:", data=response)

            # Evaluate current response
            eval_prompt = self._build_eval_prompt(
                original_request=str(message),
                current_response="\n".join(str(r) for r in response)
                if isinstance(response, list)
                else str(response),
                iteration=refinement_count,
            )

            evaluation_result = None
            async with contextlib.AsyncExitStack() as stack:
                if isinstance(self.evaluator, Agent):
                    await stack.enter_async_context(self.evaluator)

                evaluation_result = await self.evaluator_llm.generate_structured(
                    message=eval_prompt,
                    response_model=EvaluationResult,
                    model=model,
                    max_tokens=max_tokens,
                )

            # Track iteration
            self.refinement_history.append(
                {
                    "attempt": refinement_count + 1,
                    "response": response,
                    "evaluation_result": evaluation_result,
                }
            )

            logger.debug("Evaluator result:", data=evaluation_result)

            # Track best response (using enum ordering)
            if evaluation_result.rating.value > best_rating.value:
                best_rating = evaluation_result.rating
                best_response = response
                logger.debug(
                    "New best response:",
                    data={"rating": best_rating, "response": best_response},
                )

            # Check if we've reached acceptable quality
            if (
                evaluation_result.rating.value >= self.min_rating.value
                or not evaluation_result.needs_improvement
            ):
                logger.debug(
                    f"Acceptable quality {evaluation_result.rating.value} reached",
                    data={
                        "rating": evaluation_result.rating.value,
                        "needs_improvement": evaluation_result.needs_improvement,
                        "min_rating": self.min_rating.value,
                    },
                )
                break

            # Generate refined response
            refinement_prompt = self._build_refinement_prompt(
                original_request=str(message),
                current_response="\n".join(str(r) for r in response)
                if isinstance(response, list)
                else str(response),
                feedback=evaluation_result,
                iteration=refinement_count,
            )

            async with contextlib.AsyncExitStack() as stack:
                if isinstance(self.optimizer, Agent):
                    await stack.enter_async_context(self.optimizer)

                response = await self.optimizer_llm.generate(
                    message=refinement_prompt,
                    use_history=use_history,
                    max_iterations=max_iterations,
                    model=model,
                    stop_sequences=stop_sequences,
                    max_tokens=max_tokens,
                    parallel_tool_calls=parallel_tool_calls,
                )

            refinement_count += 1

        return best_response

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> str:
        """Generate an optimized response and return it as a string"""
        response = await self.generate(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        return "\n".join(self.optimizer_llm.message_str(r) for r in response)

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> ModelT:
        """Generate an optimized structured response"""
        response_str = await self.generate_str(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        return await self.optimizer.generate_structured(
            message=response_str,
            response_model=response_model,
            model=model,
            max_tokens=max_tokens,
        )

    def _build_eval_prompt(
        self, original_request: str, current_response: str, iteration: int
    ) -> str:
        """Build the evaluation prompt for the evaluator"""
        return f"""
        Evaluate the following response based on these criteria:
        {self.evaluator.instruction}

        Original Request: {original_request}
        Current Response (Iteration {iteration + 1}): {current_response}

        Provide your evaluation as a structured response with:
        1. A quality rating (EXCELLENT, GOOD, FAIR, or POOR)
        2. Specific feedback and suggestions
        3. Whether improvement is needed (true/false)
        4. Focus areas for improvement

        Rate as EXCELLENT only if no improvements are needed.
        Rate as GOOD if only minor improvements are possible.
        Rate as FAIR if several improvements are needed.
        Rate as POOR if major improvements are needed.
        """

    def _build_refinement_prompt(
        self,
        original_request: str,
        current_response: str,
        feedback: EvaluationResult,
        iteration: int,
    ) -> str:
        """Build the refinement prompt for the optimizer"""
        return f"""
        Improve your previous response based on the evaluation feedback.

        Original Request: {original_request}

        Previous Response (Iteration {iteration + 1}):
        {current_response}

        Quality Rating: {feedback.rating}
        Feedback: {feedback.feedback}
        Areas to Focus On: {", ".join(feedback.focus_areas)}

        Generate an improved version addressing the feedback while maintaining accuracy and relevance.
        """

```

### src/mcp_agent/workflows/intent_classifier/**init**.py

```py

```

### src/mcp_agent/workflows/intent_classifier/intent_classifier_base.py

```py
from abc import ABC, abstractmethod
from typing import Dict, List
from pydantic import BaseModel, Field


class Intent(BaseModel):
    """A class that represents a single intent category"""

    name: str
    """The name of the intent"""

    description: str | None = None
    """A description of what this intent represents"""

    examples: List[str] = Field(default_factory=list)
    """Example phrases or requests that match this intent"""

    metadata: Dict[str, str] = Field(default_factory=dict)
    """Additional metadata about the intent that might be useful for classification"""


class IntentClassificationResult(BaseModel):
    """A class that represents the result of intent classification"""

    intent: str
    """The classified intent name"""

    p_score: float | None = None
    """
    The probability score (i.e. 0->1) of the classification.
    This is optional and may only be provided if the classifier is probabilistic (e.g. a probabilistic binary classifier).
    """

    extracted_entities: Dict[str, str] = Field(default_factory=dict)
    """Any entities or parameters extracted from the input request that are relevant to the intent"""


class IntentClassifier(ABC):
    """
    Base class for intent classification. This can be implemented using different approaches
    like LLMs, embedding models, traditional ML classification models, or rule-based systems.

    When to use this:
        - When you need to understand the user's intention before routing or processing
        - When you want to extract structured information from natural language inputs
        - When you need to handle multiple related but distinct types of requests

    Examples:
        - Classifying customer service requests (complaint, question, feedback)
        - Understanding user commands in a chat interface
        - Determining the type of analysis requested for a dataset
    """

    def __init__(self, intents: List[Intent]):
        self.intents = {intent.name: intent for intent in intents}
        self.initialized: bool = False

        if not self.intents:
            raise ValueError("At least one intent must be provided")

    @abstractmethod
    async def classify(
        self, request: str, top_k: int = 1
    ) -> List[IntentClassificationResult]:
        """
        Classify the input request into one or more intents.

        Args:
            request: The input text to classify
            top_k: Maximum number of top intent matches to return. May return fewer.

        Returns:
            List of classification results, ordered by confidence
        """

    async def initialize(self):
        """Initialize the classifier. Override this method if needed."""
        self.initialized = True


# Example
# Define some intents
# intents = [
#     Intent(
#         name="schedule_meeting",
#         description="Schedule or set up a meeting or appointment",
#         examples=[
#             "Can you schedule a meeting with John?",
#             "Set up a call for next week",
#             "I need to arrange a meeting"
#         ]
#     ),
#     Intent(
#         name="check_calendar",
#         description="Check calendar availability or existing appointments",
#         examples=[
#             "What meetings do I have today?",
#             "Show me my calendar",
#             "Am I free tomorrow afternoon?"
#         ]
#     )
# ]

# # Initialize with OpenAI embeddings
# classifier = OpenAIEmbeddingIntentClassifier(intents=intents, model="text-embedding-3-small")

# # Or use Cohere embeddings
# classifier = OpenAIEmbeddingIntentClassifier(intents=intents, model="embed-multilingual-v3.0")

# # Classify some text
# results = await classifier.classify(
#     request="Can you set up a meeting with Sarah for tomorrow?"
#     top_k=3
# )

```

### src/mcp_agent/workflows/intent_classifier/intent_classifier_embedding.py

```py
from typing import List

from numpy import mean

from mcp_agent.workflows.embedding import (
    FloatArray,
    EmbeddingModel,
    compute_confidence,
    compute_similarity_scores,
)
from mcp_agent.workflows.intent_classifier.intent_classifier_base import (
    Intent,
    IntentClassifier,
    IntentClassificationResult,
)


class EmbeddingIntent(Intent):
    """An intent with embedding information"""

    embedding: FloatArray | None = None
    """Pre-computed embedding for this intent"""


class EmbeddingIntentClassifier(IntentClassifier):
    """
    An intent classifier that uses embedding similarity for classification.
    Supports different embedding models through the EmbeddingModel interface.

    Features:
    - Semantic similarity based classification
    - Support for example-based learning
    - Flexible embedding model support
    - Multiple similarity computation strategies
    """

    def __init__(
        self,
        intents: List[Intent],
        embedding_model: EmbeddingModel,
    ):
        super().__init__(intents=intents)
        self.embedding_model = embedding_model
        self.initialized = False

    @classmethod
    async def create(
        cls,
        intents: List[Intent],
        embedding_model: EmbeddingModel,
    ) -> "EmbeddingIntentClassifier":
        """
        Factory method to create and initialize a classifier.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            intents=intents,
            embedding_model=embedding_model,
        )
        await instance.initialize()
        return instance

    async def initialize(self):
        """
        Precompute embeddings for all intents by combining their
        descriptions and examples
        """
        if self.initialized:
            return

        for intent in self.intents.values():
            # Combine all text for a rich intent representation
            intent_texts = [intent.name, intent.description] + intent.examples

            # Get embeddings for all texts
            embeddings = await self.embedding_model.embed(intent_texts)

            # Use mean pooling to combine embeddings
            embedding = mean(embeddings, axis=0)

            # Create intents with embeddings
            self.intents[intent.name] = EmbeddingIntent(
                **intent,
                embedding=embedding,
            )

        self.initialized = True

    async def classify(
        self, request: str, top_k: int = 1
    ) -> List[IntentClassificationResult]:
        """
        Classify the input text into one or more intents

        Args:
            text: Input text to classify
            top_k: Maximum number of top matches to return

        Returns:
            List of classification results, ordered by confidence
        """
        if not self.initialized:
            await self.initialize()

        # Get embedding for input
        embeddings = await self.embedding_model.embed([request])
        request_embedding = embeddings[0]  # Take first since we only embedded one text

        results: List[IntentClassificationResult] = []
        for intent_name, intent in self.intents.items():
            if intent.embedding is None:
                continue

            similarity_scores = compute_similarity_scores(
                request_embedding, intent.embedding
            )

            # Compute overall confidence score
            confidence = compute_confidence(similarity_scores)

            results.append(
                IntentClassificationResult(
                    intent=intent_name,
                    p_score=confidence,
                )
            )

        results.sort(key=lambda x: x.p_score, reverse=True)
        return results[:top_k]

```

### src/mcp_agent/workflows/intent_classifier/intent_classifier_embedding_cohere.py

```py
from typing import List

from mcp_agent.workflows.embedding.embedding_cohere import CohereEmbeddingModel
from mcp_agent.workflows.intent_classifier.intent_classifier_base import Intent
from mcp_agent.workflows.intent_classifier.intent_classifier_embedding import (
    EmbeddingIntentClassifier,
)


class CohereEmbeddingIntentClassifier(EmbeddingIntentClassifier):
    """
    An intent classifier that uses Cohere's embedding models for computing semantic simiarity based classifications.
    """

    def __init__(
        self,
        intents: List[Intent],
        embedding_model: CohereEmbeddingModel | None = None,
    ):
        embedding_model = embedding_model or CohereEmbeddingModel()
        super().__init__(embedding_model=embedding_model, intents=intents)

    @classmethod
    async def create(
        cls,
        intents: List[Intent],
        embedding_model: CohereEmbeddingModel | None = None,
    ) -> "CohereEmbeddingIntentClassifier":
        """
        Factory method to create and initialize a classifier.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            intents=intents,
            embedding_model=embedding_model,
        )
        await instance.initialize()
        return instance

```

### src/mcp_agent/workflows/intent_classifier/intent_classifier_embedding_openai.py

```py
from typing import List

from mcp_agent.workflows.embedding.embedding_openai import OpenAIEmbeddingModel
from mcp_agent.workflows.intent_classifier.intent_classifier_base import Intent
from mcp_agent.workflows.intent_classifier.intent_classifier_embedding import (
    EmbeddingIntentClassifier,
)


class OpenAIEmbeddingIntentClassifier(EmbeddingIntentClassifier):
    """
    An intent classifier that uses OpenAI's embedding models for computing semantic simiarity based classifications.
    """

    def __init__(
        self,
        intents: List[Intent],
        embedding_model: OpenAIEmbeddingModel | None = None,
    ):
        embedding_model = embedding_model or OpenAIEmbeddingModel()
        super().__init__(embedding_model=embedding_model, intents=intents)

    @classmethod
    async def create(
        cls,
        intents: List[Intent],
        embedding_model: OpenAIEmbeddingModel | None = None,
    ) -> "OpenAIEmbeddingIntentClassifier":
        """
        Factory method to create and initialize a classifier.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            intents=intents,
            embedding_model=embedding_model,
        )
        await instance.initialize()
        return instance

```

### src/mcp_agent/workflows/intent_classifier/intent_classifier_llm.py

```py
from typing import List, Literal
from pydantic import BaseModel

from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM
from mcp_agent.workflows.intent_classifier.intent_classifier_base import (
    Intent,
    IntentClassifier,
    IntentClassificationResult,
)

DEFAULT_INTENT_CLASSIFICATION_INSTRUCTION = """
You are a precise intent classifier that analyzes user requests to determine their intended action or purpose.
Below are the available intents with their descriptions and examples:

{context}

Your task is to analyze the following request and determine the most likely intent(s). Consider:
- How well the request matches the intent descriptions and examples
- Any specific entities or parameters that should be extracted
- The confidence level in the classification

Request: {request}

Respond in JSON format:
{{
    "classifications": [
        {{
            "intent": <intent name>,
            "confidence": <float between 0 and 1>,
            "extracted_entities": {{
                "entity_name": "entity_value"
            }},
            "reasoning": <brief explanation>
        }}
    ]
}}

Return up to {top_k} most likely intents. Only include intents with reasonable confidence (>0.5).
If no intents match well, return an empty list.
"""


class LLMIntentClassificationResult(IntentClassificationResult):
    """The result of intent classification using an LLM."""

    confidence: Literal["low", "medium", "high"]
    """Confidence level of the classification"""

    reasoning: str | None = None
    """Optional explanation of why this intent was chosen"""


class StructuredIntentResponse(BaseModel):
    """The complete structured response from the LLM"""

    classifications: List[LLMIntentClassificationResult]


class LLMIntentClassifier(IntentClassifier):
    """
    An intent classifier that uses an LLM to determine the user's intent.
    Particularly useful when you need:
    - Flexible understanding of natural language
    - Detailed reasoning about classifications
    - Entity extraction alongside classification
    """

    def __init__(
        self,
        llm: AugmentedLLM,
        intents: List[Intent],
        classification_instruction: str | None = None,
    ):
        super().__init__(intents=intents)
        self.llm = llm
        self.classification_instruction = classification_instruction

    @classmethod
    async def create(
        cls,
        llm: AugmentedLLM,
        intents: List[Intent],
        classification_instruction: str | None = None,
    ) -> "LLMIntentClassifier":
        """
        Factory method to create and initialize a classifier.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            llm=llm,
            intents=intents,
            classification_instruction=classification_instruction,
        )
        await instance.initialize()
        return instance

    async def classify(
        self, request: str, top_k: int = 1
    ) -> List[LLMIntentClassificationResult]:
        if not self.initialized:
            self.initialize()

        classification_instruction = (
            self.classification_instruction or DEFAULT_INTENT_CLASSIFICATION_INSTRUCTION
        )

        # Generate the context with intent descriptions and examples
        context = self._generate_context()

        # Format the prompt with all the necessary information
        prompt = classification_instruction.format(
            context=context, request=request, top_k=top_k
        )

        # Get classification from LLM
        response = await self.llm.generate_structured(
            message=prompt, response_model=StructuredIntentResponse
        )

        if not response or not response.classifications:
            return []

        results = []
        for classification in response.classifications:
            intent = self.intents.get(classification.intent)
            if not intent:
                # Skip invalid categories
                # TODO: saqadri - log or raise an error
                continue

            results.append(classification)

        return results[:top_k]

    def _generate_context(self) -> str:
        """Generate a formatted context string describing all intents"""
        context_parts = []

        for idx, intent in enumerate(self.intents.values(), 1):
            description = (
                f"{idx}. Intent: {intent.name}\nDescription: {intent.description}"
            )

            if intent.examples:
                examples = "\n".join(f"- {example}" for example in intent.examples)
                description += f"\nExamples:\n{examples}"

            if intent.metadata:
                metadata = "\n".join(
                    f"- {key}: {value}" for key, value in intent.metadata.items()
                )
                description += f"\nAdditional Information:\n{metadata}"

            context_parts.append(description)

        return "\n\n".join(context_parts)

```

### src/mcp_agent/workflows/intent_classifier/intent_classifier_llm_anthropic.py

```py
from typing import List

from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.intent_classifier.intent_classifier_base import Intent
from mcp_agent.workflows.intent_classifier.intent_classifier_llm import (
    LLMIntentClassifier,
)

CLASSIFIER_SYSTEM_INSTRUCTION = """
You are a precise intent classifier that analyzes input requests to determine their intended action or purpose.
You are provided with a request and a list of intents to choose from.
You can choose one or more intents, or choose none if no intent is appropriate.
"""


class AnthropicLLMIntentClassifier(LLMIntentClassifier):
    """
    An LLM router that uses an Anthropic model to make routing decisions.
    """

    def __init__(
        self,
        intents: List[Intent],
        classification_instruction: str | None = None,
    ):
        anthropic_llm = AnthropicAugmentedLLM(instruction=CLASSIFIER_SYSTEM_INSTRUCTION)

        super().__init__(
            llm=anthropic_llm,
            intents=intents,
            classification_instruction=classification_instruction,
        )

    @classmethod
    async def create(
        cls,
        intents: List[Intent],
        classification_instruction: str | None = None,
    ) -> "AnthropicLLMIntentClassifier":
        """
        Factory method to create and initialize a classifier.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            intents=intents,
            classification_instruction=classification_instruction,
        )
        await instance.initialize()
        return instance

```

### src/mcp_agent/workflows/intent_classifier/intent_classifier_llm_openai.py

```py
from typing import List

from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.intent_classifier.intent_classifier_base import Intent
from mcp_agent.workflows.intent_classifier.intent_classifier_llm import (
    LLMIntentClassifier,
)

CLASSIFIER_SYSTEM_INSTRUCTION = """
You are a precise intent classifier that analyzes input requests to determine their intended action or purpose.
You are provided with a request and a list of intents to choose from.
You can choose one or more intents, or choose none if no intent is appropriate.
"""


class OpenAILLMIntentClassifier(LLMIntentClassifier):
    """
    An LLM router that uses an OpenAI model to make routing decisions.
    """

    def __init__(
        self,
        intents: List[Intent],
        classification_instruction: str | None = None,
    ):
        openai_llm = OpenAIAugmentedLLM(instruction=CLASSIFIER_SYSTEM_INSTRUCTION)

        super().__init__(
            llm=openai_llm,
            intents=intents,
            classification_instruction=classification_instruction,
        )

    @classmethod
    async def create(
        cls,
        intents: List[Intent],
        classification_instruction: str | None = None,
    ) -> "OpenAILLMIntentClassifier":
        """
        Factory method to create and initialize a classifier.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            intents=intents,
            classification_instruction=classification_instruction,
        )
        await instance.initialize()
        return instance

```

### src/mcp_agent/workflows/llm/**init**.py

```py

```

### src/mcp_agent/workflows/llm/augmented_llm.py

```py
from typing import Generic, List, Optional, Protocol, Type, TypeVar, TYPE_CHECKING

from mcp.types import (
    CallToolRequest,
    CallToolResult,
    TextContent,
)

from mcp_agent.executor.executor import Executor, AsyncioExecutor
from mcp_agent.mcp.mcp_aggregator import MCPAggregator

if TYPE_CHECKING:
    from mcp_agent.agents.agent import Agent

MessageParamT = TypeVar("MessageParamT")
"""A type representing an input message to an LLM."""

MessageT = TypeVar("MessageT")
"""A type representing an output message from an LLM."""

ModelT = TypeVar("ModelT")
"""A type representing a structured output message from an LLM."""


class Memory(Protocol, Generic[MessageParamT]):
    """
    Simple memory management for storing past interactions in-memory.
    """

    # TODO: saqadri - add checkpointing and other advanced memory capabilities

    def __init__(self): ...

    def extend(self, messages: List[MessageParamT]) -> None: ...

    def set(self, messages: List[MessageParamT]) -> None: ...

    def append(self, message: MessageParamT) -> None: ...

    def get(self) -> List[MessageParamT]: ...

    def clear(self) -> None: ...


class SimpleMemory(Memory, Generic[MessageParamT]):
    """
    Simple memory management for storing past interactions in-memory.
    """

    def __init__(self):
        self.history: List[MessageParamT] = []

    def extend(self, messages: List[MessageParamT]):
        self.history.extend(messages)

    def set(self, messages: List[MessageParamT]):
        self.history = messages.copy()

    def append(self, message: MessageParamT):
        self.history.append(message)

    def get(self) -> List[MessageParamT]:
        return self.history

    def clear(self):
        self.history = []


class AugmentedLLMProtocol(Protocol, Generic[MessageParamT, MessageT]):
    """Protocol defining the interface for augmented LLMs"""

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> List[MessageT]:
        """Request an LLM generation, which may run multiple iterations, and return the result"""

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> str:
        """Request an LLM generation and return the string representation of the result"""

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> ModelT:
        """Request a structured LLM generation and return the result as a Pydantic model."""


class AugmentedLLM(Generic[MessageParamT, MessageT]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    # TODO: saqadri - add streaming support (e.g. generate_stream)
    # TODO: saqadri - consider adding middleware patterns for pre/post processing of messages, for now we have pre/post_tool_call

    @classmethod
    async def convert_message_to_message_param(
        cls, message: MessageT, **kwargs
    ) -> MessageParamT:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        # Many LLM implementations will allow the same type for input and output messages
        return message

    def __init__(
        self,
        server_names: List[str] | None = None,
        instruction: str | None = None,
        name: str | None = None,
        agent: Optional["Agent"] = None,
        executor: Executor | None = None,
    ):
        """
        Initialize the LLM with a list of server names and an instruction.
        If a name is provided, it will be used to identify the LLM.
        If an agent is provided, all other properties are optional
        """
        self.executor = executor or AsyncioExecutor()
        self.aggregator = (
            agent if agent is not None else MCPAggregator(server_names or [])
        )
        self.name = name or (agent.name if agent else None)
        self.instruction = instruction or (
            agent.instruction if agent and isinstance(agent.instruction, str) else None
        )
        self.history: Memory[MessageParamT] = SimpleMemory[MessageParamT]()

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> List[MessageT]:
        """Request an LLM generation, which may run multiple iterations, and return the result"""

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> str:
        """Request an LLM generation and return the string representation of the result"""

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> ModelT:
        """Request a structured LLM generation and return the result as a Pydantic model."""

    async def get_last_message(self) -> MessageParamT | None:
        """
        Return the last message generated by the LLM or None if history is empty.
        This is useful for prompt chaining workflows where the last message from one LLM is used as input to another.
        """
        history = self.history.get()
        return history[-1] if history else None

    async def get_last_message_str(self) -> str | None:
        """Return the string representation of the last message generated by the LLM or None if history is empty."""
        last_message = await self.get_last_message()
        return self.message_param_str(last_message) if last_message else None

    async def pre_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest
    ) -> CallToolRequest | bool:
        """Called before a tool is executed. Return False to prevent execution."""
        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ) -> CallToolResult:
        """Called after a tool execution. Can modify the result before it's returned."""
        return result

    async def call_tool(
        self,
        request: CallToolRequest,
        tool_call_id: str | None = None,
    ) -> CallToolResult:
        """Call a tool with the given parameters and optional ID"""

        try:
            preprocess = await self.pre_tool_call(
                tool_call_id=tool_call_id,
                request=request,
            )

            if isinstance(preprocess, bool):
                if not preprocess:
                    return CallToolResult(
                        isError=True,
                        content=[
                            TextContent(
                                text=f"Error: Tool '{request.params.name}' was not allowed to run."
                            )
                        ],
                    )
            else:
                request = preprocess

            tool_name = request.params.name
            tool_args = request.params.arguments
            result = await self.aggregator.call_tool(tool_name, tool_args)

            postprocess = await self.post_tool_call(
                tool_call_id=tool_call_id, request=request, result=result
            )

            if isinstance(postprocess, CallToolResult):
                result = postprocess

            return result
        except Exception as e:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Error executing tool '{request.params.name}': {str(e)}",
                    )
                ],
            )

    def message_param_str(self, message: MessageParamT) -> str:
        """Convert an input message to a string representation."""
        return str(message)

    def message_str(self, message: MessageT) -> str:
        """Convert an output message to a string representation."""
        return str(message)

```

### src/mcp_agent/workflows/llm/augmented_llm_anthropic.py

```py
from typing import List, Type

import instructor
from anthropic import Anthropic
from anthropic.types import (
    Message,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
)

from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, ModelT
from mcp_agent.context import get_current_config
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class AnthropicAugmentedLLM(AugmentedLLM[MessageParam, Message]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    @classmethod
    async def convert_message_to_message_param(
        cls, message: Message, **kwargs
    ) -> MessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        content = []

        for content_block in message.content:
            if content_block.type == "text":
                content.append(TextBlockParam(type="text", text=content_block.text))
            elif content_block.type == "tool_use":
                content.append(
                    ToolUseBlockParam(
                        type="tool_use",
                        name=content_block.name,
                        input=content_block.input,
                        id=content_block.id,
                    )
                )

        return MessageParam(role="assistant", content=content, **kwargs)

    async def generate(
        self,
        message,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "claude-3-5-sonnet-20241022",
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = False,
    ):
        """
        Process a query using an LLM and available tools.
        The default implementation uses Claude as the LLM.
        Override this method to use a different LLM.
        """
        config = get_current_config()
        anthropic = Anthropic(api_key=config.anthropic.api_key)

        messages: List[MessageParam] = []

        if use_history:
            messages.extend(self.history.get())

        if isinstance(message, str):
            messages.append({"role": "user", "content": message})
        elif isinstance(message, list):
            messages.extend(message)
        else:
            messages.append(message)

        response = await self.aggregator.list_tools()
        available_tools: List[ToolParam] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        responses: List[Message] = []

        for i in range(max_iterations):
            arguments = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
                "system": self.instruction or None,
                "stop_sequences": stop_sequences,
                "tools": available_tools,
            }
            logger.debug(
                f"Iteration {i}: Calling {model} with messages:",
                data=messages,
            )

            executor_result = await self.executor.execute(
                anthropic.messages.create, **arguments
            )

            response = executor_result[0]

            logger.debug(
                f"Iteration {i}: {model} response:",
                data=response,
            )

            response_as_message = await self.convert_message_to_message_param(response)
            messages.append(response_as_message)
            responses.append(response)

            if response.stop_reason == "end_turn":
                logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'end_turn'"
                )
                break
            elif response.stop_reason == "stop_sequence":
                # We have reached a stop sequence
                logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'stop_sequence'"
                )
                break
            elif response.stop_reason == "max_tokens":
                # We have reached the max tokens limit
                logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'max_tokens'"
                )
                # TODO: saqadri - would be useful to return the reason for stopping to the caller
                break
            else:  # response.stop_reason == "tool_use":
                for content in response.content:
                    if content.type == "tool_use":
                        tool_name = content.name
                        tool_args = content.input
                        tool_use_id = content.id

                        tool_call_request = CallToolRequest(
                            method="tools/call",
                            params=CallToolRequestParams(
                                name=tool_name, arguments=tool_args
                            ),
                        )

                        result = await self.call_tool(
                            request=tool_call_request, tool_call_id=tool_use_id
                        )

                        messages.append(
                            MessageParam(
                                role="user",
                                content=[
                                    ToolResultBlockParam(
                                        type="tool_result",
                                        tool_use_id=tool_use_id,
                                        content=result.content,
                                        is_error=result.isError,
                                    )
                                ],
                            )
                        )

        if use_history:
            self.history.set(messages)

        logger.debug("Final response:", data=responses)

        return responses

    async def generate_str(
        self,
        message,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "claude-3-5-sonnet-20241022",
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = False,
    ) -> str:
        """
        Process a query using an LLM and available tools.
        The default implementation uses Claude as the LLM.
        Override this method to use a different LLM.
        """
        responses: List[Message] = await self.generate(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        final_text: List[str] = []

        for response in responses:
            for content in response.content:
                if content.type == "text":
                    final_text.append(content.text)
                elif content.type == "tool_use":
                    final_text.append(
                        f"[Calling tool {content.name} with args {content.input}]"
                    )

        return "\n".join(final_text)

    async def generate_structured(
        self,
        message,
        response_model: Type[ModelT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "claude-3-5-sonnet-20240620",
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> ModelT:
        # First we invoke the LLM to generate a string response
        # We need to do this in a two-step process because Instructor doesn't
        # know how to invoke MCP tools via call_tool, so we'll handle all the
        # processing first and then pass the final response through Instructor
        response = await self.generate_str(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Next we pass the text through instructor to extract structured data
        client = instructor.from_anthropic(
            Anthropic(api_key=get_current_config().anthropic.api_key),
        )

        # Extract structured data from natural language
        structured_response = client.chat.completions.create(
            model=model,
            response_model=response_model,
            messages=[{"role": "user", "content": response}],
            max_tokens=max_tokens,
        )

        return structured_response

    def message_param_str(self, message: MessageParam) -> str:
        """Convert an input message to a string representation."""

        if message.get("content"):
            content = message["content"]
            if isinstance(content, str):
                return content
            else:
                final_text: List[str] = []
                for block in content:
                    if block.text:
                        final_text.append(str(block.text))
                    else:
                        final_text.append(str(block))

                return "\n".join(final_text)

        return str(message)

    def message_str(self, message: Message) -> str:
        """Convert an output message to a string representation."""
        content = message.content

        if content:
            if isinstance(content, list):
                final_text: List[str] = []
                for block in content:
                    if block.text:
                        final_text.append(str(block.text))
                    else:
                        final_text.append(str(block))

                return "\n".join(final_text)
            else:
                return str(content)

        return str(message)

```

### src/mcp_agent/workflows/llm/augmented_llm_openai.py

```py
import json
from typing import List, Type

import instructor
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from mcp.types import (
    CallToolRequestParams,
    CallToolRequest,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)

from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, ModelT
from mcp_agent.context import get_current_config
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class OpenAIAugmentedLLM(
    AugmentedLLM[ChatCompletionMessageParam, ChatCompletionMessage]
):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    This implementation uses OpenAI's ChatCompletion as the LLM.
    """

    @classmethod
    async def convert_message_to_message_param(
        cls, message: ChatCompletionMessage, **kwargs
    ) -> ChatCompletionMessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            content=message.content,
            tool_calls=message.tool_calls,
            audio=message.audio,
            refusal=message.refusal,
            **kwargs,
        )

    async def generate(
        self,
        message,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "gpt-4o",
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ):
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """
        config = get_current_config()
        openai_client = OpenAI(api_key=config.openai.api_key)
        messages: List[ChatCompletionMessageParam] = []

        if self.instruction:
            messages.append(
                ChatCompletionSystemMessageParam(
                    role="system", content=self.instruction
                )
            )

        if use_history:
            messages.extend(self.history.get())

        if isinstance(message, str):
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=message)
            )
        elif isinstance(message, list):
            messages.extend(message)
        else:
            messages.append(message)

        response = await self.aggregator.list_tools()
        available_tools: List[ChatCompletionToolParam] = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                    # TODO: saqadri - determine if we should specify "strict" to True by default
                },
            )
            for tool in response.tools
        ]
        if not available_tools:
            available_tools = None

        responses: List[ChatCompletionMessage] = []

        for i in range(max_iterations):
            arguments = {
                "model": model,
                "messages": messages,
                "stop": stop_sequences,
                "tools": available_tools,
                "max_tokens": max_tokens,
            }

            if available_tools:
                arguments["tools"] = available_tools
                arguments["parallel_tool_calls"] = parallel_tool_calls

            logger.debug(
                f"Iteration {i}: Calling OpenAI ChatCompletion with messages:",
                data=messages,
            )

            executor_result = await self.executor.execute(
                openai_client.chat.completions.create, **arguments
            )

            response = executor_result[0]

            logger.debug(
                f"Iteration {i}: OpenAI ChatCompletion response:",
                data=response,
            )

            if not response.choices or len(response.choices) == 0:
                # No response from the model, we're done
                break

            # TODO: saqadri - handle multiple choices for more complex interactions.
            # Keeping it simple for now because multiple choices will also complicate memory management
            choice = response.choices[0]
            messages.append(choice.message)
            responses.append(choice.message)

            if choice.finish_reason == "stop":
                # We have reached the end of the conversation
                logger.debug(f"Iteration {i}: Stopping because finish_reason is 'stop'")
                break
            elif choice.finish_reason == "length":
                # We have reached the max tokens limit
                logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'length'"
                )
                # TODO: saqadri - would be useful to return the reason for stopping to the caller
                break
            elif choice.finish_reason == "content_filter":
                # The response was filtered by the content filter
                logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'content_filter'"
                )
                # TODO: saqadri - would be useful to return the reason for stopping to the caller
                break
            else:  #  choice.finish_reason in ["tool_calls", "function_call"]
                message = choice.message

                if message.content:
                    messages.append(
                        self.convert_message_to_message_param(message, name=self.name)
                    )

                if message.tool_calls:
                    # Execute all tool calls in parallel
                    tool_tasks = [
                        self.execute_tool_call(tool_call)
                        for tool_call in message.tool_calls
                    ]

                    # Wait for all tool calls to complete
                    tool_results = await self.executor.execute(*tool_tasks)
                    logger.debug(
                        f"Iteration {i}: Tool call results: {str(tool_results) if tool_results else 'None'}"
                    )

                    # Add non-None results to messages
                    for result in tool_results:
                        if isinstance(result, BaseException):
                            # Handle any unexpected exceptions during parallel execution
                            logger.error(
                                f"Warning: Unexpected error during tool execution: {result}. Continuing..."
                            )
                            continue
                        if result is not None:
                            messages.append(result)
        if use_history:
            self.history.set(messages)

        return responses

    async def generate_str(
        self,
        message,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "gpt-4o",
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ):
        """
        Process a query using an LLM and available tools.
        The default implementation uses OpenAI's ChatCompletion as the LLM.
        Override this method to use a different LLM.
        """
        responses = await self.generate(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        final_text: List[str] = []

        for response in responses:
            content = response.content
            if not content:
                continue

            if isinstance(content, str):
                final_text.append(content)
                continue

        return "\n".join(final_text)

    async def generate_structured(
        self,
        message,
        response_model: Type[ModelT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> ModelT:
        # First we invoke the LLM to generate a string response
        # We need to do this in a two-step process because Instructor doesn't
        # know how to invoke MCP tools via call_tool, so we'll handle all the
        # processing first and then pass the final response through Instructor
        response = await self.generate_str(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model or "gpt-4o",
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Next we pass the text through instructor to extract structured data
        client = instructor.from_openai(
            OpenAI(api_key=get_current_config().openai.api_key),
            mode=instructor.Mode.TOOLS_STRICT,
        )

        # Extract structured data from natural language
        structured_response = client.chat.completions.create(
            model=model or "gpt-4o",
            response_model=response_model,
            messages=[
                {"role": "user", "content": response},
            ],
        )

        return structured_response

    async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest):
        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ):
        return result

    async def execute_tool_call(
        self,
        tool_call: ChatCompletionToolParam,
    ) -> ChatCompletionToolMessageParam | None:
        """
        Execute a single tool call and return the result message.
        Returns None if there's no content to add to messages.
        """
        tool_name = tool_call.function.name
        tool_args_str = tool_call.function.arguments
        tool_call_id = tool_call.id
        tool_args = {}

        try:
            if tool_args_str:
                tool_args = json.loads(tool_args_str)
        except json.JSONDecodeError as e:
            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call_id,
                content=f"Invalid JSON provided in tool call arguments for '{tool_name}'. Failed to load JSON: {str(e)}",
            )

        tool_call_request = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=tool_name, arguments=tool_args),
        )

        result = await self.call_tool(
            request=tool_call_request, tool_call_id=tool_call_id
        )

        if result.content:
            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call_id,
                content=[mcp_content_to_openai_content(c) for c in result.content],
            )

        return None

    def message_param_str(self, message: ChatCompletionMessageParam) -> str:
        """Convert an input message to a string representation."""
        if message.get("content"):
            content = message["content"]
            if isinstance(content, str):
                return content
            else:  # content is a list
                final_text: List[str] = []
                for part in content:
                    text_part = part.get("text")
                    if text_part:
                        final_text.append(str(text_part))
                    else:
                        final_text.append(str(part))

                return "\n".join(final_text)

        return str(message)

    def message_str(self, message: ChatCompletionMessage) -> str:
        """Convert an output message to a string representation."""
        content = message.content
        if content:
            return content

        return str(message)


def mcp_content_to_openai_content(
    content: TextContent | ImageContent | EmbeddedResource,
) -> ChatCompletionContentPartTextParam:
    if isinstance(content, TextContent):
        return ChatCompletionContentPartTextParam(type="text", text=content.text)
    elif isinstance(content, ImageContent):
        # Best effort to convert an image to text
        return ChatCompletionContentPartTextParam(
            type="text", text=f"{content.mimeType}:{content.data}"
        )
    elif isinstance(content, EmbeddedResource):
        if isinstance(content.resource, TextResourceContents):
            return ChatCompletionContentPartTextParam(
                type="text", text=content.resource.text
            )
        else:  # BlobResourceContents
            return ChatCompletionContentPartTextParam(
                type="text", text=f"{content.resource.mimeType}:{content.resource.blob}"
            )
    else:
        # Last effort to convert the content to a string
        return ChatCompletionContentPartTextParam(type="text", text=str(content))

```

### src/mcp_agent/workflows/orchestrator/**init**.py

```py

```

### src/mcp_agent/workflows/orchestrator/orchestrator.py

```py
import contextlib
from typing import Any, Callable, Coroutine, List, Literal, Type

from mcp_agent.agents.agent import Agent
from mcp_agent.context import get_current_context
from mcp_agent.executor.executor import Executor
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
)
from mcp_agent.mcp_server_registry import ServerRegistry
from mcp_agent.workflows.orchestrator.orchestrator_models import (
    format_plan_result,
    format_step_result,
    NextStep,
    Plan,
    PlanResult,
    Step,
    StepResult,
    TaskWithResult,
)
from mcp_agent.workflows.orchestrator.orchestrator_prompts import (
    FULL_PLAN_PROMPT_TEMPLATE,
    ITERATIVE_PLAN_PROMPT_TEMPLATE,
    SYNTHESIZE_PLAN_PROMPT_TEMPLATE,
    TASK_PROMPT_TEMPLATE,
)
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class Orchestrator(AugmentedLLM[MessageParamT, MessageT]):
    """
    In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks,
    delegates them to worker LLMs, and synthesizes their results. It does this
    in a loop until the task is complete.

    When to use this workflow:
        - This workflow is well-suited for complex tasks where you can’t predict the
        subtasks needed (in coding, for example, the number of files that need to be
        changed and the nature of the change in each file likely depend on the task).

    Example where orchestrator-workers is useful:
        - Coding products that make complex changes to multiple files each time.
        - Search tasks that involve gathering and analyzing information from multiple sources
        for possible relevant information.
    """

    def __init__(
        self,
        llm_factory: Callable[[Agent], AugmentedLLM[MessageParamT, MessageT]],
        planner: AugmentedLLM | None = None,
        available_agents: List[Agent | AugmentedLLM] | None = None,
        executor: Executor | None = None,
        plan_type: Literal["full", "iterative"] = "full",
        server_registry: ServerRegistry | None = None,
    ):
        """
        Args:
            llm_factory: Factory function to create an LLM for a given agent
            planner: LLM to use for planning steps (if not provided, a default planner will be used)
            plan_type: "full" planning generates the full plan first, then executes. "iterative" plans the next step, and loops until success.
            available_agents: List of agents available to tasks executed by this orchestrator
            executor: Executor to use for parallel task execution (defaults to asyncio)
        """
        super().__init__(executor=executor)

        self.llm_factory = llm_factory

        self.planner = planner or llm_factory(
            agent=Agent(
                name="LLM Orchestration Planner",
                instruction="""
                You are an expert planner. Given an objective task and a list of MCP servers (which are collections of tools)
                or Agents (which are collections of servers), your job is to break down the objective into a series of steps,
                which can be performed by LLMs with access to the servers or agents.
                """,
            )
        )

        self.plan_type: Literal["full", "iterative"] = plan_type
        self.server_registry = server_registry or get_current_context().server_registry
        self.agents = {agent.name: agent for agent in available_agents or []}

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = False,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 16384,
        parallel_tool_calls: bool = True,
    ) -> List[MessageT]:
        """Request an LLM generation, which may run multiple iterations, and return the result"""

        # TODO: saqadri - history tracking is complicated in this multi-step workflow, so we will ignore it for now
        if use_history:
            raise NotImplementedError(
                "History tracking is not yet supported for orchestrator workflows"
            )

        objective = str(message)
        plan_result = await self.execute(
            objective=objective,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
        )

        return [plan_result.result]

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = False,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 16384,
        parallel_tool_calls: bool = True,
    ) -> str:
        """Request an LLM generation and return the string representation of the result"""
        result = await self.generate(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        return str(result[0])

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        use_history: bool = False,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 16384,
        parallel_tool_calls: bool = True,
    ) -> ModelT:
        """Request a structured LLM generation and return the result as a Pydantic model."""
        result_str = await self.generate_str(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        llm = self.llm_factory(
            agent=Agent(
                name="Structured Output",
                instruction="Produce a structured output given a message",
            )
        )

        structured_result = await llm.generate_structured(
            message=result_str,
            response_model=response_model,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        return structured_result

    async def execute(
        self,
        objective: str,
        max_iterations: int = 30,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 16384,
    ) -> PlanResult:
        """Execute task with result chaining between steps"""
        iterations = 0

        plan_result = PlanResult(objective=objective, step_results=[])

        while iterations < max_iterations:
            if self.plan_type == "iterative":
                # Get next plan/step
                next_step = await self._get_next_step(
                    objective=objective, plan_result=plan_result, model=model
                )
                logger.debug(f"Iteration {iterations}: Iterative plan:", data=next_step)
                plan = Plan(steps=[next_step], is_complete=next_step.is_complete)
            elif self.plan_type == "full":
                plan = await self._get_full_plan(
                    objective=objective, plan_result=plan_result
                )
                logger.debug(f"Iteration {iterations}: Full Plan:", data=plan)
            else:
                raise ValueError(f"Invalid plan type {self.plan_type}")

            plan_result.plan = plan

            if plan.is_complete:
                plan_result.is_complete = True

                # Synthesize final result into a single message
                synthesis_prompt = SYNTHESIZE_PLAN_PROMPT_TEMPLATE.format(
                    plan_result=format_plan_result(plan_result)
                )

                plan_result.result = await self.planner.generate_str(
                    message=synthesis_prompt,
                    max_iterations=1,
                    model=model,
                    stop_sequences=stop_sequences,
                    max_tokens=max_tokens,
                )

                return plan_result

            # Execute each step, collecting results
            # Note that in iterative mode this will only be a single step
            for step in plan.steps:
                step_result = await self._execute_step(
                    step=step,
                    previous_result=plan_result,
                    model=model,
                    max_iterations=max_iterations,
                    stop_sequences=stop_sequences,
                    max_tokens=max_tokens,
                )

                plan_result.add_step_result(step_result)

            logger.debug(
                f"Iteration {iterations}: Intermediate plan result:", data=plan_result
            )
            iterations += 1

        raise RuntimeError(f"Task failed to complete in {max_iterations} iterations")

    async def _execute_step(
        self,
        step: Step,
        previous_result: PlanResult,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 16384,
    ) -> StepResult:
        """Execute a step's subtasks in parallel and synthesize results"""
        step_result = StepResult(step=step, task_results=[])

        # Format previous results
        context = format_plan_result(previous_result)

        # Execute subtasks in parallel
        futures: List[Coroutine[Any, Any, str]] = []
        results = []

        async with contextlib.AsyncExitStack() as stack:
            # Set up all the tasks with their agents and LLMs
            for task in step.tasks:
                agent = self.agents.get(task.agent)
                if not agent:
                    # TODO: saqadri - should we fail the entire workflow in this case?
                    raise ValueError(f"No agent found matching {task.agent}")
                elif isinstance(agent, AugmentedLLM):
                    llm = agent
                else:
                    # Enter agent context
                    ctx_agent = await stack.enter_async_context(agent)
                    llm = await ctx_agent.attach_llm(self.llm_factory)

                task_description = TASK_PROMPT_TEMPLATE.format(
                    objective=previous_result.objective,
                    task=task.description,
                    context=context,
                )

                futures.append(
                    llm.generate_str(
                        message=task_description,
                        max_iterations=max_iterations,
                        model=model,
                        stop_sequences=stop_sequences,
                        max_tokens=max_tokens,
                    )
                )

            # Wait for all tasks to complete
            results = await self.executor.execute(*futures)

        # Store task results
        for task, result in zip(step.tasks, results):
            step_result.add_task_result(
                TaskWithResult(**task.model_dump(), result=str(result))
            )

        # Synthesize overall step result
        # TODO: saqadri - instead of running through an LLM,
        # we set the step result to the formatted results of the subtasks
        # From empirical evidence, running it through an LLM at this step can
        # lead to compounding errors since some information gets lost in the synthesis
        # synthesis_prompt = SYNTHESIZE_STEP_PROMPT_TEMPLATE.format(
        #     step_result=format_step_result(step_result)
        # )
        # synthesizer_llm = self.llm_factory(
        #     agent=Agent(
        #         name="Synthesizer",
        #         instruction="Your job is to concatenate the results of parallel tasks into a single result.",
        #     )
        # )
        # step_result.result = await synthesizer_llm.generate_str(
        #     message=synthesis_prompt,
        #     max_iterations=1,
        #     model=model,
        #     stop_sequences=stop_sequences,
        #     max_tokens=max_tokens,
        # )
        step_result.result = format_step_result(step_result)

        return step_result

    async def _get_full_plan(
        self,
        objective: str,
        plan_result: PlanResult,
        max_iterations: int = 30,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 16384,
    ) -> Plan:
        """Generate full plan considering previous results"""

        agents = "\n".join(
            [
                f"{idx}. {self._format_agent_info(agent)}"
                for idx, agent in enumerate(self.agents, 1)
            ]
        )

        prompt = FULL_PLAN_PROMPT_TEMPLATE.format(
            objective=objective,
            plan_result=format_plan_result(plan_result),
            agents=agents,
        )

        plan = await self.planner.generate_structured(
            message=prompt,
            response_model=Plan,
            max_iterations=max_iterations,
            model=model,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
        )

        return plan

    async def _get_next_step(
        self, objective: str, plan_result: PlanResult, model: str = None
    ) -> NextStep:
        """Generate just the next needed step"""

        agents = "\n".join(
            [
                f"{idx}. {self._format_agent_info(agent)}"
                for idx, agent in enumerate(self.agents, 1)
            ]
        )

        prompt = ITERATIVE_PLAN_PROMPT_TEMPLATE.format(
            objective=objective,
            plan_result=format_plan_result(plan_result),
            agents=agents,
        )

        next_step = await self.planner.generate_structured(
            message=prompt,
            response_model=NextStep,
        )
        return next_step

    def _format_server_info(self, server_name: str) -> str:
        """Format server information for display to planners"""
        server_config = self.server_registry.get_server_config(server_name)
        server_str = f"Server Name: {server_name}"
        if not server_config:
            return server_str

        description = server_config.description
        if description:
            server_str = f"{server_str}\nDescription: {description}"

        return server_str

    def _format_agent_info(self, agent_name: str) -> str:
        """Format Agent information for display to planners"""
        agent = self.agents.get(agent_name)
        if not agent:
            return ""

        servers = "\n".join(
            [
                f"- {self._format_server_info(server_name)}"
                for server_name in agent.server_names
            ]
        )

        return f"Agent Name: {agent.name}\nDescription: {agent.instruction}\nServers in Agent: {servers}"

```

### src/mcp_agent/workflows/orchestrator/orchestrator_models.py

```py
from typing import List

from pydantic import BaseModel, ConfigDict, Field

from mcp_agent.workflows.orchestrator.orchestrator_prompts import (
    PLAN_RESULT_TEMPLATE,
    STEP_RESULT_TEMPLATE,
    TASK_RESULT_TEMPLATE,
)


class Task(BaseModel):
    """An individual task that needs to be executed"""

    description: str = Field(description="Description of the task")


class ServerTask(Task):
    """An individual task that can be accomplished by one or more MCP servers"""

    servers: List[str] = Field(
        description="Names of MCP servers that the LLM has access to for this task",
        default_factory=list,
    )


class AgentTask(Task):
    """An individual task that can be accomplished by an Agent."""

    agent: str = Field(
        description="Name of Agent from given list of agents that the LLM has access to for this task",
    )


class Step(BaseModel):
    """A step containing independent tasks that can be executed in parallel"""

    description: str = Field(description="Description of the step")

    tasks: List[AgentTask] = Field(
        description="Subtasks that can be executed in parallel",
        default_factory=list,
    )


class Plan(BaseModel):
    """Plan generated by the orchestrator planner."""

    steps: List[Step] = Field(
        description="List of steps to execute sequentially",
        default_factory=list,
    )
    is_complete: bool = Field(
        description="Whether the overall plan objective is complete"
    )


class TaskWithResult(Task):
    """An individual task with its result"""

    result: str = Field(
        description="Result of executing the task", default="Task completed"
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class StepResult(BaseModel):
    """Result of executing a step"""

    step: Step = Field(description="The step that was executed", default_factory=Step)
    task_results: List[TaskWithResult] = Field(
        description="Results of executing each task", default_factory=list
    )
    result: str = Field(
        description="Result of executing the step", default="Step completed"
    )

    def add_task_result(self, task_result: TaskWithResult):
        """Add a task result to this step"""
        if not isinstance(self.task_results, list):
            self.task_results = []
        self.task_results.append(task_result)


class PlanResult(BaseModel):
    """Results of executing a plan"""

    objective: str
    """Objective of the plan"""

    plan: Plan | None = None
    """The plan that was executed"""

    step_results: List[StepResult]
    """Results of executing each step"""

    is_complete: bool = False
    """Whether the overall plan objective is complete"""

    result: str | None = None
    """Result of executing the plan"""

    def add_step_result(self, step_result: StepResult):
        """Add a step result to this plan"""
        if not isinstance(self.step_results, list):
            self.step_results = []
        self.step_results.append(step_result)


class NextStep(Step):
    """Single next step in iterative planning"""

    is_complete: bool = Field(
        description="Whether the overall plan objective is complete"
    )


def format_task_result(task_result: TaskWithResult) -> str:
    """Format a task result for display to planners"""
    return TASK_RESULT_TEMPLATE.format(
        task_description=task_result.description, task_result=task_result.result
    )


def format_step_result(step_result: StepResult) -> str:
    """Format a step result for display to planners"""
    tasks_str = "\n".join(
        f"  - {format_task_result(task)}" for task in step_result.task_results
    )
    return STEP_RESULT_TEMPLATE.format(
        step_description=step_result.step.description,
        step_result=step_result.result,
        tasks_str=tasks_str,
    )


def format_plan_result(plan_result: PlanResult) -> str:
    """Format the full plan execution state for display to planners"""
    steps_str = (
        "\n\n".join(
            f"{i + 1}:\n{format_step_result(step)}"
            for i, step in enumerate(plan_result.step_results)
        )
        if plan_result.step_results
        else "No steps executed yet"
    )

    return PLAN_RESULT_TEMPLATE.format(
        plan_objective=plan_result.objective,
        steps_str=steps_str,
        plan_status="Complete" if plan_result.is_complete else "In Progress",
        plan_result=plan_result.result if plan_result.is_complete else "In Progress",
    )

```

### src/mcp_agent/workflows/orchestrator/orchestrator_prompts.py

````py
TASK_RESULT_TEMPLATE = """Task: {task_description}
Result: {task_result}"""

STEP_RESULT_TEMPLATE = """Step: {step_description}
Step Subtasks:
{tasks_str}"""

PLAN_RESULT_TEMPLATE = """Plan Objective: {plan_objective}

Progress So Far (steps completed):
{steps_str}

Plan Current Status: {plan_status}
Plan Current Result: {plan_result}"""

FULL_PLAN_PROMPT_TEMPLATE = """You are tasked with orchestrating a plan to complete an objective.
You can analyze results from the previous steps already executed to decide if the objective is complete.
Your plan must be structured in sequential steps, with each step containing independent parallel subtasks.

Objective: {objective}

{plan_result}

If the previous results achieve the objective, return is_complete=True.
Otherwise, generate remaining steps needed.

You have access to the following MCP Servers (which are collections of tools/functions),
and Agents (which are collections of servers):

Agents:
{agents}

Generate a plan with all remaining steps needed.
Steps are sequential, but each Step can have parallel subtasks.
For each Step, specify a description of the step and independent subtasks that can run in parallel.
For each subtask specify:
    1. Clear description of the task that an LLM can execute
    2. Name of 1 Agent OR List of MCP server names to use for the task

Return your response in the following JSON structure:
    {{
        "steps": [
            {{
                "description": "Description of step 1",
                "tasks": [
                    {{
                        "description": "Description of task 1",
                        "agent": "agent_name"  # For AgentTask
                    }},
                    {{
                        "description": "Description of task 2",
                        "agent": "agent_name2"
                    }}
                ]
            }}
        ],
        "is_complete": false
    }}

You must respond with valid JSON only, with no triple backticks. No markdown formatting.
No extra text. Do not wrap in ```json code fences."""

ITERATIVE_PLAN_PROMPT_TEMPLATE = """You are tasked with determining only the next step in a plan
needed to complete an objective. You must analyze the current state and progress from previous steps
to decide what to do next.

A Step must be sequential in the plan, but can have independent parallel subtasks. Only return a single Step.

Objective: {objective}

{plan_result}

If the previous results achieve the objective, return is_complete=True.
Otherwise, generate the next Step.

You have access to the following MCP Servers (which are collections of tools/functions),
and Agents (which are collections of servers):

Agents:
{agents}

Generate the next step, by specifying a description of the step and independent subtasks that can run in parallel:
For each subtask specify:
    1. Clear description of the task that an LLM can execute
    2. Name of 1 Agent OR List of MCP server names to use for the task

Return your response in the following JSON structure:
    {{

        "description": "Description of step 1",
        "tasks": [
            {{
                "description": "Description of task 1",
                "agent": "agent_name"  # For AgentTask
            }}
        ],
        "is_complete": false
    }}

You must respond with valid JSON only, with no triple backticks. No markdown formatting.
No extra text. Do not wrap in ```json code fences."""

TASK_PROMPT_TEMPLATE = """You are part of a larger workflow to achieve the objective: {objective}.
Your job is to accomplish only the following task: {task}.

Results so far that may provide helpful context:
{context}"""

SYNTHESIZE_STEP_PROMPT_TEMPLATE = """Synthesize the results of these parallel tasks into a cohesive result:
{step_result}"""

SYNTHESIZE_PLAN_PROMPT_TEMPLATE = """Synthesize the results of executing all steps in the plan into a cohesive result:
{plan_result}"""

````

### src/mcp_agent/workflows/parallel/**init**.py

```py

```

### src/mcp_agent/workflows/parallel/fan_in.py

```py
import contextlib
from typing import Callable, Dict, List, Type

from mcp_agent.agents.agent import Agent
from mcp_agent.executor.executor import Executor, AsyncioExecutor
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
)

FanInInput = (
    # Dict of agent/source name to list of messages generated by that agent
    Dict[str, List[MessageT] | List[MessageParamT]]
    # Dict of agent/source name to string generated by that agent
    | Dict[str, str]
    # List of lists of messages generated by each agent
    | List[List[MessageT] | List[MessageParamT]]
    # List of strings generated by each agent
    | List[str]
)


class FanIn:
    """
    Aggregate results from multiple parallel tasks into a single result.

    This is a building block of the Parallel workflow, which can be used to fan out
    work to multiple agents or other parallel tasks, and then aggregate the results.

    For example, you can use FanIn to combine the results of multiple agents into a single response,
    such as a Summarization Fan-In agent that combines the outputs of multiple language models.
    """

    def __init__(
        self,
        aggregator_agent: Agent | AugmentedLLM[MessageParamT, MessageT],
        llm_factory: Callable[[Agent], AugmentedLLM[MessageParamT, MessageT]] = None,
        executor: Executor | None = None,
    ):
        """
        Initialize the FanIn with an Agent responsible for processing multiple responses into a single aggregated one.
        """

        self.executor = executor or AsyncioExecutor()
        self.llm_factory = llm_factory
        self.aggregator_agent = aggregator_agent

        if not isinstance(self.aggregator_agent, AugmentedLLM):
            if not self.llm_factory:
                raise ValueError("llm_factory is required when using an Agent")

    async def generate(
        self,
        messages: FanInInput,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> List[MessageT]:
        """
        Request fan-in agent generation from a list of messages from multiple sources/agents.
        Internally aggregates the messages and then calls the aggregator agent to generate a response.
        """
        message: (
            str | MessageParamT | List[MessageParamT]
        ) = await self.aggregate_messages(messages)

        async with contextlib.AsyncExitStack() as stack:
            if isinstance(self.aggregator_agent, AugmentedLLM):
                llm = self.aggregator_agent
            else:
                # Enter agent context
                ctx_agent = await stack.enter_async_context(self.aggregator_agent)
                llm = await ctx_agent.attach_llm(self.llm_factory)

            return await llm.generate(
                message=message,
                use_history=use_history,
                max_iterations=max_iterations,
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )

    async def generate_str(
        self,
        messages: FanInInput,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> str:
        """
        Request fan-in agent generation from a list of messages from multiple sources/agents.
        Internally aggregates the messages and then calls the aggregator agent to generate a
        response, which is returned as a string.
        """

        message: (
            str | MessageParamT | List[MessageParamT]
        ) = await self.aggregate_messages(messages)

        async with contextlib.AsyncExitStack() as stack:
            if isinstance(self.aggregator_agent, AugmentedLLM):
                llm = self.aggregator_agent
            else:
                # Enter agent context
                ctx_agent = await stack.enter_async_context(self.aggregator_agent)
                llm = await ctx_agent.attach_llm(self.llm_factory)

            return await llm.generate_str(
                message=message,
                use_history=use_history,
                max_iterations=max_iterations,
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )

    async def generate_structured(
        self,
        messages: FanInInput,
        response_model: Type[ModelT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> ModelT:
        """
        Request a structured fan-in agent generation from a list of messages
        from multiple sources/agents. Internally aggregates the messages and then calls
        the aggregator agent to generate a response, which is returned as a Pydantic model.
        """

        message: (
            str | MessageParamT | List[MessageParamT]
        ) = await self.aggregate_messages(messages)

        async with contextlib.AsyncExitStack() as stack:
            if isinstance(self.aggregator_agent, AugmentedLLM):
                llm = self.aggregator_agent
            else:
                # Enter agent context
                ctx_agent = await stack.enter_async_context(self.aggregator_agent)
                llm = await ctx_agent.attach_llm(self.llm_factory)

            return await llm.generate_structured(
                message=message,
                response_model=response_model,
                use_history=use_history,
                max_iterations=max_iterations,
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )

    async def aggregate_messages(
        self, messages: FanInInput
    ) -> str | MessageParamT | List[MessageParamT]:
        """
        Aggregate messages from multiple sources/agents into a single message to
        use with the aggregator agent generation.

        The input can be a dictionary of agent/source name to list of messages
        generated by that agent, or just the unattributed lists of messages to aggregate.

        Args:
            messages: Can be one of:
                - Dict[str, List[MessageT] | List[MessageParamT]]: Dict of agent names to messages
                - Dict[str, str]: Dict of agent names to message strings
                - List[List[MessageT] | List[MessageParamT]]: List of message lists from agents
                - List[str]: List of message strings from agents

        Returns:
            Aggregated message as string, MessageParamT or List[MessageParamT]

        Raises:
            ValueError: If input is empty or contains empty/invalid elements
        """
        # Handle dictionary inputs
        if isinstance(messages, dict):
            # Check for empty dict
            if not messages:
                raise ValueError("Input dictionary cannot be empty")

            first_value = next(iter(messages.values()))

            # Dict[str, List[MessageT] | List[MessageParamT]]
            if isinstance(first_value, list):
                if any(not isinstance(v, list) for v in messages.values()):
                    raise ValueError("All dictionary values must be lists of messages")
                # Process list of messages for each agent
                return await self.aggregate_agent_messages(messages)

            # Dict[str, str]
            elif isinstance(first_value, str):
                if any(not isinstance(v, str) for v in messages.values()):
                    raise ValueError("All dictionary values must be strings")
                # Process string outputs from each agent
                return await self.aggregate_agent_message_strings(messages)

            else:
                raise ValueError(
                    "Dictionary values must be either lists of messages or strings"
                )

        # Handle list inputs
        elif isinstance(messages, list):
            # Check for empty list
            if not messages:
                raise ValueError("Input list cannot be empty")

            first_item = messages[0]

            # List[List[MessageT] | List[MessageParamT]]
            if isinstance(first_item, list):
                if any(not isinstance(item, list) for item in messages):
                    raise ValueError("All list items must be lists of messages")
                # Process list of message lists
                return await self.aggregate_message_lists(messages)

            # List[str]
            elif isinstance(first_item, str):
                if any(not isinstance(item, str) for item in messages):
                    raise ValueError("All list items must be strings")
                # Process list of strings
                return await self.aggregate_message_strings(messages)

            else:
                raise ValueError(
                    "List items must be either lists of messages or strings"
                )

        else:
            raise ValueError(
                "Input must be either a dictionary of agent messages or a list of messages"
            )

    # Helper methods for processing different types of inputs
    async def aggregate_agent_messages(
        self, messages: Dict[str, List[MessageT] | List[MessageParamT]]
    ) -> str | MessageParamT | List[MessageParamT]:
        """
        Aggregate message lists with agent names.

        Args:
            messages: Dictionary mapping agent names to their message lists

        Returns:
            str | List[MessageParamT]: Messages formatted with agent attribution

        """

        # In the default implementation, we'll just convert the messages to a
        # single string with agent attribution
        aggregated_messages = []

        if not messages:
            return ""

        # Format each agent's messages with attribution
        for agent_name, agent_messages in messages.items():
            agent_message_strings = []
            for msg in agent_messages or []:
                if isinstance(msg, str):
                    agent_message_strings.append(f"Agent {agent_name}: {msg}")
                else:
                    # Assume it's a Message/MessageParamT and add attribution
                    agent_message_strings.append(f"Agent {agent_name}: {str(msg)}")

            aggregated_messages.append("\n".join(agent_message_strings))

        # Combine all messages with clear separation
        final_message = "\n\n".join(aggregated_messages)
        final_message = f"Aggregated responses from multiple Agents:\n\n{final_message}"
        return final_message

    async def aggregate_agent_message_strings(self, messages: Dict[str, str]) -> str:
        """
        Aggregate string outputs with agent names.

        Args:
            messages: Dictionary mapping agent names to their string outputs

        Returns:
            str: Combined string with agent attributions
        """
        if not messages:
            return ""

        # Format each agent's message with agent attribution
        aggregated_messages = [
            f"Agent {agent_name}: {message}" for agent_name, message in messages.items()
        ]

        # Combine all messages with clear separation
        final_message = "\n\n".join(aggregated_messages)
        final_message = f"Aggregated responses from multiple Agents:\n\n{final_message}"
        return final_message

    async def aggregate_message_lists(
        self, messages: List[List[MessageT] | List[MessageParamT]]
    ) -> str | MessageParamT | List[MessageParamT]:
        """
        Aggregate message lists without agent names.

        Args:
            messages: List of message lists from different agents

        Returns:
            List[MessageParamT]: List of formatted messages
        """
        aggregated_messages = []

        if not messages:
            return ""

        # Format each source's messages
        for i, source_messages in enumerate(messages, 1):
            source_message_strings = []
            for msg in source_messages or []:
                if isinstance(msg, str):
                    source_message_strings.append(f"Source {i}: {msg}")
                else:
                    # Assume it's a MessageParamT or MessageT and add source attribution
                    source_message_strings.append(f"Source {i}: {str(msg)}")

            aggregated_messages.append("\n".join(source_messages))

        # Combine all messages with clear separation
        final_message = "\n\n".join(aggregated_messages)
        final_message = (
            f"Aggregated responses from multiple sources:\n\n{final_message}"
        )
        return final_message

    async def aggregate_message_strings(self, messages: List[str]) -> str:
        """
        Aggregate string outputs without agent names.

        Args:
            messages: List of string outputs from different agents

        Returns:
            str: Combined string with source attributions
        """
        if not messages:
            return ""

        # Format each source's message with attribution
        aggregated_messages = [
            f"Source {i}: {message}" for i, message in enumerate(messages, 1)
        ]

        # Combine all messages with clear separation
        final_message = "\n\n".join(aggregated_messages)
        final_message = (
            f"Aggregated responses from multiple sources:\n\n{final_message}"
        )
        return final_message

```

### src/mcp_agent/workflows/parallel/fan_out.py

```py
import contextlib
import functools
from typing import Any, Callable, Coroutine, Dict, List, Type

from mcp_agent.agents.agent import Agent
from mcp_agent.executor.executor import Executor, AsyncioExecutor
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
)
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class FanOut:
    """
    Distribute work to multiple parallel tasks.

    This is a building block of the Parallel workflow, which can be used to fan out
    work to multiple agents or other parallel tasks, and then aggregate the results.
    """

    def __init__(
        self,
        agents: List[Agent | AugmentedLLM[MessageParamT, MessageT]] | None = None,
        functions: List[Callable[[MessageParamT], List[MessageT]]] | None = None,
        llm_factory: Callable[[Agent], AugmentedLLM[MessageParamT, MessageT]] = None,
        executor: Executor | None = None,
    ):
        """
        Initialize the FanOut with a list of agents, functions, or LLMs.
        If agents are provided, they will be wrapped in an AugmentedLLM using llm_factory if not already done so.
        If functions are provided, they will be invoked in parallel directly.
        """

        self.executor = executor or AsyncioExecutor()
        self.llm_factory = llm_factory
        self.agents = agents or []
        self.functions: List[Callable[[MessageParamT], MessageT]] = functions or []

        if not self.agents and not self.functions:
            raise ValueError(
                "At least one agent or function must be provided for fan-out to work"
            )

        if not self.llm_factory:
            for agent in self.agents:
                if not isinstance(agent, AugmentedLLM):
                    raise ValueError("llm_factory is required when using an Agent")

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> Dict[str, List[MessageT]]:
        """
        Request fan-out agent/function generations, and return the results as a dictionary.
        The keys are the names of the agents or functions that generated the results.
        """
        tasks: List[
            Callable[..., List[MessageT]] | Coroutine[Any, Any, List[MessageT]]
        ] = []
        task_names: List[str] = []
        task_results = []

        async with contextlib.AsyncExitStack() as stack:
            for agent in self.agents:
                if isinstance(agent, AugmentedLLM):
                    llm = agent
                else:
                    # Enter agent context
                    ctx_agent = await stack.enter_async_context(agent)
                    llm = await ctx_agent.attach_llm(self.llm_factory)

                tasks.append(
                    llm.generate(
                        message=message,
                        use_history=use_history,
                        max_iterations=max_iterations,
                        model=model,
                        stop_sequences=stop_sequences,
                        max_tokens=max_tokens,
                        parallel_tool_calls=parallel_tool_calls,
                    )
                )
                task_names.append(agent.name)

            # Create bound methods for regular functions
            for function in self.functions:
                tasks.append(functools.partial(function, message))
                task_names.append(function.__name__ or id(function))

            # Wait for all tasks to complete
            logger.debug("Running fan-out tasks:", data=task_names)
            task_results = await self.executor.execute(*tasks)

        logger.debug(
            "Fan-out tasks completed:", data=dict(zip(task_names, task_results))
        )
        return dict(zip(task_names, task_results))

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> Dict[str, str]:
        """
        Request fan-out agent/function generations and return the string results as a dictionary.
        The keys are the names of the agents or functions that generated the results.
        """

        def fn_result_to_string(fn, message):
            return str(fn(message))

        tasks: List[Callable[..., str] | Coroutine[Any, Any, str]] = []
        task_names: List[str] = []
        task_results = []

        async with contextlib.AsyncExitStack() as stack:
            for agent in self.agents:
                if isinstance(agent, AugmentedLLM):
                    llm = agent
                else:
                    # Enter agent context
                    ctx_agent = await stack.enter_async_context(agent)
                    llm = await ctx_agent.attach_llm(self.llm_factory)

                tasks.append(
                    llm.generate_str(
                        message=message,
                        use_history=use_history,
                        max_iterations=max_iterations,
                        model=model,
                        stop_sequences=stop_sequences,
                        max_tokens=max_tokens,
                        parallel_tool_calls=parallel_tool_calls,
                    )
                )
                task_names.append(agent.name)

            # Create bound methods for regular functions
            for function in self.functions:
                tasks.append(functools.partial(fn_result_to_string, function, message))
                task_names.append(function.__name__ or id(function))

            task_results = await self.executor.execute(*tasks)

        return dict(zip(task_names, task_results))

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> Dict[str, ModelT]:
        """
        Request a structured fan-out agent/function generation and return the result as a Pydantic model.
        The keys are the names of the agents or functions that generated the results.
        """
        tasks = []
        task_names = []
        task_results = []

        async with contextlib.AsyncExitStack() as stack:
            for agent in self.agents:
                if isinstance(agent, AugmentedLLM):
                    llm = agent
                else:
                    # Enter agent context
                    ctx_agent = await stack.enter_async_context(agent)
                    llm = await ctx_agent.attach_llm(self.llm_factory)

                tasks.append(
                    llm.generate_structured(
                        message=message,
                        response_model=response_model,
                        use_history=use_history,
                        max_iterations=max_iterations,
                        model=model,
                        stop_sequences=stop_sequences,
                        max_tokens=max_tokens,
                        parallel_tool_calls=parallel_tool_calls,
                    )
                )
                task_names.append(agent.name)

            # Create bound methods for regular functions
            for function in self.functions:
                tasks.append(functools.partial(function, message))
                task_names.append(function.__name__ or id(function))

            task_results = await self.executor.execute(*tasks)

        return dict(zip(task_names, task_results))

```

### src/mcp_agent/workflows/parallel/parallel_llm.py

```py
from typing import Any, Callable, List, Type

from mcp_agent.agents.agent import Agent
from mcp_agent.executor.executor import Executor
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
)
from mcp_agent.workflows.parallel.fan_in import FanInInput, FanIn
from mcp_agent.workflows.parallel.fan_out import FanOut


class ParallelLLM(AugmentedLLM[MessageParamT, MessageT]):
    """
    LLMs can sometimes work simultaneously on a task (fan-out)
    and have their outputs aggregated programmatically (fan-in).
    This workflow performs both the fan-out and fan-in operations using  LLMs.
    From the user's perspective, an input is specified and the output is returned.

    When to use this workflow:
        Parallelization is effective when the divided subtasks can be parallelized
        for speed (sectioning), or when multiple perspectives or attempts are needed for
        higher confidence results (voting).

    Examples:
        Sectioning:
            - Implementing guardrails where one model instance processes user queries
            while another screens them for inappropriate content or requests.

            - Automating evals for evaluating LLM performance, where each LLM call
            evaluates a different aspect of the model’s performance on a given prompt.

        Voting:
            - Reviewing a piece of code for vulnerabilities, where several different
            agents review and flag the code if they find a problem.

            - Evaluating whether a given piece of content is inappropriate,
            with multiple agents evaluating different aspects or requiring different
            vote thresholds to balance false positives and negatives.
    """

    def __init__(
        self,
        fan_in_agent: Agent | AugmentedLLM | Callable[[FanInInput], Any],
        fan_out_agents: List[Agent | AugmentedLLM] | None = None,
        fan_out_functions: List[Callable] | None = None,
        llm_factory: Callable[[Agent], AugmentedLLM] = None,
        executor: Executor | None = None,
    ):
        """
        Initialize the LLM with a list of server names and an instruction.
        If a name is provided, it will be used to identify the LLM.
        If an agent is provided, all other properties are optional
        """
        super().__init__(executor=executor)

        self.llm_factory = llm_factory
        self.fan_in_agent = fan_in_agent
        self.fan_out_agents = fan_out_agents
        self.fan_out_functions = fan_out_functions
        self.history = (
            None  # History tracking is complex in this workflow, so it is not supported
        )

        self.fan_in_fn: Callable[[FanInInput], Any] = None
        self.fan_in: FanIn = None
        if isinstance(fan_in_agent, Callable):
            self.fan_in_fn = fan_in_agent
        else:
            self.fan_in = FanIn(
                aggregator_agent=fan_in_agent,
                llm_factory=llm_factory,
                executor=executor,
            )

        self.fan_out = FanOut(
            agents=fan_out_agents,
            functions=fan_out_functions,
            llm_factory=llm_factory,
            executor=executor,
        )

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> List[MessageT] | Any:
        # First, we fan-out
        responses = await self.fan_out.generate(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Then, we fan-in
        if self.fan_in_fn:
            result = await self.fan_in_fn(responses)
        else:
            result = await self.fan_in.generate(
                messages=responses,
                use_history=use_history,
                max_iterations=max_iterations,
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )

        return result

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> str:
        """Request an LLM generation and return the string representation of the result"""

        # First, we fan-out
        responses = await self.fan_out.generate(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Then, we fan-in
        if self.fan_in_fn:
            result = str(await self.fan_in_fn(responses))
        else:
            result = await self.fan_in.generate_str(
                messages=responses,
                use_history=use_history,
                max_iterations=max_iterations,
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )
        return result

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> ModelT:
        """Request a structured LLM generation and return the result as a Pydantic model."""
        # First, we fan-out
        responses = await self.fan_out.generate(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Then, we fan-in
        if self.fan_in_fn:
            result = await self.fan_in_fn(responses)
        else:
            result = await self.fan_in.generate_structured(
                messages=responses,
                response_model=response_model,
                use_history=use_history,
                max_iterations=max_iterations,
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )
        return result

```

### src/mcp_agent/workflows/router/**init**.py

```py

```

### src/mcp_agent/workflows/router/router_base.py

```py
from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, List, TypeVar

from pydantic import BaseModel, Field
from mcp.server.fastmcp.tools import Tool as FastTool

from mcp_agent.agents.agent import Agent
from mcp_agent.context import get_current_context
from mcp_agent.mcp_server_registry import ServerRegistry
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)

ResultT = TypeVar("ResultT", bound=str | Agent | Callable)


class RouterResult(BaseModel, Generic[ResultT]):
    """A class that represents the result of a Router.route request"""

    result: ResultT
    """The router returns an MCP server name, an Agent, or a function to route the input to."""

    p_score: float | None = None
    """
    The probability score (i.e. 0->1) of the routing decision.
    This is optional and may only be provided if the router is probabilistic (e.g. a probabilistic binary classifier).
    """


class RouterCategory(BaseModel):
    """
    A class that represents a category of routing.
    Used to collect information the router needs to decide.
    """

    name: str
    """The name of the category"""

    description: str | None = None
    """A description of the category"""

    category: str | Agent | Callable
    """The class to route to"""


class ServerRouterCategory(RouterCategory):
    """A class that represents a category of routing to an MCP server"""

    tools: List[FastTool] = Field(default_factory=list)


class AgentRouterCategory(RouterCategory):
    """A class that represents a category of routing to an agent"""

    servers: List[ServerRouterCategory] = Field(default_factory=list)


class Router(ABC):
    """
    Routing classifies an input and directs it to one or more specialized followup tasks.
    This class helps to route an input to a specific MCP server,
    an Agent (an aggregation of MCP servers), or a function (any Callable).

    When to use this workflow:
        - This workflow allows for separation of concerns, and building more specialized prompts.

        - Routing works well for complex tasks where there are distinct categories that
        are better handled separately, and where classification can be handled accurately,
        either by an LLM or a more traditional classification model/algorithm.

    Examples where routing is useful:
        - Directing different types of customer service queries
        (general questions, refund requests, technical support)
        into different downstream processes, prompts, and tools.

        - Routing easy/common questions to smaller models like Claude 3.5 Haiku
        and hard/unusual questions to more capable models like Claude 3.5 Sonnet
        to optimize cost and speed.

    Args:
        routing_instruction: A string that tells the router how to route the input.
        mcp_servers_names: A list of server names to route the input to.
        agents: A list of agents to route the input to.
        functions: A list of functions to route the input to.
        server_registry: The server registry to use for resolving the server names.
    """

    def __init__(
        self,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        server_registry: ServerRegistry | None = None,
    ):
        self.routing_instruction = routing_instruction
        self.server_names = mcp_servers_names or []
        self.agents = agents or []
        self.functions = functions or []
        self.server_registry = server_registry or get_current_context().server_registry

        # A dict of categories to route to, keyed by category name.
        # These are populated in the initialize method.
        self.server_categories: Dict[str, ServerRouterCategory] = {}
        self.agent_categories: Dict[str, AgentRouterCategory] = {}
        self.function_categories: Dict[str, RouterCategory] = {}
        self.categories: Dict[str, RouterCategory] = {}
        self.initialized: bool = False

        if not self.server_names and not self.agents and not self.functions:
            raise ValueError(
                "At least one of mcp_servers_names, agents, or functions must be provided."
            )

        if self.server_names and not self.server_registry:
            raise ValueError(
                "server_registry must be provided if mcp_servers_names are provided."
            )

    @abstractmethod
    async def route(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult[str | Agent | Callable]]:
        """
        Route the input request to one or more MCP servers, agents, or functions.
        If no routing decision can be made, returns an empty list.

        Args:
            request: The input to route.
            top_k: The maximum number of top routing results to return. May return fewer.
        """

    @abstractmethod
    async def route_to_server(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult[str]]:
        """Route the input to one or more MCP servers."""

    @abstractmethod
    async def route_to_agent(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult[Agent]]:
        """Route the input to one or more agents."""

    @abstractmethod
    async def route_to_function(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult[Callable]]:
        """
        Route the input to one or more functions.

        Args:
            input: The input to route.
        """

    async def initialize(self):
        """Initialize the router categories."""

        if self.initialized:
            return

        server_categories = [
            self.get_server_category(server_name) for server_name in self.server_names
        ]
        self.server_categories = {
            category.name: category for category in server_categories
        }

        agent_categories = [self.get_agent_category(agent) for agent in self.agents]
        self.agent_categories = {
            category.name: category for category in agent_categories
        }

        function_categories = [
            self.get_function_category(function) for function in self.functions
        ]
        self.function_categories = {
            category.name: category for category in function_categories
        }

        all_categories = server_categories + agent_categories + function_categories

        self.categories = {category.name: category for category in all_categories}
        self.initialized = True

    def get_server_category(self, server_name: str) -> ServerRouterCategory:
        server_config = self.server_registry.get_server_config(server_name)

        # TODO: saqadri - Currently we only populate the server name and description.
        # To make even more high fidelity routing decisions, we can populate the
        # tools, resources and prompts that the server has access to.
        return ServerRouterCategory(
            category=server_name,
            name=server_config.name if server_config else server_name,
            description=server_config.description,
        )

    def get_agent_category(self, agent: Agent) -> AgentRouterCategory:
        agent_description = (
            agent.instruction({}) if callable(agent.instruction) else agent.instruction
        )

        return AgentRouterCategory(
            category=agent,
            name=agent.name,
            description=agent_description,
            servers=[
                self.get_server_category(server_name)
                for server_name in agent.server_names
            ],
        )

    def get_function_category(self, function: Callable) -> RouterCategory:
        tool = FastTool.from_function(function)

        return RouterCategory(
            category=function,
            name=tool.name,
            description=tool.description,
        )

    def format_category(
        self, category: RouterCategory, index: int | None = None
    ) -> str:
        """Format a category into a readable string."""

        index_str = f"{index}. " if index is not None else " "
        category_str = ""

        if isinstance(category, ServerRouterCategory):
            category_str = self._format_server_category(category)
        elif isinstance(category, AgentRouterCategory):
            category_str = self._format_agent_category(category)
        else:
            category_str = self._format_function_category(category)

        return f"{index_str}{category_str}"

    def _format_tools(self, tools: List[FastTool]) -> str:
        """Format a list of tools into a readable string."""
        if not tools:
            return "No tool information provided."

        tool_descriptions = []
        for tool in tools:
            desc = f"- {tool.name}: {tool.description}"
            tool_descriptions.append(desc)

        return "\n".join(tool_descriptions)

    def _format_server_category(self, category: ServerRouterCategory) -> str:
        """Format a server category into a readable string."""
        description = category.description or "No description provided"
        tools = self._format_tools(category.tools)
        return f"Server Category: {category.name}\nDescription: {description}\nTools in server:\n{tools}"

    def _format_agent_category(self, category: AgentRouterCategory) -> str:
        """Format an agent category into a readable string."""
        description = category.description or "No description provided"
        servers = "\n".join(
            [f"- {server.name} ({server.description})" for server in category.servers]
        )

        return f"Agent Category: {category.name}\nDescription: {description}\nServers in agent:\n{servers}"

    def _format_function_category(self, category: RouterCategory) -> str:
        """Format a function category into a readable string."""
        description = category.description or "No description provided"
        return f"Function Category: {category.name}\nDescription: {description}"

```

### src/mcp_agent/workflows/router/router_embedding.py

```py
from typing import Callable, List

from numpy import mean

from mcp_agent.agents.agent import Agent
from mcp_agent.mcp_server_registry import ServerRegistry
from mcp_agent.workflows.embedding.embedding_base import (
    EmbeddingModel,
    FloatArray,
    compute_similarity_scores,
    compute_confidence,
)
from mcp_agent.workflows.router.router_base import (
    Router,
    RouterCategory,
    RouterResult,
)


class EmbeddingRouterCategory(RouterCategory):
    """A category for embedding-based routing"""

    embedding: FloatArray | None = None
    """Pre-computed embedding for this category"""


class EmbeddingRouter(Router):
    """
    A router that uses embedding similarity to route requests to appropriate categories.
    This class helps to route an input to a specific MCP server, an Agent (an aggregation of MCP servers),
    or a function (any Callable).

    Features:
    - Semantic similarity based routing using embeddings
    - Flexible embedding model support
    - Support for formatting and combining category metadata

    Example usage:
        # Initialize router with embedding model
        router = EmbeddingRouter(
            embedding_model=OpenAIEmbeddingModel(model="text-embedding-3-small"),
            mcp_servers_names=["customer_service", "tech_support"],
        )

        # Route a request
        results = await router.route("My laptop keeps crashing")
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        server_registry: ServerRegistry | None = None,
    ):
        super().__init__(
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            server_registry=server_registry,
        )

        self.embedding_model = embedding_model

    @classmethod
    async def create(
        cls,
        embedding_model: EmbeddingModel,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        server_registry: ServerRegistry | None = None,
    ) -> "EmbeddingRouter":
        """
        Factory method to create and initialize a router.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            embedding_model=embedding_model,
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            server_registry=server_registry,
        )
        await instance.initialize()
        return instance

    async def initialize(self):
        """Initialize by computing embeddings for all categories"""

        async def create_category_with_embedding(
            category: RouterCategory,
        ) -> EmbeddingRouterCategory:
            # Get formatted text representation of category
            category_text = self.format_category(category)
            embedding = self._compute_embedding([category_text])
            category_with_embedding = EmbeddingRouterCategory(
                **category, embedding=embedding
            )

            return category_with_embedding

        if self.initialized:
            return

        # Create categories for servers, agents, and functions
        await super().initialize()
        self.initialized = False  # We are not initialized yet

        for name, category in self.server_categories.items():
            category_with_embedding = await create_category_with_embedding(category)
            self.server_categories[name] = category_with_embedding
            self.categories[name] = category_with_embedding

        for name, category in self.agent_categories.items():
            category_with_embedding = await create_category_with_embedding(category)
            self.agent_categories[name] = category_with_embedding
            self.categories[name] = category_with_embedding

        for name, category in self.function_categories.items():
            category_with_embedding = await create_category_with_embedding(category)
            self.function_categories[name] = category_with_embedding
            self.categories[name] = category_with_embedding

        self.initialized = True

    async def route(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult[str | Agent | Callable]]:
        """Route the request based on embedding similarity"""
        if not self.initialized:
            await self.initialize()

        return await self._route_with_embedding(request, top_k)

    async def route_to_server(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult[str]]:
        """Route specifically to server categories"""
        if not self.initialized:
            await self.initialize()

        results = await self._route_with_embedding(
            request,
            top_k,
            include_servers=True,
            include_agents=False,
            include_functions=False,
        )
        return [r.result for r in results[:top_k]]

    async def route_to_agent(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult[Agent]]:
        """Route specifically to agent categories"""
        if not self.initialized:
            await self.initialize()

        results = await self._route_with_embedding(
            request,
            top_k,
            include_servers=False,
            include_agents=True,
            include_functions=False,
        )
        return [r.result for r in results[:top_k]]

    async def route_to_function(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult[Callable]]:
        """Route specifically to function categories"""
        if not self.initialized:
            await self.initialize()

        results = await self._route_with_embedding(
            request,
            top_k,
            include_servers=False,
            include_agents=False,
            include_functions=True,
        )
        return [r.result for r in results[:top_k]]

    async def _route_with_embedding(
        self,
        request: str,
        top_k: int = 1,
        include_servers: bool = True,
        include_agents: bool = True,
        include_functions: bool = True,
    ) -> List[RouterResult]:
        def create_result(category: RouterCategory, request_embedding):
            if category.embedding is None:
                return None

            similarity = compute_similarity_scores(
                request_embedding, category.embedding
            )

            return RouterResult(
                p_score=compute_confidence(similarity), result=category.category
            )

        request_embedding = self._compute_embedding([request])

        results: List[RouterResult] = []
        if include_servers:
            for _, category in self.server_categories.items():
                result = create_result(category, request_embedding)
                if result:
                    results.append(result)

        if include_agents:
            for _, category in self.agent_categories.items():
                result = create_result(category, request_embedding)
                if result:
                    results.append(result)

        if include_functions:
            for _, category in self.function_categories.items():
                result = create_result(category, request_embedding)
                if result:
                    results.append(result)

        results.sort(key=lambda x: x.p_score, reverse=True)
        return results[:top_k]

    async def _compute_embedding(self, data: List[str]):
        # Get embedding for the provided text
        embeddings = await self.embedding_model.embed(data)

        # Use mean pooling to combine embeddings
        embedding = mean(embeddings, axis=0)

        return embedding

```

### src/mcp_agent/workflows/router/router_embedding_cohere.py

```py
from typing import Callable, List

from mcp_agent.agents.agent import Agent
from mcp_agent.mcp_server_registry import ServerRegistry
from mcp_agent.workflows.embedding.embedding_cohere import CohereEmbeddingModel
from mcp_agent.workflows.router.router_embedding import EmbeddingRouter


class CohereEmbeddingRouter(EmbeddingRouter):
    """
    A router that uses Cohere embedding similarity to route requests to appropriate categories.
    This class helps to route an input to a specific MCP server, an Agent (an aggregation of MCP servers),
    or a function (any Callable).
    """

    def __init__(
        self,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        server_registry: ServerRegistry | None = None,
        embedding_model: CohereEmbeddingModel | None = None,
    ):
        embedding_model = embedding_model or CohereEmbeddingModel()

        super().__init__(
            embedding_model=embedding_model,
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            server_registry=server_registry,
        )

    @classmethod
    async def create(
        cls,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        server_registry: ServerRegistry | None = None,
        embedding_model: CohereEmbeddingModel | None = None,
    ) -> "CohereEmbeddingRouter":
        """
        Factory method to create and initialize a router.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            server_registry=server_registry,
            embedding_model=embedding_model,
        )
        await instance.initialize()
        return instance

```

### src/mcp_agent/workflows/router/router_embedding_openai.py

```py
from typing import Callable, List

from mcp_agent.agents.agent import Agent
from mcp_agent.mcp_server_registry import ServerRegistry
from mcp_agent.workflows.embedding.embedding_openai import OpenAIEmbeddingModel
from mcp_agent.workflows.router.router_embedding import EmbeddingRouter


class OpenAIEmbeddingRouter(EmbeddingRouter):
    """
    A router that uses OpenAI embedding similarity to route requests to appropriate categories.
    This class helps to route an input to a specific MCP server, an Agent (an aggregation of MCP servers),
    or a function (any Callable).
    """

    def __init__(
        self,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        server_registry: ServerRegistry | None = None,
        embedding_model: OpenAIEmbeddingModel | None = None,
    ):
        embedding_model = embedding_model or OpenAIEmbeddingModel()

        super().__init__(
            embedding_model=embedding_model,
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            server_registry=server_registry,
        )

    @classmethod
    async def create(
        cls,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        server_registry: ServerRegistry | None = None,
        embedding_model: OpenAIEmbeddingModel | None = None,
    ) -> "OpenAIEmbeddingRouter":
        """
        Factory method to create and initialize a router.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            server_registry=server_registry,
            embedding_model=embedding_model,
        )
        await instance.initialize()
        return instance

```

### src/mcp_agent/workflows/router/router_llm.py

```py
from typing import Callable, List, Literal

from pydantic import BaseModel

from mcp_agent.agents.agent import Agent
from mcp_agent.mcp_server_registry import ServerRegistry
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM
from mcp_agent.workflows.router.router_base import ResultT, Router, RouterResult
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


DEFAULT_ROUTING_INSTRUCTION = """
You are a highly accurate request router that directs incoming requests to the most appropriate category.
A category is a specialized destination, such as a Function, an MCP Server (a collection of tools/functions), or an Agent (a collection of servers).
Below are the available routing categories, each with their capabilities and descriptions:

{context}

Your task is to analyze the following request and determine the most appropriate categories from the options above. Consider:
- The specific capabilities and tools each destination offers
- How well the request matches the category's description
- Whether the request might benefit from multiple categories (up to {top_k})

Request: {request}

Respond in JSON format:
{{
    "categories": [
        {{
            "category": <category name>,
            "confidence": <high, medium or low>,
            "reasoning": <brief explanation>
        }}
    ]
}}

Only include categories that are truly relevant. You may return fewer than {top_k} if appropriate.
If none of the categories are relevant, return an empty list.
"""


class LLMRouterResult(RouterResult[ResultT]):
    """A class that represents the result of an LLMRouter.route request"""

    confidence: Literal["high", "medium", "low"]
    """The confidence level of the routing decision."""

    reasoning: str | None = None
    """
    A brief explanation of the routing decision.
    This is optional and may only be provided if the router is an LLM
    """


class StructuredResponseCategory(BaseModel):
    """A class that represents a single category returned by an LLM router"""

    category: str
    """The name of the category (i.e. MCP server, Agent or function) to route the input to."""

    confidence: Literal["high", "medium", "low"]
    """The confidence level of the routing decision."""

    reasoning: str | None = None
    """A brief explanation of the routing decision."""


class StructuredResponse(BaseModel):
    """A class that represents the structured response of an LLM router"""

    categories: List[StructuredResponseCategory]
    """A list of categories to route the input to."""


class LLMRouter(Router):
    """
    A router that uses an LLM to route an input to a specific category.
    """

    def __init__(
        self,
        llm: AugmentedLLM,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        server_registry: ServerRegistry | None = None,
    ):
        super().__init__(
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            routing_instruction=routing_instruction,
            server_registry=server_registry,
        )

        self.llm = llm

    @classmethod
    async def create(
        cls,
        llm: AugmentedLLM,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        server_registry: ServerRegistry | None = None,
    ) -> "LLMRouter":
        """
        Factory method to create and initialize a router.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            llm=llm,
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            routing_instruction=routing_instruction,
            server_registry=server_registry,
        )
        await instance.initialize()
        return instance

    async def route(
        self, request: str, top_k: int = 1
    ) -> List[LLMRouterResult[str | Agent | Callable]]:
        if not self.initialized:
            await self.initialize()

        return await self._route_with_llm(request, top_k)

    async def route_to_server(
        self, request: str, top_k: int = 1
    ) -> List[LLMRouterResult[str]]:
        if not self.initialized:
            await self.initialize()

        return await self._route_with_llm(
            request,
            top_k,
            include_servers=True,
            include_agents=False,
            include_functions=False,
        )

    async def route_to_agent(
        self, request: str, top_k: int = 1
    ) -> List[LLMRouterResult[Agent]]:
        if not self.initialized:
            await self.initialize()

        return await self._route_with_llm(
            request,
            top_k,
            include_servers=False,
            include_agents=True,
            include_functions=False,
        )

    async def route_to_function(
        self, request: str, top_k: int = 1
    ) -> List[LLMRouterResult[Callable]]:
        if not self.initialized:
            await self.initialize()

        return await self._route_with_llm(
            request,
            top_k,
            include_servers=False,
            include_agents=False,
            include_functions=True,
        )

    async def _route_with_llm(
        self,
        request: str,
        top_k: int = 1,
        include_servers: bool = True,
        include_agents: bool = True,
        include_functions: bool = True,
    ) -> List[LLMRouterResult]:
        if not self.initialized:
            await self.initialize()

        routing_instruction = self.routing_instruction or DEFAULT_ROUTING_INSTRUCTION

        # Generate the categories context
        context = self._generate_context(
            include_servers=include_servers,
            include_agents=include_agents,
            include_functions=include_functions,
        )

        logger.debug(
            f"Requesting routing from LLM, \nrequest: {request} \ntop_k: {top_k} \nrouting_instruction: {routing_instruction} \ncontext={context}"
        )

        # Format the prompt with all the necessary information
        prompt = routing_instruction.format(
            context=context, request=request, top_k=top_k
        )

        # Get routes from LLM
        response = await self.llm.generate_structured(
            message=prompt,
            response_model=StructuredResponse,
        )

        # Construct the result
        if not response or not response.categories:
            return []

        result: List[LLMRouterResult] = []
        for r in response.categories:
            router_category = self.categories.get(r.category)
            if not router_category:
                # Skip invalid categories
                # TODO: saqadri - log or raise an error
                continue

            result.append(
                LLMRouterResult(
                    result=router_category.category,
                    confidence=r.confidence,
                    reasoning=r.reasoning,
                )
            )

        return result[:top_k]

    def _generate_context(
        self,
        include_servers: bool = True,
        include_agents: bool = True,
        include_functions: bool = True,
    ) -> str:
        """Generate a formatted context list of categories."""

        context_list = []
        idx = 1

        # Format all categories
        if include_servers:
            for category in self.server_categories.values():
                context_list.append(self.format_category(category, idx))
                idx += 1

        if include_agents:
            for category in self.agent_categories.values():
                context_list.append(self.format_category(category, idx))
                idx += 1

        if include_functions:
            for category in self.function_categories.values():
                context_list.append(self.format_category(category, idx))
                idx += 1

        return "\n\n".join(context_list)

```

### src/mcp_agent/workflows/router/router_llm_anthropic.py

```py
from typing import Callable, List

from mcp_agent.agents.agent import Agent
from mcp_agent.mcp_server_registry import ServerRegistry
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.router.router_llm import LLMRouter

ROUTING_SYSTEM_INSTRUCTION = """
You are a highly accurate request router that directs incoming requests to the most appropriate category.
A category is a specialized destination, such as a Function, an MCP Server (a collection of tools/functions), or an Agent (a collection of servers).
You will be provided with a request and a list of categories to choose from.
You can choose one or more categories, or choose none if no category is appropriate.
"""


class AnthropicLLMRouter(LLMRouter):
    """
    An LLM router that uses an Anthropic model to make routing decisions.
    """

    def __init__(
        self,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        server_registry: ServerRegistry | None = None,
    ):
        anthropic_llm = AnthropicAugmentedLLM(instruction=ROUTING_SYSTEM_INSTRUCTION)

        super().__init__(
            llm=anthropic_llm,
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            routing_instruction=routing_instruction,
            server_registry=server_registry,
        )

    @classmethod
    async def create(
        cls,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        server_registry: ServerRegistry | None = None,
    ) -> "AnthropicLLMRouter":
        """
        Factory method to create and initialize a router.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            routing_instruction=routing_instruction,
            server_registry=server_registry,
        )
        await instance.initialize()
        return instance

```

### src/mcp_agent/workflows/router/router_llm_openai.py

```py
from typing import Callable, List

from mcp_agent.agents.agent import Agent
from mcp_agent.mcp_server_registry import ServerRegistry
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.router.router_llm import LLMRouter

ROUTING_SYSTEM_INSTRUCTION = """
You are a highly accurate request router that directs incoming requests to the most appropriate category.
A category is a specialized destination, such as a Function, an MCP Server (a collection of tools/functions), or an Agent (a collection of servers).
You will be provided with a request and a list of categories to choose from.
You can choose one or more categories, or choose none if no category is appropriate.
"""


class OpenAILLMRouter(LLMRouter):
    """
    An LLM router that uses an OpenAI model to make routing decisions.
    """

    def __init__(
        self,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        server_registry: ServerRegistry | None = None,
    ):
        openai_llm = OpenAIAugmentedLLM(instruction=ROUTING_SYSTEM_INSTRUCTION)

        super().__init__(
            llm=openai_llm,
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            routing_instruction=routing_instruction,
            server_registry=server_registry,
        )

    @classmethod
    async def create(
        cls,
        mcp_servers_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        server_registry: ServerRegistry | None = None,
    ) -> "OpenAILLMRouter":
        """
        Factory method to create and initialize a classifier.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            mcp_servers_names=mcp_servers_names,
            agents=agents,
            functions=functions,
            routing_instruction=routing_instruction,
            server_registry=server_registry,
        )
        await instance.initialize()
        return instance

```

### src/mcp_agent/workflows/swarm/**init**.py

```py

```

### src/mcp_agent/workflows/swarm/swarm.py

```py
from typing import Callable, Dict, Generic, List, Optional
from collections import defaultdict

from pydantic import AnyUrl, BaseModel, ConfigDict
from mcp.server.fastmcp.tools import Tool as FastTool
from mcp.types import (
    CallToolRequest,
    EmbeddedResource,
    CallToolResult,
    ListToolsResult,
    TextContent,
    TextResourceContents,
    Tool,
)

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
)
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class AgentResource(EmbeddedResource):
    """
    A resource that returns an agent. Meant for use with tool calls that want to return an Agent for further processing.
    """

    agent: Optional["Agent"] = None


class AgentFunctionResultResource(EmbeddedResource):
    """
    A resource that returns an AgentFunctionResult.
    Meant for use with tool calls that return an AgentFunctionResult for further processing.
    """

    result: "AgentFunctionResult"


def create_agent_resource(agent: "Agent") -> AgentResource:
    return AgentResource(
        type="resource",
        agent=agent,
        resource=TextResourceContents(
            text=f"You are now Agent '{agent.name}'. Please review the messages and continue execution",
            uri=AnyUrl("http://fake.url"),  # Required property but not needed
        ),
    )


def create_agent_function_result_resource(
    result: "AgentFunctionResult",
) -> AgentFunctionResultResource:
    return AgentFunctionResultResource(
        type="resource",
        result=result,
        resource=TextResourceContents(
            text=result.value or result.agent.name or "AgentFunctionResult",
            uri=AnyUrl("http://fake.url"),  # Required property but not needed
        ),
    )


class SwarmAgent(Agent):
    """
    A SwarmAgent is an Agent that can spawn other agents and interactively resolve a task.
    Based on OpenAI Swarm: https://github.com/openai/swarm.

    SwarmAgents have access to tools available on the servers they are connected to, but additionally
    have a list of (possibly local) functions that can be called as tools.
    """

    def __init__(
        self,
        name: str,
        instruction: str | Callable[[Dict], str] = "You are a helpful agent.",
        server_names: list[str] = None,
        functions: List["AgentFunctionCallable"] = None,
        parallel_tool_calls: bool = True,
    ):
        super().__init__(
            name=name,
            instruction=instruction,
            server_names=server_names,
            # TODO: saqadri - figure out if Swarm can maintain connection persistence
            # It's difficult because we don't know when the agent will be done with its task
            connection_persistence=False,
        )
        self.functions = functions
        self.parallel_tool_calls = parallel_tool_calls

        # Map function names to tools
        self._function_tool_map: Dict[str, FastTool] = {}

    async def initialize(self):
        if self.initialized:
            return

        await super().initialize()
        for function in self.functions:
            tool: FastTool = FastTool.from_function(function)
            self._function_tool_map[tool.name] = tool

    async def list_tools(self) -> ListToolsResult:
        if not self.initialized:
            await self.initialize()

        result = await super().list_tools()
        for tool in self._function_tool_map.values():
            result.tools.append(
                Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.parameters,
                )
            )

        return result

    async def call_tool(
        self, name: str, arguments: dict | None = None
    ) -> CallToolResult:
        if not self.initialized:
            await self.initialize()

        if name in self._function_tool_map:
            tool = self._function_tool_map[name]
            result = await tool.run(arguments)

            logger.debug(f"Function tool {name} result:", data=result)

            if isinstance(result, Agent) or isinstance(result, SwarmAgent):
                resource = create_agent_resource(result)
                return CallToolResult(content=[resource])
            elif isinstance(result, AgentFunctionResult):
                resource = create_agent_function_result_resource(result)
                return CallToolResult(content=[resource])
            elif isinstance(result, str):
                # TODO: saqadri - this is likely meant for returning context variables
                return CallToolResult(content=[TextContent(type="text", text=result)])
            elif isinstance(result, dict):
                return CallToolResult(
                    content=[TextContent(type="text", text=str(result))]
                )
            else:
                logger.warning(f"Unknown result type: {result}, returning as text.")
                return CallToolResult(
                    content=[TextContent(type="text", text=str(result))]
                )

        return await super().call_tool(name, arguments)


class AgentFunctionResult(BaseModel):
    """
    Encapsulates the possible return values for a Swarm agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Agent | None = None
    context_variables: dict = {}

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


AgentFunctionReturnType = str | Agent | dict | AgentFunctionResult
"""A type alias for the return type of a Swarm agent function."""

AgentFunctionCallable = Callable[[], AgentFunctionReturnType]


async def create_transfer_to_agent_tool(
    agent: "Agent", agent_function: Callable[[], None]
) -> Tool:
    return Tool(
        name="transfer_to_agent",
        description="Transfer control to the agent",
        agent_resource=create_agent_resource(agent),
        agent_function=agent_function,
    )


async def create_agent_function_tool(agent_function: "AgentFunctionCallable") -> Tool:
    return Tool(
        name="agent_function",
        description="Agent function",
        agent_resource=None,
        agent_function=agent_function,
    )


class Swarm(AugmentedLLM[MessageParamT, MessageT], Generic[MessageParamT, MessageT]):
    """
    Handles orchestrating agents that can use tools via MCP servers.

    MCP version of the OpenAI Swarm class (https://github.com/openai/swarm.)
    """

    # TODO: saqadri - streaming isn't supported yet because the underlying AugmentedLLM classes don't support it
    def __init__(self, agent: SwarmAgent, context_variables: Dict[str, str] = None):
        """
        Initialize the LLM planner with an agent, which will be used as the
        starting point for the workflow.
        """
        super().__init__(agent=agent)
        self.context_variables = defaultdict(str, context_variables or {})
        self.instruction = (
            agent.instruction(self.context_variables)
            if isinstance(agent.instruction, Callable)
            else agent.instruction
        )
        logger.debug(
            f"Swarm initialized with agent {agent.name}",
            data={
                "context_variables": self.context_variables,
                "instruction": self.instruction,
            },
        )

    async def get_tool(self, tool_name: str) -> Tool | None:
        """Get the schema for a tool by name."""
        result = await self.aggregator.list_tools()
        for tool in result.tools:
            if tool.name == tool_name:
                return tool

        return None

    async def pre_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest
    ) -> CallToolRequest | bool:
        if not self.aggregator:
            # If there are no agents, we can't do anything, so we should bail
            return False

        tool = await self.get_tool(request.params.name)
        if not tool:
            logger.warning(
                f"Warning: Tool '{request.params.name}' not found in agent '{self.aggregator.name}' tools. Proceeding with original request params."
            )
            return request

        # If the tool has a "context_variables" parameter, we set it to our context variables state
        if "context_variables" in tool.inputSchema:
            logger.debug(
                f"Setting context variables on tool_call '{request.params.name}'",
                data=self.context_variables,
            )
            request.params.arguments["context_variables"] = self.context_variables

        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ) -> CallToolResult:
        contents = []
        for content in result.content:
            if isinstance(content, AgentResource):
                # Set the new agent as the current agent
                await self._set_agent(content.agent)
                contents.append(TextContent(type="text", text=content.resource.text))
            elif isinstance(content, AgentFunctionResult):
                logger.info(
                    "Updating context variables with new context variables from agent function result",
                    data=content.context_variables,
                )
                self.context_variables.update(content.context_variables)
                if content.agent:
                    # Set the new agent as the current agent
                    self._set_agent(content.agent)

                contents.append(TextContent(type="text", text=content.resource.text))
            else:
                contents.append(content)

        result.content = contents
        return result

    async def _set_agent(
        self,
        agent: SwarmAgent,
    ):
        logger.info(
            f"Switching from agent '{self.aggregator.name}' -> agent '{agent.name}'"
        )
        if self.aggregator:
            # Close the current agent
            await self.aggregator.shutdown()

        # Initialize the new agent (if it's not None)
        self.aggregator = agent

        if not self.aggregator:
            self.instruction = None
            return

        await self.aggregator.initialize()
        self.instruction = (
            agent.instruction(self.context_variables)
            if callable(agent.instruction)
            else agent.instruction
        )

```

### src/mcp_agent/workflows/swarm/swarm_anthropic.py

```py
from typing import List

from mcp_agent.workflows.swarm.swarm import Swarm
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class AnthropicSwarm(Swarm, AnthropicAugmentedLLM):
    """
    MCP version of the OpenAI Swarm class (https://github.com/openai/swarm.),
    using Anthropic's API as the LLM.
    """

    async def generate(
        self,
        message,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "claude-3-5-sonnet-20241022",
        stop_sequences: List[str] = None,
        max_tokens: int = 8192,
        parallel_tool_calls: bool = False,
    ):
        iterations = 0
        response = None
        agent_name = str(self.aggregator.name) if self.aggregator else None

        while iterations < max_iterations and agent_name:
            response = await super().generate(
                message=message
                if iterations == 0
                else "Please resolve my original request. If it has already been resolved then end turn",
                use_history=use_history,
                max_iterations=1,  # TODO: saqadri - validate
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )
            logger.debug(f"Agent: {agent_name}, response:", data=response)
            agent_name = self.aggregator.name if self.aggregator else None
            iterations += 1

        # Return final response back
        return response

```

### src/mcp_agent/workflows/swarm/swarm_openai.py

```py
from typing import List

from mcp_agent.workflows.swarm.swarm import Swarm
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class OpenAISwarm(Swarm, OpenAIAugmentedLLM):
    """
    MCP version of the OpenAI Swarm class (https://github.com/openai/swarm.), using OpenAI's ChatCompletion as the LLM.
    """

    async def generate(
        self,
        message,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "gpt-4o",
        stop_sequences: List[str] = None,
        max_tokens: int = 8192,
        parallel_tool_calls: bool = False,
    ):
        iterations = 0
        response = None
        agent_name = str(self.aggregator.name) if self.aggregator else None

        while iterations < max_iterations and agent_name:
            response = await super().generate(
                message=message
                if iterations == 0
                else "Please resolve my original request. If it has already been resolved then end turn",
                use_history=use_history,
                max_iterations=1,  # TODO: saqadri - validate
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )
            logger.debug(f"Agent: {agent_name}, response:", data=response)
            agent_name = self.aggregator.name if self.aggregator else None
            iterations += 1

        # Return final response back
        return response

```
