"""
Reading settings from environment variables and providing a settings object
for the application configuration.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnthropicSettings(BaseModel):
    """
    Settings for using Anthropic models in the MCP Agent application.
    """

    api_key: str

    model_config = ConfigDict(extra="allow")


class CohereSettings(BaseModel):
    """
    Settings for using Cohere models in the MCP Agent application.
    """

    api_key: str

    model_config = ConfigDict(extra="allow")


class OpenAISettings(BaseModel):
    """
    Settings for using OpenAI models in the MCP Agent application.
    """

    api_key: str

    model_config = ConfigDict(extra="allow")


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


class Settings(BaseSettings):
    """
    Settings class for the MCP Agent application.
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
    )  # Customize the behavior of settings here

    execution_engine: Literal["asyncio", "temporal"] = "asyncio"

    temporal: TemporalSettings | None = None
    """Settings for Temporal workflow orchestration"""

    anthropic: AnthropicSettings | None = None
    """Settings for using Anthropic models in the MCP Agent application"""

    cohere: CohereSettings | None = None
    """Settings for using Cohere models in the MCP Agent application"""

    openai: OpenAISettings | None = None
    """Settings for using OpenAI models in the MCP Agent application"""

    logger: OpenTelemetrySettings | None = OpenTelemetrySettings()
    """OpenTelemetry logging settings for the MCP Agent application"""

    usage_telemetry: UsageTelemetrySettings | None = UsageTelemetrySettings()
    """Usage tracking settings for the MCP Agent application"""

    config_yaml: str = "mcp-agent.config.yaml"
    """Path to the configuration file for the MCP Agent application"""


settings = Settings()
