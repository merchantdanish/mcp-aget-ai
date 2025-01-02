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

    otlp_endpoint: str | None = None
    """OTLP endpoint for OpenTelemetry tracing"""

    disable_usage_telemetry: bool = False
    """Disable usage tracking that is enabled by default in mcp-agent"""

    config_yaml: str = "mcp-agent.config.yaml"
    """Path to the configuration file for the MCP Agent application"""

    upstream_server_name: str | None = None


settings = Settings()
