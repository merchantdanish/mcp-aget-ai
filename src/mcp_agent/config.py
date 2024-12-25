"""
Reading settings from environment variables and providing a settings object
for the application configuration.
"""

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class TemporalSettings(BaseModel):
    """
    Temporal settings for the MCP Agent application.
    """

    host: str
    namespace: str = "default"
    task_queue: str


class Settings(BaseSettings):
    """
    Settings class for the MCP Agent application.
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
    )  # Customize the behavior of settings here

    temporal: TemporalSettings | None = None
    """Settings for Temporal workflow orchestration"""

    otlp_endpoint: str | None = None
    """OTLP endpoint for OpenTelemetry tracing"""

    disable_usage_telemetry: bool = False
    """Disable usage tracking that is enabled by default in mcp-agent"""

    config_yaml: str = "mcp-agent.config.yaml"
    """Path to the configuration file for the MCP Agent application"""

    upstream_server_name: str | None = None


settings = Settings()
