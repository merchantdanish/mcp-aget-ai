import pytest

from mcp_agent.config import get_settings, _clear_global_settings


class TestConfigEnvAliases:
    @pytest.fixture(autouse=True)
    def clear_settings(self):
        _clear_global_settings()

    def test_aliases_from_yaml_preload(self, monkeypatch):
        yaml_payload = """
openai:
  OPENAI_API_KEY: sk-openai-yaml
anthropic:
  ANTHROPIC_API_KEY: sk-anthropic-yaml
azure:
  AZURE_OPENAI_API_KEY: az-key-yaml
  AZURE_OPENAI_ENDPOINT: https://azure.openai.example
google:
  GEMINI_API_KEY: g-api-gemini-yaml
bedrock:
  AWS_ACCESS_KEY_ID: AKIA_YAML
  AWS_SECRET_ACCESS_KEY: SECRET_YAML
  AWS_SESSION_TOKEN: TOKEN_YAML
  AWS_REGION: us-east-2
  AWS_PROFILE: default
"""
        monkeypatch.setenv("MCP_APP_SETTINGS_PRELOAD", yaml_payload)
        settings = get_settings()
        assert settings.openai and settings.openai.api_key == "sk-openai-yaml"
        assert settings.anthropic and settings.anthropic.api_key == "sk-anthropic-yaml"
        assert settings.azure and settings.azure.api_key == "az-key-yaml"
        assert settings.azure.endpoint == "https://azure.openai.example"
        assert settings.google and settings.google.api_key == "g-api-gemini-yaml"
        assert settings.bedrock and settings.bedrock.aws_access_key_id == "AKIA_YAML"
        assert settings.bedrock.aws_secret_access_key == "SECRET_YAML"
        assert settings.bedrock.aws_session_token == "TOKEN_YAML"
        assert settings.bedrock.aws_region == "us-east-2"
        assert settings.bedrock.profile == "default"

    def test_aliases_from_yaml_preload_with_nested_style_keys(self, monkeypatch):
        yaml_payload = """
openai:
  openai__api_key: sk-openai-yaml-nested
anthropic:
  anthropic__api_key: sk-anthropic-yaml-nested
azure:
  azure__api_key: az-key-yaml-nested
  azure__endpoint: https://azure.nested.example
google:
  google__api_key: g-api-yaml-nested
bedrock:
  bedrock__aws_access_key_id: AKIA_YAML_NESTED
  bedrock__aws_secret_access_key: SECRET_YAML_NESTED
  bedrock__aws_session_token: TOKEN_YAML_NESTED
  bedrock__aws_region: eu-central-1
  bedrock__profile: dev
"""
        monkeypatch.setenv("MCP_APP_SETTINGS_PRELOAD", yaml_payload)
        settings = get_settings()
        assert settings.openai and settings.openai.api_key == "sk-openai-yaml-nested"
        assert (
            settings.anthropic
            and settings.anthropic.api_key == "sk-anthropic-yaml-nested"
        )
        assert settings.azure and settings.azure.api_key == "az-key-yaml-nested"
        assert settings.azure.endpoint == "https://azure.nested.example"
        assert settings.google and settings.google.api_key == "g-api-yaml-nested"
        assert (
            settings.bedrock
            and settings.bedrock.aws_access_key_id == "AKIA_YAML_NESTED"
        )
        assert settings.bedrock.aws_secret_access_key == "SECRET_YAML_NESTED"
        assert settings.bedrock.aws_session_token == "TOKEN_YAML_NESTED"
        assert settings.bedrock.aws_region == "eu-central-1"
        assert settings.bedrock.profile == "dev"
