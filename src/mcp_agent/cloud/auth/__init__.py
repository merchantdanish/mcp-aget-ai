"""Authentication module for MCP Agent Cloud.

This module provides authentication mechanisms for securely deploying MCP Agents
and managing service credentials.
"""

from mcp_agent.cloud.auth.cli_auth_service import CLIAuthService
from mcp_agent.cloud.auth.service_credential_manager import ServiceCredentialManager, CredentialStore, FileCredentialStore

__all__ = ["CLIAuthService", "ServiceCredentialManager", "CredentialStore", "FileCredentialStore"]