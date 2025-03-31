"""Authentication modules for MCP Agent Cloud."""

from mcp_agent.auth.auth_service import AuthService
from mcp_agent.auth.auth_provider import (
    AuthProvider,
    SelfContainedAuthProvider,
    OAuthProvider,
    GitHubOAuthProvider,
    CustomOAuthProvider,
)

__all__ = [
    "AuthService",
    "AuthProvider",
    "SelfContainedAuthProvider",
    "OAuthProvider", 
    "GitHubOAuthProvider",
    "CustomOAuthProvider",
]