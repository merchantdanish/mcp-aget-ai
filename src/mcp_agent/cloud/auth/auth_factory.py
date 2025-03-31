"""Authentication factory for MCP servers.

This module provides a factory for creating authentication services for MCP servers,
supporting different authentication providers and server configurations.
"""

import os
import uuid
import secrets
from typing import Dict, Any, Optional, List, Tuple

from mcp_agent.cloud.auth.auth_service import AuthService
from mcp_agent.cloud.auth.providers import SelfContainedAuthProvider, GitHubOAuthProvider, CustomOAuthProvider

class MCPAuthFactory:
    """Factory for creating authentication services for MCP servers."""
    
    def __init__(self, base_dir: str = None):
        """Initialize the auth factory.
        
        Args:
            base_dir: Base directory for storing auth data
        """
        self.base_dir = base_dir or os.path.expanduser("~/.mcp-agent-cloud/auth")
        os.makedirs(self.base_dir, exist_ok=True)
        
    async def create_server_auth(self, server_name: str, provider_type: str = "self-contained") -> Tuple[AuthService, Dict[str, Any]]:
        """Create an authentication service for a server.
        
        Args:
            server_name: Name of the server
            provider_type: Type of authentication provider
            
        Returns:
            Tuple of (auth_service, auth_config)
        """
        # Create the auth directory for this server
        server_auth_dir = os.path.join(self.base_dir, server_name)
        os.makedirs(server_auth_dir, exist_ok=True)
        
        # Create config for the provider
        provider_config = self._create_provider_config(server_name, provider_type)
        
        # Create auth config
        auth_config = {
            "providers": {
                provider_type: provider_config
            },
            "clients_file": os.path.join(server_auth_dir, "clients.json")
        }
        
        # Create auth service
        auth_service = AuthService(auth_config)
        
        # Register a default client for the server
        default_client = await self._register_default_client(auth_service, provider_type)
        auth_config["default_client"] = default_client
        
        return auth_service, auth_config
    
    def _create_provider_config(self, server_name: str, provider_type: str) -> Dict[str, Any]:
        """Create configuration for an authentication provider.
        
        Args:
            server_name: Name of the server
            provider_type: Type of authentication provider
            
        Returns:
            Provider configuration
        """
        if provider_type == "self-contained":
            # Create a default user
            return {
                "jwt_secret": secrets.token_hex(32),
                "default_user": {
                    "username": "admin",
                    "password": secrets.token_hex(8),
                    "email": "admin@example.com",
                    "role": "admin"
                }
            }
        elif provider_type == "github":
            # Get GitHub credentials from environment
            client_id = os.environ.get("GITHUB_CLIENT_ID")
            client_secret = os.environ.get("GITHUB_CLIENT_SECRET")
            
            if not client_id or not client_secret:
                client_id = f"github-client-{uuid.uuid4().hex[:8]}"
                client_secret = secrets.token_hex(16)
            
            return {
                "client_id": client_id,
                "client_secret": client_secret
            }
        elif provider_type == "custom":
            # Get custom provider configuration from environment
            return {
                "authorize_url": os.environ.get("CUSTOM_AUTHORIZE_URL", "https://auth.example.com/authorize"),
                "token_url": os.environ.get("CUSTOM_TOKEN_URL", "https://auth.example.com/token"),
                "userinfo_url": os.environ.get("CUSTOM_USERINFO_URL", "https://auth.example.com/userinfo"),
                "client_id": os.environ.get("CUSTOM_CLIENT_ID", f"custom-client-{uuid.uuid4().hex[:8]}"),
                "client_secret": os.environ.get("CUSTOM_CLIENT_SECRET", secrets.token_hex(16))
            }
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    async def _register_default_client(self, auth_service: AuthService, provider_type: str) -> Dict[str, Any]:
        """Register a default client for the server.
        
        Args:
            auth_service: Authentication service
            provider_type: Type of authentication provider
            
        Returns:
            Default client information
        """
        provider = await auth_service.get_provider(provider_type)
        if not provider:
            return {}
        
        # Register a client
        redirect_uris = ["http://localhost:8000/callback", "https://mcp-agent-cloud.example.com/callback"]
        client_data = await provider.register_client("MCP Server Default Client", redirect_uris)
        
        return {
            "client_id": client_data.get("client_id", ""),
            "client_secret": client_data.get("client_secret", ""),
            "redirect_uris": redirect_uris
        }