"""Authentication middleware for MCP servers in MCP Agent Cloud.

This module provides middleware for handling authentication in MCP server requests.
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Awaitable

from mcp_agent.cloud.auth.auth_service import AuthService

# Type for an HTTP request handler
RequestHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]

logger = logging.getLogger(__name__)

class MCPCloudAuthMiddleware:
    """Authentication middleware for MCP servers in MCP Agent Cloud."""
    
    def __init__(self, auth_service: AuthService, server_name: str):
        """Initialize the authentication middleware.
        
        Args:
            auth_service: Authentication service
            server_name: Name of the server
        """
        self.auth_service = auth_service
        self.server_name = server_name
        
    async def __call__(self, request: Dict[str, Any], next_handler: RequestHandler) -> Dict[str, Any]:
        """Process an HTTP request.
        
        Args:
            request: HTTP request data
            next_handler: Next handler in the chain
            
        Returns:
            HTTP response
        """
        # Skip authentication for authentication-related endpoints
        path = request.get("path", "")
        if (
            path.startswith("/.well-known/") or 
            path in ["/authorize", "/token", "/register", "/callback", "/health"]
        ):
            return await next_handler(request)
        
        # Check for Authorization header
        headers = request.get("headers", {})
        auth_header = headers.get("Authorization", "")
        
        if not auth_header:
            # No authorization provided, return 401
            return {
                "status": 401,
                "headers": {
                    "Content-Type": "application/json",
                    "WWW-Authenticate": "Bearer realm=\"MCP Server\", error=\"unauthorized\", error_description=\"Authentication required\""
                },
                "body": json.dumps({
                    "error": "unauthorized",
                    "error_description": "Authentication required to access this MCP server"
                })
            }
        
        # Parse Authorization header
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            # Invalid Authorization header format
            return {
                "status": 401,
                "headers": {
                    "Content-Type": "application/json",
                    "WWW-Authenticate": "Bearer realm=\"MCP Server\", error=\"invalid_token\", error_description=\"Invalid token format\""
                },
                "body": json.dumps({
                    "error": "invalid_token",
                    "error_description": "Invalid token format"
                })
            }
        
        token = parts[1]
        
        # Validate the token
        is_valid, user_data = await self.auth_service.validate_token(token)
        
        if not is_valid or not user_data:
            # Invalid token
            return {
                "status": 401,
                "headers": {
                    "Content-Type": "application/json",
                    "WWW-Authenticate": "Bearer realm=\"MCP Server\", error=\"invalid_token\", error_description=\"Invalid or expired token\""
                },
                "body": json.dumps({
                    "error": "invalid_token",
                    "error_description": "Invalid or expired token"
                })
            }
        
        # Add user data to request context
        request["user"] = user_data
        
        # Proceed to next handler
        return await next_handler(request)