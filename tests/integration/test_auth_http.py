"""Integration tests for HTTP authentication."""

import pytest
import asyncio
import json
import urllib.parse
import tempfile
import os
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from mcp_agent.auth.auth_service import AuthService
from mcp_agent.auth.http_handlers import MCPAuthHTTPHandlers
from mcp_agent.auth.middleware import MCPAuthMiddleware


class MockHTTPServer:
    """Mock HTTP server for testing."""
    
    def __init__(self, auth_service, server_name="test-server"):
        """Initialize the mock HTTP server."""
        self.auth_service = auth_service
        self.server_name = server_name
        self.middleware = MCPAuthMiddleware(auth_service, server_name)
        self.handlers = MCPAuthHTTPHandlers(auth_service, f"https://{server_name}.example.com")
        self.routes = self.handlers.get_routes()
    
    async def handle_request(self, request):
        """Handle an HTTP request."""
        # Get the route handler
        path = request.get("path", "")
        handler = self.routes.get(path)
        
        if handler:
            # Apply middleware and pass to handler
            return await self.middleware(request, handler)
        
        # Protected route needs authentication
        return await self.middleware(request, self.protected_route_handler)
    
    async def protected_route_handler(self, request):
        """Handle a protected route."""
        # This handler is protected by middleware and requires authentication
        user = request.get("user", {})
        return {
            "status": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "message": "Protected resource accessed successfully",
                "user": user
            })
        }


@pytest.fixture
def temp_auth_dir():
    """Create a temporary directory for auth files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def auth_service(temp_auth_dir):
    """Create an auth service with a temporary clients file."""
    clients_file = os.path.join(temp_auth_dir, "clients.json")
    
    # Create initial configuration
    config = {
        "providers": {
            "self-contained": {
                "jwt_secret": "integration-test-jwt-secret"
            },
            "github": {
                "client_id": "integration-test-github-client-id",
                "client_secret": "integration-test-github-client-secret",
                "jwt_secret": "integration-test-github-jwt-secret"
            }
        },
        "clients_file": clients_file
    }
    
    # Create auth service
    return AuthService(config)


@pytest.fixture
def http_server(auth_service):
    """Create a mock HTTP server with auth middleware and handlers."""
    return MockHTTPServer(auth_service)


class TestAuthHTTPIntegration:
    """Integration tests for HTTP authentication."""
    
    @pytest.mark.asyncio
    async def test_protected_routes(self, http_server, auth_service):
        """Test middleware protection of routes."""
        # 1. Create a valid token directly in the auth service
        user_data = await auth_service.providers["self-contained"].create_user(
            username="integration-test-user",
            password="integration-test-password"
        )
        
        # Generate a valid token using the provider
        token = auth_service.providers["self-contained"].generate_jwt_token(
            user_data=user_data,
            secret=auth_service.providers["self-contained"].jwt_secret
        )
        
        # Store the token in auth service 
        auth_service.tokens[token] = {
            "provider": "self-contained",
            "client_id": "test-client-id",
            "expires_at": int(time.time()) + 3600,
            "user_data": user_data
        }
        
        # 2. Access a protected resource with a valid token
        protected_request = {
            "path": "/api/protected-resource",
            "headers": {"Authorization": f"Bearer {token}"}
        }
        
        protected_response = await http_server.handle_request(protected_request)
        assert protected_response["status"] == 200
        
        # Check the response body
        response_data = json.loads(protected_response["body"])
        assert response_data["message"] == "Protected resource accessed successfully"
        assert "user" in response_data
        
        # 3. Access a protected resource with an invalid token
        invalid_request = {
            "path": "/api/protected-resource",
            "headers": {"Authorization": "Bearer invalid-token"}
        }
        
        invalid_response = await http_server.handle_request(invalid_request)
        assert invalid_response["status"] == 401
        
        # 4. Access a protected resource without a token
        no_token_request = {
            "path": "/api/protected-resource",
            "headers": {}
        }
        
        no_token_response = await http_server.handle_request(no_token_request)
        assert no_token_response["status"] == 401
    
    @pytest.mark.asyncio
    async def test_server_metadata(self, http_server):
        """Test getting the authorization server metadata."""
        # Create a request
        request = {
            "path": "/.well-known/oauth-authorization-server"
        }
        
        # Handle the request
        response = await http_server.handle_request(request)
        
        # Check the response
        assert response["status"] == 200
        assert response["headers"]["Content-Type"] == "application/json"
        
        # Parse the response body
        metadata = json.loads(response["body"])
        assert "issuer" in metadata
        assert "authorization_endpoint" in metadata
        assert "token_endpoint" in metadata
        assert "registration_endpoint" in metadata
        assert "provider_types_supported" in metadata
        assert "self-contained" in metadata["provider_types_supported"]
        assert "github" in metadata["provider_types_supported"]