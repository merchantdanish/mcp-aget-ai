"""Unit tests for the authentication middleware."""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch

from mcp_agent.auth.middleware import MCPAuthMiddleware
from mcp_agent.auth.auth_service import AuthService


@pytest.fixture
def auth_service_mock():
    """Create a mock auth service."""
    mock = Mock(spec=AuthService)
    
    # Setup validate_token method
    mock.validate_token = AsyncMock()
    
    # Default: Token is valid
    mock.validate_token.return_value = (True, {"id": "user-1234", "username": "testuser"})
    
    return mock


@pytest.fixture
def middleware(auth_service_mock):
    """Create the middleware with a mock auth service."""
    return MCPAuthMiddleware(auth_service_mock, "test-server")


@pytest.fixture
def next_handler_mock():
    """Create a mock next handler."""
    return AsyncMock(return_value={
        "status": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"message": "Handler success"})
    })


class TestMCPAuthMiddleware:
    """Test the MCP auth middleware."""
    
    @pytest.mark.asyncio
    async def test_auth_endpoints_pass_through(self, middleware, next_handler_mock):
        """Test that auth endpoints pass through the middleware without authentication."""
        # Test each auth endpoint
        auth_endpoints = [
            "/.well-known/oauth-authorization-server",
            "/authorize",
            "/token",
            "/register",
            "/callback",
            "/health"
        ]
        
        for endpoint in auth_endpoints:
            # Create a request
            request = {"path": endpoint}
            
            # Process the request
            response = await middleware(request, next_handler_mock)
            
            # Check that the next handler was called
            next_handler_mock.assert_called_with(request)
            
            # Check the response
            assert response["status"] == 200
            assert response["headers"]["Content-Type"] == "application/json"
            assert json.loads(response["body"])["message"] == "Handler success"
            
            # Reset the mock
            next_handler_mock.reset_mock()
    
    @pytest.mark.asyncio
    async def test_missing_auth_header(self, middleware, next_handler_mock):
        """Test that a request without an Authorization header is rejected."""
        # Create a request without an Authorization header
        request = {"path": "/api/resource", "headers": {}}
        
        # Process the request
        response = await middleware(request, next_handler_mock)
        
        # Check that the next handler was not called
        next_handler_mock.assert_not_called()
        
        # Check the response
        assert response["status"] == 401
        assert response["headers"]["Content-Type"] == "application/json"
        assert "WWW-Authenticate" in response["headers"]
        assert json.loads(response["body"])["error"] == "unauthorized"
    
    @pytest.mark.asyncio
    async def test_invalid_auth_header_format(self, middleware, next_handler_mock):
        """Test that a request with an invalid Authorization header format is rejected."""
        # Create a request with an invalid Authorization header
        request = {
            "path": "/api/resource",
            "headers": {"Authorization": "InvalidFormat"}
        }
        
        # Process the request
        response = await middleware(request, next_handler_mock)
        
        # Check that the next handler was not called
        next_handler_mock.assert_not_called()
        
        # Check the response
        assert response["status"] == 401
        assert response["headers"]["Content-Type"] == "application/json"
        assert "WWW-Authenticate" in response["headers"]
        assert json.loads(response["body"])["error"] == "invalid_token"
    
    @pytest.mark.asyncio
    async def test_invalid_token(self, middleware, next_handler_mock, auth_service_mock):
        """Test that a request with an invalid token is rejected."""
        # Mock an invalid token
        auth_service_mock.validate_token.return_value = (False, None)
        
        # Create a request with an invalid token
        request = {
            "path": "/api/resource",
            "headers": {"Authorization": "Bearer invalid-token"}
        }
        
        # Process the request
        response = await middleware(request, next_handler_mock)
        
        # Check that the auth service was called
        auth_service_mock.validate_token.assert_called_once_with("invalid-token")
        
        # Check that the next handler was not called
        next_handler_mock.assert_not_called()
        
        # Check the response
        assert response["status"] == 401
        assert response["headers"]["Content-Type"] == "application/json"
        assert "WWW-Authenticate" in response["headers"]
        assert json.loads(response["body"])["error"] == "invalid_token"
    
    @pytest.mark.asyncio
    async def test_valid_token(self, middleware, next_handler_mock, auth_service_mock):
        """Test that a request with a valid token is processed."""
        # Create user data
        user_data = {"id": "user-1234", "username": "testuser"}
        auth_service_mock.validate_token.return_value = (True, user_data)
        
        # Create a request with a valid token
        request = {
            "path": "/api/resource",
            "headers": {"Authorization": "Bearer valid-token"}
        }
        
        # Process the request
        response = await middleware(request, next_handler_mock)
        
        # Check that the auth service was called
        auth_service_mock.validate_token.assert_called_once_with("valid-token")
        
        # Check that user data was added to the request
        assert request["user"] == user_data
        
        # Check that the next handler was called with the modified request
        next_handler_mock.assert_called_once_with(request)
        
        # Check the response
        assert response["status"] == 200
        assert response["headers"]["Content-Type"] == "application/json"
        assert json.loads(response["body"])["message"] == "Handler success"