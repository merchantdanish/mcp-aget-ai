"""Unit tests for the HTTP handlers."""

import pytest
import asyncio
import json
import urllib.parse
from unittest.mock import Mock, AsyncMock, patch

from mcp_agent.auth.http_handlers import MCPAuthHTTPHandlers
from mcp_agent.auth.auth_service import AuthService


@pytest.fixture
def auth_service_mock():
    """Create a mock auth service."""
    mock = Mock(spec=AuthService)
    
    # Setup async methods
    metadata = {
        "issuer": "https://example.com",
        "authorization_endpoint": "https://example.com/authorize",
        "token_endpoint": "https://example.com/token",
        "jwks_uri": "https://example.com/.well-known/jwks.json",
        "registration_endpoint": "https://example.com/register",
        "scopes_supported": ["openid", "profile", "email"],
        "response_types_supported": ["code"],
        "provider_types_supported": ["self-contained", "github"]
    }
    mock.get_authorization_server_metadata = AsyncMock(return_value=metadata)
    
    # Setup handle_authorize_request
    mock.handle_authorize_request = AsyncMock(return_value={
        "redirect_to": "https://example.com/login"
    })
    
    # Setup handle_token_request
    mock.handle_token_request = AsyncMock(return_value={
        "access_token": "test-access-token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": "test-refresh-token"
    })
    
    # Setup register_client
    mock.register_client = AsyncMock(return_value={
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "client_name": "Test Client",
        "redirect_uris": ["http://localhost:8000/callback"]
    })
    
    return mock


@pytest.fixture
def http_handlers(auth_service_mock):
    """Create HTTP handlers with a mock auth service."""
    return MCPAuthHTTPHandlers(auth_service_mock, "https://example.com")


class TestMCPAuthHTTPHandlers:
    """Test the MCP auth HTTP handlers."""
    
    @pytest.mark.asyncio
    async def test_well_known_handler(self, http_handlers, auth_service_mock):
        """Test the /.well-known/oauth-authorization-server handler."""
        # Create a request
        request = {}
        
        # Handle the request
        response = await http_handlers.well_known_handler(request)
        
        # Check that the method was called
        auth_service_mock.get_authorization_server_metadata.assert_called_once_with("https://example.com")
        
        # Check the response
        assert response["status"] == 200
        assert response["headers"]["Content-Type"] == "application/json"
        
        # Parse the response body
        body = json.loads(response["body"])
        assert body["issuer"] == "https://example.com"
        assert body["authorization_endpoint"] == "https://example.com/authorize"
        assert body["token_endpoint"] == "https://example.com/token"
        
        # Test error handling
        auth_service_mock.get_authorization_server_metadata.side_effect = Exception("Test error")
        
        # Handle the request
        response = await http_handlers.well_known_handler(request)
        
        # Check the response
        assert response["status"] == 500
        assert response["headers"]["Content-Type"] == "application/json"
        
        # Parse the response body
        body = json.loads(response["body"])
        assert body["error"] == "server_error"
        assert body["error_description"] == "Test error"
    
    @pytest.mark.asyncio
    async def test_authorize_handler(self, http_handlers, auth_service_mock):
        """Test the /authorize handler."""
        # Create a request with query parameters
        query = urllib.parse.urlencode({
            "provider": "self-contained",
            "client_id": "test-client-id",
            "redirect_uri": "http://localhost:8000/callback",
            "scope": "openid profile email",
            "response_type": "code",
            "state": "test-state",
            "code_challenge": "test-challenge",
            "code_challenge_method": "S256"
        })
        request = {"query": query}
        
        # Handle the request
        response = await http_handlers.authorize_handler(request)
        
        # Check that the method was called
        auth_service_mock.handle_authorize_request.assert_called_once_with(
            provider_name="self-contained",
            client_id="test-client-id",
            redirect_uri="http://localhost:8000/callback",
            scope="openid profile email",
            response_type="code",
            state="test-state",
            code_challenge="test-challenge",
            code_challenge_method="S256"
        )
        
        # Check the response
        assert response["status"] == 302
        assert response["headers"]["Location"] == "https://example.com/login"
        
        # Test error handling
        auth_service_mock.handle_authorize_request.side_effect = Exception("Test error")
        
        # Handle the request
        response = await http_handlers.authorize_handler(request)
        
        # Check the response
        assert response["status"] == 400
        assert response["headers"]["Content-Type"] == "application/json"
        
        # Parse the response body
        body = json.loads(response["body"])
        assert body["error"] == "invalid_request"
        assert body["error_description"] == "Test error"
    
    @pytest.mark.asyncio
    async def test_token_handler_json(self, http_handlers, auth_service_mock):
        """Test the /token handler with JSON content type."""
        # Create a request with JSON body
        body = json.dumps({
            "grant_type": "authorization_code",
            "provider": "self-contained",
            "code": "test-code",
            "redirect_uri": "http://localhost:8000/callback",
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "code_verifier": "test-verifier"
        })
        request = {
            "headers": {"Content-Type": "application/json"},
            "body": body
        }
        
        # Handle the request
        response = await http_handlers.token_handler(request)
        
        # Check that the method was called
        auth_service_mock.handle_token_request.assert_called_once_with(
            grant_type="authorization_code",
            provider_name="self-contained",
            code="test-code",
            redirect_uri="http://localhost:8000/callback",
            client_id="test-client-id",
            client_secret="test-client-secret",
            refresh_token=None,
            code_verifier="test-verifier"
        )
        
        # Check the response
        assert response["status"] == 200
        assert response["headers"]["Content-Type"] == "application/json"
        
        # Parse the response body
        body = json.loads(response["body"])
        assert body["access_token"] == "test-access-token"
        assert body["token_type"] == "Bearer"
        assert body["expires_in"] == 3600
        assert body["refresh_token"] == "test-refresh-token"
    
    @pytest.mark.asyncio
    async def test_token_handler_form(self, http_handlers, auth_service_mock):
        """Test the /token handler with form content type."""
        # Create a request with form body
        form_data = {
            "grant_type": "refresh_token",
            "provider": "self-contained",
            "refresh_token": "test-refresh-token",
            "client_id": "test-client-id"
        }
        body = urllib.parse.urlencode(form_data)
        request = {
            "headers": {"Content-Type": "application/x-www-form-urlencoded"},
            "body": body
        }
        
        # Handle the request
        response = await http_handlers.token_handler(request)
        
        # Check that the method was called
        auth_service_mock.handle_token_request.assert_called_once_with(
            grant_type="refresh_token",
            provider_name="self-contained",
            code=None,
            redirect_uri=None,
            client_id="test-client-id",
            client_secret=None,
            refresh_token="test-refresh-token",
            code_verifier=None
        )
        
        # Check the response
        assert response["status"] == 200
        assert response["headers"]["Content-Type"] == "application/json"
        
        # Parse the response body
        body = json.loads(response["body"])
        assert body["access_token"] == "test-access-token"
        assert body["token_type"] == "Bearer"
        assert body["expires_in"] == 3600
        assert body["refresh_token"] == "test-refresh-token"
    
    @pytest.mark.asyncio
    async def test_token_handler_error(self, http_handlers, auth_service_mock):
        """Test the /token handler with an error response."""
        # Create a request
        request = {
            "headers": {"Content-Type": "application/json"},
            "body": "{}"
        }
        
        # Mock an error response from handle_token_request
        auth_service_mock.handle_token_request.return_value = {
            "error": "invalid_request",
            "error_description": "Missing required parameters"
        }
        
        # Handle the request
        response = await http_handlers.token_handler(request)
        
        # Check the response
        assert response["status"] == 400
        assert response["headers"]["Content-Type"] == "application/json"
        
        # Parse the response body
        body = json.loads(response["body"])
        assert body["error"] == "invalid_request"
        assert body["error_description"] == "Missing required parameters"
        
        # Test server error
        auth_service_mock.handle_token_request.side_effect = Exception("Test error")
        
        # Handle the request
        response = await http_handlers.token_handler(request)
        
        # Check the response
        assert response["status"] == 500
        assert response["headers"]["Content-Type"] == "application/json"
        
        # Parse the response body
        body = json.loads(response["body"])
        assert body["error"] == "server_error"
        assert body["error_description"] == "Test error"
    
    @pytest.mark.asyncio
    async def test_register_handler(self, http_handlers, auth_service_mock):
        """Test the /register handler."""
        # Create a request with JSON body
        body = json.dumps({
            "provider": "self-contained",
            "client_name": "Test Client",
            "redirect_uris": ["http://localhost:8000/callback"],
            "client_uri": "http://localhost:8000",
            "logo_uri": "http://localhost:8000/logo.png",
            "tos_uri": "http://localhost:8000/tos",
            "policy_uri": "http://localhost:8000/policy",
            "software_id": "test-software-id",
            "software_version": "1.0.0"
        })
        request = {
            "headers": {"Content-Type": "application/json"},
            "body": body
        }
        
        # Handle the request
        response = await http_handlers.register_handler(request)
        
        # Check that the method was called
        auth_service_mock.register_client.assert_called_once_with(
            provider_name="self-contained",
            client_name="Test Client",
            redirect_uris=["http://localhost:8000/callback"],
            client_uri="http://localhost:8000",
            logo_uri="http://localhost:8000/logo.png",
            tos_uri="http://localhost:8000/tos",
            policy_uri="http://localhost:8000/policy",
            software_id="test-software-id",
            software_version="1.0.0"
        )
        
        # Check the response
        assert response["status"] == 201
        assert response["headers"]["Content-Type"] == "application/json"
        
        # Parse the response body
        body = json.loads(response["body"])
        assert body["client_id"] == "test-client-id"
        assert body["client_secret"] == "test-client-secret"
        assert body["client_name"] == "Test Client"
        assert body["redirect_uris"] == ["http://localhost:8000/callback"]
    
    @pytest.mark.asyncio
    async def test_register_handler_missing_params(self, http_handlers):
        """Test the /register handler with missing parameters."""
        # Create a request with missing parameters
        body = json.dumps({
            "provider": "self-contained"
        })
        request = {
            "headers": {"Content-Type": "application/json"},
            "body": body
        }
        
        # Handle the request
        response = await http_handlers.register_handler(request)
        
        # Check the response
        assert response["status"] == 400
        assert response["headers"]["Content-Type"] == "application/json"
        
        # Parse the response body
        body = json.loads(response["body"])
        assert body["error"] == "invalid_request"
        assert body["error_description"] == "Missing required parameters"
    
    @pytest.mark.asyncio
    async def test_callback_handler_success(self, http_handlers):
        """Test the /callback handler with a successful authentication."""
        # Create a request with query parameters
        query = urllib.parse.urlencode({
            "code": "test-code",
            "state": "test-state"
        })
        request = {"query": query}
        
        # Handle the request
        response = await http_handlers.callback_handler(request)
        
        # Check the response
        assert response["status"] == 200
        assert response["headers"]["Content-Type"] == "text/html"
        
        # Check that the response body contains the code and state
        assert "test-code" in response["body"]
        assert "test-state" in response["body"]
        assert "Authentication Successful" in response["body"]
    
    @pytest.mark.asyncio
    async def test_callback_handler_error(self, http_handlers):
        """Test the /callback handler with an authentication error."""
        # Create a request with error parameters
        query = urllib.parse.urlencode({
            "error": "access_denied",
            "error_description": "User denied access"
        })
        request = {"query": query}
        
        # Handle the request
        response = await http_handlers.callback_handler(request)
        
        # Check the response
        assert response["status"] == 200
        assert response["headers"]["Content-Type"] == "text/html"
        
        # Check that the response body contains the error details
        assert "Authentication Error" in response["body"]
        assert "access_denied" in response["body"]
        assert "User denied access" in response["body"]
    
    def test_get_routes(self, http_handlers):
        """Test getting the routes for the HTTP handlers."""
        # Get the routes
        routes = http_handlers.get_routes()
        
        # Check that all the expected routes are present
        assert "/.well-known/oauth-authorization-server" in routes
        assert "/authorize" in routes
        assert "/token" in routes
        assert "/register" in routes
        assert "/callback" in routes
        
        # Check that the handlers are the correct methods
        assert routes["/.well-known/oauth-authorization-server"] == http_handlers.well_known_handler
        assert routes["/authorize"] == http_handlers.authorize_handler
        assert routes["/token"] == http_handlers.token_handler
        assert routes["/register"] == http_handlers.register_handler
        assert routes["/callback"] == http_handlers.callback_handler