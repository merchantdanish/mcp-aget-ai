"""Unit tests for the auth service."""

import pytest
import asyncio
import time
import os
import json
from unittest.mock import Mock, AsyncMock, patch

from mcp_agent.auth.auth_service import AuthService


class TestAuthService:
    """Test the auth service."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, auth_config):
        """Test initializing the auth service."""
        # Create an auth service
        auth_service = AuthService(auth_config)
        
        # Check that the providers were initialized
        assert "self-contained" in auth_service.providers
        assert "github" in auth_service.providers
        
        # Check that the clients and tokens are empty
        assert auth_service.clients == {}
        assert auth_service.tokens == {}
        assert auth_service.auth_codes == {}
        assert auth_service.pkce_challenges == {}
    
    @pytest.mark.asyncio
    async def test_get_provider(self, auth_service):
        """Test getting a provider."""
        # Get existing providers
        self_contained_provider = await auth_service.get_provider("self-contained")
        github_provider = await auth_service.get_provider("github")
        
        # Check that the providers were returned
        assert self_contained_provider is not None
        assert github_provider is not None
        
        # Get a non-existent provider
        custom_provider = await auth_service.get_provider("custom")
        
        # Check that no provider was returned
        assert custom_provider is None
    
    @pytest.mark.asyncio
    async def test_get_authorize_url(self, auth_service):
        """Test getting an authorization URL."""
        # Mock the provider's get_authorization_url method
        provider = await auth_service.get_provider("self-contained")
        provider.get_authorization_url = AsyncMock(return_value="https://example.com/authorize")
        
        # Get an authorization URL
        url = await auth_service.get_authorize_url(
            provider_name="self-contained",
            client_id="test-client-id",
            redirect_uri="http://localhost:8000/callback",
            scope=["openid", "profile", "email"],
            state="test-state",
            code_challenge="test-challenge",
            code_challenge_method="S256"
        )
        
        # Check that the provider's method was called with the right arguments
        provider.get_authorization_url.assert_called_once_with(
            "test-client-id",
            "http://localhost:8000/callback",
            ["openid", "profile", "email"],
            "test-state"
        )
        
        # Check that the URL was returned
        assert url == "https://example.com/authorize"
        
        # Check that the PKCE challenge was stored
        assert "test-state" in auth_service.pkce_challenges
        assert auth_service.pkce_challenges["test-state"]["code_challenge"] == "test-challenge"
        assert auth_service.pkce_challenges["test-state"]["code_challenge_method"] == "S256"
        
        # Test invalid provider
        with pytest.raises(ValueError):
            await auth_service.get_authorize_url(
                provider_name="invalid-provider",
                client_id="test-client-id",
                redirect_uri="http://localhost:8000/callback",
                scope=["openid", "profile", "email"],
                state="test-state"
            )
    
    @pytest.mark.asyncio
    async def test_exchange_code_for_token(self, auth_service):
        """Test exchanging an auth code for a token."""
        # Create an auth code
        auth_service.auth_codes["test-code"] = {
            "client_id": "test-client-id",
            "state": "test-state"
        }
        
        # Prepare the code verifier and challenge
        import hashlib
        import base64
        code_verifier = "test-verifier"
        digest = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        
        # Store a PKCE challenge that matches our verifier
        auth_service.pkce_challenges["test-state"] = {
            "code_challenge": code_challenge,
            "code_challenge_method": "S256"
        }
        
        # Mock the provider's exchange_code_for_token method
        provider = await auth_service.get_provider("self-contained")
        provider.exchange_code_for_token = AsyncMock(return_value={
            "access_token": "test-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "test-refresh-token"
        })
        
        # Exchange the auth code for a token with PKCE
        token_response = await auth_service.exchange_code_for_token(
            provider_name="self-contained",
            code="test-code",
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/callback",
            code_verifier=code_verifier
        )
        
        # Check that the provider's method was called with the right arguments
        provider.exchange_code_for_token.assert_called_once_with(
            "test-code",
            "test-client-id",
            "test-client-secret",
            "http://localhost:8000/callback"
        )
        
        # Check the token response
        assert token_response["access_token"] == "test-access-token"
        assert token_response["token_type"] == "Bearer"
        assert token_response["expires_in"] == 3600
        assert token_response["refresh_token"] == "test-refresh-token"
        
        # Check that the auth code and PKCE challenge were removed
        assert "test-code" not in auth_service.auth_codes
        assert "test-state" not in auth_service.pkce_challenges
        
        # Check that the token was stored
        assert "test-access-token" in auth_service.tokens
        assert auth_service.tokens["test-access-token"]["provider"] == "self-contained"
        assert auth_service.tokens["test-access-token"]["client_id"] == "test-client-id"
        
        # Test invalid code
        with pytest.raises(ValueError):
            await auth_service.exchange_code_for_token(
                provider_name="self-contained",
                code="invalid-code",
                client_id="test-client-id",
                client_secret="test-client-secret",
                redirect_uri="http://localhost:8000/callback"
            )
        
        # Test client ID mismatch
        auth_service.auth_codes["test-code-2"] = {
            "client_id": "other-client-id",
            "state": "test-state-2"
        }
        
        with pytest.raises(ValueError):
            await auth_service.exchange_code_for_token(
                provider_name="self-contained",
                code="test-code-2",
                client_id="test-client-id",
                client_secret="test-client-secret",
                redirect_uri="http://localhost:8000/callback"
            )
        
        # Test invalid PKCE verifier
        auth_service.auth_codes["test-code-3"] = {
            "client_id": "test-client-id",
            "state": "test-state-3"
        }
        
        auth_service.pkce_challenges["test-state-3"] = {
            "code_challenge": "different-challenge",
            "code_challenge_method": "plain"
        }
        
        with pytest.raises(ValueError):
            await auth_service.exchange_code_for_token(
                provider_name="self-contained",
                code="test-code-3",
                client_id="test-client-id",
                client_secret="test-client-secret",
                redirect_uri="http://localhost:8000/callback",
                code_verifier="different-verifier"
            )
    
    @pytest.mark.asyncio
    async def test_validate_token(self, auth_service):
        """Test validating a token."""
        # Store a token
        auth_service.tokens["test-token"] = {
            "provider": "self-contained",
            "expires_at": int(time.time()) + 3600
        }
        
        # Mock the provider's validate_token method
        provider = await auth_service.get_provider("self-contained")
        provider.validate_token = AsyncMock(return_value=(True, {"username": "testuser"}))
        
        # Validate the token
        is_valid, user_data = await auth_service.validate_token("test-token")
        
        # Check that the provider's method was called
        provider.validate_token.assert_called_once_with("test-token")
        
        # Check the result
        assert is_valid
        assert user_data["username"] == "testuser"
        
        # Test token not found
        is_valid, user_data = await auth_service.validate_token("invalid-token")
        
        assert not is_valid
        assert user_data is None
        
        # Test expired token
        auth_service.tokens["expired-token"] = {
            "provider": "self-contained",
            "expires_at": int(time.time()) - 3600
        }
        
        is_valid, user_data = await auth_service.validate_token("expired-token")
        
        assert not is_valid
        assert user_data is None
        
        # Test invalid provider
        auth_service.tokens["invalid-provider-token"] = {
            "provider": "invalid-provider",
            "expires_at": int(time.time()) + 3600
        }
        
        is_valid, user_data = await auth_service.validate_token("invalid-provider-token")
        
        assert not is_valid
        assert user_data is None
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, auth_service):
        """Test refreshing a token."""
        # Mock the provider's refresh_token method
        provider = await auth_service.get_provider("self-contained")
        provider.refresh_token = AsyncMock(return_value={
            "access_token": "new-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "new-refresh-token"
        })
        
        # Refresh the token
        token_response = await auth_service.refresh_token(
            provider_name="self-contained",
            refresh_token="test-refresh-token"
        )
        
        # Check that the provider's method was called
        provider.refresh_token.assert_called_once_with("test-refresh-token")
        
        # Check the token response
        assert token_response["access_token"] == "new-access-token"
        assert token_response["token_type"] == "Bearer"
        assert token_response["expires_in"] == 3600
        assert token_response["refresh_token"] == "new-refresh-token"
        
        # Check that the token was stored
        assert "new-access-token" in auth_service.tokens
        assert auth_service.tokens["new-access-token"]["provider"] == "self-contained"
        
        # Test invalid provider
        with pytest.raises(ValueError):
            await auth_service.refresh_token(
                provider_name="invalid-provider",
                refresh_token="test-refresh-token"
            )
    
    @pytest.mark.asyncio
    async def test_revoke_token(self, auth_service):
        """Test revoking a token."""
        # Store a token
        auth_service.tokens["test-token"] = {
            "provider": "self-contained"
        }
        
        # Mock the provider's revoke_token method
        provider = await auth_service.get_provider("self-contained")
        provider.revoke_token = AsyncMock(return_value=True)
        
        # Revoke the token
        result = await auth_service.revoke_token("test-token")
        
        # Check that the provider's method was called
        provider.revoke_token.assert_called_once_with("test-token")
        
        # Check the result
        assert result
        
        # Check that the token was removed
        assert "test-token" not in auth_service.tokens
        
        # Test token not found
        result = await auth_service.revoke_token("invalid-token")
        
        assert not result
        
        # Test provider not found
        auth_service.tokens["invalid-provider-token"] = {
            "provider": "invalid-provider"
        }
        
        result = await auth_service.revoke_token("invalid-provider-token")
        
        assert not result
    
    @pytest.mark.asyncio
    async def test_register_client(self, auth_service):
        """Test registering a client."""
        # Mock the provider's register_client method
        provider = await auth_service.get_provider("self-contained")
        provider.register_client = AsyncMock(return_value={
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
            "client_name": "Test Client",
            "redirect_uris": ["http://localhost:8000/callback"]
        })
        
        # Register a client
        client_data = await auth_service.register_client(
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
        
        # Check that the provider's method was called
        provider.register_client.assert_called_once_with(
            "Test Client",
            ["http://localhost:8000/callback"]
        )
        
        # Check the client data
        assert client_data["client_id"] == "test-client-id"
        assert client_data["client_secret"] == "test-client-secret"
        assert client_data["client_name"] == "Test Client"
        assert client_data["redirect_uris"] == ["http://localhost:8000/callback"]
        assert client_data["provider"] == "self-contained"
        assert client_data["client_uri"] == "http://localhost:8000"
        assert client_data["logo_uri"] == "http://localhost:8000/logo.png"
        assert client_data["tos_uri"] == "http://localhost:8000/tos"
        assert client_data["policy_uri"] == "http://localhost:8000/policy"
        assert client_data["software_id"] == "test-software-id"
        assert client_data["software_version"] == "1.0.0"
        assert "created_at" in client_data
        
        # Check that the client was stored
        assert "test-client-id" in auth_service.clients
        
        # Test invalid provider
        with pytest.raises(ValueError):
            await auth_service.register_client(
                provider_name="invalid-provider",
                client_name="Test Client",
                redirect_uris=["http://localhost:8000/callback"]
            )
    
    @pytest.mark.asyncio
    async def test_create_auth_code(self, auth_service):
        """Test creating an auth code."""
        # Register a provider-specific create_auth_code method for self-contained provider
        provider = await auth_service.get_provider("self-contained")
        provider.create_auth_code = AsyncMock(return_value="provider-specific-code")
        
        # Create an auth code for self-contained provider
        code = await auth_service.create_auth_code(
            provider_name="self-contained",
            client_id="test-client-id",
            redirect_uri="http://localhost:8000/callback",
            scope=["openid", "profile", "email"],
            state="test-state",
            user_data={"username": "testuser"}
        )
        
        # Check that the provider's method was called
        provider.create_auth_code.assert_called_once_with(
            {"username": "testuser"},
            "test-client-id"
        )
        
        # Check the code
        assert code == "provider-specific-code"
        
        # Create an auth code for another provider
        github_provider = await auth_service.get_provider("github")
        # github_provider doesn't have create_auth_code method
        
        code = await auth_service.create_auth_code(
            provider_name="github",
            client_id="test-client-id",
            redirect_uri="http://localhost:8000/callback",
            scope=["openid", "profile", "email"],
            state="test-state",
            user_data={"username": "testuser"}
        )
        
        # Check the code
        assert code
        assert code in auth_service.auth_codes
        assert auth_service.auth_codes[code]["provider"] == "github"
        assert auth_service.auth_codes[code]["client_id"] == "test-client-id"
        assert auth_service.auth_codes[code]["redirect_uri"] == "http://localhost:8000/callback"
        assert auth_service.auth_codes[code]["scope"] == "openid profile email"
        assert auth_service.auth_codes[code]["state"] == "test-state"
        assert auth_service.auth_codes[code]["user_data"] == {"username": "testuser"}
        assert "created_at" in auth_service.auth_codes[code]
        assert "expires_at" in auth_service.auth_codes[code]
        
        # Test invalid provider
        with pytest.raises(ValueError):
            await auth_service.create_auth_code(
                provider_name="invalid-provider",
                client_id="test-client-id",
                redirect_uri="http://localhost:8000/callback",
                scope=["openid", "profile", "email"],
                state="test-state"
            )
    
    @pytest.mark.asyncio
    async def test_get_authorization_server_metadata(self, auth_service):
        """Test getting the authorization server metadata."""
        # Get the metadata
        metadata = await auth_service.get_authorization_server_metadata("https://example.com")
        
        # Check the metadata
        assert metadata["issuer"] == "https://example.com"
        assert metadata["authorization_endpoint"] == "https://example.com/authorize"
        assert metadata["token_endpoint"] == "https://example.com/token"
        assert metadata["jwks_uri"] == "https://example.com/.well-known/jwks.json"
        assert metadata["registration_endpoint"] == "https://example.com/register"
        assert "scopes_supported" in metadata
        assert "response_types_supported" in metadata
        assert "response_modes_supported" in metadata
        assert "grant_types_supported" in metadata
        assert "token_endpoint_auth_methods_supported" in metadata
        assert "token_endpoint_auth_signing_alg_values_supported" in metadata
        assert "service_documentation" in metadata
        assert "ui_locales_supported" in metadata
        assert "op_policy_uri" in metadata
        assert "op_tos_uri" in metadata
        assert "revocation_endpoint" in metadata
        assert "revocation_endpoint_auth_methods_supported" in metadata
        assert "code_challenge_methods_supported" in metadata
        assert "provider_types_supported" in metadata
        assert "self-contained" in metadata["provider_types_supported"]
        assert "github" in metadata["provider_types_supported"]
    
    @pytest.mark.asyncio
    async def test_handle_authorize_request(self, auth_service):
        """Test handling an authorization request."""
        # Add a client
        auth_service.clients["test-client-id"] = {
            "redirect_uris": ["http://localhost:8000/callback"]
        }
        
        # Mock the provider's get_authorization_url method
        provider = await auth_service.get_provider("self-contained")
        provider.get_authorization_url = AsyncMock(return_value="https://example.com/authorize")
        
        # Handle an authorization request
        response = await auth_service.handle_authorize_request(
            provider_name="self-contained",
            client_id="test-client-id",
            redirect_uri="http://localhost:8000/callback",
            scope="openid profile email",
            response_type="code",
            state="test-state",
            code_challenge="test-challenge",
            code_challenge_method="S256"
        )
        
        # Check that the provider's method was called
        provider.get_authorization_url.assert_called_once_with(
            "test-client-id",
            "http://localhost:8000/callback",
            ["openid", "profile", "email"],
            "test-state"
        )
        
        # Check the response
        assert "redirect_to" in response
        assert response["redirect_to"] == "https://example.com/authorize"
        assert response["state"] == "test-state"
        
        # Check that the PKCE challenge was stored
        assert "test-state" in auth_service.pkce_challenges
        assert auth_service.pkce_challenges["test-state"]["code_challenge"] == "test-challenge"
        assert auth_service.pkce_challenges["test-state"]["code_challenge_method"] == "S256"
        
        # Test invalid provider
        response = await auth_service.handle_authorize_request(
            provider_name="invalid-provider",
            client_id="test-client-id",
            redirect_uri="http://localhost:8000/callback",
            scope="openid profile email",
            response_type="code"
        )
        
        assert "error" in response
        assert response["error"] == "invalid_request"
        
        # Test client not found
        response = await auth_service.handle_authorize_request(
            provider_name="self-contained",
            client_id="invalid-client-id",
            redirect_uri="http://localhost:8000/callback",
            scope="openid profile email",
            response_type="code"
        )
        
        assert "error" in response
        assert response["error"] == "invalid_client"
        
        # Test invalid redirect URI
        response = await auth_service.handle_authorize_request(
            provider_name="self-contained",
            client_id="test-client-id",
            redirect_uri="http://example.com/callback",
            scope="openid profile email",
            response_type="code"
        )
        
        assert "error" in response
        assert response["error"] == "invalid_request"
        
        # Test unsupported response type
        response = await auth_service.handle_authorize_request(
            provider_name="self-contained",
            client_id="test-client-id",
            redirect_uri="http://localhost:8000/callback",
            scope="openid profile email",
            response_type="token"
        )
        
        assert "error" in response
        assert response["error"] == "unsupported_response_type"
    
    @pytest.mark.asyncio
    async def test_handle_token_request(self, auth_service):
        """Test handling a token request."""
        # Mock the exchange_code_for_token method
        auth_service.exchange_code_for_token = AsyncMock(return_value={
            "access_token": "test-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "test-refresh-token"
        })
        
        # Mock the refresh_token method
        auth_service.refresh_token = AsyncMock(return_value={
            "access_token": "new-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "new-refresh-token"
        })
        
        # Handle an authorization_code token request
        response = await auth_service.handle_token_request(
            grant_type="authorization_code",
            provider_name="self-contained",
            code="test-code",
            redirect_uri="http://localhost:8000/callback",
            client_id="test-client-id",
            client_secret="test-client-secret",
            code_verifier="test-verifier"
        )
        
        # Check that the exchange_code_for_token method was called
        auth_service.exchange_code_for_token.assert_called_once_with(
            "self-contained",
            "test-code",
            "test-client-id",
            "test-client-secret",
            "http://localhost:8000/callback",
            "test-verifier"
        )
        
        # Check the response
        assert response["access_token"] == "test-access-token"
        assert response["token_type"] == "Bearer"
        assert response["expires_in"] == 3600
        assert response["refresh_token"] == "test-refresh-token"
        
        # Handle a refresh_token token request
        response = await auth_service.handle_token_request(
            grant_type="refresh_token",
            provider_name="self-contained",
            refresh_token="test-refresh-token",
            client_id="test-client-id"
        )
        
        # Check that the refresh_token method was called
        auth_service.refresh_token.assert_called_once_with(
            "self-contained",
            "test-refresh-token"
        )
        
        # Check the response
        assert response["access_token"] == "new-access-token"
        assert response["token_type"] == "Bearer"
        assert response["expires_in"] == 3600
        assert response["refresh_token"] == "new-refresh-token"
        
        # Test missing parameters for authorization_code
        response = await auth_service.handle_token_request(
            grant_type="authorization_code",
            provider_name="self-contained"
        )
        
        assert "error" in response
        assert response["error"] == "invalid_request"
        
        # Test missing parameters for refresh_token
        response = await auth_service.handle_token_request(
            grant_type="refresh_token",
            provider_name="self-contained"
        )
        
        assert "error" in response
        assert response["error"] == "invalid_request"
        
        # Test unsupported grant type
        response = await auth_service.handle_token_request(
            grant_type="password",
            provider_name="self-contained"
        )
        
        assert "error" in response
        assert response["error"] == "unsupported_grant_type"