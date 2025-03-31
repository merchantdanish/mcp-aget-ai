"""Unit tests for auth providers."""

import pytest
import asyncio
import time
import secrets
from unittest.mock import Mock, AsyncMock

from mcp_agent.auth.auth_provider import (
    AuthProvider,
    SelfContainedAuthProvider,
    OAuthProvider,
    GitHubOAuthProvider,
    CustomOAuthProvider,
)

class TestSelfContainedAuthProvider:
    """Test the self-contained auth provider."""
    
    @pytest.mark.asyncio
    async def test_create_user(self, self_contained_provider):
        """Test creating a user."""
        # Create a user
        user_data = await self_contained_provider.create_user(
            username="testuser",
            password="testpassword",
            user_data={"email": "test@example.com"}
        )
        
        # Check that the user was created
        assert user_data["username"] == "testuser"
        assert "password_hash" in user_data
        assert "password_salt" in user_data
        assert user_data["email"] == "test@example.com"
        assert "id" in user_data
        assert "created_at" in user_data
    
    @pytest.mark.asyncio
    async def test_authenticate_user(self, self_contained_provider):
        """Test authenticating a user."""
        # Create a user
        await self_contained_provider.create_user(
            username="testuser",
            password="testpassword"
        )
        
        # Test valid authentication
        is_valid, user_data = await self_contained_provider.authenticate_user(
            username="testuser",
            password="testpassword"
        )
        
        assert is_valid
        assert user_data["username"] == "testuser"
        
        # Test invalid username
        is_valid, user_data = await self_contained_provider.authenticate_user(
            username="wronguser",
            password="testpassword"
        )
        
        assert not is_valid
        assert user_data is None
        
        # Test invalid password
        is_valid, user_data = await self_contained_provider.authenticate_user(
            username="testuser",
            password="wrongpassword"
        )
        
        assert not is_valid
        assert user_data is None
    
    @pytest.mark.asyncio
    async def test_create_auth_code(self, self_contained_provider):
        """Test creating an auth code."""
        # Create a user
        user_data = await self_contained_provider.create_user(
            username="testuser",
            password="testpassword"
        )
        
        # Create an auth code
        auth_code = await self_contained_provider.create_auth_code(
            user_data=user_data,
            client_id="test-client-id"
        )
        
        # Check that the auth code was created
        assert auth_code
        assert auth_code in self_contained_provider.auth_codes
        assert self_contained_provider.auth_codes[auth_code]["client_id"] == "test-client-id"
        assert self_contained_provider.auth_codes[auth_code]["user_data"] == user_data
    
    @pytest.mark.asyncio
    async def test_exchange_code_for_token(self, self_contained_provider):
        """Test exchanging an auth code for a token."""
        # Create a user
        user_data = await self_contained_provider.create_user(
            username="testuser",
            password="testpassword"
        )
        
        # Register a client
        client_data = await self_contained_provider.register_client(
            client_name="Test Client",
            redirect_uris=["http://localhost:8000/callback"]
        )
        
        client_id = client_data["client_id"]
        client_secret = client_data["client_secret"]
        
        # Create an auth code
        auth_code = await self_contained_provider.create_auth_code(
            user_data=user_data,
            client_id=client_id
        )
        
        # Exchange the auth code for a token
        token_response = await self_contained_provider.exchange_code_for_token(
            code=auth_code,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri="http://localhost:8000/callback"
        )
        
        # Check the token response
        assert "access_token" in token_response
        assert token_response["token_type"] == "Bearer"
        assert "expires_in" in token_response
        assert "refresh_token" in token_response
        assert "user_id" in token_response
        
        # Verify that the auth code was removed
        assert auth_code not in self_contained_provider.auth_codes
    
    @pytest.mark.asyncio
    async def test_validate_token(self, self_contained_provider):
        """Test validating a token."""
        # Create a user
        user_data = await self_contained_provider.create_user(
            username="testuser",
            password="testpassword"
        )
        
        # Generate a token
        token = self_contained_provider.generate_jwt_token(
            user_data=user_data,
            secret=self_contained_provider.jwt_secret
        )
        
        # Add the token to the provider's tokens
        self_contained_provider.tokens[token] = {
            "user_data": user_data,
            "expires_at": int(time.time()) + 3600
        }
        
        # Validate the token
        is_valid, token_user_data = await self_contained_provider.validate_token(token)
        
        # Check the result
        assert is_valid
        assert token_user_data["username"] == user_data["username"]
        
        # Test invalid token
        is_valid, token_user_data = await self_contained_provider.validate_token("invalid-token")
        
        assert not is_valid
        assert token_user_data is None
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, self_contained_provider):
        """Test refreshing a token."""
        # Refresh a token
        token_response = await self_contained_provider.refresh_token("test-refresh-token")
        
        # Check the token response
        assert "access_token" in token_response
        assert token_response["token_type"] == "Bearer"
        assert "expires_in" in token_response
        assert "refresh_token" in token_response
    
    @pytest.mark.asyncio
    async def test_revoke_token(self, self_contained_provider):
        """Test revoking a token."""
        # Create a user
        user_data = await self_contained_provider.create_user(
            username="testuser",
            password="testpassword"
        )
        
        # Generate a token
        token = self_contained_provider.generate_jwt_token(
            user_data=user_data,
            secret=self_contained_provider.jwt_secret
        )
        
        # Add the token to the provider's tokens
        self_contained_provider.tokens[token] = {
            "user_data": user_data,
            "expires_at": int(time.time()) + 3600
        }
        
        # Revoke the token
        result = await self_contained_provider.revoke_token(token)
        
        # Check the result
        assert result
        assert token not in self_contained_provider.tokens
        
        # Test revoking an invalid token
        result = await self_contained_provider.revoke_token("invalid-token")
        
        assert not result
    
    @pytest.mark.asyncio
    async def test_register_client(self, self_contained_provider):
        """Test registering a client."""
        # Register a client
        client_data = await self_contained_provider.register_client(
            client_name="Test Client",
            redirect_uris=["http://localhost:8000/callback"]
        )
        
        # Check the client data
        assert "client_id" in client_data
        assert "client_secret" in client_data
        assert client_data["client_name"] == "Test Client"
        assert client_data["redirect_uris"] == ["http://localhost:8000/callback"]
    
    @pytest.mark.asyncio
    async def test_get_authorization_url(self, self_contained_provider):
        """Test getting an authorization URL."""
        # Get an authorization URL
        url = await self_contained_provider.get_authorization_url(
            client_id="test-client-id",
            redirect_uri="http://localhost:8000/callback",
            scope=["openid", "profile", "email"],
            state="test-state"
        )
        
        # Check the URL
        assert "client_id=test-client-id" in url
        assert "redirect_uri=http://localhost:8000/callback" in url
        assert "scope=openid%20profile%20email" in url
        assert "state=test-state" in url
        assert "response_type=code" in url


class TestGitHubOAuthProvider:
    """Test the GitHub OAuth provider."""
    
    @pytest.mark.asyncio
    async def test_exchange_code_for_token(self, github_provider):
        """Test exchanging an auth code for a token."""
        # Exchange the auth code for a token
        token_response = await github_provider.exchange_code_for_token(
            code="test-code",
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/callback"
        )
        
        # Check the token response
        assert "access_token" in token_response
        assert token_response["token_type"] == "Bearer"
        assert "scope" in token_response
        assert "refresh_token" in token_response
    
    @pytest.mark.asyncio
    async def test_validate_token(self, github_provider):
        """Test validating a token."""
        # Test valid token (starts with "gh-")
        is_valid, user_data = await github_provider.validate_token("gh-test-token")
        
        # Check the result
        assert is_valid
        assert user_data["id"] == "gh-user-1234"
        assert user_data["login"] == "github-user"
        assert user_data["name"] == "GitHub User"
        assert user_data["email"] == "user@example.com"
        
        # Test invalid token
        is_valid, user_data = await github_provider.validate_token("invalid-token")
        
        assert not is_valid
        assert user_data is None
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, github_provider):
        """Test refreshing a token."""
        # Refresh a token
        token_response = await github_provider.refresh_token("test-refresh-token")
        
        # Check the token response
        assert "access_token" in token_response
        assert token_response["token_type"] == "Bearer"
        assert "scope" in token_response
    
    @pytest.mark.asyncio
    async def test_get_authorization_url(self, github_provider):
        """Test getting an authorization URL."""
        # Get an authorization URL
        url = await github_provider.get_authorization_url(
            client_id="test-client-id",
            redirect_uri="http://localhost:8000/callback",
            scope=["openid", "profile", "email"],
            state="test-state"
        )
        
        # Check the URL
        assert github_provider.authorize_url in url
        assert "client_id=test-client-id" in url
        assert "redirect_uri=http://localhost:8000/callback" in url
        assert "scope=openid%20profile%20email" in url
        assert "state=test-state" in url
        assert "response_type=code" in url