"""Authentication providers for MCP Agent Cloud.

This module provides authentication provider classes for different authentication methods,
including self-contained, OAuth, and custom providers.
"""

import os
import time
import uuid
import json
import secrets
import hashlib
import base64
import logging
import httpx
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta

# Try to import jwt, but make it optional
try:
    import jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False
    
    # Simple JWT implementation for demo purposes
    class SimpleJWT:
        @staticmethod
        def encode(payload, secret, algorithm=None):
            """Simple implementation of JWT encoding for demo purposes."""
            # In a real implementation, this would use actual JWT encoding
            payload_str = json.dumps(payload)
            encoded = base64.urlsafe_b64encode(payload_str.encode()).decode()
            # We're not actually signing this in the demo implementation
            return f"{encoded}.{secret[:10]}"
        
        @staticmethod
        def decode(token, secret, algorithms=None):
            """Simple implementation of JWT decoding for demo purposes."""
            # In a real implementation, this would verify the signature
            try:
                parts = token.split(".")
                if len(parts) != 2:
                    raise ValueError("Invalid token format")
                
                encoded_payload = parts[0]
                # Simple check of the "signature"
                if parts[1] != secret[:10]:
                    raise ValueError("Invalid signature")
                
                # Decode the payload
                payload_str = base64.urlsafe_b64decode(encoded_payload + "==").decode()
                return json.loads(payload_str)
            except Exception as e:
                # For simplicity in the demo
                class PyJWTError(Exception):
                    pass
                raise PyJWTError(str(e))
    
    # Use our simple implementation
    jwt = SimpleJWT

# Configure logging
logger = logging.getLogger(__name__)

# Constants for JWT token generation
TOKEN_EXPIRY = 60 * 60 * 24  # 24 hours in seconds
REFRESH_TOKEN_EXPIRY = 60 * 60 * 24 * 30  # 30 days in seconds


class AuthProvider(ABC):
    """Base class for authentication providers."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize the auth provider.
        
        Args:
            name: Name of the provider
            config: Configuration for the provider
        """
        self.name = name
        self.config = config or {}
        self.client_secrets = {}  # Map of client_id -> client_secret
        self.auth_codes = {}  # Map of client_id -> auth_code
        self.tokens = {}  # Map of access_token -> user_data
        
    @abstractmethod
    async def get_authorization_url(self, client_id: str, redirect_uri: str, scope: List[str], state: str) -> str:
        """Get the authorization URL.
        
        Args:
            client_id: Client ID
            redirect_uri: Redirect URI after authorization
            scope: Requested scopes
            state: State parameter for CSRF protection
            
        Returns:
            Authorization URL
        """
        pass
    
    @abstractmethod
    async def exchange_code_for_token(self, code: str, client_id: str, client_secret: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange an authorization code for a token.
        
        Args:
            code: Authorization code
            client_id: Client ID
            client_secret: Client secret
            redirect_uri: Redirect URI
            
        Returns:
            Token response
        """
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate an access token.
        
        Args:
            token: Access token to validate
            
        Returns:
            Tuple of (is_valid, user_data)
        """
        pass
    
    @abstractmethod
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh an access token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            Token response
        """
        pass
    
    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """Revoke an access token.
        
        Args:
            token: Access token to revoke
            
        Returns:
            True if successful
        """
        pass
    
    async def register_client(self, client_name: str, redirect_uris: List[str]) -> Dict[str, Any]:
        """Register a new client.
        
        Args:
            client_name: Name of the client
            redirect_uris: Allowed redirect URIs
            
        Returns:
            Client registration data
        """
        # Generate client ID and secret
        client_id = str(uuid.uuid4())
        client_secret = secrets.token_hex(32)
        
        # Store client secret
        self.client_secrets[client_id] = client_secret
        
        return {
            "client_id": client_id,
            "client_secret": client_secret,
            "client_name": client_name,
            "redirect_uris": redirect_uris,
            "registration_time": int(time.time())
        }
        
    async def create_auth_code(self, user_data: Dict[str, Any], client_id: str) -> str:
        """Create an authorization code for a user.
        
        Args:
            user_data: User data
            client_id: Client ID
            
        Returns:
            Authorization code
        """
        # Generate a random code
        code = secrets.token_hex(16)
        
        # Store the code with user data and client ID
        self.auth_codes[code] = {
            "client_id": client_id,
            "user_data": user_data,
            "created_at": int(time.time()),
            "expires_at": int(time.time()) + 600  # 10 minutes
        }
        
        return code


class SelfContainedAuthProvider(AuthProvider):
    """Self-contained authentication provider that manages its own users and tokens."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the self-contained auth provider.
        
        Args:
            config: Configuration for the provider
        """
        super().__init__("self-contained", config)
        self.users = {}  # Map of username -> user_data
        self.jwt_secret = config.get("jwt_secret", secrets.token_hex(32))
        
        # Create a default user if specified in config
        default_user = config.get("default_user")
        if default_user:
            username = default_user.get("username", "admin")
            password = default_user.get("password", secrets.token_hex(8))
            self.users[username] = {
                "username": username,
                "password_hash": self._hash_password(password),
                "email": default_user.get("email", f"{username}@example.com"),
                "role": default_user.get("role", "admin"),
                "created_at": int(time.time())
            }
    
    def _hash_password(self, password: str) -> str:
        """Hash a password.
        
        Args:
            password: Password to hash
            
        Returns:
            Hashed password
        """
        # In a real implementation, this would use a proper password hashing algorithm
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password.
        
        Args:
            password: Password to verify
            password_hash: Hashed password
            
        Returns:
            True if password is correct
        """
        return self._hash_password(password) == password_hash
    
    async def get_authorization_url(self, client_id: str, redirect_uri: str, scope: List[str], state: str) -> str:
        """Get the authorization URL.
        
        Args:
            client_id: Client ID
            redirect_uri: Redirect URI after authorization
            scope: Requested scopes
            state: State parameter for CSRF protection
            
        Returns:
            Authorization URL
        """
        # In a real implementation, this would typically return a URL to a login page
        # For demo purposes, we'll return a dummy URL
        return f"https://auth.mcp-agent-cloud.example.com/login?client_id={client_id}&redirect_uri={redirect_uri}&state={state}&scope={'+'.join(scope)}"
    
    async def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate a user with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Tuple of (is_authenticated, user_data)
        """
        user_data = self.users.get(username)
        if not user_data:
            return False, None
        
        if not self._verify_password(password, user_data.get("password_hash", "")):
            return False, None
        
        # Return user data without password hash
        user_data_without_password = user_data.copy()
        user_data_without_password.pop("password_hash", None)
        
        return True, user_data_without_password
    
    async def exchange_code_for_token(self, code: str, client_id: str, client_secret: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange an authorization code for a token.
        
        Args:
            code: Authorization code
            client_id: Client ID
            client_secret: Client secret
            redirect_uri: Redirect URI
            
        Returns:
            Token response
        """
        # Check if the code is valid
        code_data = self.auth_codes.get(code)
        if not code_data:
            raise ValueError("Invalid authorization code")
        
        # Check if the code has expired
        if code_data.get("expires_at", 0) < int(time.time()):
            raise ValueError("Authorization code has expired")
        
        # Check if the client ID matches
        if code_data.get("client_id") != client_id:
            raise ValueError("Client ID mismatch")
        
        # Check if the client secret is valid
        if self.client_secrets.get(client_id) != client_secret:
            raise ValueError("Invalid client secret")
        
        # Generate tokens
        user_data = code_data.get("user_data", {})
        access_token = self._generate_access_token(user_data)
        refresh_token = self._generate_refresh_token(user_data)
        
        # Store access token
        self.tokens[access_token] = {
            "user_data": user_data,
            "created_at": int(time.time()),
            "expires_at": int(time.time()) + TOKEN_EXPIRY
        }
        
        # Remove the used code
        self.auth_codes.pop(code, None)
        
        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": TOKEN_EXPIRY,
            "refresh_token": refresh_token,
            "scope": "read write"
        }
    
    def _generate_access_token(self, user_data: Dict[str, Any]) -> str:
        """Generate an access token for a user.
        
        Args:
            user_data: User data
            
        Returns:
            Access token
        """
        now = int(time.time())
        payload = {
            "sub": user_data.get("username", f"user-{uuid.uuid4().hex[:8]}"),
            "iat": now,
            "exp": now + TOKEN_EXPIRY,
            "role": user_data.get("role", "user"),
            "scope": "read write"
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def _generate_refresh_token(self, user_data: Dict[str, Any]) -> str:
        """Generate a refresh token for a user.
        
        Args:
            user_data: User data
            
        Returns:
            Refresh token
        """
        now = int(time.time())
        payload = {
            "sub": user_data.get("username", f"user-{uuid.uuid4().hex[:8]}"),
            "iat": now,
            "exp": now + REFRESH_TOKEN_EXPIRY,
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    async def validate_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate an access token.
        
        Args:
            token: Access token to validate
            
        Returns:
            Tuple of (is_valid, user_data)
        """
        try:
            # Decode the token
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if token has expired
            exp = payload.get("exp", 0)
            if exp < int(time.time()):
                return False, None
            
            # Get username from token
            username = payload.get("sub")
            if not username:
                return False, None
            
            # Get user data
            user_data = self.users.get(username)
            if not user_data:
                return False, None
            
            # Return user data without password hash
            user_data_without_password = user_data.copy()
            user_data_without_password.pop("password_hash", None)
            
            return True, user_data_without_password
        except Exception as e:
            logger.error(f"Error validating token: {str(e)}")
            return False, None
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh an access token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            Token response
        """
        try:
            # Decode the refresh token
            payload = jwt.decode(refresh_token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if token has expired
            exp = payload.get("exp", 0)
            if exp < int(time.time()):
                raise ValueError("Refresh token has expired")
            
            # Check if token is a refresh token
            token_type = payload.get("type")
            if token_type != "refresh":
                raise ValueError("Not a refresh token")
            
            # Get username from token
            username = payload.get("sub")
            if not username:
                raise ValueError("Invalid token")
            
            # Get user data
            user_data = self.users.get(username)
            if not user_data:
                raise ValueError("User not found")
            
            # Generate new tokens
            access_token = self._generate_access_token(user_data)
            new_refresh_token = self._generate_refresh_token(user_data)
            
            # Store access token
            self.tokens[access_token] = {
                "user_data": user_data,
                "created_at": int(time.time()),
                "expires_at": int(time.time()) + TOKEN_EXPIRY
            }
            
            return {
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": TOKEN_EXPIRY,
                "refresh_token": new_refresh_token,
                "scope": "read write"
            }
        except Exception as e:
            logger.error(f"Error refreshing token: {str(e)}")
            raise ValueError(f"Error refreshing token: {str(e)}")
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke an access token.
        
        Args:
            token: Access token to revoke
            
        Returns:
            True if successful
        """
        # Remove the token from storage
        if token in self.tokens:
            self.tokens.pop(token, None)
            return True
        return False
    
    async def create_user(self, username: str, password: str, email: str = None, role: str = "user") -> Dict[str, Any]:
        """Create a new user.
        
        Args:
            username: Username
            password: Password
            email: Email address
            role: User role
            
        Returns:
            User data without password hash
        """
        # Check if user already exists
        if username in self.users:
            raise ValueError(f"User {username} already exists")
        
        # Create the user
        user_data = {
            "username": username,
            "password_hash": self._hash_password(password),
            "email": email or f"{username}@example.com",
            "role": role,
            "created_at": int(time.time())
        }
        
        # Store the user
        self.users[username] = user_data
        
        # Return user data without password hash
        user_data_without_password = user_data.copy()
        user_data_without_password.pop("password_hash", None)
        
        return user_data_without_password


class OAuthProvider(AuthProvider):
    """Base class for OAuth providers."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize the OAuth provider.
        
        Args:
            name: Name of the provider
            config: Configuration for the provider
        """
        super().__init__(name, config)
        
        # OAuth endpoints
        self.authorize_url = config.get("authorize_url")
        self.token_url = config.get("token_url")
        self.userinfo_url = config.get("userinfo_url")
        self.revoke_url = config.get("revoke_url")
        
        # OAuth client
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
        
        if self.client_id and self.client_secret:
            self.client_secrets[self.client_id] = self.client_secret
    
    async def get_authorization_url(self, client_id: str, redirect_uri: str, scope: List[str], state: str) -> str:
        """Get the authorization URL.
        
        Args:
            client_id: Client ID
            redirect_uri: Redirect URI after authorization
            scope: Requested scopes
            state: State parameter for CSRF protection
            
        Returns:
            Authorization URL
        """
        if not self.authorize_url:
            raise ValueError("Authorization URL not configured")
        
        # Build the authorization URL
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(scope),
            "state": state,
            "response_type": "code"
        }
        
        # Convert params to query string
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        
        return f"{self.authorize_url}?{query_string}"
    
    async def exchange_code_for_token(self, code: str, client_id: str, client_secret: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange an authorization code for a token.
        
        Args:
            code: Authorization code
            client_id: Client ID
            client_secret: Client secret
            redirect_uri: Redirect URI
            
        Returns:
            Token response
        """
        if not self.token_url:
            raise ValueError("Token URL not configured")
        
        # Data for token request
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.token_url, data=data)
                
                if response.status_code != 200:
                    raise ValueError(f"Error exchanging code: {response.text}")
                
                token_data = response.json()
                
                # Store token data
                access_token = token_data.get("access_token")
                if access_token:
                    # Get user info
                    user_data = await self._get_user_info(access_token)
                    
                    # Store token with user data
                    self.tokens[access_token] = {
                        "user_data": user_data,
                        "created_at": int(time.time()),
                        "expires_at": int(time.time()) + token_data.get("expires_in", TOKEN_EXPIRY)
                    }
                
                return token_data
        except httpx.RequestError as e:
            logger.error(f"Error exchanging code: {str(e)}")
            raise ValueError(f"Error exchanging code: {str(e)}")
    
    async def _get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information using an access token.
        
        Args:
            access_token: Access token
            
        Returns:
            User information
        """
        if not self.userinfo_url:
            return {}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.userinfo_url,
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                if response.status_code != 200:
                    logger.error(f"Error getting user info: {response.text}")
                    return {}
                
                return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error getting user info: {str(e)}")
            return {}
    
    async def validate_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate an access token.
        
        Args:
            token: Access token to validate
            
        Returns:
            Tuple of (is_valid, user_data)
        """
        # Check if we have the token in our local storage
        token_data = self.tokens.get(token)
        if token_data:
            # Check if token has expired
            if token_data.get("expires_at", 0) < int(time.time()):
                return False, None
            
            return True, token_data.get("user_data", {})
        
        # If not in local storage, try to get user info
        user_data = await self._get_user_info(token)
        if user_data:
            # Store token with user data
            self.tokens[token] = {
                "user_data": user_data,
                "created_at": int(time.time()),
                "expires_at": int(time.time()) + TOKEN_EXPIRY
            }
            
            return True, user_data
        
        return False, None
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh an access token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            Token response
        """
        if not self.token_url:
            raise ValueError("Token URL not configured")
        
        # Data for refresh request
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.token_url, data=data)
                
                if response.status_code != 200:
                    raise ValueError(f"Error refreshing token: {response.text}")
                
                token_data = response.json()
                
                # Store token data
                access_token = token_data.get("access_token")
                if access_token:
                    # Get user info
                    user_data = await self._get_user_info(access_token)
                    
                    # Store token with user data
                    self.tokens[access_token] = {
                        "user_data": user_data,
                        "created_at": int(time.time()),
                        "expires_at": int(time.time()) + token_data.get("expires_in", TOKEN_EXPIRY)
                    }
                
                return token_data
        except httpx.RequestError as e:
            logger.error(f"Error refreshing token: {str(e)}")
            raise ValueError(f"Error refreshing token: {str(e)}")
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke an access token.
        
        Args:
            token: Access token to revoke
            
        Returns:
            True if successful
        """
        # Remove from local storage
        if token in self.tokens:
            self.tokens.pop(token, None)
        
        # If revoke URL is configured, revoke the token
        if self.revoke_url:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.revoke_url,
                        data={
                            "token": token,
                            "token_type_hint": "access_token",
                            "client_id": self.client_id,
                            "client_secret": self.client_secret
                        }
                    )
                    
                    return response.status_code in (200, 204)
            except httpx.RequestError as e:
                logger.error(f"Error revoking token: {str(e)}")
                return False
        
        return True


class GitHubOAuthProvider(OAuthProvider):
    """GitHub OAuth provider."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the GitHub OAuth provider.
        
        Args:
            config: Configuration for the provider
        """
        config = config or {}
        
        # Set GitHub OAuth endpoints
        config.setdefault("authorize_url", "https://github.com/login/oauth/authorize")
        config.setdefault("token_url", "https://github.com/login/oauth/access_token")
        config.setdefault("userinfo_url", "https://api.github.com/user")
        
        super().__init__("github", config)


class CustomOAuthProvider(OAuthProvider):
    """Custom OAuth provider."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the custom OAuth provider.
        
        Args:
            config: Configuration for the provider
        """
        super().__init__("custom", config)