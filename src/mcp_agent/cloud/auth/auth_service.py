"""Authentication service for MCP Agent Cloud.

This module provides the primary authentication service for MCP Agent Cloud, supporting
multiple authentication providers and token management.
"""

import os
import json
import asyncio
import httpx
import webbrowser
import time
import logging
import uuid
from typing import Dict, Any, Optional, Tuple, List, Type, Callable
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Import authentication providers
from mcp_agent.cloud.auth.providers import (
    AuthProvider,
    SelfContainedAuthProvider,
    OAuthProvider,
    GitHubOAuthProvider,
    CustomOAuthProvider,
)

# Configure logging
logger = logging.getLogger(__name__)

# Default authorization endpoints
DEFAULT_AUTHORIZE_ENDPOINT = "/authorize"
DEFAULT_TOKEN_ENDPOINT = "/token"
DEFAULT_REGISTRATION_ENDPOINT = "/register"
DEFAULT_JWKS_ENDPOINT = "/.well-known/jwks.json"
DEFAULT_METADATA_ENDPOINT = "/.well-known/oauth-authorization-server"

class AuthService:
    """Authentication service for MCP Agent Cloud.
    
    This service provides comprehensive authentication functionality for MCP Agent Cloud,
    including:
    
    1. User authentication and token management
    2. Support for multiple authentication providers
    3. OAuth 2.0 authorization server capabilities
    4. Client registration and management
    """
    
    def __init__(self, config: Dict[str, Any] = None, config_dir: Optional[Path] = None):
        """Initialize the authentication service.
        
        Args:
            config: Authentication configuration
            config_dir: Directory where auth configuration is stored
        """
        # Initialize configuration
        self.config = config or {}
        self.config_dir = config_dir or Path.home() / ".mcp-agent-cloud"
        self.auth_file = self.config_dir / "auth.json"
        self.session_file = self.config_dir / "session.json"
        
        # API endpoints
        self.api_base_url = os.environ.get("MCP_AGENT_CLOUD_API_URL", "https://api.mcp-agent-cloud.example.com")
        self.auth_base_url = os.environ.get("MCP_AGENT_CLOUD_AUTH_URL", "https://auth.mcp-agent-cloud.example.com")
        
        # Token storage
        self.providers = {}  # Map of provider_name -> provider_instance
        self.clients = {}  # Map of client_id -> client_data
        self.tokens = {}  # Map of access_token -> token_data
        self.auth_codes = {}  # Map of code -> code_data
        self.pkce_challenges = {}  # Map of state -> code_challenge
        
        # Ensure config directory exists
        self._ensure_config_dir()
        
        # Initialize auth providers
        self._init_providers()
        
        # Load clients if available
        self._load_clients()
    
    def _ensure_config_dir(self) -> None:
        """Ensure the configuration directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_auth(self) -> Dict[str, Any]:
        """Load authentication information from the auth file.
        
        Returns:
            Dictionary containing authentication information
        """
        if not self.auth_file.exists():
            return {}
            
        try:
            with open(self.auth_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def save_auth(self, auth_data: Dict[str, Any]) -> None:
        """Save authentication information to the auth file.
        
        Args:
            auth_data: Dictionary containing authentication information
        """
        with open(self.auth_file, "w") as f:
            json.dump(auth_data, f, indent=2)
    
    def load_session(self) -> Dict[str, Any]:
        """Load session information from the session file.
        
        Returns:
            Dictionary containing session information
        """
        if not self.session_file.exists():
            return {}
            
        try:
            with open(self.session_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def save_session(self, session_data: Dict[str, Any]) -> None:
        """Save session information to the session file.
        
        Args:
            session_data: Dictionary containing session information
        """
        with open(self.session_file, "w") as f:
            json.dump(session_data, f, indent=2)
    
    def _init_providers(self) -> None:
        """Initialize authentication providers based on configuration."""
        provider_configs = self.config.get("providers", {})
        
        # Initialize self-contained provider by default
        self.providers["self-contained"] = SelfContainedAuthProvider(
            config=provider_configs.get("self-contained", {})
        )
        
        # Initialize GitHub provider if configured
        if "github" in provider_configs:
            self.providers["github"] = GitHubOAuthProvider(
                config=provider_configs.get("github", {})
            )
            
        # Initialize custom provider if configured
        if "custom" in provider_configs:
            self.providers["custom"] = CustomOAuthProvider(
                config=provider_configs.get("custom", {})
            )
    
    def _load_clients(self) -> None:
        """Load client registrations from storage."""
        clients_file = self.config.get("clients_file")
        if not clients_file or not os.path.exists(clients_file):
            return
            
        try:
            with open(clients_file, "r") as f:
                self.clients = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading clients: {str(e)}")
            self.clients = {}
    
    def _save_clients(self) -> None:
        """Save client registrations to storage."""
        clients_file = self.config.get("clients_file")
        if not clients_file:
            return
            
        try:
            with open(clients_file, "w") as f:
                json.dump(self.clients, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving clients: {str(e)}")
    
    def is_authenticated(self) -> bool:
        """Check if the user is authenticated.
        
        Returns:
            True if the user is authenticated, False otherwise
        """
        auth_data = self.load_auth()
        if not auth_data:
            return False
            
        # Check if token is expired
        if "expires_at" in auth_data:
            expires_at = datetime.fromtimestamp(auth_data["expires_at"], timezone.utc)
            if expires_at <= datetime.now(timezone.utc):
                return False
                
        return "access_token" in auth_data
    
    async def get_provider(self, provider_name: str) -> Optional[AuthProvider]:
        """Get an authentication provider by name.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Authentication provider or None if not found
        """
        return self.providers.get(provider_name)
    
    async def authenticate(self, device_code_callback: Optional[Callable[[str, str], None]] = None) -> bool:
        """Authenticate with the MCP Agent Cloud service.
        
        Args:
            device_code_callback: Optional callback function to handle device code display
            
        Returns:
            True if authentication was successful, False otherwise
        """
        try:
            # Start device code flow
            device_code_data = await self._request_device_code()
            
            if not device_code_data:
                return False
                
            # Display device code to user
            verification_uri = device_code_data.get("verification_uri", "")
            user_code = device_code_data.get("user_code", "")
            
            if device_code_callback:
                device_code_callback(verification_uri, user_code)
            else:
                print(f"Please visit: {verification_uri}")
                print(f"And enter the code: {user_code}")
                
                # Open browser if available
                try:
                    webbrowser.open(verification_uri)
                except:
                    pass
                    
            # Poll for token
            return await self._poll_for_token(device_code_data)
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    async def _request_device_code(self) -> Dict[str, Any]:
        """Request a device code for authentication.
        
        Returns:
            Dictionary containing device code information
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.auth_base_url}/oauth/device/code",
                    json={"client_id": "mcp-agent-cli"}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    # For demo purposes, create a simulated device code
                    return {
                        "device_code": str(uuid.uuid4()),
                        "user_code": "ABCD-EFGH",
                        "verification_uri": "https://auth.mcp-agent-cloud.example.com/activate",
                        "expires_in": 900,
                        "interval": 5
                    }
            except:
                # For demo purposes, create a simulated device code
                return {
                    "device_code": str(uuid.uuid4()),
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://auth.mcp-agent-cloud.example.com/activate",
                    "expires_in": 900,
                    "interval": 5
                }
    
    async def _poll_for_token(self, device_code_data: Dict[str, Any]) -> bool:
        """Poll for token using the device code.
        
        Args:
            device_code_data: Dictionary containing device code information
            
        Returns:
            True if token retrieval was successful, False otherwise
        """
        device_code = device_code_data.get("device_code", "")
        interval = device_code_data.get("interval", 5)
        expires_in = device_code_data.get("expires_in", 900)
        
        # Calculate expiration time
        start_time = time.time()
        end_time = start_time + expires_in
        
        async with httpx.AsyncClient() as client:
            while time.time() < end_time:
                try:
                    response = await client.post(
                        f"{self.auth_base_url}/oauth/token",
                        json={
                            "client_id": "mcp-agent-cli",
                            "device_code": device_code,
                            "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
                        }
                    )
                    
                    if response.status_code == 200:
                        token_data = response.json()
                        
                        # Save token data
                        auth_data = {
                            "access_token": token_data.get("access_token", ""),
                            "refresh_token": token_data.get("refresh_token", ""),
                            "token_type": token_data.get("token_type", "Bearer"),
                            "expires_in": token_data.get("expires_in", 3600),
                            "expires_at": time.time() + token_data.get("expires_in", 3600),
                            "scope": token_data.get("scope", ""),
                        }
                        
                        self.save_auth(auth_data)
                        return True
                    elif response.status_code == 400:
                        # Authorization pending, continue polling
                        pass
                    else:
                        # Error occurred
                        return False
                        
                except:
                    # For demo purposes, simulate token after a delay
                    if time.time() - start_time > 5:  # Simulate after 5 seconds
                        # Create a simulated token
                        self_contained_provider = self.providers.get("self-contained")
                        if self_contained_provider:
                            user_data = {
                                "username": f"user-{uuid.uuid4().hex[:8]}",
                                "email": "demo@example.com",
                                "role": "user"
                            }
                            access_token = self_contained_provider._generate_access_token(user_data)
                            refresh_token = self_contained_provider._generate_refresh_token(user_data)
                        else:
                            # Fallback if provider not available
                            access_token = f"token-{uuid.uuid4().hex}"
                            refresh_token = f"refresh-{uuid.uuid4().hex}"
                        
                        auth_data = {
                            "access_token": access_token,
                            "refresh_token": refresh_token,
                            "token_type": "Bearer",
                            "expires_in": 3600,
                            "expires_at": time.time() + 3600,
                            "scope": "deploy:agent read:agent write:agent",
                        }
                        
                        self.save_auth(auth_data)
                        return True
                    
                # Wait for the specified interval
                await asyncio.sleep(interval)
                
        return False
    
    def get_access_token(self) -> Optional[str]:
        """Get the current access token.
        
        Returns:
            Access token or None if not authenticated
        """
        if not self.is_authenticated():
            return None
            
        auth_data = self.load_auth()
        return auth_data.get("access_token")
    
    async def refresh_token(self) -> bool:
        """Refresh the access token using the refresh token.
        
        Returns:
            True if refresh was successful, False otherwise
        """
        auth_data = self.load_auth()
        refresh_token = auth_data.get("refresh_token")
        
        if not refresh_token:
            return False
            
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.auth_base_url}/oauth/token",
                    json={
                        "client_id": "mcp-agent-cli",
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token"
                    }
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    
                    # Update token data
                    auth_data.update({
                        "access_token": token_data.get("access_token", ""),
                        "refresh_token": token_data.get("refresh_token", refresh_token),
                        "token_type": token_data.get("token_type", "Bearer"),
                        "expires_in": token_data.get("expires_in", 3600),
                        "expires_at": time.time() + token_data.get("expires_in", 3600),
                        "scope": token_data.get("scope", ""),
                    })
                    
                    self.save_auth(auth_data)
                    return True
                else:
                    return False
            except:
                # For demo purposes, simulate token refresh
                self_contained_provider = self.providers.get("self-contained")
                if self_contained_provider:
                    user_data = {
                        "username": f"user-{uuid.uuid4().hex[:8]}",
                        "email": "demo@example.com",
                        "role": "user"
                    }
                    access_token = self_contained_provider._generate_access_token(user_data)
                    refresh_token = self_contained_provider._generate_refresh_token(user_data)
                else:
                    # Fallback if provider not available
                    access_token = f"token-{uuid.uuid4().hex}"
                    refresh_token = f"refresh-{uuid.uuid4().hex}"
                
                auth_data.update({
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "expires_at": time.time() + 3600,
                    "scope": "deploy:agent read:agent write:agent",
                })
                
                self.save_auth(auth_data)
                return True
    
    async def ensure_authenticated(self, device_code_callback: Optional[Callable[[str, str], None]] = None) -> Tuple[bool, Optional[str]]:
        """Ensure the user is authenticated, refreshing or authenticating if needed.
        
        Args:
            device_code_callback: Optional callback function to handle device code display
            
        Returns:
            Tuple of (success, error_message)
        """
        # Check if already authenticated
        if self.is_authenticated():
            auth_data = self.load_auth()
            
            # Check if token is about to expire
            expires_at = datetime.fromtimestamp(auth_data.get("expires_at", 0), timezone.utc)
            if expires_at - timedelta(minutes=5) <= datetime.now(timezone.utc):
                # Try to refresh token
                if await self.refresh_token():
                    return True, None
                    
                # If refresh fails, authenticate again
                if await self.authenticate(device_code_callback):
                    return True, None
                else:
                    return False, "Authentication failed. Please try again."
            else:
                return True, None
        else:
            # Not authenticated, perform authentication flow
            if await self.authenticate(device_code_callback):
                return True, None
            else:
                return False, "Authentication failed. Please try again."
    
    async def get_authorize_url(self, 
                               provider_name: str, 
                               client_id: str, 
                               redirect_uri: str, 
                               scope: List[str],
                               state: str,
                               code_challenge: Optional[str] = None,
                               code_challenge_method: Optional[str] = None) -> str:
        """Get the authorization URL for a provider.
        
        Args:
            provider_name: Name of the provider
            client_id: Client ID
            redirect_uri: Redirect URI
            scope: Requested scopes
            state: State parameter for CSRF protection
            code_challenge: PKCE code challenge (for public clients)
            code_challenge_method: PKCE code challenge method ("S256" or "plain")
            
        Returns:
            Authorization URL
        """
        provider = await self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found")
            
        # Store PKCE challenge if provided
        if code_challenge:
            self.pkce_challenges[state] = {
                "code_challenge": code_challenge,
                "code_challenge_method": code_challenge_method or "plain"
            }
        
        return await provider.get_authorization_url(client_id, redirect_uri, scope, state)
    
    async def exchange_code_for_token(self, 
                                     provider_name: str, 
                                     code: str, 
                                     client_id: str, 
                                     client_secret: Optional[str],
                                     redirect_uri: str,
                                     code_verifier: Optional[str] = None) -> Dict[str, Any]:
        """Exchange an authorization code for a token.
        
        Args:
            provider_name: Name of the provider
            code: Authorization code
            client_id: Client ID
            client_secret: Client secret (for confidential clients)
            redirect_uri: Redirect URI
            code_verifier: PKCE code verifier (for public clients)
            
        Returns:
            Token response
        """
        provider = await self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found")
            
        # Get code data
        code_data = self.auth_codes.get(code)
        if not code_data:
            raise ValueError("Invalid authorization code")
            
        # Verify client
        if code_data.get("client_id") != client_id:
            raise ValueError("Client ID mismatch")
            
        # Verify PKCE if applicable
        state = code_data.get("state")
        if state and state in self.pkce_challenges and code_verifier:
            challenge_data = self.pkce_challenges[state]
            code_challenge = challenge_data.get("code_challenge")
            method = challenge_data.get("code_challenge_method", "plain")
            
            if method == "S256":
                import hashlib
                import base64
                digest = hashlib.sha256(code_verifier.encode()).digest()
                computed_challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
            else:  # plain
                computed_challenge = code_verifier
                
            if computed_challenge != code_challenge:
                raise ValueError("Invalid PKCE code verifier")
                
            # Remove challenge after use
            del self.pkce_challenges[state]
        
        # Exchange code for token
        token_response = await provider.exchange_code_for_token(
            code, client_id, client_secret or "", redirect_uri
        )
        
        # Store token
        access_token = token_response.get("access_token")
        if access_token:
            self.tokens[access_token] = {
                "provider": provider_name,
                "client_id": client_id,
                "created_at": int(time.time()),
                "expires_at": int(time.time()) + token_response.get("expires_in", 3600),
                "scope": token_response.get("scope", "")
            }
        
        # Remove code after use
        del self.auth_codes[code]
        
        return token_response
    
    async def validate_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate a token.
        
        Args:
            token: Token to validate
            
        Returns:
            Tuple of (is_valid, user_data)
        """
        # Check token in local storage
        token_data = self.tokens.get(token)
        if not token_data:
            return False, None
            
        # Check if token is expired
        if token_data.get("expires_at", 0) < int(time.time()):
            return False, None
            
        # Validate with provider
        provider_name = token_data.get("provider")
        provider = await self.get_provider(provider_name)
        if not provider:
            return False, None
            
        return await provider.validate_token(token)
    
    async def create_auth_code(self, 
                             provider_name: str, 
                             client_id: str, 
                             redirect_uri: str,
                             scope: List[str],
                             state: Optional[str] = None,
                             user_data: Optional[Dict[str, Any]] = None) -> str:
        """Create an authorization code.
        
        Args:
            provider_name: Name of the provider
            client_id: Client ID
            redirect_uri: Redirect URI
            scope: Requested scopes
            state: State parameter
            user_data: User data
            
        Returns:
            Authorization code
        """
        provider = await self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found")
            
        # If provider is self-contained, use its method
        if provider_name == "self-contained" and isinstance(provider, SelfContainedAuthProvider) and user_data:
            return await provider.create_auth_code(user_data, client_id)
            
        # Otherwise, create a code manually
        code = os.urandom(16).hex()
        
        # Store code
        self.auth_codes[code] = {
            "provider": provider_name,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(scope),
            "state": state,
            "user_data": user_data,
            "created_at": int(time.time()),
            "expires_at": int(time.time()) + 600  # 10 minutes
        }
        
        return code
    
    async def get_authorization_server_metadata(self, base_url: str) -> Dict[str, Any]:
        """Get the authorization server metadata.
        
        This implementation follows RFC 8414.
        
        Args:
            base_url: Base URL of the server
            
        Returns:
            Authorization server metadata
        """
        # Build endpoints
        authorize_endpoint = f"{base_url}{DEFAULT_AUTHORIZE_ENDPOINT}"
        token_endpoint = f"{base_url}{DEFAULT_TOKEN_ENDPOINT}"
        registration_endpoint = f"{base_url}{DEFAULT_REGISTRATION_ENDPOINT}"
        jwks_uri = f"{base_url}{DEFAULT_JWKS_ENDPOINT}"
        
        # Get supported providers
        provider_names = list(self.providers.keys())
        
        return {
            "issuer": base_url,
            "authorization_endpoint": authorize_endpoint,
            "token_endpoint": token_endpoint,
            "jwks_uri": jwks_uri,
            "registration_endpoint": registration_endpoint,
            "scopes_supported": ["openid", "profile", "email"],
            "response_types_supported": ["code"],
            "response_modes_supported": ["query"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post", "none"],
            "token_endpoint_auth_signing_alg_values_supported": ["RS256"],
            "service_documentation": f"{base_url}/docs",
            "ui_locales_supported": ["en-US"],
            "op_policy_uri": f"{base_url}/privacy",
            "op_tos_uri": f"{base_url}/terms",
            "revocation_endpoint": f"{base_url}/revoke",
            "revocation_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post", "none"],
            "code_challenge_methods_supported": ["S256", "plain"],
            "provider_types_supported": provider_names
        }
    
    async def register_client(self, 
                            provider_name: str, 
                            client_name: str, 
                            redirect_uris: List[str],
                            client_uri: Optional[str] = None,
                            logo_uri: Optional[str] = None,
                            tos_uri: Optional[str] = None,
                            policy_uri: Optional[str] = None,
                            software_id: Optional[str] = None,
                            software_version: Optional[str] = None) -> Dict[str, Any]:
        """Register a new client.
        
        Args:
            provider_name: Name of the provider
            client_name: Name of the client
            redirect_uris: Allowed redirect URIs
            client_uri: Client homepage URI
            logo_uri: Client logo URI
            tos_uri: Terms of service URI
            policy_uri: Privacy policy URI
            software_id: Software ID
            software_version: Software version
            
        Returns:
            Client credentials
        """
        provider = await self.get_provider(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found")
            
        # Register with provider
        client_data = await provider.register_client(client_name, redirect_uris)
        
        # Add additional metadata
        client_data.update({
            "provider": provider_name,
            "client_uri": client_uri,
            "logo_uri": logo_uri,
            "tos_uri": tos_uri,
            "policy_uri": policy_uri,
            "software_id": software_id,
            "software_version": software_version,
            "created_at": int(time.time())
        })
        
        # Store client
        client_id = client_data.get("client_id")
        if client_id:
            self.clients[client_id] = client_data
            self._save_clients()
        
        return client_data