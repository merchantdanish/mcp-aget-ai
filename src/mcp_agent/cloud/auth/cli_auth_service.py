"""CLI Authentication service for MCP Agent Cloud.

This module provides secure terminal authentication for the MCP Agent CLI,
implementing the OAuth 2.0 Device Authorization Grant flow (RFC 8628).
"""

import os
import time
import json
import uuid
import secrets
import asyncio
import httpx
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable

# Configure logging
logger = logging.getLogger(__name__)

class CLIAuthService:
    """Service for authenticating CLI commands with MCP Agent Cloud."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the CLI authentication service.
        
        Args:
            config: Authentication configuration
        """
        self.config = config or {}
        
        # Use local servers for demo
        self.api_base_url = self.config.get("api_base_url", "http://localhost:8001")
        
        # Auth server endpoints
        self.auth_url = self.config.get("auth_url", "http://localhost:8000/auth")
        self.device_code_endpoint = self.config.get("device_code_endpoint", f"{self.auth_url}/device/code")
        self.token_endpoint = self.config.get("token_endpoint", f"{self.auth_url}/token")
        self.revoke_endpoint = self.config.get("revoke_endpoint", f"{self.auth_url}/revoke")
        
        # Token storage
        self.token_dir = Path(self.config.get("token_dir", os.path.expanduser("~/.mcp-agent-cloud")))
        self.token_file = self.token_dir / "auth_tokens.json"
        
        # Ensure the token directory exists
        os.makedirs(self.token_dir, exist_ok=True)
        
        # Load existing tokens
        self.tokens = self._load_tokens()
        
    def _load_tokens(self) -> Dict[str, Any]:
        """Load authentication tokens from storage.
        
        Returns:
            Dictionary of tokens
        """
        if not self.token_file.exists():
            return {}
            
        try:
            with open(self.token_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading tokens: {str(e)}")
            return {}
            
    def _save_tokens(self) -> None:
        """Save authentication tokens to storage."""
        try:
            with open(self.token_file, "w") as f:
                json.dump(self.tokens, f, indent=2)
                
            # Set secure permissions
            if os.name != 'nt':  # Not Windows
                os.chmod(self.token_file, 0o600)  # Read/write permissions for owner only
        except IOError as e:
            logger.error(f"Error saving tokens: {str(e)}")
            
    def get_access_token(self) -> Optional[str]:
        """Get the current access token if available and not expired.
        
        Returns:
            Access token or None if not available or expired
        """
        access_token = self.tokens.get("access_token")
        expires_at = self.tokens.get("expires_at", 0)
        
        # Check if token exists and is not expired
        if access_token and expires_at > time.time():
            return access_token
            
        return None
        
    async def refresh_token(self) -> bool:
        """Refresh the access token using the refresh token.
        
        Returns:
            True if token was refreshed successfully, False otherwise
        """
        refresh_token = self.tokens.get("refresh_token")
        if not refresh_token:
            return False
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.token_endpoint,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": "mcp-agent-cli"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Error refreshing token: {response.text}")
                    return False
                    
                token_data = response.json()
                
                # Update tokens
                self.tokens.update({
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token", refresh_token),  # Use existing refresh token if not provided
                    "expires_at": int(time.time()) + token_data.get("expires_in", 3600),
                    "scope": token_data.get("scope", ""),
                    "token_type": token_data.get("token_type", "Bearer")
                })
                
                # Save tokens
                self._save_tokens()
                
                return True
        except Exception as e:
            logger.error(f"Error refreshing token: {str(e)}")
            return False
            
    async def device_authorization_flow(self, device_code_callback: Optional[Callable[[str, str], None]] = None) -> bool:
        """Perform the OAuth 2.0 Device Authorization Grant flow.
        
        Args:
            device_code_callback: Optional callback function to display device code and verification URI
            
        Returns:
            True if authentication was successful, False otherwise
        """
        try:
            # Request device code
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.device_code_endpoint,
                    data={
                        "client_id": "mcp-agent-cli",
                        "scope": "deploy:server deploy:app deploy:workflow"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Error requesting device code: {response.text}")
                    return False
                    
                device_code_data = response.json()
                
                # Extract device code data
                device_code = device_code_data["device_code"]
                user_code = device_code_data["user_code"]
                verification_uri = device_code_data["verification_uri"]
                expires_in = device_code_data.get("expires_in", 900)  # Default 15 minutes
                interval = device_code_data.get("interval", 5)  # Default 5 seconds
                
                # Display device code and URI to the user
                if device_code_callback:
                    device_code_callback(user_code, verification_uri)
                else:
                    print(f"To authenticate, please enter this code: {user_code}")
                    print(f"Visit: {verification_uri}")
                
                # Poll for token
                max_attempts = expires_in // interval
                for attempt in range(max_attempts):
                    await asyncio.sleep(interval)
                    
                    token_response = await client.post(
                        self.token_endpoint,
                        data={
                            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                            "device_code": device_code,
                            "client_id": "mcp-agent-cli"
                        }
                    )
                    
                    if token_response.status_code == 200:
                        # Authentication successful
                        token_data = token_response.json()
                        
                        # Store tokens
                        self.tokens = {
                            "access_token": token_data["access_token"],
                            "refresh_token": token_data.get("refresh_token", ""),
                            "expires_at": int(time.time()) + token_data.get("expires_in", 3600),
                            "scope": token_data.get("scope", ""),
                            "token_type": token_data.get("token_type", "Bearer")
                        }
                        
                        # Save tokens
                        self._save_tokens()
                        
                        return True
                    elif token_response.status_code == 400:
                        error = token_response.json().get("error", "")
                        if error == "authorization_pending":
                            # User hasn't authorized yet, continue polling
                            continue
                        elif error == "slow_down":
                            # Increase polling interval
                            interval = min(interval * 2, 60)  # Max 60 seconds
                            continue
                        elif error == "expired_token":
                            # Device code expired
                            logger.error("Device code expired")
                            return False
                        elif error == "access_denied":
                            # User denied access
                            logger.error("Access denied by user")
                            return False
                        else:
                            # Other error
                            logger.error(f"Error polling for token: {token_response.text}")
                            return False
                    else:
                        # Other error
                        logger.error(f"Error polling for token: {token_response.text}")
                        return False
                
                # If we get here, polling timed out
                logger.error("Authentication timed out")
                return False
        except Exception as e:
            logger.error(f"Error in device authorization flow: {str(e)}")
            return False
            
    async def ensure_authenticated(self, device_code_callback: Optional[Callable[[str, str], None]] = None) -> Tuple[bool, Optional[str]]:
        """Ensure the user is authenticated, by refreshing the token or initiating a device authorization flow.
        
        Args:
            device_code_callback: Optional callback function to display device code and verification URI
            
        Returns:
            Tuple of (is_authenticated, error_message)
        """
        # Check if we have a valid access token
        access_token = self.get_access_token()
        if access_token:
            return True, None
            
        # Try to refresh token
        if self.tokens.get("refresh_token"):
            success = await self.refresh_token()
            if success:
                return True, None
        
        # If refresh failed or no refresh token, initiate device authorization flow
        success = await self.device_authorization_flow(device_code_callback)
        if success:
            return True, None
        else:
            return False, "Authentication failed"
            
    async def logout(self) -> bool:
        """Revoke tokens and clear stored credentials.
        
        Returns:
            True if logout was successful, False otherwise
        """
        # Revoke access token if possible
        access_token = self.tokens.get("access_token")
        if access_token:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.revoke_endpoint,
                        data={
                            "token": access_token,
                            "token_type_hint": "access_token",
                            "client_id": "mcp-agent-cli"
                        }
                    )
                    
                    # We don't care about the response too much, as we're clearing local tokens anyway
                    if response.status_code not in (200, 204):
                        logger.warning(f"Warning: Error revoking access token: {response.text}")
            except Exception as e:
                logger.warning(f"Warning: Error revoking access token: {str(e)}")
        
        # Revoke refresh token if possible
        refresh_token = self.tokens.get("refresh_token")
        if refresh_token:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.revoke_endpoint,
                        data={
                            "token": refresh_token,
                            "token_type_hint": "refresh_token",
                            "client_id": "mcp-agent-cli"
                        }
                    )
                    
                    # We don't care about the response too much, as we're clearing local tokens anyway
                    if response.status_code not in (200, 204):
                        logger.warning(f"Warning: Error revoking refresh token: {response.text}")
            except Exception as e:
                logger.warning(f"Warning: Error revoking refresh token: {str(e)}")
        
        # Clear tokens
        self.tokens = {}
        self._save_tokens()
        
        return True