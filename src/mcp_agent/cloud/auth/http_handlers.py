"""HTTP handlers for authentication endpoints in MCP Agent Cloud.

This module provides HTTP handlers for authentication-related endpoints in MCP Agent Cloud.
"""

import json
import asyncio
import urllib.parse
import logging
from typing import Dict, Any, Optional, List, Callable

from mcp_agent.cloud.auth.auth_service import AuthService

# Type for an HTTP request handler
HandlerFunc = Callable[[Dict[str, Any]], Dict[str, Any]]

logger = logging.getLogger(__name__)

class MCPCloudAuthHTTPHandlers:
    """HTTP handlers for MCP authentication endpoints in MCP Agent Cloud."""
    
    def __init__(self, auth_service: AuthService, base_url: str):
        """Initialize the HTTP handlers.
        
        Args:
            auth_service: Authentication service
            base_url: Base URL of the server
        """
        self.auth_service = auth_service
        self.base_url = base_url
        
    async def well_known_handler(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /.well-known/oauth-authorization-server endpoint.
        
        Args:
            request: HTTP request data
            
        Returns:
            HTTP response
        """
        try:
            metadata = await self.auth_service.get_authorization_server_metadata(self.base_url)
            return {
                "status": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(metadata)
            }
        except Exception as e:
            logger.error(f"Error in well_known_handler: {str(e)}")
            return {
                "status": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": "server_error",
                    "error_description": str(e)
                })
            }
    
    async def authorize_handler(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /authorize endpoint.
        
        Args:
            request: HTTP request data
            
        Returns:
            HTTP response
        """
        # Extract query parameters
        query = urllib.parse.parse_qs(request.get("query", ""))
        
        # Parse parameters
        provider_name = query.get("provider", ["self-contained"])[0]
        client_id = query.get("client_id", [""])[0]
        redirect_uri = query.get("redirect_uri", [""])[0]
        scope = query.get("scope", [""])[0]
        response_type = query.get("response_type", [""])[0]
        state = query.get("state", [""])[0] if "state" in query else None
        code_challenge = query.get("code_challenge", [""])[0] if "code_challenge" in query else None
        code_challenge_method = query.get("code_challenge_method", [""])[0] if "code_challenge_method" in query else None
        
        # Handle the request
        try:
            if response_type != "code":
                return {
                    "status": 400,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({
                        "error": "unsupported_response_type",
                        "error_description": "Only 'code' response type is supported"
                    })
                }
                
            # For self-contained provider, show login UI or handle authorization
            if provider_name == "self-contained":
                # Check if this is a login form submission
                if request.get("method") == "POST":
                    # Process login
                    body = request.get("body", "")
                    form_data = urllib.parse.parse_qs(body)
                    username = form_data.get("username", [""])[0]
                    password = form_data.get("password", [""])[0]
                    
                    # Authenticate user
                    provider = await self.auth_service.get_provider("self-contained")
                    if not provider:
                        raise ValueError("Provider not found")
                        
                    user_data = await provider.authenticate_user(username, password)
                    if not user_data:
                        # Authentication failed, show login form again
                        return self._render_login_form(
                            client_id=client_id,
                            redirect_uri=redirect_uri,
                            scope=scope,
                            state=state,
                            error="Invalid username or password"
                        )
                    
                    # Create auth code
                    code = await self.auth_service.create_auth_code(
                        provider_name="self-contained",
                        client_id=client_id,
                        redirect_uri=redirect_uri,
                        scope=scope.split(),
                        state=state,
                        user_data=user_data
                    )
                    
                    # Redirect to redirect_uri with code
                    redirect_url = f"{redirect_uri}?code={code}"
                    if state:
                        redirect_url += f"&state={state}"
                        
                    return {
                        "status": 302,
                        "headers": {"Location": redirect_url},
                        "body": ""
                    }
                else:
                    # Show login form
                    return self._render_login_form(
                        client_id=client_id,
                        redirect_uri=redirect_uri,
                        scope=scope,
                        state=state
                    )
            else:
                # For external providers, redirect to their authorization endpoint
                provider = await self.auth_service.get_provider(provider_name)
                if not provider:
                    raise ValueError(f"Provider '{provider_name}' not found")
                    
                # Get authorization URL
                auth_url = await self.auth_service.get_authorize_url(
                    provider_name=provider_name,
                    client_id=client_id,
                    redirect_uri=f"{self.base_url}/callback?client_redirect={redirect_uri}&state={state}",
                    scope=scope.split(),
                    state=state,
                    code_challenge=code_challenge,
                    code_challenge_method=code_challenge_method
                )
                
                # Redirect to auth URL
                return {
                    "status": 302,
                    "headers": {"Location": auth_url},
                    "body": ""
                }
        except Exception as e:
            logger.error(f"Error in authorize_handler: {str(e)}")
            return {
                "status": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": "invalid_request",
                    "error_description": str(e)
                })
            }
    
    def _render_login_form(self, client_id: str, redirect_uri: str, scope: str, state: Optional[str] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """Render the login form for self-contained authentication.
        
        Args:
            client_id: Client ID
            redirect_uri: Redirect URI
            scope: Requested scopes
            state: State parameter
            error: Optional error message
            
        Returns:
            HTTP response with login form
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MCP Agent Cloud - Log In</title>
            <style>
                body {{ font-family: sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 400px; margin: 40px auto; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); }}
                h1 {{ color: #333; text-align: center; margin-bottom: 20px; }}
                .form-group {{ margin-bottom: 15px; }}
                label {{ display: block; margin-bottom: 5px; color: #555; }}
                input[type="text"], input[type="password"] {{ width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }}
                button {{ display: block; width: 100%; padding: 10px; background-color: #4285f4; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }}
                .error {{ color: #d32f2f; margin-bottom: 15px; }}
                .scope-info {{ background-color: #f8f9fa; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: 14px; }}
                .client-info {{ text-align: center; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Log In</h1>
                <div class="client-info">
                    <strong>{client_id}</strong> is requesting access to your account
                </div>
                
                {"<div class='error'>" + error + "</div>" if error else ""}
                
                <div class="scope-info">
                    <strong>Requested permissions:</strong> {scope}
                </div>
                
                <form method="POST" action="{self.base_url}/authorize?client_id={client_id}&redirect_uri={redirect_uri}&scope={scope}&response_type=code{f'&state={state}' if state else ''}">
                    <div class="form-group">
                        <label for="username">Username</label>
                        <input type="text" id="username" name="username" required>
                    </div>
                    <div class="form-group">
                        <label for="password">Password</label>
                        <input type="password" id="password" name="password" required>
                    </div>
                    <button type="submit">Log In</button>
                </form>
            </div>
        </body>
        </html>
        """
        
        return {
            "status": 200,
            "headers": {"Content-Type": "text/html"},
            "body": html
        }
    
    async def token_handler(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /token endpoint.
        
        Args:
            request: HTTP request data
            
        Returns:
            HTTP response
        """
        # Extract form parameters
        content_type = request.get("headers", {}).get("Content-Type", "")
        
        if "application/json" in content_type:
            # Parse JSON body
            try:
                body = json.loads(request.get("body", "{}"))
            except json.JSONDecodeError:
                body = {}
        elif "application/x-www-form-urlencoded" in content_type:
            # Parse form body
            form_data = urllib.parse.parse_qs(request.get("body", ""))
            body = {k: v[0] for k, v in form_data.items()}
        else:
            body = {}
        
        # Parse parameters
        grant_type = body.get("grant_type", "")
        provider_name = body.get("provider", "self-contained")
        code = body.get("code")
        redirect_uri = body.get("redirect_uri")
        client_id = body.get("client_id")
        client_secret = body.get("client_secret")
        refresh_token = body.get("refresh_token")
        code_verifier = body.get("code_verifier")
        
        # Handle the request
        try:
            if grant_type == "authorization_code":
                # Exchange code for token
                if not code or not redirect_uri or not client_id:
                    raise ValueError("Missing required parameters")
                    
                token_response = await self.auth_service.exchange_code_for_token(
                    provider_name=provider_name,
                    code=code,
                    client_id=client_id,
                    client_secret=client_secret,
                    redirect_uri=redirect_uri,
                    code_verifier=code_verifier
                )
                
                return {
                    "status": 200,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps(token_response)
                }
            elif grant_type == "refresh_token":
                # Refresh token
                if not refresh_token or not client_id:
                    raise ValueError("Missing required parameters")
                    
                # Get provider
                provider = await self.auth_service.get_provider(provider_name)
                if not provider:
                    raise ValueError(f"Provider '{provider_name}' not found")
                    
                # Refresh token
                token_response = await provider.refresh_token(refresh_token, client_id, client_secret or "")
                
                return {
                    "status": 200,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps(token_response)
                }
            elif grant_type == "client_credentials":
                # Client credentials grant
                if not client_id or not client_secret:
                    raise ValueError("Missing required parameters")
                    
                # Get provider
                provider = await self.auth_service.get_provider(provider_name)
                if not provider:
                    raise ValueError(f"Provider '{provider_name}' not found")
                    
                # Get token
                token_response = await provider.client_credentials_grant(client_id, client_secret)
                
                return {
                    "status": 200,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps(token_response)
                }
            else:
                raise ValueError(f"Unsupported grant type: {grant_type}")
        except Exception as e:
            logger.error(f"Error in token_handler: {str(e)}")
            return {
                "status": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": "invalid_request",
                    "error_description": str(e)
                })
            }
    
    async def register_handler(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /register endpoint.
        
        Args:
            request: HTTP request data
            
        Returns:
            HTTP response
        """
        # Extract JSON body
        try:
            body = json.loads(request.get("body", "{}"))
        except json.JSONDecodeError:
            body = {}
        
        # Parse parameters
        provider_name = body.get("provider", "self-contained")
        client_name = body.get("client_name")
        redirect_uris = body.get("redirect_uris", [])
        client_uri = body.get("client_uri")
        logo_uri = body.get("logo_uri")
        tos_uri = body.get("tos_uri")
        policy_uri = body.get("policy_uri")
        software_id = body.get("software_id")
        software_version = body.get("software_version")
        
        # Handle the request
        try:
            if not client_name or not redirect_uris:
                return {
                    "status": 400,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({
                        "error": "invalid_request",
                        "error_description": "Missing required parameters"
                    })
                }
                
            client_data = await self.auth_service.register_client(
                provider_name=provider_name,
                client_name=client_name,
                redirect_uris=redirect_uris,
                client_uri=client_uri,
                logo_uri=logo_uri,
                tos_uri=tos_uri,
                policy_uri=policy_uri,
                software_id=software_id,
                software_version=software_version
            )
            
            return {
                "status": 201,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(client_data)
            }
        except Exception as e:
            logger.error(f"Error in register_handler: {str(e)}")
            return {
                "status": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": "server_error",
                    "error_description": str(e)
                })
            }
    
    async def callback_handler(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle /callback endpoint.
        
        Args:
            request: HTTP request data
            
        Returns:
            HTTP response
        """
        # Extract query parameters
        query = urllib.parse.parse_qs(request.get("query", ""))
        
        # Parse parameters
        code = query.get("code", [""])[0]
        state = query.get("state", [""])[0] if "state" in query else None
        error = query.get("error", [""])[0] if "error" in query else None
        client_redirect = query.get("client_redirect", [""])[0] if "client_redirect" in query else None
        
        # If this is a callback from an external provider
        if client_redirect:
            # Exchange code for token with the external provider
            provider_name = request.get("headers", {}).get("X-MCP-Provider", "github")
            
            try:
                # Get provider
                provider = await self.auth_service.get_provider(provider_name)
                if not provider:
                    raise ValueError(f"Provider '{provider_name}' not found")
                    
                # Exchange code
                token_data = await provider.exchange_code_for_token(
                    code, 
                    client_id="mcp-agent-cloud", 
                    client_secret="", 
                    redirect_uri=f"{self.base_url}/callback"
                )
                
                # Get user info
                user_data = await provider.get_user_info(token_data.get("access_token", ""))
                
                # Create auth code for client
                client_code = await self.auth_service.create_auth_code(
                    provider_name="self-contained",  # We use self-contained for the final code
                    client_id=client_redirect.split("?")[0],  # Extract client ID from redirect URI
                    redirect_uri=client_redirect,
                    scope=["openid", "profile", "email"],
                    state=state,
                    user_data=user_data
                )
                
                # Redirect to client with code
                redirect_to = f"{client_redirect}?code={client_code}"
                if state:
                    redirect_to += f"&state={state}"
                    
                return {
                    "status": 302,
                    "headers": {"Location": redirect_to},
                    "body": ""
                }
            except Exception as e:
                logger.error(f"Error in callback_handler: {str(e)}")
                # Redirect to client with error
                redirect_to = f"{client_redirect}?error=server_error&error_description={urllib.parse.quote(str(e))}"
                if state:
                    redirect_to += f"&state={state}"
                    
                return {
                    "status": 302,
                    "headers": {"Location": redirect_to},
                    "body": ""
                }
        
        # Prepare response HTML
        if error:
            error_description = query.get("error_description", [""])[0]
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authentication Error</title>
                <style>
                    body {{ font-family: sans-serif; margin: 0; padding: 20px; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    .error {{ background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 20px; border-radius: 5px; }}
                    h1 {{ color: #721c24; }}
                    .code {{ font-family: monospace; background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="error">
                        <h1>Authentication Error</h1>
                        <p>An error occurred during authentication: <strong>{error}</strong></p>
                        <p>Error description: {error_description}</p>
                    </div>
                    <p>You can close this window and return to the application.</p>
                </div>
            </body>
            </html>
            """
        else:
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authentication Successful</title>
                <style>
                    body {{ font-family: sans-serif; margin: 0; padding: 20px; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; padding: 20px; border-radius: 5px; }}
                    h1 {{ color: #155724; }}
                    .code {{ font-family: monospace; background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
                </style>
                <script>
                    window.onload = function() {{
                        window.opener.postMessage({{ code: "{code}", state: "{state or ''}" }}, "*");
                        setTimeout(function() {{ window.close(); }}, 3000);
                    }};
                </script>
            </head>
            <body>
                <div class="container">
                    <div class="success">
                        <h1>Authentication Successful</h1>
                        <p>You have successfully authenticated. You can close this window and return to the application.</p>
                    </div>
                    <p>This window will close automatically in 3 seconds.</p>
                </div>
            </body>
            </html>
            """
        
        return {
            "status": 200,
            "headers": {"Content-Type": "text/html"},
            "body": html
        }
    
    def get_routes(self) -> Dict[str, HandlerFunc]:
        """Get the routes for the HTTP handlers.
        
        Returns:
            Dictionary of route paths to handler functions
        """
        return {
            "/.well-known/oauth-authorization-server": self.well_known_handler,
            "/authorize": self.authorize_handler,
            "/token": self.token_handler,
            "/register": self.register_handler,
            "/callback": self.callback_handler
        }