"""CLI authentication commands for MCP Agent Cloud.

These commands handle platform authentication (how developers authenticate
with the MCP Agent Cloud platform). These commands are distinct from service credential
management (how deployed agents authenticate with external services), which is handled
by the credentials command group.

Platform authentication secures the deployment and management operations that developers
perform through the CLI, ensuring only authorized users can deploy and manage MCP servers.
"""

import typer
import asyncio
import webbrowser
import json
import urllib.parse
import http.server
import threading
import os
from pathlib import Path
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional, List, Dict, Any

from mcp_agent.cli.terminal import Application
from mcp_agent.cloud.auth.cli_auth_service import CLIAuthService

app = typer.Typer(help="Manage platform authentication for MCP Agent Cloud (for developers)")
application = Application()

# Global auth service instance
auth_service = CLIAuthService()

@app.command("login")
def login():
    """Authenticate with MCP Agent Cloud.
    
    This command starts a device authorization flow to authenticate your terminal
    with the MCP Agent Cloud platform.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Authenticating with MCP Agent Cloud..."),
        transient=True,
    ) as progress:
        task = progress.add_task("auth", total=None)
        
        # Start device authorization flow
        success = asyncio.run(_login())
    
    if success:
        application.console.print(Panel.fit(
            "✅ Authentication successful!\n\n"
            "Your terminal is now authenticated with MCP Agent Cloud.\n"
            "You can now use deployment commands like:\n"
            "  mcp-agent deploy server\n"
            "  mcp-agent deploy app\n"
            "  mcp-agent deploy workflow",
            title="Authentication Complete",
            border_style="green"
        ))
    else:
        application.error_console.print(Panel.fit(
            "❌ Authentication failed!\n\n"
            "Please try again later, or contact support if the issue persists.",
            title="Authentication Failed",
            border_style="red"
        ))
        raise typer.Exit(code=1)

@app.command("logout")
def logout():
    """Log out from MCP Agent Cloud.
    
    This command revokes your authentication tokens and removes them from your system.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Logging out from MCP Agent Cloud..."),
        transient=True,
    ) as progress:
        task = progress.add_task("logout", total=None)
        
        # Revoke tokens and clear credentials
        success = asyncio.run(_logout())
    
    if success:
        application.console.print(Panel.fit(
            "✅ Logout successful!\n\n"
            "Your terminal is no longer authenticated with MCP Agent Cloud.",
            title="Logout Complete",
            border_style="green"
        ))
    else:
        application.error_console.print(Panel.fit(
            "❌ Logout failed!\n\n"
            "Please try again later, or contact support if the issue persists.",
            title="Logout Failed",
            border_style="red"
        ))
        raise typer.Exit(code=1)

@app.command("status")
def status():
    """Check authentication status with MCP Agent Cloud."""
    # Check if token exists and is valid
    access_token = auth_service.get_access_token()
    token_file_exists = auth_service.token_file.exists()
    
    if access_token:
        # Get token expiration
        expires_at = auth_service.tokens.get("expires_at", 0)
        expires_in = max(0, expires_at - time.time())
        expires_hours = int(expires_in // 3600)
        expires_minutes = int((expires_in % 3600) // 60)
        expires_seconds = int(expires_in % 60)
        expires_text = f"{expires_hours}h {expires_minutes}m {expires_seconds}s"
        
        # Get token scope
        scope = auth_service.tokens.get("scope", "")
        scopes = scope.split() if scope else []
        
        # Display authentication status
        application.console.print(Panel.fit(
            f"✅ Authenticated with MCP Agent Cloud!\n\n"
            f"Token expires in: [bold]{expires_text}[/bold]\n"
            f"Scopes: [bold]{', '.join(scopes)}[/bold]",
            title="Authentication Status",
            border_style="green"
        ))
    elif token_file_exists:
        application.console.print(Panel.fit(
            "🔄 Token found but expired!\n\n"
            "Your authentication token has expired. Run 'mcp-agent deploy auth login' to reauthenticate.",
            title="Authentication Status",
            border_style="yellow"
        ))
    else:
        application.console.print(Panel.fit(
            "❌ Not authenticated!\n\n"
            "You are not authenticated with MCP Agent Cloud. Run 'mcp-agent deploy auth login' to authenticate.",
            title="Authentication Status",
            border_style="red"
        ))

async def _login() -> bool:
    """Perform the login flow.
    
    Returns:
        True if login was successful, False otherwise
    """
    # Define callback function to display device code
    def display_device_code(user_code, verification_uri):
        application.console.print(Panel.fit(
            f"To authenticate, visit: [bold blue]{verification_uri}[/bold blue]\n\n"
            f"And enter this code: [bold cyan]{user_code}[/bold cyan]",
            title="Authentication Required",
            border_style="green"
        ))
        
    # Start device authorization flow
    return await auth_service.device_authorization_flow(device_code_callback=display_device_code)

async def _logout() -> bool:
    """Perform the logout flow.
    
    Returns:
        True if logout was successful, False otherwise
    """
    return await auth_service.logout()

import time