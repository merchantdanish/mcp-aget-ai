"""CLI commands for managing service credentials for MCP servers.

This module provides commands for service authentication (how deployed agents authenticate
with external services like Gmail, GitHub, etc.). These commands are distinct from platform
authentication (how developers authenticate with MCP Agent Cloud), which is handled by
the auth command group.

Service credentials are securely stored in the MCP Agent Cloud platform and can be
accessed by deployed agents when they need to interact with external services.
"""

import os
import json
import asyncio
import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from typing import Optional, Dict, Any, List
from pathlib import Path

from mcp_agent.cli.terminal import Application
from mcp_agent.cloud.auth.cli_auth_service import CLIAuthService
from mcp_agent.cloud.auth.service_credential_manager import ServiceCredentialManager

app = typer.Typer(help="Manage service credentials for MCP servers (for deployed agents)")
application = Application()

# Global service instances
auth_service = CLIAuthService()
credential_manager = ServiceCredentialManager()

@app.command("list")
def list_credentials(
    agent_id: Optional[str] = typer.Option(None, "--agent-id", "-a", help="Filter by agent ID"),
):
    """List service credentials available to MCP servers."""
    # Authenticate first
    auth_result = asyncio.run(_ensure_authenticated())
    if not auth_result:
        application.error_console.print("[bold red]Authentication failed. Please try again.[/bold red]")
        raise typer.Exit(code=1)
    
    # Get credentials
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Loading credentials..."),
        transient=True,
    ) as progress:
        task = progress.add_task("list", total=None)
        credentials = asyncio.run(_list_credentials(agent_id))
    
    if not credentials:
        application.console.print("[yellow]No credentials found.[/yellow]")
        return
    
    # Display credentials in a table
    table = Table(title="Service Credentials")
    table.add_column("ID", style="cyan")
    table.add_column("Agent ID", style="green")
    table.add_column("Service", style="blue")
    table.add_column("Created", style="yellow")
    
    for cred in credentials:
        table.add_row(
            cred.get("id", ""),
            cred.get("agent_id", ""),
            cred.get("service_name", ""),
            cred.get("created_at", "")
        )
    
    application.console.print(table)

@app.command("add")
def add_credential(
    agent_id: str = typer.Argument(..., help="Agent ID to associate with the credential"),
    service_name: str = typer.Argument(..., help="Service name (e.g., gmail, github)"),
    credential_file: Path = typer.Option(
        None,
        "--file", 
        "-f",
        help="JSON file containing credentials",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
):
    """Add a new service credential for an MCP server."""
    # Authenticate first
    auth_result = asyncio.run(_ensure_authenticated())
    if not auth_result:
        application.error_console.print("[bold red]Authentication failed. Please try again.[/bold red]")
        raise typer.Exit(code=1)
    
    # Get credentials from file or prompt
    credentials = None
    if credential_file:
        try:
            with open(credential_file, "r") as f:
                credentials = json.load(f)
        except json.JSONDecodeError:
            application.error_console.print(f"[bold red]Error:[/bold red] Invalid JSON in file {credential_file}")
            raise typer.Exit(code=1)
        except IOError as e:
            application.error_console.print(f"[bold red]Error:[/bold red] {str(e)}")
            raise typer.Exit(code=1)
    else:
        # Prompt for credentials
        application.console.print(f"Enter credentials for {service_name} service in JSON format (press Ctrl+D when done):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        
        cred_json = "".join(lines)
        try:
            credentials = json.loads(cred_json)
        except json.JSONDecodeError:
            application.error_console.print("[bold red]Error:[/bold red] Invalid JSON")
            raise typer.Exit(code=1)
    
    # Add credential
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]Adding {service_name} credentials..."),
        transient=True,
    ) as progress:
        task = progress.add_task("add", total=None)
        credential_id = asyncio.run(_add_credential(agent_id, service_name, credentials))
    
    application.console.print(Panel.fit(
        f"✅ Successfully added {service_name} credentials for agent [bold]{agent_id}[/bold]!\n\n"
        f"Credential ID: [bold]{credential_id}[/bold]\n\n"
        f"Use this credential ID when configuring your MCP server to access the {service_name} service.",
        title="Credential Added",
        border_style="green"
    ))

@app.command("get")
def get_credential(
    credential_id: str = typer.Argument(..., help="ID of the credential to retrieve"),
):
    """Retrieve a service credential by ID."""
    # Authenticate first
    auth_result = asyncio.run(_ensure_authenticated())
    if not auth_result:
        application.error_console.print("[bold red]Authentication failed. Please try again.[/bold red]")
        raise typer.Exit(code=1)
    
    # Get credential
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Retrieving credential..."),
        transient=True,
    ) as progress:
        task = progress.add_task("get", total=None)
        credential = asyncio.run(_get_credential(credential_id))
    
    if not credential:
        application.error_console.print(f"[bold red]Error:[/bold red] Credential {credential_id} not found")
        raise typer.Exit(code=1)
    
    # Print credential
    application.console.print(Panel.fit(
        json.dumps(credential, indent=2),
        title=f"Credential {credential_id}",
        border_style="blue"
    ))

@app.command("delete")
def delete_credential(
    credential_id: str = typer.Argument(..., help="ID of the credential to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation"),
):
    """Delete a service credential."""
    # Authenticate first
    auth_result = asyncio.run(_ensure_authenticated())
    if not auth_result:
        application.error_console.print("[bold red]Authentication failed. Please try again.[/bold red]")
        raise typer.Exit(code=1)
    
    # Confirm deletion
    if not force and not typer.confirm(f"Are you sure you want to delete credential {credential_id}?"):
        application.console.print("Deletion cancelled.")
        return
    
    # Delete credential
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Deleting credential..."),
        transient=True,
    ) as progress:
        task = progress.add_task("delete", total=None)
        success = asyncio.run(_delete_credential(credential_id))
    
    if success:
        application.console.print(f"✅ Successfully deleted credential [bold]{credential_id}[/bold]")
    else:
        application.error_console.print(f"[bold red]Error:[/bold red] Failed to delete credential {credential_id}")
        raise typer.Exit(code=1)

async def _ensure_authenticated() -> bool:
    """Ensure the user is authenticated.
    
    Returns:
        True if authenticated, False otherwise
    """
    def display_device_code(user_code, verification_uri):
        application.console.print(Panel.fit(
            f"To authenticate, visit: [bold blue]{verification_uri}[/bold blue]\n\n"
            f"And enter this code: [bold cyan]{user_code}[/bold cyan]",
            title="Authentication Required",
            border_style="green"
        ))
    
    is_auth, auth_error = await auth_service.ensure_authenticated(device_code_callback=display_device_code)
    return is_auth

async def _list_credentials(agent_id: Optional[str]) -> List[Dict[str, Any]]:
    """List service credentials.
    
    Args:
        agent_id: Optional agent ID to filter by
        
    Returns:
        List of credential metadata
    """
    return await credential_manager.list_credentials(agent_id)

async def _add_credential(agent_id: str, service_name: str, credentials: Dict[str, Any]) -> str:
    """Add a new service credential.
    
    Args:
        agent_id: Agent ID
        service_name: Service name
        credentials: Credential data
        
    Returns:
        Credential ID
    """
    return await credential_manager.store_credential(agent_id, service_name, credentials)

async def _get_credential(credential_id: str) -> Optional[Dict[str, Any]]:
    """Get a service credential.
    
    Args:
        credential_id: Credential ID
        
    Returns:
        Credential data if found, None otherwise
    """
    return await credential_manager.get_credential(credential_id)

async def _delete_credential(credential_id: str) -> bool:
    """Delete a service credential.
    
    Args:
        credential_id: Credential ID
        
    Returns:
        True if deleted, False otherwise
    """
    return await credential_manager.delete_credential(credential_id)