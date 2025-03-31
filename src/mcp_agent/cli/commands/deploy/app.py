"""CLI command for deploying MCPApps as MCP servers to the cloud."""

import typer
import asyncio
import os
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown
from typing import Optional, List, Dict, Any
from pathlib import Path

from mcp_agent.cli.terminal import Application
from mcp_agent.cloud.auth.cli_auth_service import CLIAuthService
from mcp_agent.cloud.deployment.app_deployment import AppDeploymentService

app = typer.Typer(help="Deploy MCPApps as MCP servers")
application = Application()

# Global service instances
auth_service = CLIAuthService()
deployment_service = AppDeploymentService(auth_service)

@app.command("deploy")
def deploy_app(
    directory: Path = typer.Argument(
        ".", 
        help="Directory containing the MCPApp code",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Name for the deployed MCPApp (defaults to directory name)"
    ),
    region: str = typer.Option(
        "us-west",
        "--region",
        "-r",
        help="Region to deploy the MCPApp to"
    ),
):
    """Deploy an MCPApp as an MCP server to the cloud.
    
    This command packages and deploys an MCPApp as an MCP server to the MCP Agent Cloud platform.
    The app must have a valid mcp_agent.config.yaml file in its directory.
    """
    # Absolute path for the directory
    directory_path = directory.resolve()
    
    # Default to directory name if no name provided
    if not name:
        name = directory_path.name
    
    # Display deployment information
    application.console.print(f"Deploying MCPApp as MCP server: [bold cyan]{name}[/bold cyan] from [bold]{directory_path}[/bold] to region [bold]{region}[/bold]")
    
    # Handle device code authentication
    def device_code_callback(verification_uri, user_code):
        """Display device code and verification URI to the user."""
        application.console.print(Panel.fit(
            f"To authenticate, visit:\n[bold cyan]{verification_uri}[/bold cyan]\n\n"
            f"And enter the code: [bold green]{user_code}[/bold green]",
            title="Authentication Required",
            border_style="yellow"
        ))
    
    # Authenticate user first
    auth_result = asyncio.run(_ensure_authenticated(device_code_callback))
    if not auth_result:
        application.error_console.print("[bold red]Authentication failed. Please try again.[/bold red]")
        raise typer.Exit(code=1)
        
    # Run the deployment process with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Deploying MCPApp as MCP server..."),
        transient=True,
    ) as progress:
        progress.add_task("deploy", total=None)
        
        # Deploy the MCPApp as an MCP server
        result, error, deployment_info = asyncio.run(_deploy_app(directory_path, name, region))
        
    if not result:
        application.error_console.print(f"[bold red]Deployment failed:[/bold red] {error}")
        raise typer.Exit(code=1)
        
    # Display deployment result
    console_url = deployment_info.get("console_url", "https://console.mcp-agent-cloud.example.com")
    server_url = deployment_info.get("url", f"https://{name}.mcp-agent-cloud.example.com")
    
    application.console.print(Panel.fit(
        f"✅ Successfully deployed [bold]{name}[/bold] as MCP server!\n\n"
        f"Server URL: [bold]{server_url}[/bold]\n"
        f"Server ID: {deployment_info.get('id', 'unknown')}\n"
        f"Status: [bold green]{deployment_info.get('status', 'deployed')}[/bold green]\n\n"
        f"View in console: [bold cyan]{console_url}[/bold cyan]\n\n"
        f"[italic]This MCPApp is now available as an MCP server and can be accessed by other agents and applications.[/italic]",
        title="Deployment Complete",
        border_style="green"
    ))

@app.command("list")
def list_apps():
    """List all deployed MCPApps (MCP servers)."""
    # Authenticate user first
    auth_result = asyncio.run(_ensure_authenticated())
    if not auth_result:
        application.error_console.print("[bold red]Authentication failed. Please try again.[/bold red]")
        raise typer.Exit(code=1)
        
    # Run the list operation with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Retrieving agents..."),
        transient=True,
    ) as progress:
        progress.add_task("list", total=None)
        
        # List all MCPApp servers
        apps = asyncio.run(_list_apps())
        
    if not apps:
        application.console.print("No MCPApps have been deployed as MCP servers yet.")
        return
        
    # Create a table to display server information
    table = Table(title="Deployed MCPApp Servers")
    table.add_column("Name", style="cyan")
    table.add_column("ID", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Region", style="magenta")
    table.add_column("Server URL", style="yellow")
    table.add_column("Created", style="dim")
    
    # Add rows for each MCPApp server
    for app in apps:
        status_style = "green" if app.get("status") == "deployed" else "yellow"
        created_at = app.get("created_at", "unknown")
        if isinstance(created_at, str) and "T" in created_at:
            # Format ISO datetime to a more readable format
            created_at = created_at.split("T")[0]
            
        table.add_row(
            app.get("name", "unknown"),
            app.get("id", "unknown"),
            f"[{status_style}]{app.get('status', 'unknown')}[/{status_style}]",
            app.get("region", "unknown"),
            app.get("url", "unknown"),
            created_at
        )
        
    # Display the table
    application.console.print(table)
    
    # Show how to get more information
    application.console.print("\nView more details in the [bold cyan]MCP Agent Cloud Console[/bold cyan]:")
    application.console.print("https://console.mcp-agent-cloud.example.com/servers")

async def _ensure_authenticated(device_code_callback=None) -> bool:
    """Ensure the user is authenticated.
    
    Args:
        device_code_callback: Optional callback function for device code display
        
    Returns:
        True if authenticated, False otherwise
    """
    is_auth, auth_error = await auth_service.ensure_authenticated(device_code_callback)
    return is_auth

async def _deploy_app(directory: Path, name: str, region: str) -> tuple:
    """Deploy an MCPApp as an MCP server.
    
    Args:
        directory: Path to the MCPApp directory
        name: Name for the deployed MCPApp server
        region: Region to deploy the server to
        
    Returns:
        Tuple of (success, error, deployment_info)
    """
    return await deployment_service.deploy_app(directory, name, region)

async def _list_apps() -> List[Dict[str, Any]]:
    """List all deployed MCPApp servers.
    
    Returns:
        List of deployed MCPApp servers
    """
    return await deployment_service.list_apps()

# For test/demo purposes
if __name__ == "__main__":
    app()