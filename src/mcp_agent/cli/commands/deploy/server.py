import typer
import asyncio
import json
import os
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from typing import Optional, List, Dict, Any

from mcp_agent.cli.terminal import Application
from mcp_agent.mcp.server_deployment import MCPServerDeploymentManager
from mcp_agent.mcp.server_templates import SERVER_TEMPLATES

app = typer.Typer(help="Deploy MCP servers")
application = Application()

# Global deployment manager instance
deployment_manager = MCPServerDeploymentManager()

def server_type_callback(value: str):
    """Validate and autocomplete server type."""
    if value not in SERVER_TEMPLATES:
        available_types = ", ".join(SERVER_TEMPLATES.keys())
        typer.echo(f"Invalid server type: {value}")
        typer.echo(f"Available server types: {available_types}")
        raise typer.BadParameter(f"Invalid server type: {value}")
    return value

@app.command("templates")
def list_templates():
    """List available MCP server templates."""
    table = Table(title="Available MCP Server Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Description", style="green")
    
    for name, template in SERVER_TEMPLATES.items():
        template_type = "Networked" if hasattr(template, "image") else "STDIO"
        table.add_row(name, template_type, template.description)
    
    application.console.print(table)

@app.command("deploy")
def deploy_server(
    name: str = typer.Argument(..., help="Name of the MCP server"),
    server_type: str = typer.Option(
        ..., 
        "--type", 
        "-t", 
        help="Type of MCP server (fetch, filesystem, etc.)",
        autocompletion=lambda: SERVER_TEMPLATES.keys(),
        callback=server_type_callback
    ),
    region: str = typer.Option("us-west", "--region", "-r", help="Deployment region"),
    public: bool = typer.Option(False, "--public", help="Make the server publicly accessible"),
    auth_provider: str = typer.Option(
        "self-contained", 
        "--auth", 
        "-a", 
        help="Authentication provider (self-contained, github, custom)",
        autocompletion=lambda: ["self-contained", "github", "custom"]
    ),
):
    """Deploy a new MCP server to the cloud."""
    # In a real implementation, we would authenticate and load configuration here
    
    # Show template information
    template = SERVER_TEMPLATES.get(server_type)
    if template:
        template_type = "Networked" if hasattr(template, "image") else "STDIO"
        application.console.print(f"Using [bold]{template_type}[/bold] template: [bold cyan]{server_type}[/bold cyan]")
        application.console.print(f"Description: {template.description}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Deploying MCP server..."),
        transient=True,
    ) as progress:
        progress.add_task("deploy", total=None)
        # Run the actual deployment process
        server_record = asyncio.run(_deploy_server(name, server_type, region, public, auth_provider))
    
    # Get auth config
    auth_config = server_record.get("auth_config", {})
    default_client = auth_config.get("default_client", {})
    client_id = default_client.get("client_id", "N/A")
    client_secret = default_client.get("client_secret", "N/A")
    
    # Display deployment result
    application.console.print(Panel.fit(
        f"✅ Successfully deployed [bold]{name}[/bold] MCP server!\n\n"
        f"Server URL: [bold]{server_record['url']}[/bold]\n"
        f"Local URL: [bold]{server_record['local_url']}[/bold]\n"
        f"Server ID: {server_record['id']}\n"
        f"Type: {server_record['type']}\n"
        f"Status: [bold green]{server_record['status']}[/bold green]\n\n"
        f"[bold]Authentication:[/bold]\n"
        f"Provider: {auth_provider}\n"
        f"Client ID: {client_id}\n"
        f"Client Secret: {client_secret}",
        title="Deployment Complete",
        subtitle="View in console: https://console.mcp-agent-cloud.example.com"
    ))

async def _deploy_server(name: str, server_type: str, region: str, public: bool, auth_provider: str = "self-contained") -> Dict[str, Any]:
    """Deploy an MCP server using the deployment manager.
    
    Args:
        name: Name of the server
        server_type: Type of the server
        region: Deployment region
        public: Whether the server should be publicly accessible
        auth_provider: Authentication provider type
        
    Returns:
        Server record
    """
    try:
        # Configure server-specific settings based on type
        config = None
        if server_type == "filesystem":
            config = {
                "volumes": {"/data": {"bind": "/app/data", "mode": "rw"}},
                "env": {"ALLOWED_PATHS": "/app/data"}
            }
        elif server_type == "fetch":
            config = {
                "env": {"FETCH_TIMEOUT": "30000"}
            }
        
        # Add auth provider configuration
        auth_config = {
            "provider_type": auth_provider
        }
        
        # If GitHub provider, check for environment variables
        if auth_provider == "github":
            auth_config["client_id"] = os.environ.get("GITHUB_CLIENT_ID")
            auth_config["client_secret"] = os.environ.get("GITHUB_CLIENT_SECRET")
            
            if not auth_config["client_id"] or not auth_config["client_secret"]:
                application.console.print("[yellow]Warning:[/yellow] GitHub OAuth credentials not found in environment. Using dummy values.")
        
        # Add auth config to server config
        if not config:
            config = {}
        config["auth"] = auth_config
            
        # Deploy the server
        server_record = await deployment_manager.deploy_server(
            server_name=name,
            server_type=server_type,
            config=config,
            region=region,
            public=public
        )
        
        return server_record
    except Exception as e:
        application.error_console.print(f"[bold red]Error deploying server:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

@app.command("list")
def list_servers():
    """List all deployed MCP servers."""
    try:
        # Fetch the list of servers from the registry
        servers = asyncio.run(_list_servers())
        
        if not servers:
            application.console.print("No MCP servers deployed yet.")
            return
        
        # Create a table to display server information
        table = Table(title="Deployed MCP Servers")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Region", style="magenta")
        table.add_column("Public", style="yellow")
        table.add_column("URL", style="cyan")
        table.add_column("Auth", style="red")
        
        # Add rows for each server
        for server in servers:
            status_style = "green" if server["status"] == "running" else "yellow"
            public_value = "Yes" if server.get("public", False) else "No"
            url = server.get("url", f"https://{server['name']}.mcp-agent-cloud.example.com")
            local_url = server.get("local_url", "N/A")
            port = server.get("port", "8000")
            
            # Get auth info
            auth_config = server.get("auth_config", {})
            auth_providers = auth_config.get("providers", {})
            auth_provider = list(auth_providers.keys())[0] if auth_providers else "none"
            
            table.add_row(
                server["name"],
                server["type"],
                f"[{status_style}]{server['status']}[/{status_style}]",
                server.get("region", "us-west"),
                public_value,
                f"{url} (local: {port})",
                auth_provider
            )
        
        # Display the table
        application.console.print(table)
        
    except Exception as e:
        application.error_console.print(f"[bold red]Error listing servers:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

async def _list_servers() -> List[Dict[str, Any]]:
    """List all deployed MCP servers using the deployment manager.
    
    Returns:
        List of server records
    """
    return await deployment_manager.list_servers()

@app.command("stop")
def stop_server(
    name: str = typer.Argument(..., help="Name of the MCP server to stop"),
):
    """Stop a running MCP server."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Stopping MCP server {name}..."),
            transient=True,
        ) as progress:
            progress.add_task("stop", total=None)
            # Stop the server
            asyncio.run(_stop_server(name))
        
        application.console.print(f"✅ Successfully stopped [bold]{name}[/bold] MCP server!")
    except Exception as e:
        application.error_console.print(f"[bold red]Error stopping server:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

async def _stop_server(name: str):
    """Stop an MCP server using the deployment manager.
    
    Args:
        name: Name of the server to stop
    """
    await deployment_manager.stop_server(name)

@app.command("start")
def start_server(
    name: str = typer.Argument(..., help="Name of the MCP server to start"),
):
    """Start a stopped MCP server."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Starting MCP server {name}..."),
            transient=True,
        ) as progress:
            progress.add_task("start", total=None)
            # Start the server
            asyncio.run(_start_server(name))
        
        application.console.print(f"✅ Successfully started [bold]{name}[/bold] MCP server!")
    except Exception as e:
        application.error_console.print(f"[bold red]Error starting server:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

async def _start_server(name: str):
    """Start an MCP server using the deployment manager.
    
    Args:
        name: Name of the server to start
    """
    await deployment_manager.start_server(name)

@app.command("delete")
def delete_server(
    name: str = typer.Argument(..., help="Name of the MCP server to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation"),
):
    """Delete an MCP server."""
    try:
        if not force:
            confirmed = typer.confirm(f"Are you sure you want to delete the server '{name}'?")
            if not confirmed:
                application.console.print("Operation cancelled.")
                return
        
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Deleting MCP server {name}..."),
            transient=True,
        ) as progress:
            progress.add_task("delete", total=None)
            # Delete the server
            asyncio.run(_delete_server(name))
        
        application.console.print(f"✅ Successfully deleted [bold]{name}[/bold] MCP server!")
    except Exception as e:
        application.error_console.print(f"[bold red]Error deleting server:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

async def _delete_server(name: str):
    """Delete an MCP server using the deployment manager.
    
    Args:
        name: Name of the server to delete
    """
    await deployment_manager.delete_server(name)