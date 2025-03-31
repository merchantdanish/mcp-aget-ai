"""CLI command for deploying workflows as MCP servers to the cloud."""

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
from mcp_agent.cloud.deployment.workflow_deployment import WorkflowDeploymentService

app = typer.Typer(help="Deploy workflows as MCP servers")
application = Application()

# Global service instances
auth_service = CLIAuthService()
workflow_deployment_service = WorkflowDeploymentService(auth_service=auth_service)

@app.command("deploy")
def deploy_workflow(
    directory: Path = typer.Argument(
        ".", 
        help="Directory containing the workflow code",
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
        help="Name for the deployed workflow (defaults to directory name)"
    ),
    workflow_type: str = typer.Option(
        "parallel",
        "--type",
        "-t",
        help="Type of workflow (parallel, router, orchestrator, evaluator-optimizer)"
    ),
    region: str = typer.Option(
        "us-west",
        "--region",
        "-r",
        help="Region to deploy the workflow to"
    ),
):
    """Deploy a workflow as an MCP server to the cloud.
    
    This command packages and deploys a workflow as an MCP server to the MCP Agent Cloud platform.
    The workflow must have a valid mcp_agent.config.yaml file in its directory.
    """
    # Display progress information
    application.console.print(f"Deploying {workflow_type} workflow as MCP server: [bold cyan]{name or directory.name}[/bold cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Deploying workflow as MCP server..."),
        transient=True,
    ) as progress:
        task = progress.add_task("deploy", total=None)
        
        # Run deployment process
        success, error_message, deployment_info = asyncio.run(
            _deploy_workflow(directory, name, region, workflow_type)
        )
    
    # Handle deployment result
    if not success:
        application.error_console.print(f"[bold red]Deployment failed: {error_message}[/bold red]")
        raise typer.Exit(code=1)
    
    # Display deployment information
    _display_deployment_info(deployment_info)

@app.command("list")
def list_workflows():
    """List all deployed workflow servers."""
    
    application.console.print("Fetching deployed workflow servers...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Fetching workflows..."),
        transient=True,
    ) as progress:
        task = progress.add_task("fetch", total=None)
        
        # Run list process
        workflows = asyncio.run(_list_workflows())
    
    if not workflows:
        application.console.print("[yellow]No workflow servers found.[/yellow]")
        return
    
    # Display workflows in a table
    table = Table(title="Deployed Workflow Servers")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Region", style="blue")
    table.add_column("URL", style="magenta")
    
    for workflow in workflows:
        table.add_row(
            workflow.get("id", ""),
            workflow.get("name", ""),
            workflow.get("status", ""),
            workflow.get("region", ""),
            workflow.get("url", "")
        )
    
    application.console.print(table)

async def _deploy_workflow(
    directory: Path, 
    name: Optional[str], 
    region: str,
    workflow_type: str
) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Deploy a workflow as an MCP server.
    
    Args:
        directory: Path to the workflow directory
        name: Optional name for the deployed workflow
        region: Region to deploy the workflow to
        workflow_type: Type of workflow
        
    Returns:
        Tuple of (success, error_message, deployment_info)
    """
    # Ensure authentication
    is_auth, auth_error = await auth_service.ensure_authenticated(device_code_callback=_display_device_code)
    if not is_auth:
        return False, f"Authentication failed: {auth_error}", None
    
    # Deploy the workflow as an MCP server
    return await workflow_deployment_service.deploy_workflow(directory, name, region)

async def _list_workflows() -> List[Dict[str, Any]]:
    """List all deployed workflow servers.
    
    Returns:
        List of deployed workflow servers
    """
    # Ensure authentication
    is_auth, auth_error = await auth_service.ensure_authenticated(device_code_callback=_display_device_code)
    if not is_auth:
        return []
    
    # Get the list of deployed workflow servers
    return await workflow_deployment_service.list_workflows()

async def _ensure_authenticated(device_code_callback=None) -> bool:
    """Ensure the user is authenticated.
    
    Args:
        device_code_callback: Optional callback function for device code display
        
    Returns:
        True if authenticated, False otherwise
    """
    is_auth, auth_error = await auth_service.ensure_authenticated(device_code_callback)
    return is_auth

def _display_device_code(device_code: str, verification_uri: str):
    """Display the device code for authentication.
    
    Args:
        device_code: The device code for authentication
        verification_uri: The URI for verification
    """
    application.console.print(Panel.fit(
        f"To authenticate, please enter this code: [bold cyan]{device_code}[/bold cyan]\n\n"
        f"Visit: [bold blue]{verification_uri}[/bold blue]",
        title="Authentication Required",
        border_style="green"
    ))

def _display_deployment_info(deployment_info: Dict[str, Any]):
    """Display deployment information.
    
    Args:
        deployment_info: Deployment information
    """
    application.console.print(Panel.fit(
        f"✅ Successfully deployed [bold]{deployment_info.get('name')}[/bold] workflow as MCP server!\n\n"
        f"Server URL: [bold]{deployment_info.get('url')}[/bold]\n"
        f"Server ID: {deployment_info.get('id')}\n"
        f"Status: [bold green]{deployment_info.get('status')}[/bold green]",
        title="Deployment Complete",
        subtitle=f"View in console: {deployment_info.get('console_url')}"
    ))

# For test/demo purposes
if __name__ == "__main__":
    app()