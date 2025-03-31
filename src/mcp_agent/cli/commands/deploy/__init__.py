import typer
from typing import Optional

# Following the "everything is an MCP server" principle
app = typer.Typer(help="Deploy MCP servers")

from mcp_agent.cli.commands.deploy.server import app as server_app
from mcp_agent.cli.commands.deploy.auth import app as auth_app
from mcp_agent.cli.commands.deploy.app import app as app_app  # Renamed from agent to app
from mcp_agent.cli.commands.deploy.workflow import app as workflow_app
from mcp_agent.cli.commands.deploy.credentials import app as credentials_app

app.add_typer(server_app, name="server")
app.add_typer(auth_app, name="auth")
app.add_typer(app_app, name="app")  # Deploy MCPApp as MCP server
app.add_typer(workflow_app, name="workflow")  # Deploy workflow as MCP server
app.add_typer(credentials_app, name="credentials")  # Manage service credentials