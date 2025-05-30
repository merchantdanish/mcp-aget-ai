from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent
import datetime
import json

# Store server start time
SERVER_START_TIME = datetime.datetime.utcnow()

mcp = FastMCP("Resource Demo MCP Server")

# Define some static resources
STATIC_RESOURCES = {
    "demo://docs/readme": {
        "name": "README",
        "description": "A sample README file.",
        "content_type": "text/markdown",
        "content": "# Demo Resource Server\n\nThis is a sample README resource provided by the demo MCP server.",
    },
    "demo://config/settings": {
        "name": "Settings",
        "description": "Sample configuration settings.",
        "content_type": "application/json",
        "content": json.dumps(
            {"setting1": True, "setting2": 42, "mode": "demo"}, indent=2
        ),
    },
    "demo://data/users": {
        "name": "User Data",
        "description": "Sample user data in JSON format.",
        "content_type": "application/json",
        "content": json.dumps(
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
                {"id": 3, "name": "Charlie"},
            ],
            indent=2,
        ),
    },
}


@mcp.resource("demo://docs/readme")
def get_readme():
    """Provide the README file content."""
    meta = STATIC_RESOURCES["demo://docs/readme"]
    return meta["content"]


@mcp.resource("demo://config/settings")
def get_settings():
    """Provide configuration settings."""
    meta = STATIC_RESOURCES["demo://config/settings"]
    return meta["content"]


@mcp.resource("demo://data/users")
def get_users():
    """Provide user data."""
    meta = STATIC_RESOURCES["demo://data/users"]
    return meta["content"]


@mcp.resource("demo://status/health")
def get_health_status():
    """Provide dynamic health/status info (changes on each request)."""
    now = datetime.datetime.utcnow().isoformat() + "Z"
    uptime_seconds = int(
        (datetime.datetime.utcnow() - SERVER_START_TIME).total_seconds()
    )

    status = {
        "status": "ok",
        "timestamp": now,
        "uptime_seconds": uptime_seconds,
        "server_start_time": SERVER_START_TIME.isoformat() + "Z",
    }

    return json.dumps(status, indent=2)


@mcp.prompt()
def echo(message: str) -> str:
    """Echo the provided message.

    This is a simple prompt that echoes back the input message.
    """
    return f"Prompt: {message}"


def main():
    """Main entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
