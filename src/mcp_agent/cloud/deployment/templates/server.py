#!/usr/bin/env python3
import os
import asyncio
import logging
import json
from aiohttp import web
import sys
import signal

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import adapter classes
from mcp_agent.cloud.deployment.adapters.stdio_adapter import StdioAdapter, StdioServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Get server command and args from environment
SERVER_COMMAND = os.environ.get("SERVER_COMMAND", "npx")
SERVER_ARGS = os.environ.get("SERVER_ARGS", "").split()
SERVER_NAME = os.environ.get("SERVER_NAME", "mcp-server")

# Create adapter and server
stdio_server = StdioServer()


async def handle_mcp_request(request):
    """Handle MCP request by passing it to the STDIO adapter."""
    try:
        # Parse request body
        body = await request.json()
        
        # Get server name from URL path
        path_parts = request.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] == "servers":
            server_name = path_parts[1]
        else:
            # Use default server name
            server_name = SERVER_NAME
            
        # Handle request
        response = await stdio_server.handle_request(server_name, body)
        
        # Return response
        return web.json_response(response)
    except Exception as e:
        logger.error(f"Error handling request: {str(e)}")
        return web.json_response({
            "error": {
                "message": f"Error handling request: {str(e)}"
            }
        }, status=500)


async def handle_health_check(request):
    """Handle health check request."""
    return web.json_response({"status": "ok"})


async def startup_server(app):
    """Start the STDIO adapters on server startup."""
    # Add adapter for the server
    stdio_server.add_adapter(SERVER_NAME, SERVER_COMMAND, SERVER_ARGS)
    
    # Set up signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(
            sig, lambda: asyncio.create_task(shutdown_server(app))
        )
    
    logger.info(f"Started STDIO adapter for {SERVER_NAME}")


async def shutdown_server(app):
    """Stop the STDIO adapters on server shutdown."""
    await stdio_server.stop_all()
    logger.info("Stopped all STDIO adapters")


def main():
    """Main entry point for the server."""
    # Create web app
    app = web.Application()
    
    # Set up routes
    app.router.add_post("/servers/{server_name}/{endpoint:.*}", handle_mcp_request)
    app.router.add_get("/health", handle_health_check)
    
    # Set up startup and shutdown handlers
    app.on_startup.append(startup_server)
    app.on_cleanup.append(shutdown_server)
    
    # Start server
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    web.run_app(app, host=host, port=port, access_log=None)


if __name__ == "__main__":
    main()