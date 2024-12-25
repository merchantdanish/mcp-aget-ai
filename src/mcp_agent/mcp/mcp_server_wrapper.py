from typing import Any
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import PromptManager
from mcp.server.fastmcp.resources import ResourceManager
from mcp.server.fastmcp.server import Settings
from mcp.server.fastmcp.tools import ToolManager
from mcp.server.fastmcp.utilities.logging import configure_logging
from mcp.server.lowlevel import Server as MCPServer


# TODO: saqadri - submit a PR to the MCP library to add this to FastMCP directly
# TODO: saqadri - delete this class, it isn't useful, use ClientSession to interact with the server at all times
class MCPServerWrapper(FastMCP):
    """
    A FastMCP variant that uses an existing MCP server instance to initialize
    """

    # pylint: disable=W0231
    def __init__(self, server: Server | None, name: str | None = None, **settings: Any):
        """
        Initialize the FastMCP instance with an existing server instance.
        If no server instance is provided, a new one will be created.
        If a server is provided, this assumes the server is already running.
        """
        self.settings = Settings(**settings)

        self._mcp_server = (
            server if server is not None else MCPServer(name=name or "FastMCP")
        )
        self._tool_manager = ToolManager(
            warn_on_duplicate_tools=self.settings.warn_on_duplicate_tools
        )
        self._resource_manager = ResourceManager(
            warn_on_duplicate_resources=self.settings.warn_on_duplicate_resources
        )
        self._prompt_manager = PromptManager(
            warn_on_duplicate_prompts=self.settings.warn_on_duplicate_prompts
        )
        self.dependencies = self.settings.dependencies

        # If an existing server was provided, we will use it to initialize the FastMCP managers
        if server is not None:
            self._initialize_tool_manager(server)
            self._initialize_resource_manager(server)
            self._initialize_prompt_manager(server)

        # Set up MCP protocol handlers
        self._setup_handlers()

        # Configure logging
        configure_logging(self.settings.log_level)

    def _initialize_prompt_manager(self, server: Server):
        prompts = server.list_prompts()
        for prompt in prompts:
            self.add_prompt(prompt)

    def _initialize_tool_manager(self, server: Server):
        tools = server.list_tools()
        for tool in tools:
            self.add_tool(tool)

    def _initialize_resource_manager(self, server: Server):
        resources = server.list_resources()
        for resource in resources:
            self.add_resource(resource)
