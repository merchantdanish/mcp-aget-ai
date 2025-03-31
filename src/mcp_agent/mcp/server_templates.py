"""MCP server templates for different server types."""

from typing import Dict, Any, List, Optional

class ServerTemplate:
    """Base class for server templates."""
    
    def __init__(self, name: str, description: str):
        """Initialize the server template.
        
        Args:
            name: Name of the template
            description: Description of the template
        """
        self.name = name
        self.description = description
    
    def get_config(self, server_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get the container configuration for the server.
        
        Args:
            server_name: Name of the server
            config: Additional configuration
            
        Returns:
            Container configuration
        """
        # Start with default configuration
        container_config = {
            "image": f"mcp-server-{self.name}:latest",
            "command": None,
            "args": [],
            "env": {},
            "ports": {"8000/tcp": 8000},
            "volumes": {},
        }
        
        # Apply server-specific configuration
        if config:
            for key, value in config.items():
                if key in container_config and isinstance(container_config[key], dict):
                    container_config[key].update(value)
                else:
                    container_config[key] = value
        
        return container_config


class NetworkedServerTemplate(ServerTemplate):
    """Template for networked MCP servers."""
    
    def __init__(self, name: str, description: str, image: str):
        """Initialize the networked server template.
        
        Args:
            name: Name of the template
            description: Description of the template
            image: Docker image to use
        """
        super().__init__(name, description)
        self.image = image
    
    def get_config(self, server_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get the container configuration for the networked server.
        
        Args:
            server_name: Name of the server
            config: Additional configuration
            
        Returns:
            Container configuration
        """
        container_config = super().get_config(server_name, config)
        container_config["image"] = self.image
        container_config["name"] = f"mcp-server-{server_name}"
        return container_config


class StdioServerTemplate(ServerTemplate):
    """Template for STDIO-based MCP servers."""
    
    def __init__(self, name: str, description: str, command: str, args: List[str]):
        """Initialize the STDIO server template.
        
        Args:
            name: Name of the template
            description: Description of the template
            command: Command to run
            args: Arguments for the command
        """
        super().__init__(name, description)
        self.command = command
        self.args = args
    
    def get_config(self, server_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get the container configuration for the STDIO server.
        
        Args:
            server_name: Name of the server
            config: Additional configuration
            
        Returns:
            Container configuration
        """
        container_config = super().get_config(server_name, config)
        container_config["command"] = self.command
        container_config["args"] = self.args
        container_config["name"] = f"mcp-server-{server_name}"
        container_config["env"]["SERVER_NAME"] = server_name
        return container_config


# Dictionary of available server templates
SERVER_TEMPLATES = {
    # Networked server templates
    "fetch": NetworkedServerTemplate(
        name="fetch",
        description="MCP server for fetching web content",
        image="modelcontextprotocol/server-fetch:latest"
    ),
    "http": NetworkedServerTemplate(
        name="http",
        description="MCP server for making HTTP requests",
        image="modelcontextprotocol/server-http:latest"
    ),
    
    # STDIO server templates
    "filesystem": StdioServerTemplate(
        name="filesystem",
        description="MCP server for accessing the file system",
        command="npx",
        args=["@modelcontextprotocol/server-filesystem", "."]
    ),
    "shell": StdioServerTemplate(
        name="shell",
        description="MCP server for running shell commands",
        command="npx",
        args=["@modelcontextprotocol/server-shell"]
    ),
    "sqlite": StdioServerTemplate(
        name="sqlite",
        description="MCP server for SQLite database access",
        command="npx",
        args=["@modelcontextprotocol/server-sqlite", "default.db"]
    ),
}