"""Server registry for MCP Agent Cloud.

This module provides a registry for tracking MCP servers deployed to the cloud.
It supports in-memory and persistent storage options.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ServerRegistry:
    """Registry for tracking MCP servers."""
    
    def __init__(self, registry_file: str = None, redis_client = None):
        """Initialize the server registry.
        
        Args:
            registry_file: Path to the registry file for persistent storage
            redis_client: Redis client for distributed storage
        """
        self.registry_file = registry_file or os.path.expanduser("~/.mcp-agent-cloud/servers.json")
        self.redis_client = redis_client
        self.servers = {}
        
        # Ensure registry directory exists
        if registry_file:
            os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load the registry from storage."""
        # If redis client is available, use it
        if self.redis_client:
            try:
                # Get all servers from Redis
                for key in self.redis_client.keys("server:*"):
                    server_data = json.loads(self.redis_client.get(key))
                    self.servers[server_data["id"]] = server_data
            except Exception as e:
                logger.error(f"Error loading registry from Redis: {str(e)}")
                # Fall back to file-based registry
                self._load_file_registry()
        else:
            # Use file-based registry
            self._load_file_registry()
    
    def _load_file_registry(self) -> None:
        """Load the registry from a file."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
                    self.servers = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding registry file: {self.registry_file}")
                self.servers = {}
            except IOError as e:
                logger.error(f"Error reading registry file: {str(e)}")
                self.servers = {}
    
    def _save_registry(self) -> None:
        """Save the registry to storage."""
        # If redis client is available, use it
        if self.redis_client:
            try:
                # Save servers to Redis
                for server_id, server_data in self.servers.items():
                    key = f"server:{server_id}"
                    self.redis_client.set(key, json.dumps(server_data))
            except Exception as e:
                logger.error(f"Error saving registry to Redis: {str(e)}")
                # Fall back to file-based registry
                self._save_file_registry()
        else:
            # Use file-based registry
            self._save_file_registry()
    
    def _save_file_registry(self) -> None:
        """Save the registry to a file."""
        try:
            with open(self.registry_file, "w") as f:
                json.dump(self.servers, f, indent=2)
        except IOError as e:
            logger.error(f"Error writing registry file: {str(e)}")
    
    def register_server(self, server: Dict[str, Any]) -> None:
        """Register a new server.
        
        Args:
            server: Server data to register
        """
        server_id = server.get("id")
        if not server_id:
            raise ValueError("Server ID is required")
            
        self.servers[server_id] = server
        self._save_registry()
        logger.info(f"Registered server: {server_id}")
    
    def update_server(self, server_id: str, server: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing server.
        
        Args:
            server_id: ID of the server to update
            server: Updated server data
            
        Returns:
            Updated server data or None if server not found
        """
        if server_id not in self.servers:
            return None
            
        self.servers[server_id].update(server)
        self._save_registry()
        logger.info(f"Updated server: {server_id}")
        return self.servers[server_id]
    
    def get_server(self, server_id: str) -> Optional[Dict[str, Any]]:
        """Get a server by ID.
        
        Args:
            server_id: ID of the server to get
            
        Returns:
            Server data or None if server not found
        """
        return self.servers.get(server_id)
    
    def get_server_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a server by name.
        
        Args:
            name: Name of the server to get
            
        Returns:
            Server data or None if server not found
        """
        for server in self.servers.values():
            if server.get("name") == name:
                return server
        return None
    
    def delete_server(self, server_id: str) -> bool:
        """Delete a server.
        
        Args:
            server_id: ID of the server to delete
            
        Returns:
            True if server was deleted, False otherwise
        """
        if server_id not in self.servers:
            return False
            
        del self.servers[server_id]
        self._save_registry()
        logger.info(f"Deleted server: {server_id}")
        return True
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """List all registered servers.
        
        Returns:
            List of server data
        """
        return list(self.servers.values())