"""Agent deployment functionality for MCP Agent Cloud.

This module provides classes for deploying and managing MCP agents in the cloud.
"""

import asyncio
import uuid
import json
import os
import tempfile
import logging
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from mcp_agent.cloud.deployment.server_deployment import MCPServerDeploymentManager

logger = logging.getLogger(__name__)

class AgentDeploymentManager:
    """Manager for deploying and managing MCP agents."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the deployment manager.
        
        Args:
            config: Configuration for the deployment manager
        """
        self.config = config or {}
        self.server_manager = MCPServerDeploymentManager(config)
        
    async def deploy_agent(self, 
                          agent_name: str, 
                          agent_config: Dict[str, Any],
                          region: str = "us-west",
                          public: bool = False) -> Dict[str, Any]:
        """Deploy a new MCP agent.
        
        Args:
            agent_name: Name of the agent
            agent_config: Agent configuration
            region: Deployment region
            public: Whether the agent should be publicly accessible
            
        Returns:
            Agent record
        """
        # Check for required config
        if "servers" not in agent_config:
            raise ValueError("Agent configuration must include 'servers' section")
            
        # Deploy required servers
        servers = {}
        for server_name, server_config in agent_config["servers"].items():
            # Deploy server if it doesn't exist
            try:
                server_record = await self.server_manager.deploy_server(
                    server_name=f"{agent_name}-{server_name}",
                    server_type=server_config.get("type", "generic"),
                    config=server_config.get("config"),
                    region=region,
                    public=public
                )
                servers[server_name] = server_record
            except ValueError as e:
                # Server might already exist
                server_record = await self.server_manager.registry.get_server_by_name(f"{agent_name}-{server_name}")
                if server_record:
                    servers[server_name] = server_record
                else:
                    raise ValueError(f"Failed to deploy server '{server_name}': {str(e)}")
        
        # Create agent record
        agent_record = {
            "id": f"agent-{uuid.uuid4().hex[:8]}",
            "name": agent_name,
            "servers": servers,
            "config": agent_config,
            "region": region,
            "public": public,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "deployed"
        }
        
        return agent_record
    
    async def delete_agent(self, agent_name: str, delete_servers: bool = True) -> None:
        """Delete an MCP agent.
        
        Args:
            agent_name: Name of the agent
            delete_servers: Whether to delete the agent's servers
        """
        # In a real implementation, we would get the agent record from a database
        # For now, we'll just delete the servers if requested
        if delete_servers:
            # List all servers
            servers = await self.server_manager.list_servers()
            
            # Delete servers whose names start with the agent name
            for server in servers:
                if server["name"].startswith(f"{agent_name}-"):
                    await self.server_manager.delete_server(server["name"])