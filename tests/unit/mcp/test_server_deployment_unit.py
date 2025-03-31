"""Unit tests for the server deployment manager."""

import pytest
import asyncio
import os
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from mcp_agent.mcp.server_deployment import (
    MCPServerDeploymentManager,
    ContainerService,
    ServerRegistry
)
from mcp_agent.mcp.server_templates import SERVER_TEMPLATES


class TestContainerService:
    """Test the container service."""
    
    @pytest.mark.asyncio
    async def test_create_container(self, container_service_mock):
        """Test creating a container."""
        # Create a container
        container_id = await container_service_mock.create_container({
            "image": "alpine:latest",
            "name": "test-container",
            "env": {"TEST_VAR": "test_value"},
            "ports": {"8000/tcp": 8000},
            "volumes": {"/host/path": {"bind": "/container/path", "mode": "rw"}}
        })
        
        # Check that the container was created
        assert container_id == "test-container-id"
    
    @pytest.mark.asyncio
    async def test_build_stdio_container(self, container_service_mock):
        """Test building a STDIO container."""
        # Build a STDIO container
        container_id, port = await container_service_mock.build_stdio_container(
            "test-server",
            "npx",
            ["@modelcontextprotocol/server-filesystem", "."]
        )
        
        # Check that the container was created
        assert container_id == "test-container-id"
        assert port == 8000
        
        # Check that the build_stdio_container method was called
        container_service_mock.build_stdio_container.assert_called_once_with(
            "test-server",
            "npx",
            ["@modelcontextprotocol/server-filesystem", "."]
        )
    
    @pytest.mark.asyncio
    async def test_stop_container(self, container_service_mock):
        """Test stopping a container."""
        # Stop a container
        await container_service_mock.stop_container("test-container-id")
        
        # Check that the stop_container method was called
        container_service_mock.stop_container.assert_called_once_with("test-container-id")
    
    @pytest.mark.asyncio
    async def test_start_container(self, container_service_mock):
        """Test starting a container."""
        # Start a container
        await container_service_mock.start_container("test-container-id")
        
        # Check that the start_container method was called
        container_service_mock.start_container.assert_called_once_with("test-container-id")
    
    @pytest.mark.asyncio
    async def test_delete_container(self, container_service_mock):
        """Test deleting a container."""
        # Delete a container
        await container_service_mock.delete_container("test-container-id")
        
        # Check that the delete_container method was called
        container_service_mock.delete_container.assert_called_once_with("test-container-id")


class TestServerRegistry:
    """Test the server registry."""
    
    @pytest.mark.asyncio
    async def test_register_server(self, server_registry_mock):
        """Test registering a server."""
        # Create a server record
        server_record = {
            "id": "srv-test-id",
            "name": "test-server",
            "type": "fetch",
            "container_id": "test-container-id",
            "endpoint": "/servers/test-server",
            "url": "https://test-server.mcp-agent-cloud.example.com",
            "local_url": "http://localhost:8000",
            "port": 8000,
            "auth_config": {
                "providers": {
                    "self-contained": {
                        "jwt_secret": "test-jwt-secret"
                    }
                },
                "default_client": {
                    "client_id": "test-client-id",
                    "client_secret": "test-client-secret",
                    "redirect_uris": ["http://localhost:8000/callback"]
                }
            },
            "region": "us-west",
            "public": False,
            "created_at": "2025-03-31T00:00:00.000000",
            "status": "running"
        }
        
        # Register the server
        await server_registry_mock.register_server(server_record)
        
        # Check that the register_server method was called
        server_registry_mock.register_server.assert_called_once_with(server_record)
    
    @pytest.mark.asyncio
    async def test_get_server(self, server_registry_mock):
        """Test getting a server by ID."""
        # Get a server
        server = await server_registry_mock.get_server("srv-test-id")
        
        # Check that the get_server method was called
        server_registry_mock.get_server.assert_called_once_with("srv-test-id")
        
        # Check the server data
        assert server["name"] == "test-server"
        assert server["type"] == "fetch"
        assert server["container_id"] == "test-container-id"
    
    @pytest.mark.asyncio
    async def test_get_server_by_name(self, server_registry_mock):
        """Test getting a server by name."""
        # Get a server
        server = await server_registry_mock.get_server_by_name("test-server")
        
        # Check that the get_server_by_name method was called
        server_registry_mock.get_server_by_name.assert_called_once_with("test-server")
        
        # Check the server data
        assert server["id"] == "srv-test-id"
        assert server["type"] == "fetch"
        assert server["container_id"] == "test-container-id"
    
    @pytest.mark.asyncio
    async def test_list_servers(self, server_registry_mock):
        """Test listing servers."""
        # List servers
        servers = await server_registry_mock.list_servers()
        
        # Check that the list_servers method was called
        server_registry_mock.list_servers.assert_called_once()
        
        # Check the servers data
        assert len(servers) == 1
        assert servers[0]["name"] == "test-server"
        assert servers[0]["type"] == "fetch"
        assert servers[0]["container_id"] == "test-container-id"
    
    @pytest.mark.asyncio
    async def test_update_server_status(self, server_registry_mock):
        """Test updating server status."""
        # Update server status
        await server_registry_mock.update_server_status("srv-test-id", "stopped")
        
        # Check that the update_server_status method was called
        server_registry_mock.update_server_status.assert_called_once_with("srv-test-id", "stopped")
    
    @pytest.mark.asyncio
    async def test_delete_server(self, server_registry_mock):
        """Test deleting a server."""
        # Delete a server
        await server_registry_mock.delete_server("srv-test-id")
        
        # Check that the delete_server method was called
        server_registry_mock.delete_server.assert_called_once_with("srv-test-id")


class TestMCPServerDeploymentManager:
    """Test the MCP server deployment manager."""
    
    @pytest.mark.asyncio
    async def test_deploy_server(self, deployment_manager):
        """Test deploying a server."""
        # Mock the get_server_by_name method to return None
        deployment_manager.registry.get_server_by_name = AsyncMock(return_value=None)
        
        # Mock the build_networked_container method
        deployment_manager.build_networked_container = AsyncMock(return_value=("test-container-id", 8000))
        
        # Deploy a server
        server = await deployment_manager.deploy_server(
            server_name="test-server",
            server_type="fetch",
            region="us-west",
            public=False
        )
        
        # Check the server data
        assert server["name"] == "test-server"
        assert server["type"] == "fetch"
        assert server["container_id"] == "test-container-id"
        assert server["endpoint"] == "/servers/test-server"
        assert server["url"] == "https://test-server.mcp-agent-cloud.example.com"
        assert server["local_url"] == "http://localhost:8000"
        assert server["port"] == 8000
        assert server["region"] == "us-west"
        assert server["public"] is False
        assert server["status"] == "running"
        
        # Check that the registry's register_server method was called
        deployment_manager.registry.register_server.assert_called_once()
        
        # Test deploying a server with an existing name
        deployment_manager.registry.get_server_by_name = AsyncMock(return_value=server)
        
        with pytest.raises(ValueError):
            await deployment_manager.deploy_server(
                server_name="test-server",
                server_type="fetch"
            )
    
    @pytest.mark.asyncio
    async def test_deploy_stdio_server(self, deployment_manager):
        """Test deploying a STDIO-based server."""
        # Mock the get_server_by_name method to return None
        deployment_manager.registry.get_server_by_name = AsyncMock(return_value=None)
        
        # Deploy a STDIO server
        server = await deployment_manager.deploy_server(
            server_name="test-stdio-server",
            server_type="filesystem",
            region="us-west",
            public=False
        )
        
        # Check the server data
        assert server["name"] == "test-stdio-server"
        assert server["type"] == "filesystem"
        assert server["container_id"] == "test-container-id"
        assert server["endpoint"] == "/servers/test-stdio-server"
        assert server["url"] == "https://test-stdio-server.mcp-agent-cloud.example.com"
        assert server["local_url"] == "http://localhost:8000"
        assert server["port"] == 8000
        assert server["region"] == "us-west"
        assert server["public"] is False
        assert server["status"] == "running"
        
        # Check that the container service's build_stdio_container method was called
        deployment_manager.container_service.build_stdio_container.assert_called_once()
        
        # Check that the registry's register_server method was called
        deployment_manager.registry.register_server.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_server(self, deployment_manager):
        """Test stopping a server."""
        # Setup a server record
        server = {
            "id": "srv-test-id",
            "name": "test-server",
            "container_id": "test-container-id"
        }
        deployment_manager.registry.get_server_by_name = AsyncMock(return_value=server)
        
        # Stop the server
        await deployment_manager.stop_server("test-server")
        
        # Check that the container service's stop_container method was called
        deployment_manager.container_service.stop_container.assert_called_once_with("test-container-id")
        
        # Check that the registry's update_server_status method was called
        deployment_manager.registry.update_server_status.assert_called_once_with("srv-test-id", "stopped")
        
        # Test stopping a non-existent server
        deployment_manager.registry.get_server_by_name = AsyncMock(return_value=None)
        
        with pytest.raises(ValueError):
            await deployment_manager.stop_server("non-existent-server")
    
    @pytest.mark.asyncio
    async def test_start_server(self, deployment_manager):
        """Test starting a server."""
        # Setup a server record
        server = {
            "id": "srv-test-id",
            "name": "test-server",
            "container_id": "test-container-id"
        }
        deployment_manager.registry.get_server_by_name = AsyncMock(return_value=server)
        
        # Start the server
        await deployment_manager.start_server("test-server")
        
        # Check that the container service's start_container method was called
        deployment_manager.container_service.start_container.assert_called_once_with("test-container-id")
        
        # Check that the registry's update_server_status method was called
        deployment_manager.registry.update_server_status.assert_called_once_with("srv-test-id", "running")
        
        # Test starting a non-existent server
        deployment_manager.registry.get_server_by_name = AsyncMock(return_value=None)
        
        with pytest.raises(ValueError):
            await deployment_manager.start_server("non-existent-server")
    
    @pytest.mark.asyncio
    async def test_delete_server(self, deployment_manager):
        """Test deleting a server."""
        # Setup a server record
        server = {
            "id": "srv-test-id",
            "name": "test-server",
            "container_id": "test-container-id"
        }
        deployment_manager.registry.get_server_by_name = AsyncMock(return_value=server)
        
        # Delete the server
        await deployment_manager.delete_server("test-server")
        
        # Check that the container service's delete_container method was called
        deployment_manager.container_service.delete_container.assert_called_once_with("test-container-id")
        
        # Check that the registry's delete_server method was called
        deployment_manager.registry.delete_server.assert_called_once_with("srv-test-id")
        
        # Test deleting a non-existent server
        deployment_manager.registry.get_server_by_name = AsyncMock(return_value=None)
        
        with pytest.raises(ValueError):
            await deployment_manager.delete_server("non-existent-server")
    
    @pytest.mark.asyncio
    async def test_list_servers(self, deployment_manager):
        """Test listing servers."""
        # List servers
        servers = await deployment_manager.list_servers()
        
        # Check that the registry's list_servers method was called
        deployment_manager.registry.list_servers.assert_called_once()
        
        # Check the servers data
        assert len(servers) == 1
        assert servers[0]["name"] == "test-server"
        assert servers[0]["type"] == "fetch"
        assert servers[0]["container_id"] == "test-container-id"
    
    def test_get_container_config(self, deployment_manager):
        """Test getting container configuration."""
        # Get container configuration for a template-based server type
        config = deployment_manager._get_container_config("fetch")
        
        # Check the configuration
        assert config["image"] == "modelcontextprotocol/server-fetch:latest"
        assert config["name"] == "mcp-server-fetch"
        
        # Get container configuration for a non-template server type
        config = deployment_manager._get_container_config("custom")
        
        # Check the configuration
        assert config["image"] == "mcp-server-custom:latest"
        assert config["command"] == "npx"
        assert config["args"] == ["@modelcontextprotocol/server-custom"]
        
        # Get container configuration with additional config
        config = deployment_manager._get_container_config("fetch", {"env": {"TEST_VAR": "test_value"}})
        
        # Check the configuration
        assert config["image"] == "modelcontextprotocol/server-fetch:latest"
        assert config["env"]["TEST_VAR"] == "test_value"