"""Integration tests for the server deployment manager."""

import pytest
import asyncio
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from mcp_agent.mcp.server_deployment import (
    MCPServerDeploymentManager,
    ContainerService,
    ServerRegistry
)
from mcp_agent.mcp.auth_factory import MCPAuthFactory


@pytest.fixture
def temp_registry_dir():
    """Create a temporary directory for the registry file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def registry_file(temp_registry_dir):
    """Create a registry file path."""
    return os.path.join(temp_registry_dir, "registry.json")


@pytest.fixture
def container_service():
    """Create a container service."""
    # Create a class with same interface as ContainerService but with simpler implementation
    class TestContainerService:
        async def create_container(self, container_config):
            return "test-container-id"
            
        async def build_stdio_container(self, server_name, command, args):
            return ("test-container-id", 8000)
            
        async def stop_container(self, container_id):
            pass
            
        async def start_container(self, container_id):
            pass
            
        async def delete_container(self, container_id):
            pass
            
    yield TestContainerService()


@pytest.fixture
def server_registry(registry_file):
    """Create a server registry with a temporary registry file."""
    registry = ServerRegistry(registry_file)
    yield registry


@pytest.fixture
def auth_factory():
    """Create a test auth factory."""
    class TestAuthFactory:
        async def create_server_auth(self, server_name, auth_provider_type="self-contained"):
            auth_service = "test-auth-service-instance"  # a string instead of a mock
            auth_config = {
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
            }
            return (auth_service, auth_config)
    
    yield TestAuthFactory()


@pytest.fixture
def deployment_manager(container_service, server_registry, auth_factory):
    """Create a deployment manager with test dependencies."""
    manager = MCPServerDeploymentManager()
    manager.container_service = container_service
    manager.registry = server_registry
    manager.auth_factory = auth_factory
    
    # Create HTTP servers dict - use a simple dict instead of mocks
    manager.http_servers = {}
    
    # Mock find_free_port to always return 8000
    async def mock_find_free_port(*args, **kwargs):
        return 8000
    
    manager.find_free_port = mock_find_free_port
    
    # Mock build_networked_container to return a fixed container ID and port
    async def mock_build_networked_container(server_name, server_type):
        return ("test-container-id", 8000)
    
    manager.build_networked_container = mock_build_networked_container
    
    yield manager


class TestServerDeploymentIntegration:
    """Integration tests for server deployment."""
    
    @pytest.mark.asyncio
    async def test_full_server_lifecycle(self, deployment_manager, registry_file):
        """Test the full lifecycle of a server: deploy, start, stop, delete."""
        # 1. Deploy a new server
        server = await deployment_manager.deploy_server(
            server_name="integration-test-server",
            server_type="fetch",
            region="us-west",
            public=False
        )
        
        # Check that the server was deployed
        assert server["name"] == "integration-test-server"
        assert server["type"] == "fetch"
        assert server["container_id"] == "test-container-id"
        assert "id" in server
        assert server["status"] == "running"
        
        # Check that the server was registered in the registry file
        with open(registry_file, "r") as f:
            registry_data = json.load(f)
            assert server["id"] in registry_data
            assert registry_data[server["id"]]["name"] == "integration-test-server"
        
        # 2. Stop the server
        await deployment_manager.stop_server("integration-test-server")
        
        # Get the server record to check its status
        servers = await deployment_manager.list_servers()
        server = [s for s in servers if s["name"] == "integration-test-server"][0]
        
        # Check that the server status is stopped
        assert server["status"] == "stopped"
        
        # 3. Start the server
        await deployment_manager.start_server("integration-test-server")
        
        # Get the server record to check its status
        servers = await deployment_manager.list_servers()
        server = [s for s in servers if s["name"] == "integration-test-server"][0]
        
        # Check that the server status is running
        assert server["status"] == "running"
        
        # 4. Delete the server
        await deployment_manager.delete_server("integration-test-server")
        
        # Get the servers to check that the server was deleted
        servers = await deployment_manager.list_servers()
        
        # Check that the server was deleted
        assert len([s for s in servers if s["name"] == "integration-test-server"]) == 0
        
        # Check that the server was removed from the registry file
        with open(registry_file, "r") as f:
            registry_data = json.load(f)
            assert len(registry_data) == 0
    
    @pytest.mark.asyncio
    async def test_deploy_multiple_servers(self, deployment_manager):
        """Test deploying multiple servers with different configurations."""
        # Deploy a networked server
        networked_server = await deployment_manager.deploy_server(
            server_name="integration-fetch-server",
            server_type="fetch",
            region="us-west",
            public=False
        )
        
        # Deploy a STDIO-based server
        stdio_server = await deployment_manager.deploy_server(
            server_name="integration-filesystem-server",
            server_type="filesystem",
            region="us-west",
            public=False
        )
        
        # Deploy a third server with custom configuration
        custom_server = await deployment_manager.deploy_server(
            server_name="integration-custom-server",
            server_type="custom",
            config={
                "env": {"CUSTOM_VAR": "custom_value"},
                "volumes": {"/host/custom": {"bind": "/container/custom", "mode": "ro"}}
            },
            region="us-east",
            public=True
        )
        
        # Get all servers
        servers = await deployment_manager.list_servers()
        
        # Check that all servers were deployed
        assert len(servers) == 3
        
        # Check the networked server
        networked = [s for s in servers if s["name"] == "integration-fetch-server"][0]
        assert networked["type"] == "fetch"
        assert networked["region"] == "us-west"
        assert networked["public"] is False
        
        # Check the STDIO server
        stdio = [s for s in servers if s["name"] == "integration-filesystem-server"][0]
        assert stdio["type"] == "filesystem"
        assert stdio["region"] == "us-west"
        assert stdio["public"] is False
        
        # Check the custom server
        custom = [s for s in servers if s["name"] == "integration-custom-server"][0]
        assert custom["type"] == "custom"
        assert custom["region"] == "us-east"
        assert custom["public"] is True
        
        # Clean up
        await deployment_manager.delete_server("integration-fetch-server")
        await deployment_manager.delete_server("integration-filesystem-server")
        await deployment_manager.delete_server("integration-custom-server")
    
    @pytest.mark.asyncio
    async def test_server_error_handling(self, deployment_manager):
        """Test error handling when deploying and managing servers."""
        # 1. Test deploying a server with an existing name
        await deployment_manager.deploy_server(
            server_name="duplicate-server",
            server_type="fetch"
        )
        
        # Deploying a server with the same name should raise an error
        with pytest.raises(ValueError):
            await deployment_manager.deploy_server(
                server_name="duplicate-server",
                server_type="fetch"
            )
        
        # 2. Test operating on a non-existent server
        with pytest.raises(ValueError):
            await deployment_manager.stop_server("non-existent-server")
        
        with pytest.raises(ValueError):
            await deployment_manager.start_server("non-existent-server")
        
        with pytest.raises(ValueError):
            await deployment_manager.delete_server("non-existent-server")
        
        # Clean up
        await deployment_manager.delete_server("duplicate-server")
    
    @pytest.mark.asyncio
    async def test_server_registry_persistence(self, deployment_manager, registry_file):
        """Test that the server registry persists server information."""
        # Deploy a server
        await deployment_manager.deploy_server(
            server_name="persistence-test-server",
            server_type="fetch"
        )
        
        # Check that the server is in the registry file
        with open(registry_file, "r") as f:
            registry_data = json.load(f)
            assert len(registry_data) == 1
            server_id = list(registry_data.keys())[0]
            assert registry_data[server_id]["name"] == "persistence-test-server"
        
        # Create a new registry with the same file
        new_registry = ServerRegistry(registry_file)
        
        # Check that the new registry has the server
        assert len(new_registry.servers) == 1
        server_id = list(new_registry.servers.keys())[0]
        assert new_registry.servers[server_id]["name"] == "persistence-test-server"
        
        # Create a new deployment manager with the new registry
        new_manager = MCPServerDeploymentManager()
        new_manager.registry = new_registry
        
        # Check that the new manager can see the server
        servers = await new_manager.list_servers()
        assert len(servers) == 1
        assert servers[0]["name"] == "persistence-test-server"
        
        # Clean up
        await deployment_manager.delete_server("persistence-test-server")