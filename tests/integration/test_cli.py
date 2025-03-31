"""Integration tests for the CLI."""

import pytest
import asyncio
import os
import json
import tempfile
from unittest.mock import Mock, AsyncMock, patch
from typer.testing import CliRunner

from mcp_agent.cli.main import app
from mcp_agent.mcp.server_deployment import MCPServerDeploymentManager



@pytest.fixture
def runner():
    """Create a CliRunner for testing."""
    return CliRunner()


@pytest.fixture
def mock_deployment_manager():
    """Create a mock deployment manager for CLI testing."""
    # Create a mock deployment manager
    mock = Mock(spec=MCPServerDeploymentManager)
    
    # Mock the deploy_server method
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
    mock.deploy_server = AsyncMock(return_value=server_record)
    
    # Mock the list_servers method
    mock.list_servers = AsyncMock(return_value=[server_record])
    
    # Mock the stop_server, start_server, and delete_server methods
    mock.stop_server = AsyncMock()
    mock.start_server = AsyncMock()
    mock.delete_server = AsyncMock()
    
    return mock


@pytest.fixture
def patch_asyncio_run():
    """Patch asyncio.run to execute coroutines directly."""
    def mock_asyncio_run(coro):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)
            loop.close()
    
    with patch("asyncio.run", mock_asyncio_run):
        yield


class TestCLI:
    """Test the CLI commands."""
    
    def test_list_templates(self, runner):
        """Test the 'templates' command."""
        # Run the command with the correct path: deploy -> server -> templates
        result = runner.invoke(app, ["deploy", "server", "templates"])
        
        
        # Check the result
        assert result.exit_code == 0
        assert "Available MCP Server Templates" in result.stdout
        assert "fetch" in result.stdout
        assert "filesystem" in result.stdout
    
    def test_deploy_server(self, runner, mock_deployment_manager, patch_asyncio_run):
        """Test the 'deploy' command."""
        # Patch the deployment manager to use our mock
        with patch("mcp_agent.cli.commands.deploy.server.deployment_manager", mock_deployment_manager):
            # Run the command with the correct path: deploy -> server -> deploy
            result = runner.invoke(app, ["deploy", "server", "deploy", "test-server", "--type", "fetch"])
            
            # Check the result
            assert result.exit_code == 0
            assert "Successfully deployed test-server MCP server!" in result.stdout
            assert "Server URL: https://test-server.mcp-agent-cloud.example.com" in result.stdout
            
            # Check that the deploy_server method was called
            mock_deployment_manager.deploy_server.assert_called_once()
            
            # Check the arguments for deploy_server
            args, kwargs = mock_deployment_manager.deploy_server.call_args
            assert kwargs["server_name"] == "test-server"
            assert kwargs["server_type"] == "fetch"
    
    def test_list_servers(self, runner, mock_deployment_manager, patch_asyncio_run):
        """Test the 'list' command."""
        # Patch the deployment manager to use our mock
        with patch("mcp_agent.cli.commands.deploy.server.deployment_manager", mock_deployment_manager):
            # Run the command with the correct path: deploy -> server -> list
            result = runner.invoke(app, ["deploy", "server", "list"])
            
            # Check the result
            assert result.exit_code == 0
            assert "Deployed MCP Servers" in result.stdout
            assert "test-server" in result.stdout
            assert "fetch" in result.stdout
            
            # Check that the list_servers method was called
            mock_deployment_manager.list_servers.assert_called_once()
    
    def test_stop_server(self, runner, mock_deployment_manager, patch_asyncio_run):
        """Test the 'stop' command."""
        # Patch the deployment manager to use our mock
        with patch("mcp_agent.cli.commands.deploy.server.deployment_manager", mock_deployment_manager):
            # Run the command with the correct path: deploy -> server -> stop
            result = runner.invoke(app, ["deploy", "server", "stop", "test-server"])
            
            # Check the result
            assert result.exit_code == 0
            assert "Successfully stopped test-server MCP server!" in result.stdout
            
            # Check that the stop_server method was called
            mock_deployment_manager.stop_server.assert_called_once_with("test-server")
    
    def test_start_server(self, runner, mock_deployment_manager, patch_asyncio_run):
        """Test the 'start' command."""
        # Patch the deployment manager to use our mock
        with patch("mcp_agent.cli.commands.deploy.server.deployment_manager", mock_deployment_manager):
            # Run the command with the correct path: deploy -> server -> start
            result = runner.invoke(app, ["deploy", "server", "start", "test-server"])
            
            # Check the result
            assert result.exit_code == 0
            assert "Successfully started test-server MCP server!" in result.stdout
            
            # Check that the start_server method was called
            mock_deployment_manager.start_server.assert_called_once_with("test-server")
    
    def test_delete_server(self, runner, mock_deployment_manager, patch_asyncio_run):
        """Test the 'delete' command."""
        # Patch the deployment manager to use our mock
        with patch("mcp_agent.cli.commands.deploy.server.deployment_manager", mock_deployment_manager):
            # Run the command with the correct path: deploy -> server -> delete and --force to bypass confirmation
            result = runner.invoke(app, ["deploy", "server", "delete", "test-server", "--force"])
            
            # Check the result
            assert result.exit_code == 0
            assert "Successfully deleted test-server MCP server!" in result.stdout
            
            # Check that the delete_server method was called
            mock_deployment_manager.delete_server.assert_called_once_with("test-server")
    
    def test_delete_server_confirmation(self, runner, mock_deployment_manager, patch_asyncio_run):
        """Test the 'delete' command with confirmation."""
        # Patch the deployment manager to use our mock
        with patch("mcp_agent.cli.commands.deploy.server.deployment_manager", mock_deployment_manager):
            # Run the command with the correct path: deploy -> server -> delete without --force to trigger confirmation
            result = runner.invoke(app, ["deploy", "server", "delete", "test-server"], input="y\n")
            
            # Check the result
            assert result.exit_code == 0
            assert "Are you sure you want to delete the server 'test-server'?" in result.stdout
            assert "Successfully deleted test-server MCP server!" in result.stdout
            
            # Check that the delete_server method was called
            mock_deployment_manager.delete_server.assert_called_once_with("test-server")
            
            # Reset the mock
            mock_deployment_manager.delete_server.reset_mock()
            
            # Run the command with the correct path: deploy -> server -> delete without --force but answer 'n' to confirmation
            result = runner.invoke(app, ["deploy", "server", "delete", "test-server"], input="n\n")
            
            # Check the result
            assert result.exit_code == 0
            assert "Are you sure you want to delete the server 'test-server'?" in result.stdout
            assert "Operation cancelled." in result.stdout
            
            # Check that the delete_server method was not called
            mock_deployment_manager.delete_server.assert_not_called()