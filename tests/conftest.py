"""Shared pytest fixtures for all tests."""

import os
import json
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from pathlib import Path
from typing import Dict, Any, List, Optional

from mcp_agent.auth.auth_service import AuthService
from mcp_agent.auth.auth_provider import SelfContainedAuthProvider, GitHubOAuthProvider
from mcp_agent.mcp.server_deployment import MCPServerDeploymentManager
from mcp_agent.mcp.server_templates import SERVER_TEMPLATES

# Create a temporary directory for test files
@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path

# Auth fixtures
@pytest.fixture
def auth_config():
    """Create a basic auth configuration."""
    return {
        "providers": {
            "self-contained": {
                "jwt_secret": "test-jwt-secret"
            },
            "github": {
                "client_id": "test-github-client-id",
                "client_secret": "test-github-client-secret",
                "jwt_secret": "test-github-jwt-secret"
            }
        },
        "clients_file": None  # No persistence for tests
    }

@pytest.fixture
def auth_service(auth_config):
    """Create an auth service with test configuration."""
    return AuthService(auth_config)

@pytest.fixture
def self_contained_provider():
    """Create a self-contained auth provider for testing."""
    config = {
        "jwt_secret": "test-jwt-secret"
    }
    return SelfContainedAuthProvider(config=config)

@pytest.fixture
def github_provider():
    """Create a GitHub auth provider for testing."""
    config = {
        "client_id": "test-github-client-id",
        "client_secret": "test-github-client-secret",
        "jwt_secret": "test-github-jwt-secret"
    }
    return GitHubOAuthProvider(config=config)

# MCP server fixtures
@pytest.fixture
def container_service_mock():
    """Create a mock container service."""
    mock = Mock()
    
    # Setup async methods
    mock.create_container = AsyncMock(return_value="test-container-id")
    mock.build_stdio_container = AsyncMock(return_value=("test-container-id", 8000))
    mock.stop_container = AsyncMock()
    mock.start_container = AsyncMock()
    mock.delete_container = AsyncMock()
    
    # Setup client property
    mock.client = Mock()
    
    return mock

@pytest.fixture
def auth_factory_mock():
    """Create a mock auth factory."""
    mock = Mock()
    
    # Setup create_server_auth method
    auth_service_mock = Mock()
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
    mock.create_server_auth = AsyncMock(return_value=(auth_service_mock, auth_config))
    
    return mock

@pytest.fixture
def server_registry_mock():
    """Create a mock server registry."""
    mock = Mock()
    
    # Setup server data
    test_server = {
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
    
    # Setup methods
    mock.register_server = AsyncMock()
    mock.get_server = AsyncMock(return_value=test_server)
    mock.get_server_by_name = AsyncMock(return_value=test_server)
    mock.list_servers = AsyncMock(return_value=[test_server])
    mock.update_server_status = AsyncMock()
    mock.delete_server = AsyncMock()
    
    return mock

@pytest.fixture
def deployment_manager(container_service_mock, auth_factory_mock, server_registry_mock):
    """Create a deployment manager with mocked dependencies."""
    manager = MCPServerDeploymentManager()
    manager.container_service = container_service_mock
    manager.auth_factory = auth_factory_mock
    manager.registry = server_registry_mock
    
    # Add a dummy HTTP server
    manager.http_servers = {
        "test-server": Mock()
    }
    
    return manager

# Integration test fixtures
@pytest.fixture
def integration_temp_dir(tmp_path):
    """Create a temporary directory for integration test files."""
    # Create subdirectories
    (tmp_path / "auth").mkdir()
    (tmp_path / "registry").mkdir()
    (tmp_path / "templates").mkdir()
    
    return tmp_path

@pytest.fixture
def integration_auth_service(integration_temp_dir):
    """Create an auth service for integration tests."""
    clients_file = integration_temp_dir / "auth" / "clients.json"
    
    config = {
        "providers": {
            "self-contained": {
                "jwt_secret": "integration-test-jwt-secret"
            }
        },
        "clients_file": str(clients_file)
    }
    
    return AuthService(config)

@pytest.fixture
def integration_deployment_manager(integration_temp_dir):
    """Create a deployment manager for integration tests."""
    # Create real registry file
    registry_file = integration_temp_dir / "registry" / "servers.json"
    
    # Create manager with configuration
    config = {
        "base_dir": str(integration_temp_dir),
        "registry_file": str(registry_file)
    }
    
    return MCPServerDeploymentManager(config)