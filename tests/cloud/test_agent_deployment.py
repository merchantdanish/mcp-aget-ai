"""Test the agent deployment functionality."""

import os
import pytest
import uuid
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

from mcp_agent.cloud.auth.auth_service import AuthService
from mcp_agent.cloud.deployment.agent_deployment import AgentDeploymentService

@pytest.fixture
def auth_service():
    """Create a mock authentication service."""
    with patch("mcp_agent.cloud.auth.auth_service.AuthService", autospec=True) as mock:
        # Set up mock auth service
        mock_service = mock.return_value
        mock_service.is_authenticated.return_value = True
        mock_service.get_access_token.return_value = "mock-token-" + uuid.uuid4().hex
        mock_service.ensure_authenticated = AsyncMock(return_value=(True, None))
        
        yield mock_service

@pytest.fixture
def deployment_service(auth_service):
    """Create a mock deployment service with the auth service."""
    service = AgentDeploymentService(auth_service)
    return service

@pytest.fixture
def test_agent_dir():
    """Create a temporary directory with a test agent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal agent structure
        agent_dir = Path(tmpdir) / "test-agent"
        agent_dir.mkdir()
        
        # Create config file
        config_content = """
name: test-agent
description: A test agent for deployment
execution_engine: asyncio
version: 1.0.0
"""
        with open(agent_dir / "mcp_agent.config.yaml", "w") as f:
            f.write(config_content)
            
        # Create a Python file
        with open(agent_dir / "main.py", "w") as f:
            f.write("""
import asyncio
from mcp_agent.app import MCPApp

app = MCPApp(name="test-agent")

async def main():
    await app.initialize()
    # Your agent code here
    
if __name__ == "__main__":
    asyncio.run(main())
""")
        
        yield agent_dir

@pytest.mark.asyncio
async def test_validate_agent_directory(deployment_service, test_agent_dir):
    """Test validating an agent directory."""
    # Should be valid
    is_valid, error = await deployment_service.validate_agent_directory(test_agent_dir)
    assert is_valid
    assert error is None
    
    # Test invalid directory
    is_valid, error = await deployment_service.validate_agent_directory(Path("/nonexistent"))
    assert not is_valid
    assert error is not None

@pytest.mark.asyncio
async def test_package_agent(deployment_service, test_agent_dir):
    """Test packaging an agent."""
    # Should package successfully
    is_packaged, error, package_data = await deployment_service.package_agent(test_agent_dir)
    assert is_packaged
    assert error is None
    assert package_data is not None
    assert len(package_data) > 0

@pytest.mark.asyncio
async def test_deploy_agent(deployment_service, test_agent_dir):
    """Test deploying an agent."""
    # Mock the internal methods
    deployment_service._upload_and_deploy = AsyncMock(return_value={
        "id": "agent-" + uuid.uuid4().hex[:8],
        "name": "test-agent",
        "status": "deployed",
        "url": "https://test-agent.mcp-agent-cloud.example.com"
    })
    
    # Should deploy successfully
    is_deployed, error, deployment_info = await deployment_service.deploy_agent(test_agent_dir)
    assert is_deployed
    assert error is None
    assert deployment_info is not None
    assert "id" in deployment_info
    assert "url" in deployment_info