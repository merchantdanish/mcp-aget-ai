import pytest
import anyio
import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Tuple, Dict
from unittest.mock import patch, AsyncMock, MagicMock, call

import mcp.types as types
from mcp_agent.mcp.websocket import websocket_client
from mcp_agent.config import MCPServerSettings, MCPServerAuthSettings


class MockWebSocket:
    """Mock WebSocket connection for testing."""

    def __init__(self, messages: List[str] = None):
        self.messages = messages or []
        self.sent_messages = []
        self._message_iterator = None

    async def send(self, data: str):
        """Store sent messages."""
        self.sent_messages.append(data)

    def __aiter__(self):
        """Return self as an async iterator."""
        self._message_iterator = iter(self.messages)
        return self

    async def __anext__(self):
        """Return the next message or raise StopAsyncIteration."""
        try:
            return next(self._message_iterator)
        except StopIteration:
            raise StopAsyncIteration


@asynccontextmanager
async def mock_ws_connect(*args, **kwargs):
    """Mock the websockets.connect function."""
    mock_ws = MockWebSocket([
        json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 1}),
        json.dumps({"jsonrpc": "2.0", "result": {"protocolVersion": "2023-12-07"}, "id": 1})
    ])
    yield mock_ws


@pytest.mark.asyncio
async def test_websocket_client():
    """Test that websocket_client correctly handles JSON-RPC messages."""

    with patch("mcp_agent.mcp.websocket.ws_connect", new=mock_ws_connect):
        async with websocket_client("ws://localhost:8000") as (read_stream, write_stream):
            # Test receiving a message
            message = await read_stream.receive()
            assert isinstance(message, types.JSONRPCMessage)
            assert message.method == "initialize"

            # Test sending a message
            response = types.JSONRPCMessage(
                jsonrpc="2.0",
                result={"message": "Hello World"},
                id=1
            )
            await write_stream.send(response)

            # Wait a bit for the message to be processed
            await anyio.sleep(0.1)

            # Receive the second message
            message = await read_stream.receive()
            assert isinstance(message, types.JSONRPCMessage)
            assert message.result["protocolVersion"] == "2023-12-07"


@pytest.mark.asyncio
async def test_websocket_client_with_auth():
    """Test that websocket_client correctly handles authentication."""

    with patch("mcp_agent.mcp.websocket.ws_connect", side_effect=mock_ws_connect) as mock_connect:
        auth_settings = {"api_key": "test-api-key"}
        async with websocket_client("ws://localhost:8000", auth_settings=auth_settings) as (read_stream, write_stream):
            # Verify that ws_connect was called with the correct headers
            mock_connect.assert_called_once()
            _, kwargs = mock_connect.call_args
            assert "extra_headers" in kwargs
            assert kwargs["extra_headers"]["Authorization"] == "Bearer test-api-key"


@pytest.mark.asyncio
async def test_transport_factory_websocket():
    """Test that the websocket transport is correctly configured in the connection manager."""

    from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager

    # Create a configuration with websocket transport
    config = MCPServerSettings(
        name="test-server",
        transport="websocket",
        url="ws://localhost:8000"
    )

    # Mock the MCPConnectionManager to avoid actual connections
    with patch("mcp_agent.mcp.mcp_connection_manager.websocket_client", new=AsyncMock()) as mock_websocket:
        # Create a mock registry
        mock_registry = MagicMock()
        mock_registry.get_server_config.return_value = config

        # Create a connection manager with the mock registry
        connection_manager = MCPConnectionManager(mock_registry)

        # Call launch_server which will use our transport_context_factory
        with patch.object(connection_manager, "_tg", new=MagicMock()):
            with patch.object(connection_manager, "running_servers", new={}):
                await connection_manager.launch_server("test-server", MagicMock())

        # Check that websocket_client was called with the right URL and no auth
        mock_websocket.assert_called_once_with("ws://localhost:8000", auth_settings=None)


@pytest.mark.asyncio
async def test_transport_factory_websocket_with_auth():
    """Test that the websocket transport passes authentication settings correctly."""

    from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager

    # Create a configuration with websocket transport and auth
    config = MCPServerSettings(
        name="test-server",
        transport="websocket",
        url="ws://localhost:8000",
        auth=MCPServerAuthSettings(api_key="test-api-key")
    )

    # Mock the MCPConnectionManager to avoid actual connections
    with patch("mcp_agent.mcp.mcp_connection_manager.websocket_client", new=AsyncMock()) as mock_websocket:
        # Create a mock registry
        mock_registry = MagicMock()
        mock_registry.get_server_config.return_value = config

        # Create a connection manager with the mock registry
        connection_manager = MCPConnectionManager(mock_registry)

        # Call launch_server which will use our transport_context_factory
        with patch.object(connection_manager, "_tg", new=MagicMock()):
            with patch.object(connection_manager, "running_servers", new={}):
                await connection_manager.launch_server("test-server", MagicMock())

        # Check that websocket_client was called with auth settings
        mock_websocket.assert_called_once()
        args, kwargs = mock_websocket.call_args
        assert args[0] == "ws://localhost:8000"
        assert kwargs["auth_settings"] == {"api_key": "test-api-key"}


@pytest.mark.asyncio
async def test_missing_url_websocket():
    """Test that an error is raised when url is missing for websocket transport."""

    from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager

    # Create a configuration with websocket transport but no URL
    config = MCPServerSettings(
        name="test-server",
        transport="websocket",
        url=None
    )

    # Mock the MCPConnectionManager to avoid actual connections
    mock_registry = MagicMock()
    mock_registry.get_server_config.return_value = config

    # Create a connection manager with the mock registry
    connection_manager = MCPConnectionManager(mock_registry)

    # Call launch_server which will use our transport_context_factory
    with patch.object(connection_manager, "_tg", new=MagicMock()):
        with patch.object(connection_manager, "running_servers", new={}):
            with pytest.raises(ValueError, match="URL is required for WebSocket transport"):
                await connection_manager.launch_server("test-server", MagicMock())