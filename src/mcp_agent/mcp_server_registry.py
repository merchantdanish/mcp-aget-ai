"""
This module defines a `ServerRegistry` class for managing MCP server configurations
and initialization logic.

The class loads server configurations from a YAML file,
supports dynamic registration of initialization hooks, and provides methods for
server initialization.
"""

from datetime import timedelta
from typing import Callable, List, Dict

import yaml
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import BaseModel
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client


class AuthConfig(BaseModel):
    """Represents authentication configuration for a server."""

    api_key: str | None = None

    class Config:
        """Allow extra properties for the configuration."""

        extra = "allow"


class ServerConfig(BaseModel):
    """
    Represents the configuration for an individual server.
    """

    transport: str
    """The transport mechanism (e.g., "stdio", "sse")."""

    command: str | None = None
    """The command to execute the server (e.g. npx)."""

    args: List[str] | None = None
    """The arguments for the server command."""

    read_timeout_seconds: int | None = None
    """The timeout in seconds for the server connection."""

    url: str | None = None
    """The URL for the server (e.g. for SSE transport)."""

    auth: AuthConfig | None = None
    """The authentication configuration for the server."""


InitHookCallable = Callable[[ClientSession | None, AuthConfig | None], bool]
"""
A type alias for an initialization hook function that is invoked after MCP server initialization.

Args:
    session (ClientSession | None): The client session for the server connection.
    auth (AuthConfig | None): The authentication configuration for the server.

Returns:
    bool: Result of the post-init hook (false indicates failure).
"""


class ClosableClientSession:
    """
    A wrapper around MCP's ClientSession that tracks the underlying streams
    and provides a `close` method for proper cleanup.
    """

    def __init__(
        self,
        read_stream: MemoryObjectReceiveStream,
        write_stream: MemoryObjectSendStream,
        read_timeout_seconds: timedelta | None = None,
    ):
        self._read_stream = read_stream
        self._write_stream = write_stream
        self.session = ClientSession(
            read_stream=read_stream,
            write_stream=write_stream,
            read_timeout_seconds=read_timeout_seconds,
        )

    async def close(self) -> bool:
        """
        Close the underlying streams to release resources.
        """
        await self._read_stream.aclose()
        await self._write_stream.aclose()
        print("Session closed successfully.")
        return True


class ServerRegistry:
    """
    A registry for managing server configurations and initialization logic.

    The `ServerRegistry` class is responsible for loading server configurations
    from a YAML file, registering initialization hooks, initializing servers,
    and executing post-initialization hooks dynamically.

    Attributes:
        config_path (str): Path to the YAML configuration file.
        registry (Dict[str, ServerConfig]): Loaded server configurations.
        init_hooks (Dict[str, InitHookCallable]): Registered initialization hooks.
    """

    def __init__(self, config_path: str):
        """
        Initialize the ServerRegistry with a configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.registry = self.load_registry(config_path)
        self.init_hooks: dict[str, InitHookCallable] = {}
        self.sessions: dict[str, ClosableClientSession] = {}

    def load_registry(self, config_path: str | None = None) -> Dict[str, ServerConfig]:
        """
        Load the YAML configuration file and validate it.

        Returns:
            Dict[str, ServerConfig]: A dictionary of server configurations.

        Raises:
            ValueError: If the configuration is invalid.
        """

        if config_path is not None:
            self.config_path = config_path

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        servers = {
            name: ServerConfig(**config) for name, config in data["servers"].items()
        }

        return servers

    async def initialize(self):
        """
        Initialize all servers in the registry. If any sessions are already active, they will be closed first.

        Returns:
            Dict[str, ClientSession]: A dictionary of client sessions for each server.
        """

        for server_name in self.sessions:
            is_closed = await self.close_session(server_name)
            if not is_closed:
                print(
                    f"Warning: Failed to close session for server '{server_name}'. Continuing with reinitialization..."
                )

        for server_name in self.registry:
            session = await self.initialize_server(server_name)
            self.execute_init_hook(server_name, session)
            self.sessions[server_name] = session

        sessions = {name: self.sessions[name].session for name in self.sessions}

        return sessions

    async def initialize_server(
        self, server_name: str, init_hook: InitHookCallable = None
    ) -> ClientSession:
        """
        Initialize a server based on its configuration.

        Args:
            server_name (str): The name of the server to initialize.

        Returns:
            StdioServerParameters: The server parameters for stdio transport.

        Raises:
            ValueError: If the server is not found or has an unsupported transport.
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        if server_name in self.sessions:
            print(
                f"Session already exists for server '{server_name}'. Reinitializing..."
            )
            await self.close_session(server_name)

        config = self.registry[server_name]

        read_timeout_seconds = (
            timedelta(config.read_timeout_seconds)
            if config.read_timeout_seconds
            else None
        )

        if config.transport == "stdio":
            if not config.command or not config.args:
                raise ValueError(
                    f"Command and args are required for stdio transport: {server_name}"
                )

            server_params = StdioServerParameters(
                command=config.command, args=config.args
            )

            async with stdio_client(server_params) as (read_stream, write_stream):
                closable_session = ClosableClientSession(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    read_timeout_seconds=read_timeout_seconds,
                )
                session = closable_session.session
                print(f"Connected to server '{server_name}' using stdio transport.")
                if init_hook:
                    print(f"Executing init hook for '{server_name}'")
                    init_hook(session, config.auth)
                self.sessions[server_name] = closable_session
                return session

        elif config.transport == "sse":
            if not config.url:
                raise ValueError(f"URL is required for SSE transport: {server_name}")

            # Use sse_client to get the read and write streams
            async with sse_client(config.url) as (read_stream, write_stream):
                closable_session = ClosableClientSession(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    read_timeout_seconds=read_timeout_seconds,
                )
                session = closable_session.session

                print(f"Connected to server '{server_name}' using SSE transport.")
                if init_hook:
                    print(f"Executing init hook for '{server_name}'")
                    init_hook(session, config.auth)
                self.sessions[server_name] = closable_session
                return session

        # Unsupported transport
        else:
            raise ValueError(f"Unsupported transport: {config.transport}")

    async def close_session(self, server_name: str) -> bool:
        """
        Close the session for a specific server by closing its underlying streams.

        Args:
            server_name (str): The name of the server to close.

        Raises:
            ValueError: If the server has no active session.
        """
        if server_name not in self.sessions:
            raise ValueError(f"No active session found for server '{server_name}'.")

        session = self.sessions[server_name]
        try:
            # Close the underlying streams
            await session.close()
            print(f"Closed session for server '{server_name}'.")
        except Exception as e:
            print(f"Error while closing session for server '{server_name}': {e}")
            return False
        finally:
            del self.sessions[server_name]

        return True

    def register_init_hook(self, server_name: str, hook: InitHookCallable) -> None:
        """
        Register an initialization hook for a specific server. This will get called
        after the server is initialized.

        Args:
            server_name (str): The name of the server.
            hook (callable): The initialization function to register.
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        self.init_hooks[server_name] = hook

    def execute_init_hook(self, server_name: str, session=None) -> bool:
        """
        Execute the initialization hook for a specific server.

        Args:
            server_name (str): The name of the server.
            session: The session object to pass to the initialization hook.
        """
        if server_name in self.init_hooks:
            hook = self.init_hooks[server_name]
            config = self.registry[server_name]
            print(f"Executing init hook for '{server_name}'")
            return hook(session, config.auth)
        else:
            print(f"No init hook registered for '{server_name}'")
