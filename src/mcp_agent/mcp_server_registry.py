"""
This module defines a `ServerRegistry` class for managing MCP server configurations
and initialization logic.

The class loads server configurations from a YAML file,
supports dynamic registration of initialization hooks, and provides methods for
server initialization.
"""

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Callable, Coroutine, List, Literal, Dict, AsyncGenerator

import anyio
import yaml
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import BaseModel, ConfigDict
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client


class AuthConfig(BaseModel):
    """Represents authentication configuration for a server."""

    api_key: str | None = None

    model_config = ConfigDict(extra="allow")


class ServerConfig(BaseModel):
    """
    Represents the configuration for an individual server.
    """

    transport: Literal["stdio", "sse"] = "stdio"
    """The transport mechanism."""

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

ReceiveLoopCallable = Callable[[ClientSession], Coroutine[None, None, None]]
"""
A type alias for a receive loop function that processes incoming messages from the server.
"""


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

    @asynccontextmanager
    async def start_server(
        self,
        server_name: str,
        client_session_constructor: Callable[
            [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
            ClientSession,
        ] = ClientSession,
    ) -> AsyncGenerator[ClientSession]:
        """
        Starts the server process based on its configuration. To initialize, call initialize_server

        Args:
            server_name (str): The name of the server to initialize.

        Returns:
            StdioServerParameters: The server parameters for stdio transport.

        Raises:
            ValueError: If the server is not found or has an unsupported transport.
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

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
                session = client_session_constructor(
                    read_stream,
                    write_stream,
                    read_timeout_seconds,
                )
                print(f"Connected to server '{server_name}' using stdio transport.")
                try:
                    yield session
                finally:
                    print("Closing session...")

        elif config.transport == "sse":
            if not config.url:
                raise ValueError(f"URL is required for SSE transport: {server_name}")

            # Use sse_client to get the read and write streams
            async with sse_client(config.url) as (read_stream, write_stream):
                session = client_session_constructor(
                    read_stream,
                    write_stream,
                    read_timeout_seconds,
                )
                print(f"Connected to server '{server_name}' using SSE transport.")
                try:
                    yield session
                finally:
                    print("Closing session...")

        # Unsupported transport
        else:
            raise ValueError(f"Unsupported transport: {config.transport}")

    @asynccontextmanager
    async def initialize_server(
        self,
        server_name: str,
        receive_loop: ReceiveLoopCallable,
        client_session_constructor: Callable[
            [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
            ClientSession,
        ] = ClientSession,
        init_hook: InitHookCallable = None,
    ) -> AsyncGenerator[ClientSession]:
        """
        Initialize a server based on its configuration.
        After initialization, also calls any registered or provided initialization hook for the server.

        Args:
            server_name (str): The name of the server to initialize.
            receive_loop (ReceiveLoopCallable): The message loop function for processing incoming messages.
            init_hook (InitHookCallable): Optional initialization hook function to call after initialization.

        Returns:
            StdioServerParameters: The server parameters for stdio transport.

        Raises:
            ValueError: If the server is not found or has an unsupported transport.
        """

        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        config = self.registry[server_name]

        async with (
            self.start_server(
                server_name, client_session_constructor=client_session_constructor
            ) as session,
            anyio.create_task_group() as tg,
        ):
            # We start the message loop in a separate task group
            tg.start_soon(receive_loop, session)

            print(f"Initializing server '{server_name}'...")
            await session.initialize()
            print(f"Initialized server '{server_name}'.")

            intialization_callback = (
                init_hook if init_hook is not None else self.init_hooks.get(server_name)
            )

            if intialization_callback:
                print(f"Executing init hook for '{server_name}'")
                intialization_callback(session, config.auth)
            yield session

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
