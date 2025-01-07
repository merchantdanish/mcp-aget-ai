"""
This module defines a `ServerRegistry` class for managing MCP server configurations
and initialization logic.

The class loads server configurations from a YAML file,
supports dynamic registration of initialization hooks, and provides methods for
server initialization.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Callable, Coroutine, Dict, AsyncGenerator

import anyio
import anyio.abc
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.types import ServerNotification

from mcp_agent.config import (
    get_settings,
    MCPServerAuthSettings,
    MCPServerSettings,
    Settings,
)
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)

InitHookCallable = Callable[[ClientSession | None, MCPServerAuthSettings | None], bool]
"""
A type alias for an initialization hook function that is invoked after MCP server initialization.

Args:
    session (ClientSession | None): The client session for the server connection.
    auth (MCPServerAuthSettings | None): The authentication configuration for the server.

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
        registry (Dict[str, MCPServerSettings]): Loaded server configurations.
        init_hooks (Dict[str, InitHookCallable]): Registered initialization hooks.
    """

    def __init__(self, config: Settings | None = None, config_path: str | None = None):
        """
        Initialize the ServerRegistry with a configuration file.

        Args:
            config (Settings): The Settings object containing the server configurations.
            config_path (str): Path to the YAML configuration file.
        """
        self.registry = (
            self.load_registry_from_file(config_path)
            if config is None
            else config.mcp.servers
        )
        self.init_hooks: Dict[str, InitHookCallable] = {}
        self.connection_manager = MCPConnectionManager(self)

    def load_registry_from_file(
        self, config_path: str | None = None
    ) -> Dict[str, MCPServerSettings]:
        """
        Load the YAML configuration file and validate it.

        Returns:
            Dict[str, MCPServerSettings]: A dictionary of server configurations.

        Raises:
            ValueError: If the configuration is invalid.
        """

        servers = get_settings(config_path).mcp.servers or {}
        return servers

    @asynccontextmanager
    async def start_server(
        self,
        server_name: str,
        client_session_constructor: Callable[
            [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
            ClientSession,
        ] = ClientSession,
    ) -> AsyncGenerator[ClientSession, None]:
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
                logger.info(
                    f"Connected to server '{server_name}' using stdio transport."
                )
                try:
                    yield session
                finally:
                    logger.info("Closing session...")

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
                logger.info(f"Connected to server '{server_name}' using SSE transport.")
                try:
                    yield session
                finally:
                    logger.info("Closing session...")

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
    ) -> AsyncGenerator[ClientSession, None]:
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

            logger.info(f"Initializing server '{server_name}'...")
            await session.initialize()
            logger.info(f"Initialized server '{server_name}'.")

            intialization_callback = (
                init_hook if init_hook is not None else self.init_hooks.get(server_name)
            )

            if intialization_callback:
                logger.info(f"Executing init hook for '{server_name}'")
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
            logger.info(f"Executing init hook for '{server_name}'")
            return hook(session, config.auth)
        else:
            logger.info(f"No init hook registered for '{server_name}'")

    def get_server_config(self, server_name: str) -> MCPServerSettings | None:
        """
        Get the configuration for a specific server.

        Args:
            server_name (str): The name of the server.

        Returns:
            MCPServerSettings: The server configuration.
        """
        return self.registry.get(server_name)


class ServerConnection:
    """
    Represents a long-lived MCP server connection, including:
      - The ClientSession
      - The background receive loop task group
      - The cleanup functionality
    """

    def __init__(
        self,
        session: ClientSession,
        task_group: anyio.abc.TaskGroup,
        transport_context: Any | None = None,
    ):
        self.session = session
        self._task_group = task_group
        self._transport_context = transport_context

    async def disconnect(self):
        """
        Cancel background tasks and clean up the connection.
        """
        try:
            # Cancel the task group first
            self._task_group.cancel_scope.cancel()
            await self._task_group.__aexit__(None, None, None)

            # Clean up transport context if it exists
            if self._transport_context:
                await self._transport_context.__aexit__(None, None, None)

            # Note: ClientSession doesn't have a close method
            # The session cleanup is handled by the transport context cleanup

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            raise


class MCPConnectionManager:
    """
    Manages persistent connections to multiple MCP servers.
    """

    def __init__(self, server_registry: ServerRegistry):
        self.server_registry = server_registry
        self.running_servers: Dict[str, ServerConnection] = {}
        self._lock = asyncio.Lock()

    async def default_receive_loop(self, session: ClientSession):
        """
        A default message receive loop to handle messages from the server.
        """
        async for message in session.incoming_messages:
            if isinstance(message, Exception):
                logger.error(f"Error in receive loop: {message}")
                continue
            elif isinstance(message, ServerNotification):
                logger.info(f"Received notification: {message}")
                continue
            else:
                # This is a message request (RequestResponder[ServerRequest, ClientResult])
                logger.info(f"Received message request: {message}")
                continue

    async def launch_server(
        self,
        server_name: str,
        client_session_constructor: Callable[
            [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
            ClientSession,
        ] = ClientSession,
        receive_loop: ReceiveLoopCallable | None = None,
        init_hook: InitHookCallable | None = None,
    ) -> ServerConnection:
        """
        Connect to a server and return a RunningServer instance that will persist
        until explicitly disconnected.
        """
        config = self.server_registry.registry.get(server_name)

        if not config:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        read_timeout = (
            timedelta(seconds=config.read_timeout_seconds)
            if config.read_timeout_seconds
            else None
        )

        # Create task group for background tasks
        task_group = await anyio.create_task_group().__aenter__()

        try:
            # Handle different transport types
            if config.transport == "stdio":
                if not config.command or not config.args:
                    raise ValueError(
                        f"Command and args required for stdio transport: {server_name}"
                    )

                server_params = StdioServerParameters(
                    command=config.command, args=config.args
                )
                # Get stdio context and streams
                stdio_ctx = stdio_client(server_params)

                read_stream, write_stream = await stdio_ctx.__aenter__()  # pylint: disable=E1101

                # Create the session
                session = client_session_constructor(
                    read_stream, write_stream, read_timeout
                )

                transport_context = stdio_ctx

            elif config.transport == "sse":
                if not config.url:
                    raise ValueError(f"URL required for SSE transport: {server_name}")

                # Get SSE context and streams
                sse_ctx = sse_client(config.url)
                read_stream, write_stream = await sse_ctx.__aenter__()  # pylint: disable=E1101

                # Create the session
                session = client_session_constructor(
                    read_stream, write_stream, read_timeout
                )

                transport_context = sse_ctx

            else:
                raise ValueError(f"Unsupported transport: {config.transport}")

            # Initialize the session
            logger.info(f"Initializing server '{server_name}'...")
            await session.initialize()
            logger.info(f"Initialized server '{server_name}'.")

            # Start the receive loop
            receive_loop_func = receive_loop or self.default_receive_loop
            task_group.start_soon(receive_loop_func, session)

            # Run init hook if provided
            intialization_callback = init_hook or self.server_registry.init_hooks.get(
                server_name
            )
            if intialization_callback:
                logger.info(f"Running init hook for '{server_name}'")
                intialization_callback(session, config.auth)

            # Create ServerConnection instance
            running_server = ServerConnection(
                session=session,
                task_group=task_group,
                transport_context=transport_context,
            )

            # Store in our dictionary of running servers
            self.running_servers[server_name] = running_server

            logger.info(f"Server '{server_name}' is up and running!")
            return running_server

        except Exception as e:
            # Clean up on error
            logger.error(f"Error launching server '{server_name}': {e}")
            await task_group.__aexit__(None, None, None)
            if "transport_context" in locals():
                await transport_context.__aexit__(None, None, None)
            raise

    async def get_server(
        self,
        server_name: str,
        client_session_constructor: Callable = ClientSession,
        receive_loop: ReceiveLoopCallable | None = None,
        init_hook: InitHookCallable | None = None,
    ) -> ServerConnection:
        """
        Get a running server instance, launching it if needed.
        """
        async with self._lock:
            server_connection = self.running_servers.get(server_name)
            if server_connection:
                return server_connection

            return await self.launch_server(
                server_name=server_name,
                client_session_constructor=client_session_constructor,
                receive_loop=receive_loop,
                init_hook=init_hook,
            )

    async def disconnect_server(self, server_name: str):
        """
        Disconnect a specific server if it's running.
        """
        async with self._lock:
            server_connection = self.running_servers.get(server_name)
            if server_connection:
                await server_connection.disconnect()
                del self.running_servers[server_name]

    async def disconnect_all(self):
        """
        Disconnect all running servers.
        """
        async with self._lock:
            for server_name in list(self.running_servers.keys()):
                server_connection = self.running_servers.get(server_name)
                if server_connection:
                    await server_connection.disconnect()
                    del self.running_servers[server_name]
