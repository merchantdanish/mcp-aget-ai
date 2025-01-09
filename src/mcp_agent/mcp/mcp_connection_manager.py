import anyio
from anyio import Event, create_task_group, Lock
from datetime import timedelta
from typing import Any, Callable, Dict, TYPE_CHECKING

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client

from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp_agent.mcp_server_registry import ServerRegistry

logger = get_logger(__name__)


class ServerConnection:
    def __init__(
        self,
        server_name: str,
        transport_type: str,
        transport_context_factory: Callable[[], Any],
        session_factory: Callable[[Any, Any], "ClientSession"],
        init_hook: Callable[["ClientSession"], None] | None = None,
    ):
        self.server_name = server_name
        self.transport_type = transport_type
        self._transport_context_factory = transport_context_factory
        self._session_factory = session_factory
        self._init_hook = init_hook

        self.session: "ClientSession" | None = None

        # Signal that session is fully up and initialized
        self._initialized_event = Event()

        # Signal we want to shut down
        self._shutdown_event = Event()

    def request_shutdown(self) -> None:
        self._shutdown_event.set()


async def _server_lifecycle_task(server_conn: ServerConnection) -> None:
    """
    Runs inside the MCPConnectionManager's shared TaskGroup.
    """
    server_name = server_conn.server_name
    try:
        transport_context = server_conn._transport_context_factory()

        async with transport_context as (read_stream, write_stream):
            # Build a session
            session = server_conn._session_factory(read_stream, write_stream)
            server_conn.session = session

            async with session:
                logger.info(f"{server_name}: Initializing server session...")
                await session.initialize()
                logger.info(f"{server_name}: Session initialized.")

                # If there's an init hook, run it
                if server_conn._init_hook:
                    logger.info(f"{server_name}: Executing init hook.")
                    server_conn._init_hook(session)

                # Now the session is ready for use
                server_conn._initialized_event.set()

                # Wait until we’re asked to shut down
                await server_conn._shutdown_event.wait()

    except Exception as exc:
        logger.error(
            f"{server_name}: Lifecycle task encountered an error: {exc}", exc_info=True
        )
        # If there's an error, we should also set the event so that
        # 'get_server' won't hang
        server_conn._initialized_event.set()
        raise
    finally:
        logger.debug(f"{server_name}: _lifecycle_task is exiting.")


class MCPConnectionManager:
    def __init__(self, server_registry: "ServerRegistry"):
        self.server_registry = server_registry
        self.running_servers: Dict[str, ServerConnection] = {}
        self._lock = Lock()
        self._tg: anyio.abc.TaskGroup | None = None

    async def __aenter__(self):
        self._tg = create_task_group()
        await self._tg.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.info("MCPConnectionManager: shutting down all server tasks...")
        if self._tg:
            await self._tg.__aexit__(exc_type, exc_val, exc_tb)
        self._tg = None

    async def launch_server(
        self,
        server_name: str,
        client_session_constructor: Callable[
            [
                anyio.abc.ObjectReceiveStream,
                anyio.abc.ObjectSendStream,
                timedelta | None,
            ],
            "ClientSession",
        ],
        init_hook: Callable[["ClientSession", Any], None] | None = None,
    ) -> ServerConnection:
        if not self._tg:
            raise RuntimeError(
                "MCPConnectionManager must be used inside an 'async with' block."
            )

        config = self.server_registry.registry.get(server_name)
        if not config:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        logger.debug(
            f"{server_name}: Found server configuration=", data=config.model_dump()
        )

        read_timeout = (
            timedelta(seconds=config.read_timeout_seconds)
            if config.read_timeout_seconds
            else None
        )

        def transport_context_factory():
            if config.transport == "stdio":
                # if you removed 'env=' from your code, do it here as well
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                )
                return stdio_client(server_params)
            elif config.transport == "sse":
                return sse_client(config.url)
            else:
                raise ValueError(f"Unsupported transport: {config.transport}")

        def session_factory(read_stream, write_stream):
            return client_session_constructor(read_stream, write_stream, read_timeout)

        # Merge user’s init_hook with server_registry’s init_hook if needed
        final_init_hook = init_hook or self.server_registry.init_hooks.get(server_name)

        def maybe_run_init_hook(session: "ClientSession"):
            if final_init_hook:
                logger.info(f"{server_name}: Executing init hook")
                final_init_hook(session, config.auth)

        server_conn = ServerConnection(
            server_name=server_name,
            transport_type=config.transport,
            transport_context_factory=transport_context_factory,
            session_factory=session_factory,
            init_hook=maybe_run_init_hook,
        )

        async with self._lock:
            # Check if already running
            if server_name in self.running_servers:
                return self.running_servers[server_name]

            self.running_servers[server_name] = server_conn
            self._tg.start_soon(_server_lifecycle_task, server_conn)

        logger.info(f"{server_name}: Persistent connection started!")
        return server_conn

    async def get_server(
        self,
        server_name: str,
        client_session_constructor: Callable[
            [
                anyio.abc.ObjectReceiveStream,
                anyio.abc.ObjectSendStream,
                timedelta | None,
            ],
            "ClientSession",
        ],
        init_hook: Callable[["ClientSession", Any], None] | None = None,
    ) -> ServerConnection:
        # Get or launch the connection
        server_conn = await self.launch_server(
            server_name=server_name,
            client_session_constructor=client_session_constructor,
            init_hook=init_hook,
        )

        # Wait until it's fully initialized, or an error occurs
        await server_conn._initialized_event.wait()

        # If the session is still None, it means the lifecycle task crashed
        if server_conn.session is None:
            raise RuntimeError(
                f"Failed to initialize server '{server_name}'; "
                "check logs for errors."
            )
        return server_conn

    async def disconnect_server(self, server_name: str) -> None:
        logger.info(f"{server_name}: Disconnecting persistent connection to server...")

        async with self._lock:
            server_conn = self.running_servers.pop(server_name, None)
        if server_conn:
            server_conn.request_shutdown()
            logger.info(
                f"{server_name}: Shutdown signal sent (lifecycle task will exit)."
            )
        else:
            logger.info(f"{server_name}: No persistent connection found.")

    async def disconnect_all(self) -> None:
        logger.info("Disconnecting all persistent server connections...")
        async with self._lock:
            for conn in self.running_servers.values():
                conn.request_shutdown()
            self.running_servers.clear()
        logger.info("All persistent server connections signaled to disconnect.")
