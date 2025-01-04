import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncGenerator, Callable

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.shared.session import RequestResponder
from mcp.types import (
    CreateMessageRequest,
    CreateMessageResult,
    ErrorData,
    ServerNotification,
    ServerRequest,
    ClientResult,
    TextContent,
)

from mcp_agent.mcp_server_registry import ReceiveLoopCallable, ServerRegistry
from mcp_agent.context import get_current_context, get_current_config

logger = logging.getLogger(__name__)


class MCPAgentClientSession(ClientSession):
    """
    MCP Agent framework acts as a client to the servers providing tools/resources/prompts for the agent workloads.
    This is a simple client session for those server connections, and supports
        - handling sampling requests
        - notifications

    Developers can extend this class to add more custom functionality as needed
    """

    async def _received_request(
        self, responder: RequestResponder[ServerRequest, ClientResult]
    ) -> None:
        request = responder.request.root

        if isinstance(request, CreateMessageRequest):
            return await self.handle_sampling_request(request, responder)

        # Handle other requests as usual
        await super()._received_request(responder)

    async def handle_sampling_request(
        self,
        request: CreateMessageRequest,
        responder: RequestResponder[ServerRequest, ClientResult],
    ):
        ctx = get_current_context()
        config = get_current_config()
        session = ctx.upstream_session
        if session is None:
            # TODO: saqadri - consider whether we should be handling the sampling request here as a client
            print(
                f"Error: No upstream client available for sampling requests. Request: {request}"
            )
            try:
                from anthropic import AsyncAnthropic

                client = AsyncAnthropic(api_key=config.anthropic.api_key)

                params = request.params
                response = await client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=params.maxTokens,
                    messages=[
                        {
                            "role": m.role,
                            "content": m.content.text
                            if hasattr(m.content, "text")
                            else m.content.data,
                        }
                        for m in params.messages
                    ],
                    system=getattr(params, "systemPrompt", None),
                    temperature=getattr(params, "temperature", 0.7),
                    stop_sequences=getattr(params, "stopSequences", None),
                )

                await responder.respond(
                    CreateMessageResult(
                        model="claude-3-sonnet-20240229",
                        role="assistant",
                        content=TextContent(type="text", text=response.content[0].text),
                    )
                )
            except Exception as e:
                logger.error(f"Error handling sampling request: {e}")
                await responder.respond(ErrorData(code=-32603, message=str(e)))
        else:
            try:
                # If a session is available, we'll pass-through the sampling request to the upstream client
                result = await session.send_request(
                    request=ServerRequest(request), result_type=CreateMessageResult
                )

                # Pass the result from the upstream client back to the server. We just act as a pass-through client here.
                await responder.send_result(result)
            except Exception as e:
                await responder.send_error(code=-32603, message=str(e))


async def receive_loop(session: ClientSession):
    """
    A default message receive loop to handle messages from the server.
    Developers can extend this function to add more custom message handling as needed
    """
    logger.info("Starting receive loop")
    async for message in session.incoming_messages:
        if isinstance(message, Exception):
            logger.error("Error: %s", message)
            continue
        elif isinstance(message, ServerNotification):
            logger.info("Received notification from server: %s", message)
            continue
        else:
            # This is a message request (RequestResponder[ServerRequest, ClientResult])
            # TODO: saqadri - handle this message request as needed
            continue


@asynccontextmanager
async def gen_client(
    server_name: str,
    client_session_constructor: Callable[
        [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
        ClientSession,
    ] = MCPAgentClientSession,
    server_registry: ServerRegistry | None = None,
    message_receive_loop: ReceiveLoopCallable = receive_loop,
) -> AsyncGenerator[ClientSession, None]:
    """
    Create a client session to the specified server.
    Handles server startup, initialization, and message receive loop setup.
    If required, callers can specify their own message receive loop and ClientSession class constructor to customize further.
    """
    ctx = get_current_context()
    server_registry = server_registry or ctx.server_registry

    if not server_registry:
        raise ValueError(
            "Server registry not found in the context. Please specify one either on this method, or in the context."
        )

    async with server_registry.initialize_server(
        server_name=server_name,
        receive_loop=message_receive_loop,
        client_session_constructor=client_session_constructor,
    ) as session:
        yield session


async def connect(
    server_name: str,
    client_session_constructor: Callable[
        [MemoryObjectReceiveStream, MemoryObjectSendStream, timedelta | None],
        ClientSession,
    ] = MCPAgentClientSession,
    server_registry: ServerRegistry | None = None,
    message_receive_loop: ReceiveLoopCallable = receive_loop,
) -> ClientSession:
    """
    Create a persistent client session to the specified server.
    Handles server startup, initialization, and message receive loop setup.
    If required, callers can specify their own message receive loop and ClientSession class constructor to customize further.
    """
    ctx = get_current_context()
    server_registry = server_registry or ctx.server_registry

    if not server_registry:
        raise ValueError(
            "Server registry not found in the context. Please specify one either on this method, or in the context."
        )

    server_connection = await server_registry.connection_manager.get_server(
        server_name=server_name,
        client_session_constructor=client_session_constructor,
        receive_loop=message_receive_loop,
    )

    return server_connection.session


async def disconnect(
    server_name: str | None,
    server_registry: ServerRegistry | None = None,
) -> None:
    """
    Disconnect from the specified server. If server_name is None, disconnect from all servers.
    """
    ctx = get_current_context()
    server_registry = server_registry or ctx.server_registry

    if not server_registry:
        raise ValueError(
            "Server registry not found in the context. Please specify one either on this method, or in the context."
        )

    if server_name:
        await server_registry.connection_manager.disconnect_server(
            server_name=server_name
        )
    else:
        await server_registry.connection_manager.disconnect_all_servers()
