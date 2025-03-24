"""
Custom implementation of stdio_client that handles stderr through rich console.
"""

from contextlib import asynccontextmanager
from typing import TextIO

import subprocess
import anyio
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio.streams.text import TextReceiveStream
from mcp.client.stdio import StdioServerParameters, get_default_environment
import mcp.types as types
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def stdio_client_with_rich_stderr(
    server: StdioServerParameters, errlog: int | TextIO = subprocess.PIPE
):
    """
    Client transport for stdio: this will connect to a server by spawning a
    process and communicating with it over stdin/stdout.
    """
    read_stream: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
    read_stream_writer: MemoryObjectSendStream[types.JSONRPCMessage | Exception]

    write_stream: MemoryObjectSendStream[types.JSONRPCMessage]
    write_stream_reader: MemoryObjectReceiveStream[types.JSONRPCMessage]

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    process = await anyio.open_process(
        [server.command, *server.args],
        env=(
            {**get_default_environment(), **server.env}
            if server.env is not None
            else get_default_environment()
        ),
        stderr=errlog,
        cwd=server.cwd,
    )

    async def stdout_reader():
        assert process.stdout, "Opened process is missing stdout"

        try:
            async with read_stream_writer:
                buffer = ""
                async for chunk in TextReceiveStream(
                    process.stdout,
                    encoding=server.encoding,
                    errors=server.encoding_error_handler,
                ):
                    lines = (buffer + chunk).split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        try:
                            message = types.JSONRPCMessage.model_validate_json(line)
                        except Exception as exc:
                            await read_stream_writer.send(exc)
                            continue

                        await read_stream_writer.send(message)
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def stderr_reader():
        assert process.stderr, "Opened process is missing stderr"
        try:
            async for chunk in TextReceiveStream(
                process.stderr,
                encoding=server.encoding,
                errors=server.encoding_error_handler,
            ):
                if chunk.strip():
                    # Let the logging system handle the formatting consistently
                    try:
                        logger.event(
                            "info", "mcpserver.stderr", chunk.rstrip(), None, {}
                        )
                    except Exception as e:
                        logger.error(
                            f"Error in stderr_reader handling output: {e}",
                            exc_info=True,
                        )
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()
        except Exception as e:
            logger.error(f"Unexpected error in stderr_reader: {e}", exc_info=True)

    async def stdin_writer():
        assert process.stdin, "Opened process is missing stdin"

        try:
            async with write_stream_reader:
                async for message in write_stream_reader:
                    json = message.model_dump_json(by_alias=True, exclude_none=True)
                    await process.stdin.send(
                        (json + "\n").encode(
                            encoding=server.encoding,
                            errors=server.encoding_error_handler,
                        )
                    )
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async with (
        anyio.create_task_group() as tg,
        process,
    ):
        tg.start_soon(stdout_reader)
        tg.start_soon(stdin_writer)
        tg.start_soon(stderr_reader)
        yield read_stream, write_stream


# TODO: saqadri (FA1) - See if the following is sufficient.
# @asynccontextmanager
# async def stdio_client_with_rich_stderr(
#     server: StdioServerParameters, errlog: TextIO = sys.stderr
# ):
#     """
#     A wrapper around the original stdio_client that captures stderr and routes it through our logger.

#     This implementation creates a custom wrapped errlog that forwards messages to our rich logger
#     while still using the original stdio_client function.

#     Args:
#         server: The server parameters for the stdio connection
#         errlog: Fallback error log (if rich logging fails)
#     """
#     # Instead of creating a full TextIO implementation, wrap the existing errlog
#     original_write = errlog.write

#     def rich_logger_write(text):
#         """Intercepts write calls to the errlog and logs them through our rich logger"""
#         print("rich_logger_write: HERE: ", text)
#         if text and text.strip():
#             try:
#                 logger.event("info", "mcpserver.stderr", text.rstrip(), None, {})
#             except Exception as e:
#                 logger.error(f"Error handling stderr output: {e}")
#                 # Fall back to original behavior

#         # Always call the original write method to maintain original behavior
#         return original_write(text)

#     # Replace the write method temporarily
#     errlog.write = rich_logger_write

#     try:
#         # Use the original stdio_client with our wrapped errlog
#         async with stdio_client(server, errlog=errlog) as (read_stream, write_stream):
#             yield read_stream, write_stream
#     finally:
#         # Restore the original write method
#         errlog.write = original_write
