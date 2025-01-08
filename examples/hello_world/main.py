import asyncio

from mcp_agent.context import get_current_context
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.gen_client import gen_client, connect, disconnect


async def example_usage():
    context = get_current_context()
    logger = get_logger(__name__)
    logger.info("Hello, world!")
    logger.info("Current config:", data=context.config.model_dump())

    # Use an async context manager to connect to the fetch server
    # At the end of the block, the connection will be closed automatically
    async with gen_client("fetch") as fetch_client:
        logger.info("fetch: Connected to server, calling list_tools...")
        result = await fetch_client.list_tools()
        logger.info("Tools available:", data=result.model_dump())

    # Connect to the filesystem server using a persistent connection via connect/disconnect
    # This is useful when you need to make multiple requests to the same server
    try:
        filesystem_client = await connect(server_name="filesystem")
        logger.info(
            "filesystem: Connected to server with persistent connection, calling list_tools..."
        )
        result = await filesystem_client.list_tools()
        logger.info("Tools available:", data=result.model_dump())
    finally:
        await disconnect(server_name="filesystem")
        logger.info("filesystem: Disconnected from server.")


if __name__ == "__main__":
    asyncio.run(example_usage())
