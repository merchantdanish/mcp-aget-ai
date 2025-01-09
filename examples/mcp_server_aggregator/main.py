import asyncio
import os
from pathlib import Path

from mcp_agent.context import get_current_context
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.mcp_aggregator import MCPAggregator


async def example_usage_persistent():
    logger = get_logger("mcp_server_aggregator.example_usage")

    context = get_current_context()
    logger.info("Hello, world! Let's create an MCP aggregator (server-of-servers)...")
    logger.info("Current config:", data=context.config.model_dump())

    # Add the current directory to the filesystem server's args
    context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

    # Create an MCP aggregator that connects to the fetch and filesystem servers
    aggregator = None

    try:
        aggregator = await MCPAggregator.create(
            server_names=["fetch", "filesystem"],
            connection_persistence=True,  # By default connections are torn down after each call
        )
        # Call list_tools on the aggregator, which will search all servers for the tool
        logger.info("Aggregator: Calling list_tools...")
        result = await aggregator.list_tools()
        logger.info("Tools available:", data=result.model_dump())

        # Call read_file on the aggregator, which will search all servers for the tool
        result = await aggregator.call_tool(
            name="read_file",
            arguments={"path": str(Path.cwd() / "README.md")},
        )
        logger.info("read_file result:", data=result.model_dump())

        # Call fetch.fetch on the aggregator
        # (i.e. server-namespacing -- fetch is the servername, which exposes fetch tool)
        result = await aggregator.call_tool(
            name="fetch.fetch",
            arguments={"url": "https://jsonplaceholder.typicode.com/todos/1"},
        )
        logger.info("fetch result:", data=result.model_dump())
    finally:
        logger.info("Closing all server connections on aggregator...")
        await aggregator.close()


async def example_usage():
    logger = get_logger("mcp_server_aggregator.example_usage")

    context = get_current_context()
    logger.info("Hello, world! Let's create an MCP aggregator (server-of-servers)...")
    logger.info("Current config:", data=context.config.model_dump())

    # Add the current directory to the filesystem server's args
    context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

    # Create an MCP aggregator that connects to the fetch and filesystem servers
    aggregator = None

    try:
        aggregator = await MCPAggregator.create(
            server_names=["fetch", "filesystem"],
            connection_persistence=False,
        )
        # Call list_tools on the aggregator, which will search all servers for the tool
        logger.info("Aggregator: Calling list_tools...")
        result = await aggregator.list_tools()
        logger.info("Tools available:", data=result.model_dump())

        # Call read_file on the aggregator, which will search all servers for the tool
        result = await aggregator.call_tool(
            name="read_file",
            arguments={"path": str(Path.cwd() / "README.md")},
        )
        logger.info("read_file result:", data=result.model_dump())

        # Call fetch.fetch on the aggregator
        # (i.e. server-namespacing -- fetch is the servername, which exposes fetch tool)
        result = await aggregator.call_tool(
            name="fetch.fetch",
            arguments={"url": "https://jsonplaceholder.typicode.com/todos/1"},
        )
        logger.info("fetch result:", data=result.model_dump())
    finally:
        logger.info("Closing all server connections on aggregator...")
        await aggregator.close()


if __name__ == "__main__":
    asyncio.run(example_usage())
