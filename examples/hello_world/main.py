import asyncio

# from mcp_agent.mcp_server_registry import ServerRegistry, MCPConnectionManager
# from mcp_agent.context import get_current_context, get_current_config
# from mcp_agent.logging.logger import get_logger


# logger = get_logger("hello_world")


async def example_usage():
    # logger.info("Hello, world!")
    print("Hello, world!")

    # # Initialize your ServerRegistry
    # registry = ServerRegistry("path/to/config.yaml")

    # # Create connection manager
    # manager = MCPConnectionManager(registry)

    # try:
    #     # Launch a server
    #     server = await manager.get_server("example-server")

    #     # Use the session
    #     result = await server.session.list_tools()
    #     print(f"Tools available: {result}")

    #     # Server stays connected until explicitly disconnected
    #     await asyncio.sleep(60)  # Do other work...

    #     # When done with a specific server
    #     await manager.disconnect_server("example-server")

    #     # Or disconnect all servers
    #     await manager.disconnect_all()

    # except Exception as e:
    #     print(f"Error: {e}")
    #     await manager.disconnect_all()


if __name__ == "__main__":
    asyncio.run(example_usage())
