import asyncio
import sys
from datetime import datetime

from rich.console import Console

# from mcp_agent.mcp_server_registry import ServerRegistry, MCPConnectionManager
from mcp_agent.context import get_current_context, configure_logger
from mcp_agent.config import Settings, LoggerSettings
from mcp_agent.logging.logger import get_logger, LoggingConfig, LoggingListener
from mcp_agent.logging.transport import ConsoleTransport, AsyncEventBus
from mcp_agent.logging.events import Event, EventFilter

# logger = get_logger("hello_world")


async def example_usage():
    # config = get_settings()
    # await configure_logger(config)
    context = get_current_context()
    logger = get_logger("hello_world.example_usage")
    logger.info("Hello, world!")
    logger.info(f"Current context: {context}")

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


async def test_logger():
    print("\n")
    # Create minimal settings
    settings = Settings(
        logger=LoggerSettings(
            type="console", level="info", batch_size=1, flush_interval=0.1
        )
    )

    # Configure logging
    await configure_logger(settings)

    # Use logger
    logger = get_logger("hello_world")
    logger.info("Hello, world!")

    # Give time for async processing
    await asyncio.sleep(5.0)

    # Clean up
    await LoggingConfig.shutdown()


class DebugConsoleTransport(ConsoleTransport):
    async def send_event(self, event: Event):
        print("DebugConsoleTransport.send_event called with:", event, file=sys.stderr)
        await super().send_event(event)


class DebugEventBus(AsyncEventBus):
    async def emit(self, event: Event):
        print(f"DebugEventBus.emit called with: {event}", file=sys.stderr)
        print(
            f"Current transport: {self.transport.__class__.__name__}", file=sys.stderr
        )
        await super().emit(event)


async def example_usage2():
    try:
        print("1. Creating transport...", file=sys.stderr)
        transport = DebugConsoleTransport()

        print("2. Getting event bus...", file=sys.stderr)
        # Create our debug bus instead of getting the default one
        AsyncEventBus._instance = DebugEventBus(transport=transport)
        bus = AsyncEventBus.get()
        print(f"Event bus transport type: {type(bus.transport)}", file=sys.stderr)

        print("3. Setting up listener...", file=sys.stderr)
        event_filter = EventFilter(min_level="info")
        logging_listener = LoggingListener(event_filter=event_filter)
        bus.add_listener("logging", logging_listener)

        print("4. Starting event bus...", file=sys.stderr)
        await bus.start()

        print("5. Configuring logging...", file=sys.stderr)
        await LoggingConfig.configure(
            event_filter=event_filter,
            transport=transport,
            batch_size=1,
            flush_interval=0.1,
        )

        print("6. Getting logger...", file=sys.stderr)
        logger = get_logger("hello_world")

        print("7. Sending test message...", file=sys.stderr)
        # Try both direct and logger methods
        print("7a. Direct transport test...", file=sys.stderr)
        await transport.send_event(
            Event(
                type="info",
                namespace="direct_test",
                message="Direct transport test",
                data={"method": "direct"},
            )
        )

        print("7b. Logger test...", file=sys.stderr)
        logger.info("Hello, world!", test_data="example")

        print("8. Waiting for processing...", file=sys.stderr)
        await asyncio.sleep(1)

        print("9. Checking final bus state...", file=sys.stderr)
        print(f"Final bus transport: {type(bus.transport)}", file=sys.stderr)
        print(f"Active listeners: {list(bus.listeners.keys())}", file=sys.stderr)

        print("10. Cleaning up...", file=sys.stderr)
        await LoggingConfig.shutdown()

        print("11. Done.", file=sys.stderr)

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}", file=sys.stderr)
        raise


async def test_console_transport():
    print("Testing console transport directly...")

    # Create transport
    transport = ConsoleTransport()

    # Create a test event
    event = Event(
        type="info",
        namespace="test",
        message="Direct console transport test",
        timestamp=datetime.now(),
        data={"test": "data"},
    )

    # Send event directly
    print("Sending test event...")
    await transport.send_event(event)

    print("Test complete.")


async def test_transport():
    # First test rich console directly
    print("Testing rich console...")
    console = Console()
    console.print("[bold red]Rich console test[/bold red]")

    # Then test our transport
    print("\nTesting console transport...")
    transport = ConsoleTransport()

    # Create and send a test event
    event = Event(
        type="info",
        namespace="test",
        message="Direct transport test message",
        timestamp=datetime.now(),
        data={"test": "data"},
    )

    await transport.send_event(event)

    print("\nTest complete.")


async def setup_logging():
    """Set up the logging system with console output."""
    # Create transport
    transport = ConsoleTransport()

    # Set up event bus with transport
    bus = AsyncEventBus.get(transport=transport)

    # Add logging listener
    event_filter = EventFilter(min_level="info")
    bus.add_listener("logging", LoggingListener(event_filter=event_filter))

    # Start the event bus
    await bus.start()

    # Configure logging system
    await LoggingConfig.configure(
        event_filter=event_filter, transport=transport, batch_size=1, flush_interval=0.1
    )

    return bus


async def example_usage3():
    # Set up logging
    await setup_logging()

    # Use logger
    logger = get_logger("hello_world")
    logger.info("Hello, world!", data={"example": "test"})

    # Give time for async processing
    await asyncio.sleep(0.2)

    # Clean up
    await LoggingConfig.shutdown()


if __name__ == "__main__":
    asyncio.run(example_usage())
