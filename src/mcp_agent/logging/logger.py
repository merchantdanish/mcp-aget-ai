"""
Logger module for the MCP Agent, which provides:
- Local + optional remote event transport
- Async event bus
- OpenTelemetry tracing decorators (for distributed tracing)
- Automatic injection of trace_id/span_id into events
- Developer-friendly Logger that can be used anywhere
"""

import asyncio
import time

from contextlib import asynccontextmanager, contextmanager

from mcp_agent.logging.events import Event, EventContext, EventType
from mcp_agent.logging.transport import AsyncEventBus


class Logger:
    """
    Developer-friendly logger that sends events to the AsyncEventBus.
    - `type` is a broad category (INFO, ERROR, etc.).
    - `name` can be a custom domain-specific event name, e.g. "ORDER_PLACED".
    """

    def __init__(self, namespace: str):
        self.namespace = namespace
        self.event_bus = AsyncEventBus.get()

    async def debug(
        self,
        message: str,
        name: str | None = None,
        context: EventContext = None,
        **data,
    ):
        await self.event("debug", name, message, context, data)

    async def info(
        self,
        message: str,
        name: str | None = None,
        context: EventContext = None,
        **data,
    ):
        await self.event("info", name, message, context, data)

    async def warning(
        self,
        message: str,
        name: str | None = None,
        context: EventContext = None,
        **data,
    ):
        await self.event("warning", name, message, context, data)

    async def error(
        self,
        message: str,
        name: str | None = None,
        context: EventContext = None,
        **data,
    ):
        await self.event("error", name, message, context, data)

    async def progress(
        self,
        message: str,
        name: str | None = None,
        percentage: float = None,
        context: EventContext = None,
        **data,
    ):
        merged_data = dict(percentage=percentage, **data)
        await self.event("progress", name, message, context, merged_data)

    async def event(
        self,
        etype: EventType,
        ename: str | None,
        message: str,
        context: EventContext | None,
        data: dict,
    ):
        evt = Event(
            type=etype,
            name=ename,
            namespace=self.namespace,
            message=message,
            context=context,
            data=data,
        )
        await self.event_bus.emit(evt)


@contextmanager
def event_context(
    logger: Logger,
    message: str,
    event_type: EventType = "info",
    name: str | None = None,
    **data,
):
    """
    Times a synchronous block, logs an event after completion.
    Because logger methods are async, we schedule the final log.
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time

        async def finalize():
            await logger.event(
                event_type,
                name,
                f"{message} finished in {duration:.3f}s",
                None,
                {"duration": duration, **data},
            )

        asyncio.get_event_loop().create_task(finalize())


@asynccontextmanager
async def async_event_context(
    logger: Logger,
    message: str,
    event_type: EventType = "info",
    name: str | None = None,
    **data,
):
    """
    Times an asynchronous block, logs an event after completion.
    Because logger methods are async, we schedule the final log.
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        await logger.event(
            event_type,
            name,
            f"{message} finished in {duration:.3f}s",
            None,
            {"duration": duration, **data},
        )


##########
# Example
##########


# class Agent:
#     """Shows how to combine Logger with OTel's @telemetry.traced decorator."""

#     def __init__(self, name: str):
#         self.logger = Logger(f"agent.{name}")

#     @telemetry.traced("agent.call_tool", kind=SpanKind.CLIENT)
#     async def call_tool(self, tool_name: str, **kwargs):
#         await self.logger.info(
#             f"Calling tool '{tool_name}'", name="TOOL_CALL_START", **kwargs
#         )
#         await asyncio.sleep(random.uniform(0.1, 0.3))
#         # Possibly do real logic here
#         await self.logger.debug(
#             f"Completed tool call '{tool_name}'", name="TOOL_CALL_END"
#         )


# class Workflow:
#     """Example workflow that logs multiple steps, also with optional tracing."""

#     def __init__(self, name: str, steps: List[str]):
#         self.logger = Logger(f"workflow.{name}")
#         self.steps = steps

#     @telemetry.traced("workflow.run", kind=SpanKind.INTERNAL)
#     async def run(self):
#         await self.logger.info(
#             "Workflow started", name="WORKFLOW_START", steps=len(self.steps)
#         )
#         for i, step_name in enumerate(self.steps, start=1):
#             pct = round((i / len(self.steps)) * 100, 2)
#             await self.logger.progress(
#                 f"Executing {step_name}", name="WORKFLOW_STEP", percentage=pct
#             )
#             await asyncio.sleep(random.uniform(0.1, 0.3))
#             await self.logger.milestone(
#                 f"Completed {step_name}", name="WORKFLOW_MILESTONE", step_index=i
#             )
#         await self.logger.status("Workflow complete", name="WORKFLOW_DONE")


# ###############################################################################
# # 10) Demo Main
# ###############################################################################


# async def main():
#     # 1) Configure Python logging
#     logging.basicConfig(level=logging.INFO)

#     # 2) Get the event bus and add local listeners
#     bus = AsyncEventBus.get()
#     bus.add_listener("logging", LoggingListener())
#     bus.add_listener("batching", BatchingListener(batch_size=3, flush_interval=2.0))

#     # 3) Optionally set up distributed transport
#     # configure_distributed("https://my-remote-logger.example.com")

#     # 4) Start the event bus
#     await bus.start()

#     # 5) Run example tasks
#     agent = Agent("assistant")
#     workflow = Workflow("demo_flow", ["init", "process", "cleanup"])

#     agent_task = asyncio.create_task(agent.call_tool("my-tool", foo="bar"))
#     workflow_task = asyncio.create_task(workflow.run())

#     # Also demonstrate timed context manager
#     logger = Logger("misc")
#     with event_context(
#         logger, "SynchronousBlock", event_type="info", name="SYNCHRONOUS_BLOCK"
#     ):
#         time.sleep(0.5)  # do a blocking operation

#     # Wait for tasks
#     await asyncio.gather(agent_task, workflow_task)

#     # 6) Stop the bus (flush & close)
#     await bus.stop()


# if __name__ == "__main__":
#     asyncio.run(main())
