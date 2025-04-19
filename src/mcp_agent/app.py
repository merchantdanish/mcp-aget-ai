import functools
from typing import Any, Dict, Optional, Type, TypeVar, Callable, TYPE_CHECKING
from datetime import timedelta
import asyncio
import sys
import uuid
from contextlib import asynccontextmanager

from mcp import ServerSession
from mcp_agent.core.context import Context, initialize_context, cleanup_context
from mcp_agent.config import Settings
from mcp_agent.logging.event_progress import ProgressAction
from mcp_agent.logging.logger import get_logger
from mcp_agent.executor.workflow_signal import SignalWaitCallback
from mcp_agent.human_input.types import HumanInputCallback
from mcp_agent.human_input.handler import console_input_callback
from mcp_agent.workflows.llm.llm_selector import ModelSelector

if TYPE_CHECKING:
    from mcp_agent.agents.agent_config import AgentConfig
    from mcp_agent.executor.workflow import Workflow

R = TypeVar("R")


class MCPApp:
    """
    Main application class that manages global state and can host workflows.

    Example usage:
        app = MCPApp()

        @app.workflow
        class MyWorkflow(Workflow[str]):
            @app.task
            async def my_task(self):
                pass

            async def run(self):
                await self.my_task()

        async with app.run() as running_app:
            workflow = MyWorkflow()
            result = await workflow.execute()
    """

    def __init__(
        self,
        name: str = "mcp_application",
        description: str | None = None,
        settings: Optional[Settings] | str = None,
        human_input_callback: Optional[HumanInputCallback] = console_input_callback,
        signal_notification: Optional[SignalWaitCallback] = None,
        upstream_session: Optional["ServerSession"] = None,
        model_selector: ModelSelector = None,
    ):
        """
        Initialize the application with a name and optional settings.
        Args:
            name: Name of the application
            description: Description of the application. If you expose the MCPApp as an MCP server,
                provide a detailed description, since it will be used as the server's description.
            settings: Application configuration - If unspecified, the settings are loaded from mcp_agent.config.yaml.
                If this is a string, it is treated as the path to the config file to load.
            human_input_callback: Callback for handling human input
            signal_notification: Callback for getting notified on workflow signals/events.
            upstream_session: Upstream session if the MCPApp is running as a server to an MCP client.
            initialize_model_selector: Initializes the built-in ModelSelector to help with model selection. Defaults to False.
        """
        self.name = name
        self.description = description or "MCP Agent Application"

        # We use these to initialize the context in initialize()
        self._config_or_path = settings
        self._human_input_callback = human_input_callback
        self._signal_notification = signal_notification
        self._upstream_session = upstream_session
        self._model_selector = model_selector

        self._workflows: Dict[str, Type["Workflow"]] = {}  # id to workflow class
        self._pending_workflows: Dict[str, tuple[Type, tuple, dict]] = {}
        self._pending_workflow_run_methods: Dict[str, tuple] = {}
        self._agent_configs: Dict[
            str, "AgentConfig"
        ] = {}  # name to agent configuration
        self._logger = None
        self._context: Optional[Context] = None
        self._initialized = False

        try:
            # Set event loop policy for Windows
            if sys.platform == "win32":
                import asyncio

                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        finally:
            pass

    @property
    def context(self) -> Context:
        if self._context is None:
            raise RuntimeError(
                "MCPApp not initialized, please call initialize() first, or use async with app.run()."
            )
        return self._context

    @property
    def config(self):
        return self._context.config

    @property
    def server_registry(self):
        return self._context.server_registry

    @property
    def executor(self):
        return self._context.executor

    @property
    def engine(self):
        return self.executor.execution_engine

    @property
    def upstream_session(self):
        return self._context.upstream_session

    @upstream_session.setter
    def upstream_session(self, value):
        self._context.upstream_session = value

    @property
    def workflows(self):
        return self._workflows

    @property
    def tasks(self):
        return self.context.task_registry.list_activities()

    @property
    def logger(self):
        if self._logger is None:
            session_id = self._context.session_id if self._context else None
            self._logger = get_logger(f"mcp_agent.{self.name}", session_id=session_id)
        return self._logger

    async def initialize(self):
        """Initialize the application."""
        if self._initialized:
            return

        # Generate a session ID first
        session_id = str(uuid.uuid4())

        # Pass the session ID to initialize_context
        self._context = await initialize_context(
            self._config_or_path, store_globally=True, session_id=session_id
        )

        # Set the properties that were passed in the constructor
        self._context.human_input_handler = self._human_input_callback
        self._context.signal_notification = self._signal_notification
        self._context.upstream_session = self._upstream_session
        self._context.model_selector = self._model_selector

        # Store a reference to this app instance in the context for easier access
        self._context.app = self

        # Initialise pending workflow run
        if self.context and self.context.executor:
            decorator_registry = self._context.decorator_registry
            engine_type = self._context.executor.execution_engine

            workflow_run_decorator = decorator_registry.get_workflow_run_decorator(
                engine_type
            )

            if workflow_run_decorator:
                for function_id, (
                    fn,
                    fn_kwargs,
                ) in self._pending_workflow_run_methods.items():
                    # Find which workflow class this method belongs to
                    module_class_name, method_name = function_id.rsplit(".", 1)
                    module_name, class_name = module_class_name.rsplit(".", 1)

                    # Look through workflows to find the matching class
                    for workflow_cls in self._workflows.values():
                        if (
                            workflow_cls.__module__ == module_name
                            and workflow_cls.__name__ == class_name
                        ):
                            # Decorate the method and store on the class
                            decorated_method = workflow_run_decorator(fn, **fn_kwargs)
                            setattr(
                                workflow_cls,
                                f"_decorated_{method_name}",
                                decorated_method,
                            )
                            break

        # Initialise pending workflows
        for workflow_id, (cls, args, kwargs) in self._pending_workflows.items():
            if self.context and self.context.executor:
                decorator_registry = self.context.decorator_registry
                engine_type = self.context.executor.execution_engine

                workflow_defn_decorator = (
                    decorator_registry.get_workflow_defn_decorator(engine_type)
                )

                # TODO: jerron - Setting sandboxed=False is a workaround to silence temporal's RestrictedWorkflowAccessError.
                # Can we make this work without having to run outside sandbox environment?
                # This is not ideal as it could lead to non-deterministic behavior.

                # Apply the engine-specific decorator if available
                if workflow_defn_decorator:
                    self._workflows[workflow_id] = workflow_defn_decorator(
                        cls,
                        sandboxed=False,
                        *args,
                        **kwargs,
                    )
                    self.context.workflow_registry.register(workflow_id, cls)

        self._initialized = True
        self.logger.info(
            "MCPAgent initialized",
            data={
                "progress_action": "Running",
                "target": self.name,
                "agent_name": "mcp_application_loop",
                "session_id": session_id,
            },
        )

    async def cleanup(self):
        """Cleanup application resources."""
        if not self._initialized:
            return

        # Updatre progress display before logging is shut down
        self.logger.info(
            "MCPAgent cleanup",
            data={
                "progress_action": ProgressAction.FINISHED,
                "target": self.name or "mcp_app",
                "agent_name": "mcp_application_loop",
            },
        )

        try:
            await cleanup_context()
        except asyncio.CancelledError:
            self.logger.debug("Cleanup cancelled during shutdown")

        self._context = None
        self._initialized = False

    @asynccontextmanager
    async def run(self):
        """
        Run the application. Use as context manager.

        Example:
            async with app.run() as running_app:
                # App is initialized here
                pass
        """
        await self.initialize()
        try:
            yield self
        finally:
            await self.cleanup()

    def workflow(
        self, cls: Type, *args, workflow_id: str | None = None, **kwargs
    ) -> Type:
        """
        Decorator for a workflow class. By default it's a no-op,
        but different executors can use this to customize behavior
        for workflow registration.

        Example:
            If Temporal is available & we use a TemporalExecutor,
            this decorator will wrap with temporal_workflow.defn.
        """
        cls._app = self

        workflow_id = workflow_id or cls.__name__
        self._pending_workflows[workflow_id] = (cls, args, kwargs)
        self._workflows[workflow_id] = cls

        return cls

    def workflow_run(self, fn: Callable[..., R], **kwargs) -> Callable[..., R]:
        """
        Decorator for a workflow's main 'run' method.
        Different executors can use this to customize behavior for workflow execution.

        Example:
            If Temporal is in use, this gets converted to @workflow.run.
        """
        # Generate a unique ID for this function
        function_id = f"{fn.__module__}.{fn.__qualname__}"

        # Store the function and its kwargs for later decoration
        self._pending_workflow_run_methods[function_id] = (fn, kwargs)

        # Return a wrapper that checks if we're initialized or defers to original function
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            # Get the class from the first argument (self)
            if args and hasattr(args[0], "__class__"):
                workflow_cls = args[0].__class__
                method_name = fn.__name__

                # Check if this class has the decorated method
                decorated_method = getattr(
                    workflow_cls, f"_decorated_{method_name}", None
                )
                if decorated_method:
                    # Use the decorated method if available
                    return await decorated_method(*args, **kwargs)

            # Otherwise just call the original function
            return await fn(*args, **kwargs)

        return wrapper

    def workflow_task(
        self,
        name: str | None = None,
        schedule_to_close_timeout: timedelta | None = None,
        retry_policy: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable[..., R]], Callable[..., R]]:
        """
        Decorator to mark a function as a workflow task,
        automatically registering it in the global activity registry.

        Args:
            name: Optional custom name for the activity
            schedule_to_close_timeout: Maximum time the task can take to complete
            retry_policy: Retry policy configuration
            **kwargs: Additional metadata passed to the activity registration

        Returns:
            Decorated function that preserves async and typing information

        Raises:
            TypeError: If the decorated function is not async
            ValueError: If the retry policy or timeout is invalid
        """

        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            if not asyncio.iscoroutinefunction(func):
                raise TypeError(f"Function {func.__name__} must be async.")

            actual_name = name or f"{func.__module__}.{func.__qualname__}"
            timeout = schedule_to_close_timeout or timedelta(minutes=10)
            metadata = {
                "activity_name": actual_name,
                "schedule_to_close_timeout": timeout,
                "retry_policy": retry_policy or {},
                **kwargs,
            }
            activity_registry = self.context.task_registry
            activity_registry.register(actual_name, func, metadata)

            setattr(func, "is_workflow_task", True)
            setattr(func, "execution_metadata", metadata)

            # TODO: saqadri - determine if we need this
            # Preserve metadata through partial application
            # @functools.wraps(func)
            # async def wrapper(*args: Any, **kwargs: Any) -> R:
            #     result = await func(*args, **kwargs)
            #     return cast(R, result)  # Ensure type checking works

            # # Add metadata that survives partial application
            # wrapper.is_workflow_task = True  # type: ignore
            # wrapper.execution_metadata = metadata  # type: ignore

            # # Make metadata accessible through partial
            # def __getattr__(name: str) -> Any:
            #     if name == "is_workflow_task":
            #         return True
            #     if name == "execution_metadata":
            #         return metadata
            #     raise AttributeError(f"'{func.__name__}' has no attribute '{name}'")

            # wrapper.__getattr__ = __getattr__  # type: ignore

            # return wrapper

            return func

        return decorator

    def is_workflow_task(self, func: Callable[..., Any]) -> bool:
        """
        Check if a function is marked as a workflow task.
        This gets set for functions that are decorated with @workflow_task."""
        return bool(getattr(func, "is_workflow_task", False))

    @property
    def agent_configs(self) -> Dict[str, "AgentConfig"]:
        """Get the dictionary of registered agent configurations."""
        return self._agent_configs

    def register_agent_config(self, config: "AgentConfig") -> "AgentConfig":
        """
        Register an agent configuration with the application.

        Args:
            config: The agent configuration to register

        Returns:
            The registered configuration
        """
        self._agent_configs[config.name] = config
        return config
