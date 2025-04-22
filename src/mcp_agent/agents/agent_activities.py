import asyncio
import functools
import inspect
from typing import Callable, Dict, Iterable, Optional, TYPE_CHECKING


from mcp.types import (
    ListToolsResult,
)

from mcp_agent.agents.agent_config import AgentConfig
from mcp_agent.core.context_dependent import ContextDependent
from mcp_agent.mcp.mcp_aggregator import MCPAggregator

if TYPE_CHECKING:
    from mcp_agent.agents.agent import Agent
    from mcp_agent.app import MCPApp
    from mcp_agent.core.context import Context


def bind_agent_tasks(
    app: "MCPApp",
    tasks: "AgentTasks",
    *,
    include: Iterable[str] = (
        "initialize_agent",
        "shutdown_agent",
        "shutdown_all_agents",
        "list_tools",
    ),
) -> None:
    """
    Turn the selected *bound methods* of `tasks` into Temporal activities.

    Each generated wrapper:
      • is an `async def` with only *args / **kwargs
      • captures the bound method in its closure (so `self` is correct)
      • keeps the original name so the activity is easy to read in Temporal UI
    """

    def make_wrapper(bound: Callable) -> Callable:
        @functools.wraps(bound)  # preserves __name__ / __qualname__
        async def _wrapper(*args, **kwargs):  # ← no keyword‑only params!
            return await bound(*args, **kwargs)

        return _wrapper

    for meth_name in include:
        bound_method = getattr(tasks, meth_name)

        if not inspect.iscoroutinefunction(bound_method):
            raise TypeError(f"{meth_name} must be `async def`")

        wrapper = make_wrapper(bound_method)

        # Register as Temporal activity
        app.workflow_task(name=meth_name)(wrapper)


class AgentTasks(ContextDependent):
    """
    Class for any external tasks that the agent needs to perform (e.g. connect to MCP servers, etc.)
    """

    agents: Dict[str, "Agent"] = {}
    server_aggregators: Dict[str, MCPAggregator] = {}
    agents_lock: asyncio.Lock = asyncio.Lock()

    def __init__(
        self,
        context: Optional["Context"] = None,
        app: Optional["MCPApp"] = None,
        **kwargs,
    ):
        super().__init__(
            context=context,
            **kwargs,
        )

        if app is None:
            print("Warning: No app provided to AgentTasks. Using context.app instead.")

        if context is None:
            print("Warning: No context provided to AgentTasks.")

        if context.app is None:
            print("Warning: No app found in context. Using default app.")

        self.app = app or context.app

        # # # Register each method as a workflow task
        # self.app.workflow_task(name="initialize_agent")(self.initialize_agent)
        # self.app.workflow_task()(self.shutdown_agent)
        # self.app.workflow_task()(self.shutdown_all_agents)
        # self.app.workflow_task()(self.list_tools)

    # These methods need to be decorated at the class level, not in __init__
    # @staticmethod
    # def register_as_workflow_task(app: "MCPApp"):
    #     """Class decorator to register all relevant methods as workflow tasks"""

    #     def decorator(cls):
    #         if not app:
    #             return cls

    #         # app.workflow_task2(name="initialize_agent2")(cls.initialize_agent2)
    #         app.workflow_task(name="initialize_agent")(cls.initialize_agent)
    #         app.workflow_task()(cls.shutdown_agent)
    #         app.workflow_task()(cls.shutdown_all_agents)
    #         app.workflow_task(name="list_tools")(cls.list_tools)
    #         return cls

    #     return decorator

    async def initialize_agent(
        self, agent_config: AgentConfig, force: bool = False
    ) -> bool:
        """
        Initialize an agent with the given configuration.
        This method is called when the agent is created.
        """

        print("Initializing agent with params:", agent_config)

        # agent_config_args = params.get("agent_config")
        # agent_config = AgentConfig(**agent_config_args)
        # force: bool = params.get("force", False)

        print("Initializing agent:", agent_config.name)

        if agent_config.name in self.agents and not force:
            return self.agents[agent_config.name]

        # server_aggregator = MCPAggregator(
        #     server_names=agent_config.server_names,
        #     connection_persistence=agent_config.connection_persistence,
        #     context=self.context,
        #     name=agent_config.name,
        # )

        agent = agent_config.create_agent(self.context)

        print("Agent created:", agent_config.name)

        await agent.initialize()

        print("Agent initialized:", agent_config.name)

        self.agents[agent_config.name] = agent

        return True

    async def shutdown_agent(self, agent_name: str) -> None:
        """
        Shutdown the agent with the given name.
        This method is called when the agent is no longer needed.
        """

        if agent_name in self.agents:
            await self.agents[agent_name].shutdown()
            del self.agents[agent_name]

    async def shutdown_all_agents(self) -> None:
        """
        Shutdown all agents.
        This method is called when the application is shutting down.
        """

        async with self.agents_lock:
            for agent in self.agents.values():
                await agent.shutdown()
            self.agents.clear()

    async def list_tools(self, agent_name: str) -> ListToolsResult:
        """
        List the tools available to the agent.
        This method is called when the agent needs to list its tools.
        """

        # agent_name: str = params.get("agent_name")
        # server_name: str | None = params.get("server_name")

        async with self.agents_lock:
            if agent_name not in self.agents:
                raise ValueError(f"Agent {agent_name} not found")

            agent = self.agents[agent_name]
            res = await agent.list_tools()

            return res
