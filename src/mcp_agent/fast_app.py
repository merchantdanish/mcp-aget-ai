"""
FastMCPApp - Extended MCPApp with declarative agent and workflow configuration.
"""

from typing import Callable, List

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent_config import (
    AgentConfig,
    AugmentedLLMConfig,
    ParallelLLMConfig,
    OrchestratorLLMConfig,
    RouterConfig,
    EvaluatorOptimizerConfig,
)


class FastMCPApp(MCPApp):
    """
    Extension of MCPApp with declarative agent configuration and workflow patterns.
    Provides decorators for easily defining agents and workflows.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def agent(
        self,
        name: str,
        instruction: str,
        server_names: List[str] = None,
        llm_factory: Callable = None,
        **kwargs,
    ):
        """
        Decorator to define a basic agent.

        Example:
            @app.agent("finder", "You find information", ["fetch", "filesystem"])
            def finder_config(config):
                config.llm_config = AugmentedLLMConfig(factory=OpenAIAugmentedLLM)
                return config

        Args:
            name: The name of the agent
            instruction: The agent's instruction/system prompt
            server_names: List of MCP servers the agent can access
            llm_factory: Optional LLM factory to initialize the agent with
            **kwargs: Additional parameters for the agent

        Returns:
            Decorator function that registers the agent configuration
        """

        def decorator(config_fn):
            # Create basic config
            config = AgentConfig(
                name=name, instruction=instruction, server_names=server_names or []
            )

            # Add LLM config if provided
            if llm_factory:
                config.llm_config = AugmentedLLMConfig(factory=llm_factory)

            # Apply any extra configuration from the function
            config = config_fn(config)

            # Register the configuration
            self._agent_configs[name] = config
            return config

        return decorator

    def parallel(
        self,
        name: str,
        instruction: str,
        fan_in: str,
        fan_out: List[str],
        llm_factory: Callable = None,
        **kwargs,
    ):
        """
        Decorator to define a parallel workflow agent.

        Example:
            @app.parallel("researcher", "Research team",
                         fan_in="aggregator", fan_out=["finder", "writer"])
            def researcher_config(config):
                config.parallel_config.concurrent = True
                return config

        Args:
            name: The name of the workflow agent
            instruction: The agent's instruction/system prompt
            fan_in: Name of the agent to use for aggregating results
            fan_out: List of agent names to distribute work to
            llm_factory: Optional LLM factory for the workflow
            **kwargs: Additional parameters

        Returns:
            Decorator function that registers the parallel workflow configuration
        """

        def decorator(config_fn):
            # Create basic config with parallel workflow
            config = AgentConfig(
                name=name,
                instruction=instruction,
                parallel_config=ParallelLLMConfig(
                    fan_in_agent=fan_in, fan_out_agents=fan_out
                ),
            )

            # Add LLM config if provided
            if llm_factory:
                config.llm_config = AugmentedLLMConfig(factory=llm_factory)

            # Apply any extra configuration
            config = config_fn(config)

            # Register the configuration
            self._agent_configs[name] = config
            return config

        return decorator

    def orchestrator(
        self,
        name: str,
        instruction: str,
        available_agents: List[str],
        llm_factory: Callable = None,
        **kwargs,
    ):
        """
        Decorator to define an orchestrator workflow agent.

        Example:
            @app.orchestrator("manager", "Project manager",
                            available_agents=["finder", "writer", "analyst"])
            def manager_config(config):
                config.orchestrator_config.max_iterations = 5
                return config

        Args:
            name: The name of the workflow agent
            instruction: The agent's instruction/system prompt
            available_agents: List of agent names the orchestrator can use
            llm_factory: Optional LLM factory for the workflow
            **kwargs: Additional parameters

        Returns:
            Decorator function that registers the orchestrator workflow configuration
        """

        def decorator(config_fn):
            config = AgentConfig(
                name=name,
                instruction=instruction,
                orchestrator_config=OrchestratorLLMConfig(
                    available_agents=available_agents
                ),
            )

            if llm_factory:
                config.llm_config = AugmentedLLMConfig(factory=llm_factory)

            config = config_fn(config)
            self._agent_configs[name] = config
            return config

        return decorator

    def router(
        self,
        name: str,
        instruction: str,
        agent_names: List[str],
        router_type: str = "llm",
        llm_factory: Callable = None,
        **kwargs,
    ):
        """
        Decorator to define a router workflow agent.

        Example:
            @app.router("dispatcher", "Routes tasks to specialists",
                      agent_names=["finder", "writer", "coder"])
            def dispatcher_config(config):
                config.router_config.top_k = 2
                return config

        Args:
            name: The name of the workflow agent
            instruction: The agent's instruction/system prompt
            agent_names: List of agent names the router can dispatch to
            router_type: Type of router ("llm" or "embedding")
            llm_factory: Optional LLM factory for the workflow
            **kwargs: Additional parameters

        Returns:
            Decorator function that registers the router workflow configuration
        """

        def decorator(config_fn):
            config = AgentConfig(
                name=name,
                instruction=instruction,
                router_config=RouterConfig(
                    agent_names=agent_names, router_type=router_type
                ),
            )

            if llm_factory:
                config.llm_config = AugmentedLLMConfig(factory=llm_factory)

            config = config_fn(config)
            self._agent_configs[name] = config
            return config

        return decorator

    def evaluator_optimizer(
        self,
        name: str,
        instruction: str,
        evaluator: str,
        optimizer: str,
        llm_factory: Callable = None,
        **kwargs,
    ):
        """
        Decorator to define an evaluator-optimizer workflow agent.

        Example:
            @app.evaluator_optimizer("quality_team", "Ensures high quality output",
                                   evaluator="critic", optimizer="writer")
            def quality_team_config(config):
                config.evaluator_optimizer_config.min_rating = "good"
                return config

        Args:
            name: The name of the workflow agent
            instruction: The agent's instruction/system prompt
            evaluator: Name of the agent to use as evaluator
            optimizer: Name of the agent to use as optimizer
            llm_factory: Optional LLM factory for the workflow
            **kwargs: Additional parameters

        Returns:
            Decorator function that registers the evaluator-optimizer workflow configuration
        """

        def decorator(config_fn):
            config = AgentConfig(
                name=name,
                instruction=instruction,
                evaluator_optimizer_config=EvaluatorOptimizerConfig(
                    evaluator_agent=evaluator, optimizer_agent=optimizer
                ),
            )

            if llm_factory:
                config.llm_config = AugmentedLLMConfig(factory=llm_factory)

            config = config_fn(config)
            self._agent_configs[name] = config
            return config

        return decorator
