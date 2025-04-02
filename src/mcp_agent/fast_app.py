"""
FastMCPApp - Extended MCPApp with declarative agent and workflow configuration.
"""

from typing import Callable, List, Optional, Tuple

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_config import (
    AgentConfig,
    AugmentedLLMConfig,
    ParallelWorkflowConfig,
    OrchestratorWorkflowConfig,
    RouterWorkflowConfig,
    EvaluatorOptimizerWorkflowConfig,
    SwarmWorkflowConfig,
)
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM


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
                parallel_config=ParallelWorkflowConfig(
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
                orchestrator_config=OrchestratorWorkflowConfig(
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
                router_config=RouterWorkflowConfig(
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
                evaluator_optimizer_config=EvaluatorOptimizerWorkflowConfig(
                    evaluator_agent=evaluator, optimizer_agent=optimizer
                ),
            )

            if llm_factory:
                config.llm_config = AugmentedLLMConfig(factory=llm_factory)

            config = config_fn(config)
            self._agent_configs[name] = config
            return config

        return decorator

    def swarm(
        self,
        name: str,
        instruction: str,
        agents: List[str],
        llm_factory: Callable = None,
        **kwargs,
    ):
        """
        Decorator to define a swarm workflow agent.

        Example:
            @app.swarm("team", "A collaborative team of agents",
                     agents=["leader", "researcher", "writer"])
            def team_config(config):
                config.swarm_config.context_variables = {"priority": "accuracy"}
                return config

        Args:
            name: The name of the workflow agent
            instruction: The agent's instruction/system prompt
            agents: List of agent names in the swarm
            llm_factory: Optional LLM factory for the workflow
            **kwargs: Additional parameters

        Returns:
            Decorator function that registers the swarm workflow configuration
        """

        def decorator(config_fn):
            config = AgentConfig(
                name=name,
                instruction=instruction,
                swarm_config=SwarmWorkflowConfig(agents=agents),
            )

            if llm_factory:
                config.llm_config = AugmentedLLMConfig(factory=llm_factory)

            config = config_fn(config)
            self._agent_configs[name] = config
            return config

        return decorator

    async def create_agent(self, name: str) -> Tuple[Agent, Optional[AugmentedLLM]]:
        """
        Create an agent with its configured workflow.

        Args:
            name: The name of the registered agent configuration

        Returns:
            Tuple of (agent instance, augmented LLM or workflow instance)

        Raises:
            ValueError: If no agent configuration with the given name exists
        """
        if name not in self._agent_configs:
            raise ValueError(f"No agent configuration named '{name}' is registered")

        config = self._agent_configs[name]

        # Create and initialize the basic agent
        agent = config.create_agent(context=self._context)
        await agent.initialize()

        # Handle different workflow types with type-safe configs
        workflow_type = config.get_workflow_type()

        if workflow_type is None and config.llm_config is not None:
            # Basic agent with simple LLM
            llm_instance = await config.llm_config.create_llm()
            llm = await agent.attach_llm(lambda: llm_instance)
            return agent, llm

        elif workflow_type == "parallel":
            # Create a Parallel workflow with type-safe config
            parallel_config = config.parallel_config
            from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM

            # Get referenced agents
            fan_in_agent, _ = await self.create_agent(parallel_config.fan_in_agent)
            fan_out_agents = []
            for agent_name in parallel_config.fan_out_agents:
                fan_out_agent, _ = await self.create_agent(agent_name)
                fan_out_agents.append(fan_out_agent)

            # Get LLM factory
            llm_factory = config.llm_config.factory if config.llm_config else None

            # Create parallel workflow
            parallel = ParallelLLM(
                fan_in_agent=fan_in_agent,
                fan_out_agents=fan_out_agents,
                llm_factory=llm_factory,
                concurrent=parallel_config.concurrent,
                synchronize_fan_out_models=parallel_config.synchronize_fan_out_models,
                **parallel_config.extra_params,
            )

            return agent, parallel

        elif workflow_type == "orchestrator":
            # Create an Orchestrator workflow with type-safe config
            orchestrator_config = config.orchestrator_config
            from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator

            # Get referenced agents
            available_agents = []
            for agent_name in orchestrator_config.available_agents:
                available_agent, _ = await self.create_agent(agent_name)
                available_agents.append(available_agent)

            # Get the LLM factory
            llm_factory = config.llm_config.factory if config.llm_config else None

            # Optional planner agent
            planner = None
            if orchestrator_config.planner_agent:
                planner, _ = await self.create_agent(orchestrator_config.planner_agent)

            # Create the orchestrator
            orchestrator = Orchestrator(
                llm_factory=llm_factory,
                available_agents=available_agents,
                planner=planner,
                max_iterations=orchestrator_config.max_iterations,
                **orchestrator_config.extra_params,
            )

            return agent, orchestrator

        elif workflow_type == "router":
            # Create a Router workflow with type-safe config
            router_config = config.router_config

            # Get referenced agents
            agents = []
            for agent_name in router_config.agent_names:
                agent_inst, _ = await self.create_agent(agent_name)
                agents.append(agent_inst)

            # Determine which router implementation to use
            if router_config.router_type == "llm":
                from mcp_agent.workflows.router.router_llm import LLMRouter

                # Get LLM factory
                llm_factory = config.llm_config.factory if config.llm_config else None
                llm_instance = None
                if llm_factory:
                    llm_instance = await config.llm_config.create_llm()

                # Create the router
                router = LLMRouter(
                    llm=llm_instance, agents=agents, **router_config.extra_params
                )

            else:  # embedding router
                # Create the router (implementation depends on embedding model)
                if router_config.embedding_model == "cohere":
                    from mcp_agent.workflows.router.router_embedding_cohere import (
                        CohereEmbeddingRouter,
                    )

                    router = CohereEmbeddingRouter(
                        agents=agents, **router_config.extra_params
                    )
                else:
                    from mcp_agent.workflows.router.router_embedding_openai import (
                        OpenAIEmbeddingRouter,
                    )

                    router = OpenAIEmbeddingRouter(
                        agents=agents, **router_config.extra_params
                    )

            return agent, router

        elif workflow_type == "evaluator_optimizer":
            # Create an Evaluator-Optimizer workflow with type-safe config
            eo_config = config.evaluator_optimizer_config
            from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
                EvaluatorOptimizerLLM,
                QualityRating,
            )

            # Get referenced agents
            evaluator_agent, _ = await self.create_agent(eo_config.evaluator_agent)
            optimizer_agent, _ = await self.create_agent(eo_config.optimizer_agent)

            # Get LLM factory
            llm_factory = config.llm_config.factory if config.llm_config else None

            # Parse min_rating string to enum
            min_rating = QualityRating.GOOD  # Default
            try:
                min_rating = QualityRating[eo_config.min_rating.upper()]
            except (KeyError, AttributeError):
                pass

            # Create the evaluator-optimizer
            eo = EvaluatorOptimizerLLM(
                evaluator=evaluator_agent,
                optimizer=optimizer_agent,
                llm_factory=llm_factory,
                min_rating=min_rating,
                max_iterations=eo_config.max_iterations,
                **eo_config.extra_params,
            )

            return agent, eo

        elif workflow_type == "swarm":
            # Create a Swarm workflow with type-safe config
            swarm_config = config.swarm_config

            # Choose the swarm implementation based on LLM factory
            llm_factory = config.llm_config.factory if config.llm_config else None

            if not llm_factory:
                raise ValueError("A LLM factory is required for Swarm workflow")

            # Get the factory class name to determine which Swarm implementation to use
            factory_class_name = llm_factory.__name__

            if "Anthropic" in factory_class_name:
                from mcp_agent.workflows.swarm.swarm_anthropic import AnthropicSwarm

                # Get the primary agent
                primary_agent, _ = await self.create_agent(swarm_config.agents[0])

                # Create the swarm
                swarm = AnthropicSwarm(
                    agent=primary_agent,
                    context_variables=swarm_config.context_variables,
                    **swarm_config.extra_params,
                )
            else:
                # Default to OpenAI swarm
                from mcp_agent.workflows.swarm.swarm_openai import OpenAISwarm

                # Get the primary agent
                primary_agent, _ = await self.create_agent(swarm_config.agents[0])

                # Create the swarm
                swarm = OpenAISwarm(
                    agent=primary_agent,
                    context_variables=swarm_config.context_variables,
                    **swarm_config.extra_params,
                )

            return agent, swarm

        else:
            # No workflow or LLM config, just return the basic agent
            return agent, None
