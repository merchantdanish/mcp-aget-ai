"""
Agent configuration classes for declarative agent definition.
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    Generic,
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field, ConfigDict

from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, RequestParams
from mcp_agent.human_input.types import HumanInputCallback

if TYPE_CHECKING:
    from mcp_agent.agents.agent import Agent
    from mcp_agent.context import Context

# Define TypeVar for LLM types
LLM = TypeVar("LLM", bound=AugmentedLLM)

# region AugmentedLLM configs

# TODO: saqadri - Use these in the constructors for the respective classes


class AugmentedLLMConfig(BaseModel, Generic[LLM]):
    """
    Configuration for creating an AugmentedLLM instance.
    Provides type-safe configuration for different LLM providers.
    """

    # The factory function or class that creates the LLM
    factory: Callable[..., LLM]

    # Model-specific parameters
    provider_params: Dict[str, Any] = Field(default_factory=dict)

    # Request parameters used in generate calls
    default_request_params: Optional[RequestParams] = None

    model_config = ConfigDict(extra=True, arbitrary_types_allowed=True)

    async def create_llm(self) -> LLM:
        """
        Create an LLM instance using this configuration.

        Returns:
            An instance of the configured LLM type
        """
        params = {}
        # Combine common parameters with provider-specific parameters
        # ...

        # Add any additional provider-specific parameters
        params.update(self.provider_params)

        # Create the LLM instance
        return self.factory(**params)


class BasicAgentConfig(BaseModel):
    """
    Configuration for a basic agent with an LLM.
    This contains all the parameters needed to create a standard Agent
    without any complex workflow pattern.
    """

    name: str
    instruction: Union[str, Callable[[Dict], str]] = "You are a helpful agent."
    server_names: List[str] = Field(default_factory=list)
    functions: List[Callable] = Field(default_factory=list)
    connection_persistence: bool = True
    human_input_callback: Optional[HumanInputCallback] = None
    llm_config: Optional[AugmentedLLMConfig] = None
    extra_kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ParallelLLMConfig(BaseModel):
    """Type-safe configuration for ParallelLLM workflow pattern."""

    fan_in_agent: str  # Name of the agent to use for fan-in
    fan_out_agents: List[str]  # Names of agents to use for fan-out
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class OrchestratorLLMConfig(BaseModel):
    """Type-safe configuration for Orchestrator workflow pattern."""

    available_agents: List[str]  # Names of agents available to the orchestrator
    planner_agent: Optional[str] = None  # Optional custom planner agent
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class RouterConfig(BaseModel):
    """Type-safe configuration for Router workflow pattern."""

    agent_names: List[str]  # Names of agents to route between
    top_k: int = 1
    router_type: Literal["llm", "embedding"] = "llm"
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class EvaluatorOptimizerConfig(BaseModel):
    """Type-safe configuration for Evaluator-Optimizer workflow pattern."""

    evaluator_agent: str  # Name of the agent to use as evaluator
    optimizer_agent: str  # Name of the agent to use as optimizer
    min_rating: str = "excellent"  # Minimum quality rating to accept
    max_refinements: int = 3  # Maximum number of refinements
    extra_params: Dict[str, Any] = Field(default_factory=dict)


# endregion


class AgentConfig(BaseModel):
    """
    Complete configuration for an agent, which can be basic or use a complex workflow pattern.
    Only one workflow configuration should be set.
    """

    name: str
    instruction: Union[str, Callable[[Dict], str]] = "You are a helpful agent."
    server_names: List[str] = Field(default_factory=list)
    functions: List[Callable] = Field(default_factory=list)
    connection_persistence: bool = True
    human_input_callback: Optional[HumanInputCallback] = None

    # LLM config for either basic agent or workflow LLM factory
    llm_config: Optional[AugmentedLLMConfig] = None

    # Workflow configuration - only one should be set
    parallel_config: Optional[ParallelLLMConfig] = None
    orchestrator_config: Optional[OrchestratorLLMConfig] = None
    router_config: Optional[RouterConfig] = None
    evaluator_optimizer_config: Optional[EvaluatorOptimizerConfig] = None

    # Additional kwargs
    extra_kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def create_agent(self, context: Optional["Context"] = None) -> "Agent":
        """
        Create a basic agent instance.
        This doesn't initialize the agent or attach an LLM.

        Args:
            context: Optional Context to pass to the Agent

        Returns:
            Instantiated Agent object without initialization
        """
        from mcp_agent.agents.agent import Agent

        return Agent(
            name=self.name,
            instruction=self.instruction,
            server_names=self.server_names,
            functions=self.functions,
            connection_persistence=self.connection_persistence,
            human_input_callback=self.human_input_callback,
            context=context,
            **self.extra_kwargs,
        )

    def get_agent_type(self) -> Optional[str]:
        """
        Get the type of workflow this agent uses, if any.

        Returns:
            String identifier of workflow type or None for basic agents
        """
        configs = [
            ("parallel", self.parallel_config),
            ("orchestrator", self.orchestrator_config),
            ("router", self.router_config),
            ("evaluator_optimizer", self.evaluator_optimizer_config),
            ("swarm", self.swarm_config),
        ]

        for name, config in configs:
            if config is not None:
                return name

        return None

    def get_workflow_config(self) -> Optional[Any]:
        """
        Get the workflow configuration object.

        Returns:
            The appropriate workflow configuration object or None
        """
        workflow_type = self.get_agent_type()
        if workflow_type == "parallel":
            return self.parallel_config
        elif workflow_type == "orchestrator":
            return self.orchestrator_config
        elif workflow_type == "router":
            return self.router_config
        elif workflow_type == "evaluator_optimizer":
            return self.evaluator_optimizer_config
        elif workflow_type == "swarm":
            return self.swarm_config
        return None


async def create_agent(name: str, context: "Context") -> "Agent":
    """
    Create an agent with its configured workflow.

    Args:
        name: The name of the registered agent configuration

    Returns:
        Tuple of (agent instance, augmented LLM or workflow instance)

    Raises:
        ValueError: If no agent configuration with the given name exists
    """
    agent_configs = context.app._agent_configs
    if not agent_configs:
        raise ValueError("No AgentConfig's were found")

    # Check if the agent name is registered
    config = agent_configs.get(name)
    if not config:
        raise ValueError(f"No agent configuration named '{name}' is registered")

    # Create and initialize the agent
    agent = config.create_agent(context=context)
    await agent.initialize()

    # Create and attach the AugmentedLLM workflow if applicable
    workflow_type = config.get_agent_type()
    llm_factory = config.llm_config.factory if config.llm_config else None

    if workflow_type is None and config.llm_config is not None:
        # Basic agent with simple LLM
        llm_instance = await config.llm_config.create_llm()
        llm = await agent.attach_llm(lambda: llm_instance)
        return agent, llm
    elif workflow_type == "parallel":
        from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM  # pylint: disable=C0415

        parallel_config = config.parallel_config

        # Get referenced agents
        fan_in_agent = await create_agent(
            name=parallel_config.fan_in_agent, context=context
        )
        fan_out_agents = []
        for agent_name in parallel_config.fan_out_agents:
            fan_out_agent = await create_agent(name=agent_name, context=context)
            fan_out_agents.append(fan_out_agent)

        # Create parallel workflow
        parallel = ParallelLLM(
            fan_in_agent=fan_in_agent,
            fan_out_agents=fan_out_agents,
            llm_factory=llm_factory,
            context=context,
            **parallel_config.extra_params,
        )

        # Attach the parallel workflow to the agent
        await agent.attach_llm(llm=parallel)
        return agent
    elif workflow_type == "orchestrator":
        from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator  # pylint: disable=C0415

        orchestrator_config = config.orchestrator_config

        # Get referenced agents
        available_agents = []
        for agent_name in orchestrator_config.available_agents:
            available_agent = await create_agent(name=agent_name, context=context)
            available_agents.append(available_agent)

        # Optional planner agent
        planner = None
        if orchestrator_config.planner_agent:
            planner = await create_agent(
                name=orchestrator_config.planner_agent, context=context
            )

        # Create the orchestrator
        orchestrator = Orchestrator(
            llm_factory=llm_factory,
            available_agents=available_agents,
            planner=planner,
            context=context,
            **orchestrator_config.extra_params,
        )

        # Attach the orchestrator workflow to the agent
        await agent.attach_llm(llm=orchestrator)
        return agent
    elif workflow_type == "evaluator_optimizer":
        from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (  # pylint: disable=C0415
            EvaluatorOptimizerLLM,
            QualityRating,
        )

        eo_config = config.evaluator_optimizer_config

        evaluator_agent = await create_agent(
            name=eo_config.evaluator_agent, context=context
        )
        optimizer_agent = await create_agent(
            name=eo_config.optimizer_agent, context=context
        )

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
    else:
        # No workflow or LLM config, just return the basic agent
        return agent
