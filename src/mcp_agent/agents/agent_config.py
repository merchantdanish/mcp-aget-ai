"""
Agent configuration classes for declarative agent definition.
"""

from typing import TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union, Generic

from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, RequestParams
from mcp_agent.human_input.types import HumanInputCallback

if TYPE_CHECKING:
    from mcp_agent.agents.agent import Agent

# Define TypeVar for LLM types
LLM = TypeVar("LLM", bound=AugmentedLLM)


class AugmentedLLMConfig(BaseModel, Generic[LLM]):
    """
    Configuration for creating an AugmentedLLM instance.
    Provides type-safe configuration for different LLM providers.
    """

    # The factory function or class that creates the LLM
    factory: Callable[..., LLM]

    # Common parameters that apply to most LLMs
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    # Model-specific parameters
    provider_params: Dict[str, Any] = Field(default_factory=dict)

    # Request parameters used in generate calls
    default_request_params: Optional[RequestParams] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def create_llm(self) -> LLM:
        """
        Create an LLM instance using this configuration.

        Returns:
            An instance of the configured LLM type
        """
        # Combine common parameters with provider-specific parameters
        params = {}
        if self.model:
            params["model"] = self.model
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

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


# Type-safe configs for each workflow pattern


class ParallelWorkflowConfig(BaseModel):
    """Type-safe configuration for ParallelLLM workflow pattern."""

    fan_in_agent: str  # Name of the agent to use for fan-in
    fan_out_agents: List[str]  # Names of agents to use for fan-out
    concurrent: bool = True
    synchronize_fan_out_models: bool = False
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class OrchestratorWorkflowConfig(BaseModel):
    """Type-safe configuration for Orchestrator workflow pattern."""

    available_agents: List[str]  # Names of agents available to the orchestrator
    max_iterations: int = 10
    planner_agent: Optional[str] = None  # Optional custom planner agent
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class RouterWorkflowConfig(BaseModel):
    """Type-safe configuration for Router workflow pattern."""

    agent_names: List[str]  # Names of agents to route between
    top_k: int = 1
    router_type: Literal["llm", "embedding"] = "llm"
    embedding_model: Optional[str] = None  # For embedding-based router
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class EvaluatorOptimizerWorkflowConfig(BaseModel):
    """Type-safe configuration for Evaluator-Optimizer workflow pattern."""

    evaluator_agent: str  # Name of the agent to use as evaluator
    optimizer_agent: str  # Name of the agent to use as optimizer
    min_rating: str = "excellent"  # Minimum quality rating to accept
    max_iterations: int = 5
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class SwarmWorkflowConfig(BaseModel):
    """Type-safe configuration for Swarm workflow pattern."""

    agents: List[str]  # Names of agents in the swarm
    context_variables: Dict[str, Any] = Field(default_factory=dict)
    extra_params: Dict[str, Any] = Field(default_factory=dict)


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
    parallel_config: Optional[ParallelWorkflowConfig] = None
    orchestrator_config: Optional[OrchestratorWorkflowConfig] = None
    router_config: Optional[RouterWorkflowConfig] = None
    evaluator_optimizer_config: Optional[EvaluatorOptimizerWorkflowConfig] = None
    swarm_config: Optional[SwarmWorkflowConfig] = None

    # Additional kwargs
    extra_kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def create_agent(self, context=None) -> "Agent":
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

    def get_workflow_type(self) -> Optional[str]:
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
        workflow_type = self.get_workflow_type()
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
