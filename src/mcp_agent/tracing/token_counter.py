"""
Token counting and cost tracking system for MCP Agent framework.
Provides hierarchical tracking of token usage across agents and subagents.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set, Union, Tuple, Awaitable
from datetime import datetime
from collections import defaultdict
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
import atexit
import asyncio

from mcp_agent.workflows.llm.llm_selector import load_default_models, ModelInfo
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TokenUsageBase:
    """Base class for token usage information"""

    input_tokens: int = 0
    """Number of tokens in the input/prompt"""

    output_tokens: int = 0
    """Number of tokens in the output/completion"""

    total_tokens: int = 0
    """Total number of tokens (input + output)"""

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class TokenUsage(TokenUsageBase):
    """Token usage for a single LLM call with metadata"""

    model_name: Optional[str] = None
    """Name of the model used (e.g., 'gpt-4o', 'claude-3-opus')"""

    model_info: Optional[ModelInfo] = None
    """Full model metadata including provider, costs, capabilities"""

    timestamp: datetime = field(default_factory=datetime.now)
    """When this usage was recorded"""


@dataclass
class WatchConfig:
    """Configuration for watching a node"""

    watch_id: str
    """Unique identifier for this watch"""

    callback: Union[
        Callable[["TokenNode", TokenUsage], None],
        Callable[["TokenNode", TokenUsage], Awaitable[None]],
    ]
    """Callback function: (node, aggregated_usage) -> None or async version"""

    node: Optional["TokenNode"] = None
    """Specific node instance to watch"""

    node_name: Optional[str] = None
    """Node name to watch (used if node not provided)"""

    node_type: Optional[str] = None
    """Node type to watch (used if node not provided)"""

    threshold: Optional[int] = None
    """Only trigger callback when total tokens exceed this threshold"""

    throttle_ms: Optional[int] = None
    """Minimum milliseconds between callbacks for the same node"""

    include_subtree: bool = True
    """Whether to trigger on changes in subtree or just direct usage"""

    is_async: bool = False
    """Whether the callback is async"""

    _last_triggered: Dict[str, float] = field(default_factory=dict)
    """Track last trigger time per node for throttling"""


@dataclass
class TokenNode:
    """Node in the token usage tree"""

    name: str
    """Name of this node (e.g., agent name, workflow name)"""

    node_type: str
    """Type of node: 'app', 'workflow', 'agent', 'llm'
    
    Hierarchy:
    - 'app': Root level application (MCPApp)
    - 'workflow': Workflow class instances (e.g., BasicAgentWorkflow, ParallelWorkflow)
    - 'agent': Higher-order AugmentedLLM instances (e.g., Orchestrator, EvaluatorOptimizer, ParallelLLM)
    - 'llm': Base AugmentedLLM classes (e.g., OpenAIAugmentedLLM, AnthropicAugmentedLLM)
    """

    parent: Optional["TokenNode"] = None
    """Parent node in the tree"""

    children: List["TokenNode"] = field(default_factory=list)
    """Child nodes"""

    usage: TokenUsage = field(default_factory=TokenUsage)
    """Direct token usage by this node (not including children)"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata for this node"""

    _cached_aggregate: Optional[TokenUsage] = field(default=None, init=False)
    """Cached aggregate usage to avoid deep recursion"""

    _cache_valid: bool = field(default=False, init=False)
    """Whether the cached aggregate is valid"""

    def add_child(self, child: "TokenNode") -> None:
        """Add a child node"""
        child.parent = self
        self.children.append(child)
        # Invalidate cache when structure changes
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        """Invalidate cache for this node and all ancestors"""
        self._cache_valid = False
        self._cached_aggregate = None
        if self.parent:
            self.parent.invalidate_cache()

    def aggregate_usage(self) -> TokenUsage:
        """Recursively aggregate usage from this node and all children (with caching)"""
        try:
            # Return cached value if valid
            if self._cache_valid and self._cached_aggregate is not None:
                return self._cached_aggregate

            # Compute aggregated usage
            total = TokenUsage(
                input_tokens=self.usage.input_tokens,
                output_tokens=self.usage.output_tokens,
                total_tokens=self.usage.total_tokens,
            )

            for child in self.children:
                try:
                    child_usage = child.aggregate_usage()
                    total.input_tokens += child_usage.input_tokens
                    total.output_tokens += child_usage.output_tokens
                    total.total_tokens += child_usage.total_tokens
                except Exception as e:
                    logger.error(f"Error aggregating usage for child {child.name}: {e}")

            # Cache the result
            self._cached_aggregate = total
            self._cache_valid = True

            return total
        except Exception as e:
            logger.error(f"Error in aggregate_usage: {e}")
            return TokenUsage()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        # Direct usage info
        usage_dict = {
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "model_name": self.usage.model_name,
            "timestamp": self.usage.timestamp.isoformat(),
        }

        # Include model info if available
        if self.usage.model_info:
            usage_dict["model_info"] = {
                "name": self.usage.model_info.name,
                "provider": self.usage.model_info.provider,
                "description": self.usage.model_info.description,
                "context_window": self.usage.model_info.context_window,
                "tool_calling": self.usage.model_info.tool_calling,
                "structured_outputs": self.usage.model_info.structured_outputs,
            }

        # Aggregated usage (including children)
        aggregated = self.aggregate_usage()

        aggregate_usage_dict = {
            "input_tokens": aggregated.input_tokens,
            "output_tokens": aggregated.output_tokens,
            "total_tokens": aggregated.total_tokens,
        }

        return {
            "name": self.name,
            "type": self.node_type,
            "usage": usage_dict,
            "aggregate_usage": aggregate_usage_dict,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class ModelUsageSummary:
    """Summary of usage for a specific model"""

    model_name: str
    """Name of the model"""

    usage: TokenUsageBase
    """Token usage for this model"""

    cost: float
    """Total cost in USD for this model's usage"""

    provider: Optional[str] = None
    """Provider of the model (e.g., 'openai', 'anthropic')"""

    model_info: Optional[Dict[str, Any]] = None
    """Serialized ModelInfo metadata (capabilities, context window, etc.)"""


@dataclass
class ModelUsageDetail(ModelUsageSummary):
    """Detailed usage for a specific model including which nodes used it"""

    nodes: List[TokenNode] = field(default_factory=list)
    """List of nodes that directly used this model"""

    @property
    def total_tokens(self) -> int:
        """Total tokens used by this model"""
        return self.usage.total_tokens

    @property
    def input_tokens(self) -> int:
        """Input tokens used by this model"""
        return self.usage.input_tokens

    @property
    def output_tokens(self) -> int:
        """Output tokens used by this model"""
        return self.usage.output_tokens


@dataclass
class TokenSummary:
    """Complete summary of token usage across all models and nodes"""

    usage: TokenUsageBase
    """Total token usage across all models"""

    cost: float
    """Total cost in USD across all models"""

    model_usage: Dict[str, ModelUsageSummary]
    """Usage breakdown by model. Key is 'model_name (provider)' or just 'model_name'"""

    usage_tree: Optional[Dict[str, Any]] = None
    """Hierarchical view of usage by node (serialized TokenNode tree)"""


@dataclass
class NodeSummary:
    """Summary of a node's token usage"""

    name: str
    """Name of the node"""

    node_type: str
    """Type of node: 'agent', 'workflow', etc."""

    usage: TokenUsageBase
    """Total token usage for this node (including children)"""


@dataclass
class NodeTypeUsage:
    """Token usage aggregated by node type (e.g., all agents, all workflows, etc.)"""

    node_type: str
    """Type of node: 'agent', 'workflow', etc."""

    node_count: int
    """Number of nodes of this type"""

    usage: TokenUsageBase
    """Combined token usage for all nodes of this type"""


@dataclass
class NodeUsageDetail:
    """Detailed breakdown of a node's token usage"""

    name: str
    """Name of the node"""

    node_type: str
    """Type of node: 'agent', 'workflow', etc."""

    direct_usage: TokenUsageBase
    """Token usage directly by this node (not including children)"""

    usage: TokenUsageBase
    """Total token usage including all descendants"""

    usage_by_node_type: Dict[str, NodeTypeUsage]
    """Usage breakdown by child node type (e.g., {'agent': NodeTypeUsage(...), 'workflow': NodeTypeUsage(...)})"""

    child_usage: List[NodeSummary]
    """Usage summary for each direct child node"""


class TokenCounter:
    """
    Hierarchical token counter with cost calculation.
    Thread-safe implementation for tracking token usage across the call stack.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._stack: List[TokenNode] = []
        self._root: Optional[TokenNode] = None
        self._current: Optional[TokenNode] = None

        # Load model costs
        self._models: List[ModelInfo] = load_default_models()
        self._model_costs = self._build_cost_lookup()
        self._model_lookup = {model.name: model for model in self._models}
        self._models_by_provider = self._build_provider_lookup()

        # Cache for model lookups to avoid repeated fuzzy matching
        # Key: (model_name, provider), Value: ModelInfo or None
        self._model_cache: Dict[Tuple[str, Optional[str]], Optional[ModelInfo]] = {}

        # Track total usage by (model_name, provider) tuple
        self._usage_by_model: Dict[Tuple[str, Optional[str]], TokenUsage] = defaultdict(
            TokenUsage
        )

        # Watch configurations
        self._watches: Dict[str, WatchConfig] = {}
        self._node_watches: Dict[int, Set[str]] = defaultdict(
            set
        )  # node_id -> watch_ids

        # Thread pool for sync callback execution
        self._callback_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="token-watch"
        )
        # Track if we're running in an event loop
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Register cleanup on shutdown
        atexit.register(self._cleanup_executor)

    def _build_cost_lookup(self) -> Dict[str, Dict[str, float]]:
        """Build lookup table for model costs"""
        cost_lookup = {}

        for model in self._models:
            if model.metrics.cost.blended_cost_per_1m is not None:
                blended_cost = model.metrics.cost.blended_cost_per_1m
            elif (
                model.metrics.cost.input_cost_per_1m is not None
                and model.metrics.cost.output_cost_per_1m is not None
            ):
                # Default 3:1 input:output ratio
                blended_cost = (
                    model.metrics.cost.input_cost_per_1m * 3
                    + model.metrics.cost.output_cost_per_1m
                ) / 4
            else:
                blended_cost = 1.0  # Fallback

            cost_lookup[model.name] = {
                "blended_cost_per_1m": blended_cost,
                "input_cost_per_1m": model.metrics.cost.input_cost_per_1m,  # Keep None if not set
                "output_cost_per_1m": model.metrics.cost.output_cost_per_1m,  # Keep None if not set
            }

        return cost_lookup

    def _build_provider_lookup(self) -> Dict[str, Dict[str, ModelInfo]]:
        """Build lookup table for models by provider"""
        provider_models: Dict[str, Dict[str, ModelInfo]] = {}
        for model in self._models:
            if model.provider not in provider_models:
                provider_models[model.provider] = {}
            provider_models[model.provider][model.name] = model
        return provider_models

    def find_model_info(
        self, model_name: str, provider: Optional[str] = None
    ) -> Optional[ModelInfo]:
        """
        Find ModelInfo by name and optionally provider.

        Args:
            model_name: Name of the model
            provider: Optional provider to help disambiguate

        Returns:
            ModelInfo if found, None otherwise
        """
        # Check cache first
        cache_key = (model_name, provider)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # Try exact match first
        model_info = self._model_lookup.get(model_name)
        if model_info:
            # If provider specified, check if it matches
            if (
                provider is None
                or provider == model_info.provider
                or provider.lower() == model_info.provider.lower()
            ):
                # Cache the result
                self._model_cache[cache_key] = model_info
                return model_info

        # If provider is specified, search within that provider's models
        provider_models: Dict[str, ModelInfo] = (
            self._models_by_provider.get(provider, None) if provider else None
        )
        if provider and not provider_models:
            # If no provider models, try case-insensitive match
            for key, models in self._models_by_provider.items():
                if key.lower() == provider.lower():
                    provider_models = models
                    break

        if provider_models:
            # Try exact match within provider
            if model_name in provider_models:
                result = provider_models[model_name]
                self._model_cache[cache_key] = result
                return result

            # Try fuzzy match within provider - prefer longer matches
            best_match = None
            best_match_score = 0

            for known_name, known_model in provider_models.items():
                score = 0

                # Calculate match score
                if model_name == known_name:
                    score = 1000  # Exact match
                elif known_name.startswith(model_name):
                    # Prefer matches where search term is a prefix (e.g., gpt-4o-mini matches gpt-4o-mini-2024-07-18)
                    score = 500 + (len(model_name) / len(known_name) * 100)
                elif model_name in known_name:
                    score = len(model_name) / len(known_name) * 100
                elif known_name in model_name:
                    score = (
                        len(known_name) / len(model_name) * 50
                    )  # Lower score for partial matches

                if score > best_match_score:
                    best_match = known_model
                    best_match_score = score

            if best_match:
                self._model_cache[cache_key] = best_match
                return best_match

        # Try fuzzy match across all models - prefer longer matches
        best_match = None
        best_match_score = 0

        for known_name, known_model in self._model_lookup.items():
            score = 0

            # Calculate match score
            if model_name == known_name:
                score = 1000  # Exact match
            elif known_name.startswith(model_name):
                # Prefer matches where search term is a prefix (e.g., gpt-4o-mini matches gpt-4o-mini-2024-07-18)
                score = 500 + (len(model_name) / len(known_name) * 100)
            elif model_name in known_name:
                score = len(model_name) / len(known_name) * 100
            elif known_name in model_name:
                score = (
                    len(known_name) / len(model_name) * 50
                )  # Lower score for partial matches

            # Boost score if provider matches
            if (
                score > 0
                and provider
                and provider.lower() in known_model.provider.lower()
            ):
                score += 50

            if score > best_match_score:
                best_match = known_model
                best_match_score = score

        if best_match:
            # Cache the result
            self._model_cache[cache_key] = best_match
            return best_match

        # Cache the None result too to avoid repeated searches
        self._model_cache[cache_key] = None
        return None

    def push(
        self, name: str, node_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Push a new context onto the stack.
        This is called when entering a new scope (app, workflow, agent, etc).
        """
        try:
            with self._lock:
                node = TokenNode(
                    name=name, node_type=node_type, metadata=metadata or {}
                )

                if self._current:
                    self._current.add_child(node)
                else:
                    # This is the root
                    self._root = node

                self._stack.append(node)
                self._current = node

                # logger.debug(f"Pushed token context: {name} ({node_type})")
        except Exception as e:
            logger.error(f"Error in TokenCounter.push: {e}", exc_info=True)
            # Continue execution - don't break the program

    def pop(self) -> Optional[TokenNode]:
        """
        Pop the current context from the stack.
        Returns the popped node with aggregated usage.
        """
        try:
            with self._lock:
                if not self._stack:
                    logger.warning("Attempted to pop from empty token stack")
                    return None

                node = self._stack.pop()
                self._current = self._stack[-1] if self._stack else None

                return node
        except Exception as e:
            logger.error(f"Error in TokenCounter.pop: {e}", exc_info=True)
            return None

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        model_info: Optional[ModelInfo] = None,
    ) -> None:
        """
        Record token usage at the current stack level.
        This is called by AugmentedLLM after each LLM call.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: Name of the model (e.g., "gpt-4", "claude-3-opus")
            provider: Optional provider name to help disambiguate models
            model_info: Optional full ModelInfo object with metadata
        """
        try:
            # Validate inputs
            input_tokens = int(input_tokens) if input_tokens is not None else 0
            output_tokens = int(output_tokens) if output_tokens is not None else 0

            with self._lock:
                if not self._current:
                    logger.warning("No current token context, creating root")
                    try:
                        self.push("root", "app")
                    except Exception as e:
                        logger.error(f"Failed to create root context: {e}")
                        return

                # If we have model_name but no model_info, try to look it up
                if model_name and not model_info:
                    try:
                        model_info = self.find_model_info(model_name, provider)
                    except Exception as e:
                        logger.debug(f"Failed to find model info for {model_name}: {e}")

                # Update current node's usage
                if self._current and hasattr(self._current, "usage"):
                    self._current.usage.input_tokens += input_tokens
                    self._current.usage.output_tokens += output_tokens
                    self._current.usage.total_tokens += input_tokens + output_tokens

                    # Store model information
                    if model_name and not self._current.usage.model_name:
                        self._current.usage.model_name = model_name
                    if model_info and not self._current.usage.model_info:
                        self._current.usage.model_info = model_info

                    # logger.debug(
                    #     f"Recording {input_tokens + output_tokens} tokens for node {self._current.name} "
                    #     f"({self._current.node_type}), total before: {self._current.usage.total_tokens - input_tokens - output_tokens}"
                    # )

                    # Only invalidate the current node's cache (not ancestors)
                    # This prevents cascade invalidation up the tree
                    self._current._cache_valid = False
                    self._current._cached_aggregate = None
                    # logger.debug(
                    #     f"Invalidated cache for {self._current.name} (targeted)"
                    # )

                    # Trigger watches which will handle ancestor updates
                    self._trigger_watches(self._current)
                    # logger.debug(f"Triggered watches for {self._current.name}")

                # Track global usage by model and provider
                if model_name:
                    try:
                        # Use provider from model_info if available, otherwise use the passed provider
                        provider_key = (
                            model_info.provider
                            if model_info and hasattr(model_info, "provider")
                            else provider
                        )
                        usage_key = (model_name, provider_key)

                        model_usage = self._usage_by_model[usage_key]
                        model_usage.input_tokens += input_tokens
                        model_usage.output_tokens += output_tokens
                        model_usage.total_tokens += input_tokens + output_tokens
                        model_usage.model_name = model_name
                        if model_info and not model_usage.model_info:
                            model_usage.model_info = model_info
                    except Exception as e:
                        logger.error(f"Failed to track global usage: {e}")

                # logger.debug(
                #     f"Recorded {input_tokens + output_tokens} tokens "
                #     f"(in: {input_tokens}, out: {output_tokens}) "
                #     f"for {getattr(self._current, 'name', 'unknown')} using {model_name or 'unknown model'}"
                # )
        except Exception as e:
            logger.error(f"Error in TokenCounter.record_usage: {e}", exc_info=True)
            # Continue execution - don't break the program

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        provider: Optional[str] = None,
    ) -> float:
        """Calculate cost for given token usage"""
        try:
            # Validate inputs
            input_tokens = max(0, int(input_tokens) if input_tokens is not None else 0)
            output_tokens = max(
                0, int(output_tokens) if output_tokens is not None else 0
            )

            # Look up the model to get accurate cost
            try:
                model_info = self.find_model_info(model_name, provider)
                if model_info:
                    model_name = model_info.name
            except Exception as e:
                logger.debug(f"Failed to find model info: {e}")

            if not model_name or model_name not in self._model_costs:
                logger.info(
                    f"Model {model_name} not found in costs, using default estimate"
                )
                return (input_tokens + output_tokens) * 0.5 / 1_000_000

            costs = self._model_costs.get(model_name, {})

            input_cost_per_1m = costs.get("input_cost_per_1m")
            output_cost_per_1m = costs.get("output_cost_per_1m")

            if input_cost_per_1m is not None and output_cost_per_1m is not None:
                input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
                output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
                total_cost = input_cost + output_cost
                # logger.debug(
                #     f"Using input/output costs: input_cost=${input_cost:.6f}, output_cost=${output_cost:.6f}, total=${total_cost:.6f}"
                # )
                return total_cost
            else:
                total_tokens = input_tokens + output_tokens
                blended_cost_per_1m = costs.get("blended_cost_per_1m", 0.5)
                blended_cost = (total_tokens / 1_000_000) * blended_cost_per_1m
                # logger.debug(
                #     f"Using blended cost: total_tokens={total_tokens}, blended_cost_per_1m={blended_cost_per_1m}, total=${blended_cost:.6f}"
                # )
                return blended_cost
        except Exception as e:
            logger.warning(f"Error in TokenCounter.calculate_cost: {e}", exc_info=True)
            # Return a default cost estimate
            return (input_tokens + output_tokens) * 0.5 / 1_000_000

    def get_current_path(self) -> List[str]:
        """Get the current stack path (e.g., ['app', 'workflow', 'agent'])"""
        with self._lock:
            return [node.name for node in self._stack]

    def get_tree(self) -> Optional[Dict[str, Any]]:
        """Get the full token usage tree"""
        with self._lock:
            if self._root:
                return self._root.to_dict()
            return None

    def get_summary(self) -> TokenSummary:
        """Get summary of token usage and costs"""
        try:
            with self._lock:
                total_cost = 0.0
                model_costs: Dict[str, ModelUsageSummary] = {}

                # Calculate costs per model
                for (model_name, provider_key), usage in self._usage_by_model.items():
                    try:
                        # Use the provider from the key (which came from record_usage)
                        # Fall back to model_info.provider if key's provider is None
                        provider = provider_key
                        if provider is None and usage.model_info:
                            provider = getattr(usage.model_info, "provider", None)

                        # logger.debug(
                        #     f"Calculating cost for {model_name} from {provider}"
                        # )
                        # logger.debug(
                        #     f"Usage - input: {usage.input_tokens}, output: {usage.output_tokens}, total: {usage.total_tokens}"
                        # )

                        cost = self.calculate_cost(
                            model_name,
                            usage.input_tokens,
                            usage.output_tokens,
                            provider,
                        )

                        # logger.debug(f"get_summary: Calculated cost: ${cost:.6f}")
                        total_cost += cost

                        # Create model info dict if available
                        model_info_dict = None
                        if usage.model_info:
                            try:
                                model_info_dict = {
                                    "provider": getattr(
                                        usage.model_info, "provider", None
                                    ),
                                    "description": getattr(
                                        usage.model_info, "description", None
                                    ),
                                    "context_window": getattr(
                                        usage.model_info, "context_window", None
                                    ),
                                    "tool_calling": getattr(
                                        usage.model_info, "tool_calling", None
                                    ),
                                    "structured_outputs": getattr(
                                        usage.model_info, "structured_outputs", None
                                    ),
                                }
                            except Exception as e:
                                logger.debug(f"Failed to extract model info: {e}")

                        model_summary = ModelUsageSummary(
                            model_name=model_name,
                            provider=provider,
                            usage=TokenUsageBase(
                                input_tokens=usage.input_tokens,
                                output_tokens=usage.output_tokens,
                                total_tokens=usage.total_tokens,
                            ),
                            cost=cost,
                            model_info=model_info_dict,
                        )

                        # Create a descriptive key for the summary
                        if provider:
                            summary_key = f"{model_name} ({provider})"
                        else:
                            summary_key = model_name

                        model_costs[summary_key] = model_summary
                    except Exception as e:
                        logger.error(f"Error processing model {model_name}: {e}")
                        continue

                # Get total usage
                total_usage = TokenUsage()
                if self._root:
                    try:
                        total_usage = self._root.aggregate_usage()
                    except Exception as e:
                        logger.error(f"Error aggregating total usage: {e}")

                return TokenSummary(
                    usage=TokenUsageBase(
                        input_tokens=total_usage.input_tokens,
                        output_tokens=total_usage.output_tokens,
                        total_tokens=total_usage.total_tokens,
                    ),
                    cost=total_cost,
                    model_usage=model_costs,
                    usage_tree=self.get_tree() if self._root else None,
                )
        except Exception as e:
            logger.error(f"Error in get_summary: {e}", exc_info=True)
            # Return empty summary on error
            return TokenSummary(
                usage=TokenUsageBase(),
                cost=0.0,
                model_usage={},
                usage_tree=None,
            )

    def reset(self) -> None:
        """Reset all token tracking"""
        with self._lock:
            self._stack.clear()
            self._root = None
            self._current = None
            self._usage_by_model.clear()
            self._watches.clear()
            self._node_watches.clear()
            logger.debug("Token counter reset")

    def find_node(
        self, name: str, node_type: Optional[str] = None
    ) -> Optional[TokenNode]:
        """
        Find a node by name and optionally type.

        Args:
            name: The name of the node to find
            node_type: Optional node type to filter by

        Returns:
            The first matching node, or None if not found
        """
        with self._lock:
            if not self._root:
                return None

            return self._find_node_recursive(self._root, name, node_type)

    def _find_node_recursive(
        self, node: TokenNode, name: str, node_type: Optional[str] = None
    ) -> Optional[TokenNode]:
        """Recursively search for a node"""
        try:
            # Check current node
            if node.name == name and (node_type is None or node.node_type == node_type):
                return node

            # Search children
            for child in node.children:
                try:
                    result = self._find_node_recursive(child, name, node_type)
                    if result:
                        return result
                except Exception as e:
                    logger.debug(f"Error searching child node: {e}")
                    continue

            return None
        except Exception as e:
            logger.error(f"Error in _find_node_recursive: {e}")
            return None

    def find_nodes_by_type(self, node_type: str) -> List[TokenNode]:
        """
        Find all nodes of a specific type.

        Args:
            node_type: The type of nodes to find (e.g., 'agent', 'workflow', 'llm_call')

        Returns:
            List of matching nodes
        """
        with self._lock:
            if not self._root:
                return []

            nodes = []
            self._find_nodes_by_type_recursive(self._root, node_type, nodes)
            return nodes

    def _find_nodes_by_type_recursive(
        self, node: TokenNode, node_type: str, nodes: List[TokenNode]
    ) -> None:
        """Recursively collect nodes by type"""
        if node.node_type == node_type:
            nodes.append(node)

        for child in node.children:
            self._find_nodes_by_type_recursive(child, node_type, nodes)

    def get_node_usage(
        self, name: str, node_type: Optional[str] = None
    ) -> Optional[TokenUsage]:
        """
        Get aggregated token usage for a specific node (including its children).

        Args:
            name: The name of the node
            node_type: Optional node type to filter by

        Returns:
            Aggregated TokenUsage for the node and its children, or None if not found
        """
        with self._lock:
            node = self.find_node(name, node_type)
            if node:
                return node.aggregate_usage()
            return None

    def get_node_cost(self, name: str, node_type: Optional[str] = None) -> float:
        """
        Calculate the total cost for a specific node (including its children).

        Args:
            name: The name of the node
            node_type: Optional node type to filter by

        Returns:
            Total cost for the node and its children
        """
        with self._lock:
            node = self.find_node(name, node_type)
            if not node:
                return 0.0

            return self._calculate_node_cost(node)

    def _calculate_node_cost(self, node: TokenNode) -> float:
        """Calculate cost for a node and its children"""
        try:
            total_cost = 0.0

            # If this node has direct usage with a model, calculate its cost
            if node.usage.model_name:
                provider = None
                if node.usage.model_info:
                    provider = getattr(node.usage.model_info, "provider", None)

                try:
                    cost = self.calculate_cost(
                        node.usage.model_name,
                        node.usage.input_tokens,
                        node.usage.output_tokens,
                        provider,
                    )
                    total_cost += cost
                except Exception as e:
                    logger.error(f"Error calculating cost for node {node.name}: {e}")

            # Add costs from children
            for child in node.children:
                try:
                    total_cost += self._calculate_node_cost(child)
                except Exception as e:
                    logger.error(f"Error calculating cost for child {child.name}: {e}")
                    continue

            return total_cost
        except Exception as e:
            logger.error(f"Error in _calculate_node_cost: {e}")
            return 0.0

    def get_app_usage(self) -> Optional[TokenUsage]:
        """Get total token usage for the entire application (root node)"""
        with self._lock:
            if self._root:
                return self._root.aggregate_usage()
            return None

    def get_agent_usage(self, name: str) -> Optional[TokenUsage]:
        """Get token usage for a specific agent"""
        return self.get_node_usage(name, "agent")

    def get_workflow_usage(self, name: str) -> Optional[TokenUsage]:
        """Get token usage for a specific workflow"""
        return self.get_node_usage(name, "workflow")

    def get_current_usage(self) -> Optional[TokenUsage]:
        """Get token usage for the current context"""
        with self._lock:
            if self._current:
                return self._current.aggregate_usage()
            return None

    def get_node_subtree(
        self, name: str, node_type: Optional[str] = None
    ) -> Optional[TokenNode]:
        """
        Get a node and its entire subtree.

        Args:
            name: The name of the node
            node_type: Optional node type to filter by

        Returns:
            The node with all its children, or None if not found
        """
        return self.find_node(name, node_type)

    def find_node_by_metadata(
        self,
        metadata_key: str,
        metadata_value: Any,
        node_type: Optional[str] = None,
        return_all_matches: bool = False,
    ) -> Optional[TokenNode] | List[TokenNode]:
        """
        Find a node by a specific metadata key-value pair.

        Args:
            metadata_key: The metadata key to search for
            metadata_value: The value to match
            node_type: Optional node type to filter by
            return_all_matches: If True, return all matching nodes; if False, return first match

        Returns:
            If return_all_matches is False: The first matching node, or None if not found
            If return_all_matches is True: List of all matching nodes (empty if none found)
        """
        with self._lock:
            if not self._root:
                return [] if return_all_matches else None

            matches = []
            self._find_node_by_metadata_recursive(
                self._root, metadata_key, metadata_value, node_type, matches
            )

            if return_all_matches:
                return matches
            else:
                return matches[0] if matches else None

    def _find_node_by_metadata_recursive(
        self,
        node: TokenNode,
        metadata_key: str,
        metadata_value: Any,
        node_type: Optional[str],
        matches: List[TokenNode],
    ) -> None:
        """Recursively search for nodes by metadata"""
        try:
            # Check if this node matches
            if node_type is None or node.node_type == node_type:
                # Safely check metadata
                if (
                    hasattr(node, "metadata")
                    and node.metadata is not None
                    and metadata_key in node.metadata
                    and node.metadata.get(metadata_key) == metadata_value
                ):
                    matches.append(node)

            # Search children
            for child in node.children:
                try:
                    self._find_node_by_metadata_recursive(
                        child, metadata_key, metadata_value, node_type, matches
                    )
                except Exception as e:
                    logger.debug(f"Error searching child node: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in _find_node_by_metadata_recursive: {e}")

    def get_app_node(self) -> Optional[TokenNode]:
        """Get the root application node"""
        with self._lock:
            return self._root if self._root and self._root.node_type == "app" else None

    def get_workflow_node(
        self,
        name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        run_id: Optional[str] = None,
        return_all_matches: bool = False,
    ) -> Optional[TokenNode] | List[TokenNode]:
        """
        Get a specific workflow node.

        Args:
            name: Name of the workflow
            workflow_id: Optional workflow_id to find specific workflow instances
            run_id: Optional run_id to find a specific workflow run (takes precedence)
            return_all_matches: If True, return all matching nodes

        Returns:
            The workflow node(s) if found
        """
        # Priority: run_id > workflow_id > name
        if run_id:
            return self.find_node_by_metadata(
                "run_id", run_id, "workflow", return_all_matches
            )
        elif workflow_id:
            return self.find_node_by_metadata(
                "workflow_id", workflow_id, "workflow", return_all_matches
            )
        elif name:
            if return_all_matches:
                return (
                    self.find_nodes_by_type("workflow")
                    if name == "*"
                    else [
                        n for n in self.find_nodes_by_type("workflow") if n.name == name
                    ]
                )
            else:
                return self.find_node(name, "workflow")
        else:
            return [] if return_all_matches else None

    def get_agent_node(
        self, name: str, return_all_matches: bool = False
    ) -> Optional[TokenNode] | List[TokenNode]:
        """
        Get a specific agent (higher-order AugmentedLLM) node.

        Args:
            name: Name of the agent
            return_all_matches: If True, return all matching nodes

        Returns:
            The agent node(s) if found
        """
        if return_all_matches:
            return [n for n in self.find_nodes_by_type("agent") if n.name == name]
        else:
            return self.find_node(name, "agent")

    def get_llm_node(
        self, name: str, return_all_matches: bool = False
    ) -> Optional[TokenNode] | List[TokenNode]:
        """
        Get a specific LLM (base AugmentedLLM) node.

        Args:
            name: Name of the LLM
            return_all_matches: If True, return all matching nodes

        Returns:
            The LLM node(s) if found
        """
        if return_all_matches:
            return [n for n in self.find_nodes_by_type("llm") if n.name == name]
        else:
            return self.find_node(name, "llm")

    def get_node_breakdown(
        self, name: str, node_type: Optional[str] = None
    ) -> Optional[NodeUsageDetail]:
        """
        Get a detailed breakdown of token usage for a node and its children.

        Args:
            name: The name of the node
            node_type: Optional node type to filter by

        Returns:
            NodeUsageDetail with breakdown by child type and direct children, or None if not found
        """
        with self._lock:
            node = self.find_node(name, node_type)
            if not node:
                return None

            # Group children by type
            children_by_type: Dict[str, List[TokenNode]] = defaultdict(list)
            for child in node.children:
                children_by_type[child.node_type].append(child)

            # Calculate usage by child type
            usage_by_node_type: Dict[str, NodeTypeUsage] = {}
            for child_type, children in children_by_type.items():
                type_usage = TokenUsage()
                for child in children:
                    child_usage = child.aggregate_usage()
                    type_usage.input_tokens += child_usage.input_tokens
                    type_usage.output_tokens += child_usage.output_tokens
                    type_usage.total_tokens += child_usage.total_tokens

                usage_by_node_type[child_type] = NodeTypeUsage(
                    node_type=child_type,
                    node_count=len(children),
                    usage=TokenUsageBase(
                        input_tokens=type_usage.input_tokens,
                        output_tokens=type_usage.output_tokens,
                        total_tokens=type_usage.total_tokens,
                    ),
                )

            # Add individual children info
            child_usage: List[NodeSummary] = []
            for child in node.children:
                child_aggregated = child.aggregate_usage()
                child_usage.append(
                    NodeSummary(
                        name=child.name,
                        node_type=child.node_type,
                        usage=TokenUsageBase(
                            input_tokens=child_aggregated.input_tokens,
                            output_tokens=child_aggregated.output_tokens,
                            total_tokens=child_aggregated.total_tokens,
                        ),
                    )
                )

            # Get aggregated usage for the node
            aggregated = node.aggregate_usage()

            return NodeUsageDetail(
                name=node.name,
                node_type=node.node_type,
                direct_usage=TokenUsageBase(
                    input_tokens=node.usage.input_tokens,
                    output_tokens=node.usage.output_tokens,
                    total_tokens=node.usage.total_tokens,
                ),
                usage=TokenUsageBase(
                    input_tokens=aggregated.input_tokens,
                    output_tokens=aggregated.output_tokens,
                    total_tokens=aggregated.total_tokens,
                ),
                usage_by_node_type=usage_by_node_type,
                child_usage=child_usage,
            )

    def get_agents_breakdown(self) -> Dict[str, TokenUsage]:
        """Get token usage breakdown by agent"""
        agents = self.find_nodes_by_type("agent")
        breakdown = {}
        for agent in agents:
            usage = agent.aggregate_usage()
            breakdown[agent.name] = usage
        return breakdown

    def get_workflows_breakdown(self) -> Dict[str, TokenUsage]:
        """Get token usage breakdown by workflow"""
        workflows = self.find_nodes_by_type("workflow")
        breakdown = {}
        for workflow in workflows:
            usage = workflow.aggregate_usage()
            breakdown[workflow.name] = usage
        return breakdown

    def get_models_breakdown(self) -> List[ModelUsageDetail]:
        """
        Get detailed breakdown of usage by model.

        Returns:
            List of ModelUsageDetail containing usage details and nodes for each model
        """
        with self._lock:
            if not self._root:
                return []

            # Collect all nodes that have model usage
            model_nodes: Dict[Tuple[str, Optional[str]], List[TokenNode]] = defaultdict(
                list
            )
            self._collect_model_nodes(self._root, model_nodes)

            # Build ModelUsageDetail for each model
            breakdown: List[ModelUsageDetail] = []

            for (model_name, provider), nodes in model_nodes.items():
                # Calculate total usage for this model
                total_input = 0
                total_output = 0

                for node in nodes:
                    total_input += node.usage.input_tokens
                    total_output += node.usage.output_tokens

                total_tokens = total_input + total_output
                total_cost = self.calculate_cost(
                    model_name, total_input, total_output, provider
                )

                breakdown.append(
                    ModelUsageDetail(
                        model_name=model_name,
                        provider=provider,
                        usage=TokenUsageBase(
                            input_tokens=total_input,
                            output_tokens=total_output,
                            total_tokens=total_tokens,
                        ),
                        cost=total_cost,
                        model_info=None,
                        nodes=nodes,
                    )
                )

            # Sort by total tokens descending
            breakdown.sort(key=lambda x: x.total_tokens, reverse=True)

            return breakdown

    def _collect_model_nodes(
        self,
        node: TokenNode,
        model_nodes: Dict[Tuple[str, Optional[str]], List[TokenNode]],
    ) -> None:
        """Recursively collect nodes that have model usage"""
        # If this node has model usage, add it
        if node.usage.model_name:
            provider = None
            if node.usage.model_info:
                provider = node.usage.model_info.provider

            key = (node.usage.model_name, provider)
            model_nodes[key].append(node)

        # Recurse to children
        for child in node.children:
            self._collect_model_nodes(child, model_nodes)

    def watch(
        self,
        callback: Union[
            Callable[[TokenNode, TokenUsage], None],
            Callable[[TokenNode, TokenUsage], Awaitable[None]],
        ],
        node: Optional[TokenNode] = None,
        node_name: Optional[str] = None,
        node_type: Optional[str] = None,
        threshold: Optional[int] = None,
        throttle_ms: Optional[int] = None,
        include_subtree: bool = True,
    ) -> str:
        """
        Watch a node or nodes for token usage changes.

        Args:
            callback: Function called when usage changes: (node, aggregated_usage) -> None
            node: Specific node instance to watch (highest priority)
            node_name: Node name pattern to watch (used if node not provided)
            node_type: Node type to watch (used if node not provided)
            threshold: Only trigger when total tokens exceed this value
            throttle_ms: Minimum milliseconds between callbacks for the same node
            include_subtree: Whether to trigger on subtree changes or just direct usage

        Returns:
            watch_id: Unique identifier for this watch (use to unwatch)

        Examples:
            # Watch a specific node
            watch_id = counter.watch(callback, node=my_node)

            # Watch all workflow nodes
            watch_id = counter.watch(callback, node_type="workflow")

            # Watch with threshold
            watch_id = counter.watch(callback, node_name="my_agent", threshold=1000)
        """
        with self._lock:
            watch_id = str(uuid.uuid4())

            # Detect if callback is async by checking if it's a coroutine function
            is_async = asyncio.iscoroutinefunction(callback)

            config = WatchConfig(
                watch_id=watch_id,
                callback=callback,
                node=node,
                node_name=node_name,
                node_type=node_type,
                threshold=threshold,
                throttle_ms=throttle_ms,
                include_subtree=include_subtree,
                is_async=is_async,
            )

            self._watches[watch_id] = config

            # If watching a specific node, track it
            if node:
                self._node_watches[id(node)].add(watch_id)

            # Try to get the current event loop if we're in async context
            try:
                self._event_loop = asyncio.get_running_loop()
            except RuntimeError:
                # No event loop running, will use thread pool for sync callbacks
                pass

            logger.debug(
                f"Added watch {watch_id} for node={node_name}, type={node_type}, async={is_async}"
            )
            return watch_id

    def unwatch(self, watch_id: str) -> bool:
        """
        Remove a watch.

        Args:
            watch_id: The watch identifier returned by watch()

        Returns:
            True if watch was removed, False if not found
        """
        with self._lock:
            config = self._watches.pop(watch_id, None)
            if not config:
                return False

            # Remove from node-specific tracking
            if config.node:
                node_id = id(config.node)
                if node_id in self._node_watches:
                    self._node_watches[node_id].discard(watch_id)
                    if not self._node_watches[node_id]:
                        del self._node_watches[node_id]

            logger.debug(f"Removed watch {watch_id}")
            return True

    def _cleanup_executor(self) -> None:
        """Clean up thread pool executor on shutdown"""
        try:
            self._callback_executor.shutdown(wait=True, cancel_futures=False)
        except Exception as e:
            logger.error(f"Error shutting down callback executor: {e}")

    def _trigger_watches(self, node: TokenNode) -> None:
        """Trigger watches for a node and its ancestors"""
        try:
            callbacks_to_execute: List[Tuple[WatchConfig, TokenNode, TokenUsage]] = []
            # logger.debug(f"_trigger_watches called for {node.name} ({node.node_type})")

            with self._lock:
                current = node
                triggered_nodes = set()
                is_original_node = True

                # Walk up the tree to collect watches that need triggering
                while current:
                    if id(current) in triggered_nodes:
                        break
                    triggered_nodes.add(id(current))

                    # Invalidate this node's cache to ensure fresh aggregation
                    # This is more targeted than cascade invalidation
                    current._cache_valid = False
                    current._cached_aggregate = None

                    # Get aggregated usage with fresh data
                    usage = current.aggregate_usage()

                    # Check all watches
                    for watch_id, config in self._watches.items():
                        try:
                            # Check if this watch applies to the current node
                            if not self._watch_matches_node(config, current):
                                continue

                            # For ancestor nodes, only trigger if include_subtree is True
                            if not is_original_node and not config.include_subtree:
                                continue

                            # Check threshold
                            if (
                                config.threshold
                                and usage.total_tokens < config.threshold
                            ):
                                continue

                            # Check throttling
                            node_key = f"{id(current)}"
                            if config.throttle_ms:
                                last_triggered = config._last_triggered.get(node_key, 0)
                                now = time.time() * 1000  # milliseconds
                                if now - last_triggered < config.throttle_ms:
                                    continue
                                config._last_triggered[node_key] = now

                            # Clone the usage data to avoid issues with cache updates
                            usage_copy = TokenUsage(
                                input_tokens=usage.input_tokens,
                                output_tokens=usage.output_tokens,
                                total_tokens=usage.total_tokens,
                                model_name=usage.model_name,
                                model_info=usage.model_info,
                            )

                            # Queue callback for execution outside lock
                            callbacks_to_execute.append((config, current, usage_copy))
                            logger.debug(
                                f"Queued watch {config.watch_id} for {current.name} ({current.node_type}) "
                                f"with {usage_copy.total_tokens} tokens"
                            )

                        except Exception as e:
                            logger.error(f"Error processing watch {watch_id}: {e}")

                    # Move to parent to check watches on ancestors
                    current = current.parent
                    is_original_node = False

            # Execute callbacks outside the lock
            for config, callback_node, callback_usage in callbacks_to_execute:
                self._execute_callback(config, callback_node, callback_usage)

        except Exception as e:
            logger.error(f"Error in _trigger_watches: {e}", exc_info=True)

    def _execute_callback(
        self, config: WatchConfig, node: TokenNode, usage: TokenUsage
    ) -> None:
        """Execute a callback, detecting async context at runtime"""
        try:
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop and not loop.is_closed():
                if config.is_async:
                    # Use the captured loop explicitly
                    task = loop.create_task(
                        self._execute_async_callback_safely(
                            config.callback, node, usage
                        )
                    )
                    # Add error handling to the task
                    task.add_done_callback(self._handle_task_exception)
                else:
                    # Run sync callback in executor to avoid blocking
                    loop.run_in_executor(
                        self._callback_executor,
                        self._execute_callback_safely,
                        config.callback,
                        node,
                        usage,
                    )
            else:
                # No event loop or closed loop
                if config.is_async:
                    logger.debug(
                        f"Async callback {config.watch_id} called outside event loop context. "
                        "Executing with asyncio.run in thread pool."
                    )
                    # Execute in thread pool with asyncio.run
                    self._callback_executor.submit(
                        lambda: asyncio.run(
                            self._execute_async_callback_safely(
                                config.callback, node, usage
                            )
                        )
                    )
                else:
                    # Execute sync callback in thread pool
                    self._callback_executor.submit(
                        self._execute_callback_safely, config.callback, node, usage
                    )
        except Exception as e:
            logger.error(f"Error executing callback: {e}", exc_info=True)

    def _handle_task_exception(self, task: asyncio.Task) -> None:
        """Handle exceptions from async tasks"""
        try:
            task.result()
        except Exception as e:
            logger.error(f"Async task error: {e}", exc_info=True)

    def _execute_callback_safely(
        self,
        callback: Callable[[TokenNode, TokenUsage], None],
        node: TokenNode,
        usage: TokenUsage,
    ) -> None:
        """Execute a sync watch callback safely in thread pool"""
        try:
            callback(node, usage)
        except Exception as e:
            logger.error(f"Watch callback error: {e}", exc_info=True)

    async def _execute_async_callback_safely(
        self,
        callback: Callable[[TokenNode, TokenUsage], Awaitable[None]],
        node: TokenNode,
        usage: TokenUsage,
    ) -> None:
        """Execute an async watch callback safely"""
        try:
            await callback(node, usage)
        except Exception as e:
            logger.error(f"Async watch callback error: {e}", exc_info=True)

    def _watch_matches_node(self, config: WatchConfig, node: TokenNode) -> bool:
        """Check if a watch configuration matches a specific node"""
        # Specific node instance match
        if config.node:
            return config.node is node

        # Node type match
        if config.node_type and node.node_type != config.node_type:
            return False

        # Node name match
        if config.node_name and node.name != config.node_name:
            return False

        # If no specific criteria, it matches all nodes
        return True
