"""
Token counting and cost tracking system for MCP Agent framework.
Provides hierarchical tracking of token usage across agents and subagents.
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

from mcp_agent.workflows.llm.llm_selector import load_default_models, ModelInfo
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TokenUsage:
    """Token usage for a single LLM call"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_name: Optional[str] = None
    model_info: Optional[ModelInfo] = None  # Full model metadata
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class TokenNode:
    """Node in the token usage tree"""

    name: str
    node_type: str  # 'app', 'workflow', 'agent', 'llm_call'
    parent: Optional["TokenNode"] = None
    children: List["TokenNode"] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "TokenNode") -> None:
        """Add a child node"""
        child.parent = self
        self.children.append(child)

    def aggregate_usage(self) -> TokenUsage:
        """Recursively aggregate usage from this node and all children"""
        total = TokenUsage(
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            total_tokens=self.usage.total_tokens,
        )

        for child in self.children:
            child_usage = child.aggregate_usage()
            total.input_tokens += child_usage.input_tokens
            total.output_tokens += child_usage.output_tokens
            total.total_tokens += child_usage.total_tokens

        return total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
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

        return {
            "name": self.name,
            "type": self.node_type,
            "usage": usage_dict,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children],
        }


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

        # Track total usage by (model_name, provider) tuple
        self._usage_by_model: Dict[tuple[str, Optional[str]], TokenUsage] = defaultdict(
            TokenUsage
        )

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
                "input_cost_per_1m": model.metrics.cost.input_cost_per_1m
                or blended_cost,
                "output_cost_per_1m": model.metrics.cost.output_cost_per_1m
                or blended_cost,
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
        # Try exact match first
        model_info = self._model_lookup.get(model_name)
        if model_info:
            # If provider specified, check if it matches
            if provider is None or provider == model_info.provider:
                return model_info

        # If provider is specified, search within that provider's models
        if provider and provider in self._models_by_provider:
            provider_models = self._models_by_provider[provider]
            # Try exact match within provider
            if model_name in provider_models:
                return provider_models[model_name]

            # Try fuzzy match within provider
            for known_name, known_model in provider_models.items():
                if model_name in known_name or known_name in model_name:
                    return known_model

        # Try fuzzy match across all models
        for known_name, known_model in self._model_lookup.items():
            if model_name in known_name or known_name in model_name:
                # If provider specified, prefer models from that provider
                if provider and provider.lower() in known_model.provider.lower():
                    return known_model

        # Last resort: fuzzy match without provider preference
        for known_name, known_model in self._model_lookup.items():
            if model_name in known_name or known_name in model_name:
                return known_model

        return None

    def push(
        self, name: str, node_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Push a new context onto the stack.
        This is called when entering a new scope (app, workflow, agent, etc).
        """
        with self._lock:
            node = TokenNode(name=name, node_type=node_type, metadata=metadata or {})

            if self._current:
                self._current.add_child(node)
            else:
                # This is the root
                self._root = node

            self._stack.append(node)
            self._current = node

            logger.debug(f"Pushed token context: {name} ({node_type})")

    def pop(self) -> Optional[TokenNode]:
        """
        Pop the current context from the stack.
        Returns the popped node with aggregated usage.
        """
        with self._lock:
            if not self._stack:
                logger.warning("Attempted to pop from empty token stack")
                return None

            node = self._stack.pop()
            self._current = self._stack[-1] if self._stack else None

            # Log aggregated usage for this node
            usage = node.aggregate_usage()
            logger.debug(
                f"Popped token context: {node.name} ({node.node_type}) - "
                f"Total: {usage.total_tokens} tokens "
                f"(input: {usage.input_tokens}, output: {usage.output_tokens})"
            )

            return node

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
        with self._lock:
            if not self._current:
                logger.warning("No current token context, creating root")
                self.push("root", "app")

            # If we have model_name but no model_info, try to look it up
            if model_name and not model_info:
                model_info = self.find_model_info(model_name, provider)

            # Update current node's usage
            self._current.usage.input_tokens += input_tokens
            self._current.usage.output_tokens += output_tokens
            self._current.usage.total_tokens += input_tokens + output_tokens

            # Store model information
            if model_name and not self._current.usage.model_name:
                self._current.usage.model_name = model_name
            if model_info and not self._current.usage.model_info:
                self._current.usage.model_info = model_info

            # Track global usage by model and provider
            if model_name:
                # Use provider from model_info if available, otherwise use the passed provider
                provider_key = model_info.provider if model_info else provider
                usage_key = (model_name, provider_key)

                model_usage = self._usage_by_model[usage_key]
                model_usage.input_tokens += input_tokens
                model_usage.output_tokens += output_tokens
                model_usage.total_tokens += input_tokens + output_tokens
                model_usage.model_name = model_name
                if model_info and not model_usage.model_info:
                    model_usage.model_info = model_info

            logger.debug(
                f"Recorded {input_tokens + output_tokens} tokens "
                f"(in: {input_tokens}, out: {output_tokens}) "
                f"for {self._current.name} using {model_name or 'unknown model'}"
            )

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        provider: Optional[str] = None,
    ) -> float:
        """Calculate cost for given token usage"""
        # Look up the model to get accurate cost
        model_info = self.find_model_info(model_name, provider)
        if model_info:
            model_name = model_info.name

        if not model_name or model_name not in self._model_costs:
            # Default estimate
            return (input_tokens + output_tokens) * 0.5 / 1_000_000

        costs = self._model_costs[model_name]

        if costs["input_cost_per_1m"] and costs["output_cost_per_1m"]:
            input_cost = (input_tokens / 1_000_000) * costs["input_cost_per_1m"]
            output_cost = (output_tokens / 1_000_000) * costs["output_cost_per_1m"]
            return input_cost + output_cost
        else:
            total_tokens = input_tokens + output_tokens
            return (total_tokens / 1_000_000) * costs["blended_cost_per_1m"]

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

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of token usage and costs"""
        with self._lock:
            total_cost = 0.0
            model_costs = {}

            # Calculate costs per model
            for (model_name, provider_key), usage in self._usage_by_model.items():
                # Use the provider from the key (which came from record_usage)
                # Fall back to model_info.provider if key's provider is None
                provider = provider_key
                if provider is None and usage.model_info:
                    provider = usage.model_info.provider

                cost = self.calculate_cost(
                    model_name, usage.input_tokens, usage.output_tokens, provider
                )
                total_cost += cost

                model_dict = {
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                    "cost": cost,
                    "provider": provider,
                }

                # Include model info if available
                if usage.model_info:
                    model_dict["model_info"] = {
                        "provider": usage.model_info.provider,
                        "description": usage.model_info.description,
                        "context_window": usage.model_info.context_window,
                        "tool_calling": usage.model_info.tool_calling,
                        "structured_outputs": usage.model_info.structured_outputs,
                    }

                # Create a descriptive key for the summary
                if provider:
                    summary_key = f"{model_name} ({provider})"
                else:
                    summary_key = model_name

                model_costs[summary_key] = model_dict

            # Get total usage
            total_usage = TokenUsage()
            if self._root:
                total_usage = self._root.aggregate_usage()

            return {
                "total_usage": {
                    "input_tokens": total_usage.input_tokens,
                    "output_tokens": total_usage.output_tokens,
                    "total_tokens": total_usage.total_tokens,
                },
                "total_cost": total_cost,
                "by_model": model_costs,
                "tree": self.get_tree() if self._root else None,
            }

    def reset(self) -> None:
        """Reset all token tracking"""
        with self._lock:
            self._stack.clear()
            self._root = None
            self._current = None
            self._usage_by_model.clear()
            logger.debug("Token counter reset")
