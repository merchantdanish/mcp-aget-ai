from .swarm import Swarm
from ..workflows.augmented_llm_anthropic import AnthropicAugmentedLLM


class AnthropicSwarm(Swarm, AnthropicAugmentedLLM):
    """
    MCP version of the OpenAI Swarm class (https://github.com/openai/swarm.), using Anthropic's API as the LLM.
    """
