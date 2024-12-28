from mcp_agent.agents.swarm import Swarm
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM


class AnthropicSwarm(Swarm, AnthropicAugmentedLLM):
    """
    MCP version of the OpenAI Swarm class (https://github.com/openai/swarm.), using Anthropic's API as the LLM.
    """
