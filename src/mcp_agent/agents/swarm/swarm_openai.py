from mcp_agent.agents.swarm.swarm import Swarm
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


class OpenAISwarm(Swarm, OpenAIAugmentedLLM):
    """
    MCP version of the OpenAI Swarm class (https://github.com/openai/swarm.), using OpenAI's ChatCompletion as the LLM.
    """
