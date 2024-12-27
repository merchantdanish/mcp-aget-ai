# TODO: saqadri - consider moving these to a separate file so they can be imported on-demand
from .swarm import Swarm
from ..workflows.augmented_llm_openai import OpenAIAugmentedLLM


class OpenAISwarm(Swarm, OpenAIAugmentedLLM):
    """
    MCP version of the OpenAI Swarm class (https://github.com/openai/swarm.), using OpenAI's ChatCompletion as the LLM.
    """
