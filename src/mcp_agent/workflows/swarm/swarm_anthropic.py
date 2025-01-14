from typing import List

from mcp_agent.workflows.swarm.swarm import Swarm
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class AnthropicSwarm(Swarm, AnthropicAugmentedLLM):
    """
    MCP version of the OpenAI Swarm class (https://github.com/openai/swarm.),
    using Anthropic's API as the LLM.
    """

    async def generate(
        self,
        message,
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = "claude-3-5-sonnet-20241022",
        stop_sequences: List[str] = None,
        max_tokens: int = 8192,
        parallel_tool_calls: bool = False,
    ):
        iterations = 0
        response = None
        agent_name = str(self.aggregator.name) if self.aggregator else None

        while iterations < max_iterations and agent_name:
            response = await super().generate(
                message=message
                if iterations == 0
                else "Please resolve my original request. If it has already been resolved then end turn",
                use_history=use_history,
                max_iterations=1,  # TODO: saqadri - validate
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )
            logger.debug(f"Agent: {agent_name}, response:", data=response)
            agent_name = self.aggregator.name if self.aggregator else None
            iterations += 1

        # Return final response back
        return response
