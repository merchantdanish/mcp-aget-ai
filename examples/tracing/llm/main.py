import asyncio
import time

from pydantic import BaseModel

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_anthropic import MessageParam
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM


# Settings loaded from mcp_agent.config.yaml/mcp_agent.secrets.yaml
app = MCPApp(name="llm_tracing_example")


class CountryInfo(BaseModel):
    """Model representing structured data for country information."""

    capital: str
    population: int


async def llm_tracing():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        # Direct LLM usage
        openai_llm = OpenAIAugmentedLLM(
            name="openai_llm",
            default_request_params=RequestParams(maxTokens=1024),
        )

        result = await openai_llm.generate_str(
            message="What is the capital of France?",
        )
        logger.info(f"openai_llm result: {result}")

        await openai_llm.select_model(RequestParams(model="gpt-4"))
        result = await openai_llm.generate_str(
            message="What is the capital of Belgium?",
        )
        logger.info(f"openai_llm result: {result}")

        structured = await openai_llm.generate_structured(
            MessageParam(
                role="user",
                content="Give JSON representing the the capitals and populations of the following countries: France, Ireland, Italy",
            ),
            response_model=CountryInfo,
        )
        logger.info(f"openai_llm structured result: {structured}")

        # Agent-integrated LLM
        llm_agent = Agent(name="llm_agent")
        async with llm_agent:
            llm = await llm_agent.attach_llm(AnthropicAugmentedLLM)
            result = await llm.generate("What is the capital of Germany?")
            logger.info(f"llm_agent result: {result}")

            structured = await llm.generate_structured(
                MessageParam(
                    role="user",
                    content="Give JSON representing the the capitals and populations of the following countries: France, Germany, Belgium",
                ),
                response_model=CountryInfo,
            )
            logger.info(f"llm_agent structured result: {structured}")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(llm_tracing())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
