import asyncio
import os
import time

from mcp_agent.app import MCPApp
from mcp_agent.config import (
    Settings,
    LoggerSettings,
    MCPSettings,
    MCPServerSettings,
    OpenAISettings,
    AnthropicSettings,
)
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

settings = Settings(
    execution_engine="asyncio",
    logger=LoggerSettings(type="file", level="debug"),
    mcp=MCPSettings(
        servers={
            "fetch": MCPServerSettings(
                command="uvx",
                args=["mcp-server-fetch"],
            ),
            "filesystem": MCPServerSettings(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem"],
            ),
        }
    ),
    openai=OpenAISettings(
        api_key="sk-my-openai-api-key",
        default_model="gpt-4o-mini",
    ),
    anthropic=AnthropicSettings(
        api_key="sk-my-anthropic-api-key",
    ),
)

# Settings can either be specified programmatically,
# or loaded from mcp_agent.config.yaml/mcp_agent.secrets.yaml
app = MCPApp(name="mcp_basic_agent")  # settings=settings)


async def example_usage():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        finder_agent = Agent(
            name="agent",
            instruction="""A good assistant""",
            server_names=["everything"],
        )

        async with finder_agent:
            logger.info("finder: Connected to server, calling list_tools...")
            tools = await finder_agent.list_tools()
            logger.info("Tools available:", data=tools.model_dump())

            logger.info("finder: Connected to server, calling list_resources...")
            resources = await finder_agent.list_resources()
            logger.info("Resources available:", data=resources.model_dump())

            resource = await finder_agent.read_resource(
                "everart://images", "everything"
            )
            logger.info("Resource result:", data=resource.model_dump())

            llm = await finder_agent.attach_llm(OpenAIAugmentedLLM)

            # Keep resources
            result = await llm.generate_str(
                message="Print the contents of mcp_agent.config.yaml verbatim",
                resource_uri="everart://images",
            )
            result = await llm.generate()
            logger.info(f"mcp_agent.config.yaml contents: {result}")

            # # Let's switch the same agent to a different LLM
            # llm = await finder_agent.attach_llm(AnthropicAugmentedLLM)

            # result = await llm.generate_str(
            #     message="Print the first 2 paragraphs of https://modelcontextprotocol.io/introduction",
            # )
            # logger.info(f"First 2 paragraphs of Model Context Protocol docs: {result}")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
