import asyncio
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

        # --- Example: Using the resource-demo MCP server ---
        resource_agent = Agent(
            name="resource_agent",
            instruction="Demo agent for MCP resource primitives",
            server_names=["resource-demo"],
        )

        async with resource_agent:
            logger.info(
                "resource_agent: Connected to resource-demo, calling list_resources..."
            )
            resources = await resource_agent.list_resources()
            resource_uris = {
                resource.name: str(resource.uri) for resource in resources.resources
            }
            logger.info(
                "Resources available from resource-demo:",
                data=resource_uris,
            )

            llm = await resource_agent.attach_llm(OpenAIAugmentedLLM)
            res = await llm.generate_str(
                "Summarise what is in my resources",
                resource_uris=[
                    resource_uris["resource-demo_get_readme"],
                    resource_uris["resource-demo_get_users"],
                ],
            )
            logger.info(f"Resource summary: {res}")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
