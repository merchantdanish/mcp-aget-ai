import asyncio
import time

from mcp_agent.app import MCPApp
from mcp_agent.config import (
    Settings,
    LoggerSettings,
    MCPSettings,
    MCPServerSettings,
    OpenAISettings,
)
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

settings = Settings(
    execution_engine="asyncio",
    logger=LoggerSettings(type="console", level="debug"),
    mcp=MCPSettings(
        servers={
            "demo-server": MCPServerSettings(
                command="uvx", args=["run", "demo_server.py"]
            )
        }
    ),
    openai=OpenAISettings(
        api_key="sk-my-openai-api-key",
        default_model="gpt-4o-mini",
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

        # --- Example: Using the demo-server MCP server ---
        agent = Agent(
            name="agent",
            instruction="Demo agent for MCP resource and prompt primitives",
            server_names=["demo-server"],
        )

        async with agent:
            # List all resources from demo-server server
            resources = await agent.list_resources("demo-server")
            resource_uris = {
                resource.name: str(resource.uri) for resource in resources.resources
            }
            logger.info(
                "Resources available from demo-server:",
                data=resource_uris,
            )

            # List all prompts from demo-server server
            prompts = await agent.list_prompts("demo-server")
            logger.info(
                "Prompts available from demo-server:",
                data=prompts.model_dump(),
            )

            # Attach a resource to the agent
            await agent.attach_resource(
                resource_uris["demo-server_get_readme"],
                "demo-server",
            )

            # Attach a prompt to the agent
            await agent.attach_prompt(
                prompts.prompts[0].name,
                {"message": "My name is John Doe."},
            )

            llm = await agent.attach_llm(OpenAIAugmentedLLM)
            res = await llm.generate_str("Summarise what are my prompts and resources?")
            logger.info(f"Summary: {res}")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
