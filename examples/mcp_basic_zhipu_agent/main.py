import asyncio

from mcp_agent.config import (
    Settings,
    MCPSettings,
    MCPServerSettings,
    ZhipuSettings,
    LoggerSettings,
)
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_zhipu import ZhipuAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams


async def run():
    """Run the finder agent example."""

    # Create settings
    settings = Settings(
        mcp=MCPSettings(
            servers={
                "fetch": MCPServerSettings(
                    command="uvx",
                    args=["mcp-server-fetch"],
                ),
                "hotnews": MCPServerSettings(
                    command="npx",
                    args=["-y", "@mcpflow.io/mcp-hotnews-mcp-server"],
                ),
                "time": MCPServerSettings(
                    command="uvx",
                    args=["mcp-server-time", "--local-timezone=America/New_York"],
                ),
            }
        ),
        execution_engine="asyncio",
        logger=LoggerSettings(type="console", level="info"),
        zhipu=ZhipuSettings(
            api_key="<your_api_key>",
            default_model="glm-4-flashx-250414",  # Use the same model as in augmented_llm_zhipu.py
        ),
    )

    # Initialize the app with settings
    app = MCPApp(name="mcp_basic_zhipu_agent", settings=settings)

    # Run the app
    async with app.run():
        # Create an agent that can load different LLMs - Use more concise prompts
        finder_agent = Agent(
            name="finder",
            instruction="""You are an assistant that can use tools to answer questions.
            """,
            server_names=["time"],
        )

        # list tools
        tools = await finder_agent.list_tools()
        print("Tools available:", tools)

        # Initialize the agent
        async with finder_agent:
            # Create the base agent with default model
            llm = await finder_agent.attach_llm(ZhipuAugmentedLLM)

            # create request parameters - Explicitly specify the model
            request_params = RequestParams(
                model="glm-4-flashx-250414",  # Explicitly specify the model to maintain consistency with augmented_llm_zhipu.py
                temperature=0.1,
                maxTokens=4096,
                systemPrompt=None,  # Don't use systemPrompt to avoid duplication with the instruction
            )

            try:
                # Use a very explicit query
                result = await llm.generate_str(
                    message="What time is it in New York? Use the time_get_current_time tool with timezone parameter set to 'America/New_York'.",
                    request_params=request_params,
                    force_tools=True,
                )
                print("\n==== Response using tool ====")
                print(result)
            except Exception as e:
                print(f"Error during model generation: {e}")


if __name__ == "__main__":
    asyncio.run(run())
