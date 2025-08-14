import asyncio

from mcp_agent.app import MCPApp
from mcp_agent.workflows.factory import (
    load_agent_specs_from_file,
    create_router_llm,
)


async def main():
    async with MCPApp(name="factory_demo").run() as agent_app:
        context = agent_app.context
        # Add current directory to filesystem server (if needed by your setup)
        context.config.mcp.servers["filesystem"].args.extend(["."])

        specs = load_agent_specs_from_file(
            "examples/workflows/factory/agents.yaml", context=context
        )

        router = await create_router_llm(
            server_names=["filesystem", "fetch"],
            agents=specs,
            provider="openai",
            context=context,
        )

        res = await router.generate_str("Find the README and summarize it")
        print("Routing result:", res)


if __name__ == "__main__":
    asyncio.run(main())
