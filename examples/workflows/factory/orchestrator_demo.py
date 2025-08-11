import asyncio

from mcp_agent.app import MCPApp
from mcp_agent.workflows.factory import (
    load_agent_specs_from_file,
    create_llm,
    create_orchestrator,
)
from mcp.types import ModelPreferences


async def main():
    async with MCPApp(name="orchestrator_demo").run() as agent_app:
        context = agent_app.context
        context.config.mcp.servers["filesystem"].args.extend(["."])

        specs = load_agent_specs_from_file(
            "examples/workflows/factory/agents.yaml", context=context
        )

        # Build an LLM with a specific model id
        planner_llm = create_llm(
            agent_name="planner",
            provider="openai",
            model_preferences="openai:gpt-4o-mini",
            context=context,
        )

        orch = create_orchestrator(
            available_agents=[planner_llm, *specs],
            provider="anthropic",
            model_preferences=ModelPreferences(
                costPriority=0.2, speedPriority=0.3, intelligencePriority=0.5
            ),
            context=context,
        )

        result = await orch.generate_str("Summarize key components in this repository.")
        print("Orchestrator result:\n", result)


if __name__ == "__main__":
    asyncio.run(main())


