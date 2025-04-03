import asyncio
import time

from mcp_agent.app import MCPApp
from mcp_agent.config import MCPServerSettings
from mcp_agent.mcp.gen_client import gen_client


async def main():
    # Create MCPApp to get the server registry
    app = MCPApp(name="workflow_mcp_client")
    async with app.run() as client_app:
        logger = client_app.logger
        context = client_app.context

        # Connect to the workflow server
        logger.info("Connecting to workflow server...")

        # Override the server configuration to point to our local script
        context.server_registry.registry["basic_agent_server"] = MCPServerSettings(
            name="basic_agent_server",
            description="Local workflow server running the basic agent example",
            command="uv",
            args=["run", "basic_agent_server.py"],
        )

        # Connect to the workflow server
        async with gen_client("basic_agent_server", context.server_registry) as server:
            # List available tools
            tools_result = await server.list_tools()
            logger.info(
                "Available tools:",
                data={"tools": [tool.name for tool in tools_result.tools]},
            )

            # List available workflows
            logger.info("Fetching available workflows...")
            workflows_response = await server.call_tool("workflows/list", {})

            workflows = {}
            if workflows_response.content and len(workflows_response.content) > 0:
                workflows_text = workflows_response.content[0].text
                try:
                    # Try to parse the response as JSON if it's a string
                    import json

                    workflows = json.loads(workflows_text)
                except (json.JSONDecodeError, TypeError):
                    # If it's not valid JSON, just use the text
                    logger.info("Received workflows text:", data=workflows_text)
                    workflows = {"workflows_text": workflows_text}

            logger.info(
                "Available workflows:", data={"workflows": list(workflows.keys())}
            )


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
