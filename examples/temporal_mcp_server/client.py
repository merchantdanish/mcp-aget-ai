import asyncio
import time
from mcp.types import CallToolResult
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
            args=[
                "run",
                "basic_agent_server.py",
            ],
        )

        # Connect to the workflow server
        async with gen_client("basic_agent_server", context.server_registry) as server:
            # Call the BasicAgentWorkflow
            run_result = await server.call_tool(
                "workflows/BasicAgentWorkflow/run",
                arguments={
                    "run_parameters": {
                        "input": "Print the first 2 paragraphs of https://modelcontextprotocol.io/introduction"
                    }
                },
            )

            print(run_result)


def _tool_result_to_json(tool_result: CallToolResult):
    if tool_result.content and len(tool_result.content) > 0:
        text = tool_result.content[0].text
        try:
            # Try to parse the response as JSON if it's a string
            import json

            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            # If it's not valid JSON, just use the text
            return None


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
