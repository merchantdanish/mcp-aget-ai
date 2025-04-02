import asyncio
import time

from mcp_agent.app import MCPApp
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
        context.server_registry.add_server(
            "workflow_server",
            command="uv",
            args=["run", "server.py"],
            description="Local workflow server exposing data processing and summarization workflows",
        )

        # Connect to the workflow server
        async with gen_client("workflow_server", context.server_registry) as server:
            # List available tools
            tools = await server.list_tools()
            logger.info(
                "Available tools:", data={"tools": [tool.name for tool in tools]}
            )

            # List available workflows
            logger.info("Fetching available workflows...")
            workflows_response = await server.call_tool("workflows/list", {})

            workflows = {}
            if workflows_response.content and len(workflows_response.content) > 0:
                workflows = workflows_response.content[0].text

            logger.info(
                "Available workflows:", data={"workflows": list(workflows.keys())}
            )

            # Run summarization workflow
            logger.info("Running the SummarizationWorkflowRegistered workflow...")
            sample_text = """
            The Model Context Protocol (MCP) is a standardized API for AI assistants to communicate with tools
            and services in their context. This protocol standardizes the way assistants access data through
            tool definitions, tools calls, and file/URL content. It is designed to make it easy for developers
            to give AI assistants access to data and tools, and for AI assistants to understand how to interact
            with those tools. The protocol defines a consistent pattern for tool discovery, invocation,
            and response handling that works across different AI assistant implementations.
            """

            # Start the summarization workflow
            workflow_run_response = await server.call_tool(
                "workflows/SummarizationWorkflowRegistered/run",
                {
                    "args": {
                        "content": sample_text,
                        "max_length": 200,
                        "style": "technical",
                        "key_points": 3,
                    }
                },
            )

            if workflow_run_response.content and len(workflow_run_response.content) > 0:
                workflow_result = workflow_run_response.content[0].text
                workflow_id = workflow_result.get("workflow_id")
                logger.info(
                    "Summarization workflow started", data={"workflow_id": workflow_id}
                )

                # Wait for workflow to complete
                logger.info("Waiting for workflow to complete...")
                await asyncio.sleep(5)

                # Check workflow status
                status_response = await server.call_tool(
                    "workflows/SummarizationWorkflowRegistered/get_status",
                    {"workflow_instance_id": workflow_id},
                )

                if status_response.content and len(status_response.content) > 0:
                    status = status_response.content[0].text

                    if status.get("completed", False) and "result" in status:
                        logger.info("Workflow completed!")
                        result = status.get("result", {})

                        if "value" in result:
                            summary = result["value"].get(
                                "summary", "No summary available"
                            )
                            key_points = result["value"].get(
                                "key_points", "No key points available"
                            )

                            logger.info("Summary:", data={"summary": summary})
                            logger.info("Key Points:", data={"key_points": key_points})
                    else:
                        logger.info("Workflow status:", data={"status": status})

            # Run data processor workflow
            logger.info("Running the DataProcessorWorkflowRegistered workflow...")

            # Use a URL that the server's fetch tool can access
            data_workflow_response = await server.call_tool(
                "workflows/DataProcessorWorkflowRegistered/run",
                {
                    "args": {
                        "source": "https://modelcontextprotocol.io/introduction",
                        "analysis_prompt": "Analyze what MCP is and its key benefits",
                        "output_format": "markdown",
                    }
                },
            )

            if (
                data_workflow_response.content
                and len(data_workflow_response.content) > 0
            ):
                workflow_result = data_workflow_response.content[0].text
                workflow_id = workflow_result.get("workflow_id")
                logger.info(
                    "Data processor workflow started", data={"workflow_id": workflow_id}
                )

                # Wait for workflow to complete (this might take longer)
                logger.info("Waiting for data processor workflow to complete...")
                max_wait = 30  # Maximum wait time in seconds
                wait_interval = 5  # Check every 5 seconds

                for _ in range(max_wait // wait_interval):
                    await asyncio.sleep(wait_interval)

                    # Check workflow status
                    status_response = await server.call_tool(
                        "workflows/DataProcessorWorkflowRegistered/get_status",
                        {"workflow_instance_id": workflow_id},
                    )

                    if status_response.content and len(status_response.content) > 0:
                        status = status_response.content[0].text

                        if status.get("completed", False):
                            result = status.get("result", {})
                            logger.info("Data processor workflow completed!")

                            if "value" in result:
                                logger.info(
                                    "Processed Data:",
                                    data={"data": result["value"][:500] + "..."},
                                )
                            break

                        # If failed, break early
                        if status.get("error"):
                            logger.error(
                                "Workflow failed:", data={"error": status.get("error")}
                            )
                            break
                else:
                    logger.warning(
                        "Workflow took too long to complete, giving up after waiting"
                    )


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
