# MCP Azure Agent Example - "Finder" Agent

This example demonstrates how to create and run a basic "Finder" Agent using Azure OpenAI model and MCP. The Agent has access to the `fetch` MCP server, enabling it to retrieve information from URLs.

## Setup

Check out the [Azure Python SDK docs](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-inference-readme?view=azure-python-preview#getting-started) to obtain the following values:

- `endpoint`
- `api_key`
- `api_version` (optional)


## Running the Agent

To run the "Finder" agent, navigate to the example directory and execute:

```bash
cd examples/mcp_basic_azure_agent

uv run main.py
```