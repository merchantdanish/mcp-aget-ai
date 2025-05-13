# MCP Streamable HTTP example

This example shows mcp-agent usage with a Streamable HTTP server (using the [example server](https://github.com/modelcontextprotocol/python-sdk/tree/main/examples/servers/simple-streamablehttp-stateless) in the `mcp-python` repo).

To run:

1. Start the server: `uv run stateless_server.py`
2. (in a new CLI terminal) Run the mcp-agent application: `uv run main.py`

The server should connect, initialize and list its tools.
