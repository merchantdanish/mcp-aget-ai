# SSE example

This example shows how to use tracing for an SSE server with mcp-agent.

- `server.py` is a simple server that runs on localhost:8000
- `main.py` is the mcp-agent client that uses the SSE server.py

### Exporting to Collector

For this example, [install Jaeger locally](https://www.jaegertracing.io/docs/2.5/getting-started/) and then update the `mcp_agent.config.yaml` for this example to have `otel.otlp_endpoint` point to the collector endpoint (e.g. `http://localhost:4318/v1/traces` is the default for Jaeger via HTTP). Make sure the otlp endpoint in `server.py`
matches this endpoint.

## Run Example

To run, open two terminals:

1. `uv run server.py`
2. `uv run main.py`

<img width="1848" alt="image" src="https://github.com/user-attachments/assets/94c1e17c-a8d7-4455-8008-8f02bc404c28" />
