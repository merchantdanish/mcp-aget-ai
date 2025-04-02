# Workflow MCP Server Example

This example demonstrates how to:

1. Create custom workflows using the MCP Agent framework
2. Expose those workflows as MCP tools via an MCP server
3. Connect to the workflow MCP server from a client application

## Concepts Demonstrated

- Using the `Workflow` base class to create custom workflows
- Registering workflows with an `MCPApp`
- Exposing workflows as MCP tools using `app_server.py`
- Connecting to a workflow server using `gen_client`
- Running workflows and monitoring their progress

## Workflows in this Example

1. **DataProcessorWorkflow**: A workflow that processes data in three steps:
   - Finding and retrieving content from a source (file or URL)
   - Analyzing the content
   - Formatting the results

2. **SummarizationWorkflow**: A workflow that summarizes text content:
   - Generates a concise summary
   - Extracts key points
   - Returns structured data

## How to Run

1. Copy the example secrets file:
   ```
   cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml
   ```

2. Edit `mcp_agent.secrets.yaml` to add your API keys.

3. Run the client, which will automatically start the server:
   ```
   uv run client.py
   ```

## Code Structure

- `server.py`: Defines the workflows and creates the MCP server
- `client.py`: Connects to the server and runs the workflows
- `mcp_agent.config.yaml`: Configuration for MCP servers and other settings
- `mcp_agent.secrets.yaml`: Secret API keys (not included in repository)

## Understanding the Code

### Workflow Definition

Workflows are defined by subclassing the `Workflow` base class and implementing:
- The `run` method containing the main workflow logic
- Optional `initialize` and `cleanup` methods for setup and teardown

```python
class DataProcessorWorkflow(Workflow[str]):
    async def run(self, source: str, analysis_prompt: Optional[str] = None, output_format: Optional[str] = None) -> WorkflowResult[str]:
        # Workflow implementation...
```

### Registering a Workflow

Workflows are registered with the MCPApp using the `@app.workflow` decorator:

```python
app = MCPApp(name="workflow_mcp_server")

@app.workflow
class DataProcessorWorkflowRegistered(DataProcessorWorkflow):
    pass
```

### Exposing Workflows as Tools

The MCP server automatically exposes workflows as tools, creating endpoints for:
- Running a workflow: `workflows/{workflow_id}/run`
- Checking status: `workflows/{workflow_id}/get_status`
- Controlling workflow execution: `workflows/{workflow_id}/pause`, `workflows/{workflow_id}/resume`, `workflows/{workflow_id}/cancel`

### Connecting to the Workflow Server

The client connects to the workflow server using the `gen_client` function:

```python
async with gen_client("workflow_server", context.server_registry) as server:
    # Connect and use the server
```