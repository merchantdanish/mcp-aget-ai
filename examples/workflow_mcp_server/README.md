# Workflow MCP Server Example

This example demonstrates three approaches to creating agents and workflows:

1. Traditional workflow-based approach with manual agent creation
2. Programmatic agent configuration using AgentConfig
3. Declarative agent configuration using FastMCPApp decorators

All three approaches can use `app_server.py` to expose the agents and workflows as an MCP server.

## Concepts Demonstrated

- Using the `Workflow` base class to create custom workflows
- Registering workflows with an `MCPApp`
- Creating and registering agent configurations with both programmatic and declarative approaches
- Exposing workflows and agents as MCP tools using `app_server.py`
- Connecting to a workflow server using `gen_client`
- Lazy instantiation of agents from configurations when their tools are called

## Components in this Example

1. **DataProcessorWorkflow**: A traditional workflow that processes data in three steps:

   - Finding and retrieving content from a source (file or URL)
   - Analyzing the content
   - Formatting the results

2. **SummarizationWorkflow**: A traditional workflow that summarizes text content:

   - Generates a concise summary
   - Extracts key points
   - Returns structured data

3. **Research Team**: A parallel workflow created using the agent configuration system:

   - Uses a fan-in/fan-out pattern with multiple specialized agents
   - Demonstrates declarative workflow pattern configuration

4. **Specialist Router**: A router workflow created using FastMCPApp decorators:
   - Routes requests to specialized agents based on content
   - Shows how to use the decorator syntax for workflow creation

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

### Approach 1: Traditional Workflow Definition

Workflows are defined by subclassing the `Workflow` base class and implementing:

- The `run` method containing the main workflow logic
- `initialize` and `cleanup` methods for setup and teardown
- Optionally a custom `create` class method for specialized instantiation

```python
class DataProcessorWorkflow(Workflow[str]):
    @classmethod
    async def create(cls, executor: Executor, name: str | None = None, **kwargs: Any) -> "DataProcessorWorkflow":
        # Custom instantiation logic
        workflow = cls(executor=executor, name=name, **kwargs)
        await workflow.initialize()
        return workflow

    async def initialize(self):
        # Set up resources like agents and LLMs

    async def run(self, source: str, analysis_prompt: Optional[str] = None, output_format: Optional[str] = None) -> WorkflowResult[str]:
        # Workflow implementation...

    async def cleanup(self):
        # Clean up resources
```

The base `Workflow` class provides a default implementation of `create()` that handles basic initialization, but workflows can override this for specialized setup. Our example shows both approaches:

1. `DataProcessorWorkflow` overrides the `create()` method to implement custom initialization
2. `SummarizationWorkflow` uses the default implementation from the base class

Workflows are registered with the MCPApp using the `@app.workflow` decorator:

```python
app = MCPApp(name="workflow_mcp_server")

@app.workflow
class DataProcessorWorkflowRegistered(DataProcessorWorkflow):
    pass
```

### Approach 2: Programmatic Agent Configuration

Agent configurations can be created programmatically using Pydantic models:

```python
# Create a basic agent configuration
research_agent_config = AgentConfig(
    name="researcher",
    instruction="You are a helpful research assistant that finds information and presents it clearly.",
    server_names=["fetch", "filesystem"],
    llm_config=AugmentedLLMConfig(
        factory=OpenAIAugmentedLLM,
        model="gpt-4o",
        temperature=0.7
    )
)

# Create a parallel workflow configuration
research_team_config = AgentConfig(
    name="research_team",
    instruction="You are a research team that produces high-quality, accurate content.",
    parallel_config=ParallelWorkflowConfig(
        fan_in_agent="editor",
        fan_out_agents=["summarizer", "fact_checker"],
        concurrent=True
    )
)

# Register the configurations with the app
app.register_agent_config(research_agent_config)
app.register_agent_config(research_team_config)
```

### Approach 3: Declarative Agent Configuration with FastMCPApp

FastMCPApp provides decorators for creating agent configurations in a more declarative style:

```python
fast_app = FastMCPApp(name="fast_workflow_mcp_server")

# Basic agent with OpenAI LLM
@fast_app.agent("assistant", "You are a helpful assistant that answers questions concisely.",
              server_names=["calculator"])
def assistant_config(config):
    config.llm_config = AugmentedLLMConfig(
        factory=OpenAIAugmentedLLM,
        model="gpt-4o",
        temperature=0.7
    )
    return config

# Router workflow with specialist agents
@fast_app.router("specialist_router", "You route requests to the appropriate specialist.",
               agent_names=["mathematician", "programmer", "writer"])
def router_config(config):
    config.llm_config = AugmentedLLMConfig(
        factory=OpenAIAugmentedLLM,
        model="gpt-4o"
    )
    config.router_config.top_k = 1
    return config
```

### Exposing Workflows and Agents as Tools

The MCP server automatically exposes both workflows and agent configurations as tools:

**Workflow tools**:

- Running a workflow: `workflows/{workflow_id}/run`
- Checking status: `workflows/{workflow_id}/get_status`
- Controlling workflow execution: `workflows/{workflow_id}/pause`, `workflows/{workflow_id}/resume`, `workflows/{workflow_id}/cancel`

**Agent tools**:

- Running an agent: `agents/{agent_name}/generate`
- Getting string response: `agents/{agent_name}/generate_str`
- Getting structured response: `agents/{agent_name}/generate_structured`

Agent configurations are lazily instantiated when their tools are called. If the agent is already active, the existing instance is reused.

### Connecting to the Workflow Server

The client connects to the workflow server using the `gen_client` function:

```python
async with gen_client("workflow_server", context.server_registry) as server:
    # Connect and use the server
```

You can then call both workflow and agent tools through this client connection.
