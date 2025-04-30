# Temporal Workflow Example

This example demonstrates how to use [Temporal](https://temporal.io/) as the execution engine for MCP Agent workflows. Temporal is a microservice orchestration platform that helps developers build and operate reliable applications at scale.

## Overview

This example showcases:
- Defining a workflow using MCP Agent's workflow decorators
- Running the workflow using Temporal as the execution engine
- Setting up a Temporal worker to process workflow tasks

The sample workflow is simple - it takes a string input and returns the uppercase version of that string. While basic, it demonstrates the core concepts of using Temporal with MCP Agent.

## Prerequisites

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) package manager
- A running Temporal server (see setup instructions below)

## Setting Up Temporal Server

Before running this example, you need to have a Temporal server running. The easiest way to get started is using the Temporal CLI:

1. Install the Temporal CLI by following the instructions at: https://docs.temporal.io/cli/

2. Start a local Temporal server:
   ```bash
   temporal server start-dev
   ```

This will start a Temporal server on `localhost:7233` (the default address configured in `mcp_agent.config.yaml`).

You can also use the Temporal Web UI to monitor your workflows by visiting `http://localhost:8233` in your browser.

## Configuration

The example uses the configuration in `mcp_agent.config.yaml`, which includes:

- Temporal server address: `localhost:7233`
- Namespace: `default`
- Task queue: `mcp-agent`
- Maximum concurrent activities: 10

## Running the Example

To run this example, you'll need to:

1. Start the Temporal server (as described above)

2. In a separate terminal, start the worker:
   ```bash
   uv run run_worker.py
   ```
   
   The worker will register the workflow with Temporal and wait for tasks to execute.

3. In another terminal, run the main application:
   ```bash
   uv run main.py
   ```

   This will start a workflow execution that sends "Hello, World!" to the workflow, which will convert it to uppercase and return the result.

## Expected Output

When you run the main application, you should see output similar to:

```
INFO:__main__:Running SimpleWorkflow with input: Hello, World!
HELLO, WORLD!
```

## How It Works

### Workflow Definition

The workflow is defined in `main.py` using the `@app.workflow` and `@app.workflow_run` decorators:

```python
@app.workflow
class SimpleWorkflow(Workflow[str]):
    @app.workflow_run
    async def run(self, input_data: str) -> WorkflowResult[str]:
        logger.info(f"Running SimpleWorkflow with input: {input_data}")
        result = input_data.upper()
        return WorkflowResult(value=result)
```

### Worker Setup

The worker is set up in `run_worker.py` using the `TemporalExecutor`'s `start_worker` method:

```python
async def main():
    async with app.run() as running_app:
        executor: TemporalExecutor = running_app.executor
        await executor.start_worker()
```

### Workflow Execution

The workflow is executed in `main.py` by starting it with the executor and waiting for the result:

```python
async def main():
    async with app.run() as agent_app:
        executor: TemporalExecutor = agent_app.executor
        handle = await executor.start_workflow("SimpleWorkflow", "Hello, World!")
        a = await handle.result()
        print(a)
```

## Additional Resources

- [Temporal Documentation](https://docs.temporal.io/)
- [MCP Agent Documentation](https://github.com/modelcontextprotocol/mcp-agent)
- Other workflow examples in this repository:
  - `examples/workflow_parallel`
  - `examples/workflow_router`
  - `examples/workflow_swarm`
