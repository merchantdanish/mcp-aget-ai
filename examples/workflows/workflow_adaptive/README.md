# Adaptive Workflow Example

This example demonstrates the Adaptive Workflow, which implements a multi-agent system based on the Claude Deep Research architecture.

## Overview

The Adaptive Workflow dynamically creates and coordinates specialized agents to accomplish complex objectives. Unlike the Orchestrator workflow which uses a fixed pattern, the Adaptive Workflow:

- Analyzes tasks to determine the appropriate approach
- Creates specialized subagents on-demand with focused instructions
- Executes tasks in parallel when possible
- Tracks costs and time budgets to prevent runaway execution
- Learns from past executions to improve future performance

## Key Features

1. **Dynamic Agent Creation**: Instead of pre-defined agents, creates specialized subagents based on task requirements
2. **Non-cascading Limits**: Each subagent has its own iteration limit that doesn't cascade, preventing exponential growth
3. **Resource Budgets**: Time and cost budgets ensure controlled execution
4. **Learning System**: Tracks successful patterns to optimize future runs
5. **Parallel Execution**: Independent subtasks run simultaneously for efficiency

## Running the Example

1. Copy the configuration files:
```bash
cp mcp_agent.config.yaml config/
cp mcp_agent.secrets.yaml.example config/mcp_agent.secrets.yaml
```

2. Edit `config/mcp_agent.secrets.yaml` with your API keys:
```yaml
openai:
  api_key: "your-openai-api-key"
```

3. Run the example:
```bash
python main.py
```

## Configuration

The workflow can be configured with:

- `time_budget`: Maximum execution time (default: 5 minutes)
- `cost_budget`: Maximum cost in dollars (default: $2.00)
- `max_iterations`: Maximum iterations for the lead agent (default: 10)
- `max_subagents`: Maximum total subagents to create (default: 15)
- `enable_parallel`: Whether to run tasks in parallel (default: true)
- `enable_learning`: Whether to learn from executions (default: true)

## Examples Included

1. **Research Task**: Gathers information about multi-agent architectures
2. **Action Task**: Creates a configuration file based on requirements
3. **Fast Mode**: Demonstrates custom parameters for speed-optimized queries

## Metrics

The workflow tracks:
- Task type (research, action, or hybrid)
- Number of iterations
- Subagents created
- Token usage and costs
- Task success/failure rates

## Comparison with Orchestrator

| Feature | Orchestrator | Adaptive Workflow |
|---------|--------------|-------------------|
| Agent Creation | Pre-defined agents | Dynamic, on-demand |
| Iteration Limits | Cascading (can explode) | Non-cascading per agent |
| Strategy | Fixed planning modes | Adaptive based on task |
| Resource Control | Basic | Time & cost budgets |
| Learning | No | Yes, tracks patterns |
| Parallelism | Limited | Full parallel execution |

## Advanced Usage

You can customize the workflow behavior by:

1. **Model Preferences**: Adjust the balance between speed, cost, and intelligence
2. **Custom Strategies**: Override the default strategy selection
3. **Memory Persistence**: Enable filesystem-based memory for long-running tasks
4. **Server Selection**: Limit which MCP servers are available to subagents

See `main.py` for examples of these customizations.