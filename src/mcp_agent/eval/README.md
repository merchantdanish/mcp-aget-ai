# MCP Agent Evaluation Framework

This framework provides tools for evaluating MCP agents across different usage scenarios.

## Original Metrics Framework

The original evaluation framework measures agent performance using:

- **Progress Rate**: How effectively an agent makes progress through subgoals
- **Grounding Accuracy**: How accurately an agent uses tools and references facts
- **Task Completion**: Whether the agent successfully completes the task
- **Turn Efficiency**: How efficiently the agent completes tasks (fewer turns is better)

## Simple Agent Tester

A simplified agent tester is also available for quick evaluations:

```bash
# Run the test_agent.py script with the agent name
python -m src.mcp_agent.eval.test_agent --agent mcp_basic_slack_agent
```

This will:

1. Run a series of standard test queries against the agent
2. Measure response times and quality metrics
3. Store results in the agent's eval_results directory
4. Generate visualization data

## Output Structure

Results are stored in the agent's directory under `eval_results/`:

```
examples/mcp_basic_slack_agent/eval_results/
├── evaluation_20250416_123456.json     # Detailed evaluation results
└── visualizations/                     # Visualization data
    └── metrics_20250416_123456.json    # Metrics for visualization
```

## Framework Structure

- `evaluate.py`: Main script for comprehensive evaluations
- `metrics.py`: Core evaluation metrics definitions
- `runner.py`: Orchestrates the evaluation process
- `scenario_tasks.py`: Defines specific evaluation tasks for different scenarios
- `visualize.py`: Generates visualizations of evaluation results
- `test_agent.py`: Simple agent testing script
- `run_evaluations.py`: Script to run evaluations on multiple agents

## Extending the Framework

The evaluation framework can be extended with:

- Additional metrics and measurements
- Domain-specific test cases
- Integration with other evaluation tools
- Custom visualization generators

● To run the evaluation for each agent in the examples/ directory, you can
use the simulate_eval.py script I created. Here's how to do it:

1. Run evaluations for individual agents:

# Evaluate a specific agent

uv run python src/mcp_agent/eval/simulate_eval.py --agent <agent_name>

# Examples:

uv run python src/mcp_agent/eval/simulate_eval.py --agent mcp_basic_agent
uv run python src/mcp_agent/eval/simulate_eval.py --agent
mcp_basic_azure_agent
uv run python src/mcp_agent/eval/simulate_eval.py --agent mcp_researcher

2. To evaluate all agents, you can create a simple script:
   for agent in $(ls examples | grep mcp\_); do
   uv run python src/mcp_agent/eval/simulate_eval.py --agent $agent
   done

3. After running evaluations, generate a comparison report:
   uv run python src/mcp_agent/eval/eval_report.py

The evaluation results will be stored in each agent's directory under
eval*results/ folder, with a structure like:
examples/<agent_name>/eval_results/
├── evaluation*<timestamp>.json # Detailed evaluation results
└── visualizations/ # Visualization data
└── metrics\_<timestamp>.json # Metrics for visualization

The comparison report will show rankings of agents by success rate and
response time, helping you identify which agents perform best across the
test cases.

● For the MCP agent evaluation framework I created, here are the evaluative
criteria being measured:

1. Response Time:


    - Measures how long the agent takes to respond to queries
    - Lower is better
    - Captured in seconds for each test case

2. Success Rate:


    - Measures if responses contain expected keywords/content
    - Higher is better
    - Calculated as percentage of test cases that match expected criteria

3. Response Length:


    - Character count of responses
    - Helps analyze verbosity and comprehensiveness

4. Tool Usage Count:


    - Number of times the agent uses tools to complete tasks
    - Helps evaluate efficiency in tool utilization
