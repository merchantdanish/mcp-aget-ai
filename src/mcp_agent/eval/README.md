# Trace Evaluation Guide

This guide explains how to evaluate traces in the `trace/` directory using mcp-agent's evaluation tools.

## Prerequisites

- Traces should be in JSONL format
- Python environment with mcp-agent installed

## Basic Usage

```bash
# Run basic trace analysis on a trace file
python -m mcp_agent.eval.eval_runner analyze-trace trace/your_trace_file.jsonl

# Analyze agent components in a trace
python -m mcp_agent.eval.eval_runner analyze-agent trace/your_trace_file.jsonl

# Generate tool metrics report
python -m mcp_agent.eval.eval_runner tool-metrics trace/your_trace_file.jsonl

# Run workflow evaluation
python -m mcp_agent.eval.eval_runner eval-workflow trace/your_trace_file.jsonl
```

## Output Formats

The analysis tools generate structured reports that include:

- Trace timeline and structure
- Agent component interactions
- LLM operations and costs
- Tool usage metrics (frequency, duration, success rates)
- Workflow step evaluation

## Advanced Usage

For customized analysis, you can use the underlying components directly:

```python
from mcp_agent.eval.trace_analyzer import TraceAnalyzer
from mcp_agent.eval.agent_trace_analysis import AgentTraceAnalysis
from mcp_agent.eval.tool_metrics import ToolMetricsAnalyzer

# Load and analyze a trace file
analyzer = TraceAnalyzer("trace/your_trace_file.jsonl")
analysis = analyzer.analyze()

# Generate agent component analysis
agent_analysis = AgentTraceAnalysis(analysis)
agent_report = agent_analysis.analyze()

# Generate tool metrics
tool_analyzer = ToolMetricsAnalyzer(analysis)
tool_metrics = tool_analyzer.analyze()
```

For more information, refer to the module documentation in the `mcp_agent.eval` package.