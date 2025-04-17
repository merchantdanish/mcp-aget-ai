"""
MCP Agent evaluation framework.

This package contains tools for evaluating MCP agents across different scenarios:
- metrics.py: Core evaluation metrics
- scenario_tasks.py: Task definitions for different scenarios
- runner.py: Agent evaluation runner
- visualize.py: Visualization utilities
- evaluate.py: Main evaluation script
- test_agent.py: Simple agent testing script
"""

from .test_agent import AgentEvaluator

__all__ = ["AgentEvaluator"]