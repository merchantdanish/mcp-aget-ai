"""
Adaptive Workflow Example

This example demonstrates the Adaptive Workflow, which implements
a multi-agent system based on the Claude Deep Research architecture.
"""

import asyncio
import os
from datetime import timedelta

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.adaptive import AdaptiveWorkflow
from mcp.types import ModelPreferences
from rich import print


async def main():
    """Run adaptive workflow examples"""
    async with MCPApp("config/mcp_agent.config.yaml") as app:
        # Define the LLM factory
        def llm_factory(agent: Agent) -> OpenAIAugmentedLLM:
            return OpenAIAugmentedLLM(
                agent=agent,
                context=app.context,
            )

        # Create the adaptive workflow
        workflow = AdaptiveWorkflow(
            llm_factory=llm_factory,
            name="AdaptiveResearchWorkflow",
            available_servers=app.context.servers.keys(),  # Use all configured servers
            time_budget=timedelta(minutes=5),
            cost_budget=2.0,
            max_iterations=10,
            max_subagents=15,
            enable_parallel=True,
            enable_learning=True,
            context=app.context,
        )

        # Example 1: Research task
        print("\n[bold cyan]Example 1: Research Task[/bold cyan]")
        print("Researching multi-agent architectures...\n")

        research_result = await workflow.generate_str(
            message="What are the key architectural patterns in multi-agent systems? Focus on coordination mechanisms and communication protocols."
        )

        print(f"[green]Result:[/green]\n{research_result}\n")

        # Show metrics
        if workflow._current_memory:
            memory = workflow._current_memory
            print("\n[bold]Metrics:[/bold]")
            print(f"- Task type: {memory.task_type}")
            print(f"- Iterations: {memory.iterations}")
            print(f"- Subagents created: {memory.total_subagents_created}")
            print(f"- Total cost: ${memory.total_cost:.3f}")
            print(
                f"- Total tokens: {memory.total_input_tokens + memory.total_output_tokens:,}"
            )

            # Show task breakdown
            completed = sum(
                1 for t in memory.completed_tasks if t.status == "completed"
            )
            failed = sum(1 for t in memory.completed_tasks if t.status == "failed")
            print(f"- Tasks: {completed} completed, {failed} failed")

        # Example 2: Action task
        print("\n[bold cyan]Example 2: Action Task[/bold cyan]")
        print("Creating a configuration file...\n")

        action_result = await workflow.generate_str(
            message="Create a sample YAML configuration file for a multi-agent system with 3 agents: researcher, analyzer, and reporter. Include their roles and capabilities."
        )

        print(f"[green]Result:[/green]\n{action_result}\n")

        # Example 3: Hybrid task
        print("\n[bold cyan]Example 3: Hybrid Task[/bold cyan]")
        print("Research then action task...\n")

        hybrid_result = await workflow.generate_str(
            message="Analyze our codebase for potential security vulnerabilities in API endpoints, then create a plan to address the top 3 issues"
        )

        print(f"[green]Result:[/green]\n{hybrid_result}\n")

        # Example 4: Fast mode with custom parameters
        print("\n[bold cyan]Example 4: Fast Mode with Custom Parameters[/bold cyan]")
        print("Quick query with speed prioritized...\n")

        fast_params = RequestParams(
            modelPreferences=ModelPreferences(
                speedPriority=0.8,
                intelligencePriority=0.1,
                costPriority=0.1,
            ),
            max_iterations=3,
        )

        fast_result = await workflow.generate_str(
            message="List the main components of the MCP (Model Context Protocol)",
            request_params=fast_params,
        )

        print(f"[green]Result:[/green]\n{fast_result}\n")

        # Example 5: Structured output
        print("\n[bold cyan]Example 5: Structured Output[/bold cyan]")
        print("Getting structured results...\n")

        from pydantic import BaseModel, Field
        from typing import List

        class SecurityAudit(BaseModel):
            vulnerabilities: List[str] = Field(
                description="List of identified vulnerabilities"
            )
            severity_levels: List[str] = Field(
                description="Severity for each vulnerability"
            )
            recommendations: List[str] = Field(description="Recommended fixes")
            estimated_effort: str = Field(description="Overall effort estimate")

        audit_result = await workflow.generate_structured(
            message="Audit our authentication system for security issues",
            response_model=SecurityAudit,
        )

        print(
            f"[green]Structured Result:[/green]\n{audit_result.model_dump_json(indent=2)}\n"
        )


if __name__ == "__main__":
    # Change to the example directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    asyncio.run(main())
