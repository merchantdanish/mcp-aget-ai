"""
Example usage of the Adaptive Workflow
"""

import asyncio
from datetime import timedelta
from typing import List

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp.types import ModelPreferences
from mcp_agent.workflows.adaptive import AdaptiveWorkflow
from mcp_agent.workflows.llm.augmented_llm import RequestParams


async def main():
    """Example of using the Adaptive Workflow"""

    # Define the LLM factory that creates LLMs for agents
    def llm_factory(agent: Agent) -> OpenAIAugmentedLLM:
        return OpenAIAugmentedLLM(
            agent=agent,
            # Model will be selected based on task requirements
        )

    # Create the adaptive workflow
    workflow = AdaptiveWorkflow(
        llm_factory=llm_factory,
        name="ResearchWorkflow",
        available_servers=[
            "filesystem",
            "github",
            "slack",
            "google-drive",
        ],
        time_budget=timedelta(minutes=10),
        cost_budget=5.0,
        max_iterations=15,
        max_subagents=20,
        enable_parallel=True,
        enable_learning=True,
    )

    # Example 1: Research task - information gathering
    print("=== Example 1: Research Task ===")
    research_result = await workflow.generate_str(
        message="What are the key differences between transformer and mamba architectures in LLMs? Include recent benchmarks and practical trade-offs.",
    )
    print(f"Result:\n{research_result}\n")

    # Example 2: Action task - making changes
    print("=== Example 2: Action Task ===")
    action_result = await workflow.generate_str(
        message="Update our documentation to include a troubleshooting section for common MCP server connection issues",
    )
    print(f"Result:\n{action_result}\n")

    # Example 3: Hybrid task - research then act
    print("=== Example 3: Hybrid Task ===")
    hybrid_result = await workflow.generate_str(
        message="Analyze our codebase for potential security vulnerabilities in API endpoints, then create a plan to address the top 3 issues",
    )
    print(f"Result:\n{hybrid_result}\n")

    # Example 4: Using custom request parameters
    print("=== Example 4: With Custom Parameters ===")

    # For a task where we want faster responses over perfect accuracy
    fast_params = RequestParams(
        modelPreferences=ModelPreferences(
            speedPriority=0.8,
            intelligencePriority=0.1,
            costPriority=0.1,
        ),
        max_iterations=5,  # Fewer iterations for speed
        parallel_tool_calls=True,
    )

    fast_result = await workflow.generate_str(
        message="List all Python files in our project that import tensorflow",
        request_params=fast_params,
    )
    print(f"Fast Result:\n{fast_result}\n")

    # Example 5: Getting structured output
    print("=== Example 5: Structured Output ===")
    from pydantic import BaseModel, Field

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
    print(f"Structured Result:\n{audit_result.model_dump_json(indent=2)}\n")

    # Example 6: Demonstrating learning - run similar tasks
    print("=== Example 6: Learning from Previous Runs ===")

    # First run
    result1 = await workflow.generate_str(
        message="Find all TODO comments in our Python codebase and categorize them by priority",
    )

    # Second similar run - should be faster/more efficient due to learning
    result2 = await workflow.generate_str(
        message="Find all FIXME comments in our Python codebase and categorize them by module",
    )

    print("The second run should have been more efficient due to learned patterns\n")

    # Show metrics if available
    if hasattr(workflow, "_current_memory") and workflow._current_memory:
        memory = workflow._current_memory
        print("=== Workflow Metrics (Last Run) ===")
        print(f"Total iterations: {memory.iterations}")
        print(f"Subagents created: {memory.total_subagents_created}")
        print(f"Total cost: ${memory.total_cost:.2f}")
        print(
            f"Total tokens: {memory.total_input_tokens + memory.total_output_tokens:,}"
        )

        # Show task distribution
        print("\nTask Status Distribution:")
        completed = sum(1 for t in memory.completed_tasks if t.status == "completed")
        failed = sum(1 for t in memory.completed_tasks if t.status == "failed")
        print(f"  Completed: {completed}")
        print(f"  Failed: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
