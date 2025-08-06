"""
Example of using Temporal as the execution engine for MCP Agent workflows.
This example demonstrates how to create a workflow using the app.workflow and app.workflow_run
decorators, and how to run it using the Temporal executor.
"""

import asyncio

from mcp_agent.agents.agent import Agent
from mcp_agent.executor.temporal import TemporalExecutor
from mcp_agent.executor.workflow import Workflow, WorkflowResult
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from mcp_agent.tracing.token_counter import TokenNode, TokenSummary
from mcp_agent.core.context import Context

from main import app

SHORT_STORY = """
The Battle of Glimmerwood

In the heart of Glimmerwood, a mystical forest knowed for its radiant trees, a small village thrived. 
The villagers, who were live peacefully, shared their home with the forest's magical creatures, 
especially the Glimmerfoxes whose fur shimmer like moonlight.

One fateful evening, the peace was shaterred when the infamous Dark Marauders attack. 
Lead by the cunning Captain Thorn, the bandits aim to steal the precious Glimmerstones which was believed to grant immortality.

Amidst the choas, a young girl named Elara stood her ground, she rallied the villagers and devised a clever plan.
Using the forests natural defenses they lured the marauders into a trap. 
As the bandits aproached the village square, a herd of Glimmerfoxes emerged, blinding them with their dazzling light, 
the villagers seized the opportunity to captured the invaders.

Elara's bravery was celebrated and she was hailed as the "Guardian of Glimmerwood". 
The Glimmerstones were secured in a hidden grove protected by an ancient spell.

However, not all was as it seemed. The Glimmerstones true power was never confirm, 
and whispers of a hidden agenda linger among the villagers.
"""


@app.workflow
class ParallelWorkflow(Workflow[str]):
    """
    A simple workflow that demonstrates the basic structure of a Temporal workflow.
    """

    @app.workflow_run
    async def run(self, input: str) -> WorkflowResult[str]:
        """
        Run the workflow, processing the input data.

        Args:
            input_data: The data to process

        Returns:
            A WorkflowResult containing the processed data
        """

        proofreader = Agent(
            name="proofreader",
            instruction=""""Review the short story for grammar, spelling, and punctuation errors.
            Identify any awkward phrasing or structural issues that could improve clarity. 
            Provide detailed feedback on corrections.""",
        )

        fact_checker = Agent(
            name="fact_checker",
            instruction="""Verify the factual consistency within the story. Identify any contradictions,
            logical inconsistencies, or inaccuracies in the plot, character actions, or setting. 
            Highlight potential issues with reasoning or coherence.""",
        )

        style_enforcer = Agent(
            name="style_enforcer",
            instruction="""Analyze the story for adherence to style guidelines.
            Evaluate the narrative flow, clarity of expression, and tone. Suggest improvements to 
            enhance storytelling, readability, and engagement.""",
        )

        grader = Agent(
            name="grader",
            instruction="""Compile the feedback from the Proofreader, Fact Checker, and Style Enforcer
            into a structured report. Summarize key issues and categorize them by type. 
            Provide actionable recommendations for improving the story, 
            and give an overall grade based on the feedback.""",
        )

        parallel = ParallelLLM(
            fan_in_agent=grader,
            fan_out_agents=[proofreader, fact_checker, style_enforcer],
            llm_factory=OpenAIAugmentedLLM,
            context=app.context,
        )

        result = await parallel.generate_str(
            message=f"Student short story submission: {input}",
        )

        return WorkflowResult(value=result)


def display_node_tree(
    node: TokenNode, indent="", is_last=True, context: Context = None
):
    """Display a node and its children in a tree structure with token usage"""
    # Connector symbols
    connector = "└── " if is_last else "├── "

    # Get node usage
    usage = node.aggregate_usage()

    # Calculate cost if context available
    cost_str = ""
    if context and context.token_counter and node.usage.model_name:
        cost = context.token_counter.calculate_cost(
            node.usage.model_name,
            node.usage.input_tokens,
            node.usage.output_tokens,
            node.usage.model_info.provider if node.usage.model_info else None,
        )
        if cost > 0:
            cost_str = f" (${cost:.4f})"

    # Display node info
    print(f"{indent}{connector}{node.name} [{node.node_type}]")
    print(
        f"{indent}{'    ' if is_last else '│   '}├─ Total: {usage.total_tokens:,} tokens{cost_str}"
    )
    print(f"{indent}{'    ' if is_last else '│   '}├─ Input: {usage.input_tokens:,}")
    print(f"{indent}{'    ' if is_last else '│   '}└─ Output: {usage.output_tokens:,}")

    # If node has model info, show it
    if node.usage.model_name:
        model_str = node.usage.model_name
        if node.usage.model_info and node.usage.model_info.provider:
            model_str += f" ({node.usage.model_info.provider})"
        print(f"{indent}{'    ' if is_last else '│   '}   Model: {model_str}")

    # Process children
    if node.children:
        print(f"{indent}{'    ' if is_last else '│   '}")
        child_indent = indent + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            display_node_tree(child, child_indent, i == len(node.children) - 1, context)


async def display_token_summary(context: Context):
    """Display comprehensive token usage summary"""
    if not context.token_counter:
        print("\nNo token counter available")
        return

    summary: TokenSummary = await context.token_counter.get_summary()

    print("\n" + "=" * 60)
    print("TOKEN USAGE SUMMARY")
    print("=" * 60)

    # Display usage tree
    if summary.usage_tree:
        print("\nToken Usage Tree:")
        print("-" * 40)
        root_node = TokenNode(**summary.usage_tree)
        display_node_tree(root_node, context=context)

    # Total usage
    print("\nTotal Usage:")
    print(f"  Total tokens: {summary.usage.total_tokens:,}")
    print(f"  Input tokens: {summary.usage.input_tokens:,}")
    print(f"  Output tokens: {summary.usage.output_tokens:,}")
    print(f"  Total cost: ${summary.cost:.4f}")

    # Breakdown by model
    if summary.model_usage:
        print("\nBreakdown by Model:")
        for model_key, data in summary.model_usage.items():
            print(f"  {model_key}:")
            print(
                f"    Tokens: {data.usage.total_tokens:,} (input: {data.usage.input_tokens:,}, output: {data.usage.output_tokens:,})"
            )
            print(f"    Cost: ${data.cost:.4f}")

    print("\n" + "=" * 60)


async def main():
    async with app.run() as orchestrator_app:
        context = orchestrator_app.context
        executor: TemporalExecutor = orchestrator_app.executor

        handle = await executor.start_workflow(
            "ParallelWorkflow",
            SHORT_STORY,
        )
        result = await handle.result()
        print("\n=== WORKFLOW RESULT ===")
        print(result)

        # Display token usage summary
        await display_token_summary(context)


if __name__ == "__main__":
    import time

    start = time.time()
    asyncio.run(main())
    end = time.time()
    t = end - start

    print(f"\nTotal run time: {t:.2f}s")
