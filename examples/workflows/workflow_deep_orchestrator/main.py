#!/usr/bin/env python
"""
Deep Orchestrator Example - Assignment Grader with Full State Visibility

This example demonstrates the Deep Orchestrator (AdaptiveOrchestrator) with:
- Dynamic agent creation and caching
- Knowledge extraction and accumulation
- Budget tracking (tokens, cost, time)
- Task queue management with dependencies
- Policy-driven execution control
- Full state visibility throughout execution
"""

import asyncio
import os
import time
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns
from rich import box

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.deep_orchestrator.orchestrator import AdaptiveOrchestrator
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

console = Console()


class DeepOrchestratorMonitor:
    """Monitor to expose all internal state of the Deep Orchestrator"""

    def __init__(self, orchestrator: AdaptiveOrchestrator):
        self.orchestrator = orchestrator
        self.start_time = time.time()

    def get_budget_table(self) -> Table:
        """Get budget status as a table"""
        budget = self.orchestrator.budget
        usage = budget.get_usage_pct()
        budget.get_remaining()

        table = Table(title="💰 Budget", box=box.ROUNDED, show_header=True)
        table.add_column("Resource", style="cyan")
        table.add_column("Used", style="yellow")
        table.add_column("Limit", style="green")
        table.add_column("Usage %", style="magenta")

        # Tokens
        table.add_row(
            "Tokens",
            f"{budget.tokens_used:,}",
            f"{budget.max_tokens:,}",
            f"{usage['tokens']:.1%}",
        )

        # Cost
        table.add_row(
            "Cost",
            f"${budget.cost_incurred:.3f}",
            f"${budget.max_cost:.2f}",
            f"{usage['cost']:.1%}",
        )

        # Time
        elapsed = datetime.now(budget.start_time.tzinfo) - budget.start_time
        elapsed_minutes = elapsed.total_seconds() / 60
        table.add_row(
            "Time",
            f"{elapsed_minutes:.1f} min",
            f"{budget.max_time_minutes} min",
            f"{usage['time']:.1%}",
        )

        return table

    def get_queue_tree(self) -> Tree:
        """Get task queue as a tree"""
        queue = self.orchestrator.queue
        tree = Tree("📋 Task Queue")

        # Completed steps
        if queue.completed_steps:
            completed = tree.add("[green]✅ Completed Steps")
            for step in queue.completed_steps[-3:]:  # Last 3
                step_node = completed.add(f"[dim]{step.description}")
                for task in step.tasks[:2]:  # First 2 tasks
                    status_icon = "✓" if task.status == "completed" else "✗"
                    step_node.add(f"[dim]{status_icon} {task.description[:50]}...")

        # Pending steps
        if queue.pending_steps:
            pending = tree.add("[yellow]⏳ Pending Steps")
            for step in queue.pending_steps[:3]:  # Next 3
                step_node = pending.add(step.description)
                for task in step.tasks[:2]:  # First 2 tasks
                    step_node.add(f"• {task.description[:50]}...")

        # Queue summary
        tree.add(f"[blue]📊 {queue.get_progress_summary()}")

        return tree

    def get_memory_panel(self) -> Panel:
        """Get memory status as a panel"""
        memory = self.orchestrator.memory
        stats = memory.get_stats()

        lines = [
            f"[cyan]Artifacts:[/cyan] {stats['artifacts']}",
            f"[cyan]Knowledge Items:[/cyan] {stats['knowledge_items']}",
            f"[cyan]Task Results:[/cyan] {stats['task_results']}",
            f"[cyan]Categories:[/cyan] {stats['knowledge_categories']}",
            f"[cyan]Est. Tokens:[/cyan] {stats['estimated_tokens']:,}",
        ]

        # Add recent knowledge items
        if memory.knowledge:
            lines.append("\n[yellow]Recent Knowledge:[/yellow]")
            for item in memory.knowledge[-3:]:
                lines.append(f"  • {item.key[:40]}: {str(item.value)[:40]}...")

        content = "\n".join(lines)
        return Panel(content, title="🧠 Memory", border_style="blue")

    def get_agents_table(self) -> Table:
        """Get agent cache status"""
        cache = self.orchestrator.agent_cache

        table = Table(title="🤖 Agent Cache", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Cached Agents", str(len(cache.cache)))
        table.add_row("Cache Hits", str(cache.hits))
        table.add_row("Cache Misses", str(cache.misses))

        if cache.hits + cache.misses > 0:
            hit_rate = cache.hits / (cache.hits + cache.misses)
            table.add_row("Hit Rate", f"{hit_rate:.1%}")

        # Show cached agent names
        if cache.cache:
            agent_names = []
            for key, agent in list(cache.cache.items())[:3]:
                agent_names.append(agent.name)
            if agent_names:
                table.add_row("Recent", ", ".join(agent_names))

        return table

    def get_policy_panel(self) -> Panel:
        """Get policy engine status"""
        policy = self.orchestrator.policy

        lines = [
            f"[cyan]Consecutive Failures:[/cyan] {policy.consecutive_failures}/{policy.max_consecutive_failures}",
            f"[cyan]Total Successes:[/cyan] {policy.total_successes}",
            f"[cyan]Total Failures:[/cyan] {policy.total_failures}",
            f"[cyan]Failure Rate:[/cyan] {policy.get_failure_rate():.1%}",
        ]

        return Panel("\n".join(lines), title="⚙️ Policy Engine", border_style="yellow")

    def get_status_summary(self) -> Panel:
        """Get overall status summary"""
        elapsed = time.time() - self.start_time

        lines = [
            f"[cyan]Objective:[/cyan] {self.orchestrator.objective[:100]}...",
            f"[cyan]Iteration:[/cyan] {self.orchestrator.iteration}/{self.orchestrator.max_iterations}",
            f"[cyan]Replans:[/cyan] {self.orchestrator.replan_count}/{self.orchestrator.max_replans}",
            f"[cyan]Elapsed:[/cyan] {elapsed:.1f}s",
        ]

        return Panel("\n".join(lines), title="📊 Status", border_style="green")


def create_display_layout() -> Layout:
    """Create the display layout"""
    layout = Layout()

    # Main structure
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=8),
    )

    # Body section - main monitoring
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="center", ratio=1),
        Layout(name="right", ratio=1),
    )

    # Footer section - detailed state
    layout["footer"].split_row(
        Layout(name="queue", ratio=2),
        Layout(name="memory", ratio=1),
    )

    return layout


def update_display(layout: Layout, monitor: DeepOrchestratorMonitor):
    """Update the display with current state"""

    # Header
    layout["header"].update(
        Panel("🚀 Deep Orchestrator - Assignment Grader", style="bold blue")
    )

    # Left column - Budget
    layout["left"].update(monitor.get_budget_table())

    # Center column - Status and Policy
    center_content = Columns(
        [
            monitor.get_status_summary(),
            monitor.get_policy_panel(),
        ]
    )
    layout["center"].update(center_content)

    # Right column - Agents
    layout["right"].update(monitor.get_agents_table())

    # Footer left - Queue
    layout["queue"].update(monitor.get_queue_tree())

    # Footer right - Memory
    layout["memory"].update(monitor.get_memory_panel())


async def main():
    """Run the Deep Orchestrator example"""

    # Initialize MCP App
    app = MCPApp(name="deep_orchestrator_example")

    async with app.run() as mcp_app:
        context = mcp_app.context
        logger = mcp_app.logger

        # Configure filesystem server with current directory
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        console.print("\n[bold cyan]🚀 Deep Orchestrator Example[/bold cyan]")
        console.print(
            "This demonstrates all the advanced features with full state visibility\n"
        )

        # Create some predefined agents (optional - orchestrator can create its own)
        predefined_agents = {
            "FileExpert": Agent(
                name="FileExpert",
                instruction="""I specialize in file operations and content management.
                I can read, write, and analyze files efficiently.""",
                server_names=["filesystem"],
                context=context,
            ),
            "StyleChecker": Agent(
                name="StyleChecker",
                instruction="""I am an expert in writing style and formatting standards.
                I check for APA compliance and provide detailed feedback.""",
                server_names=["fetch"],
                context=context,
            ),
            "Proofreader": Agent(
                name="Proofreader",
                instruction="""I specialize in grammar, spelling, and clarity.
                I provide detailed corrections and suggestions.""",
                server_names=["filesystem"],
                context=context,
            ),
        }

        # Create the Deep Orchestrator with all features enabled
        orchestrator = AdaptiveOrchestrator(
            llm_factory=OpenAIAugmentedLLM,
            name="DeepAssignmentGrader",
            available_agents=predefined_agents,
            available_servers=list(context.servers.keys())
            if hasattr(context, "servers")
            else ["filesystem", "fetch"],
            max_iterations=15,
            max_replans=2,
            enable_filesystem=True,  # Enable workspace
            enable_parallel=True,  # Enable parallel execution
            max_task_retries=2,  # Retry failed tasks
            context=context,
        )

        # Configure budget limits
        orchestrator.budget.max_tokens = 50000
        orchestrator.budget.max_cost = 5.0
        orchestrator.budget.max_time_minutes = 10

        # Create monitor for state visibility
        monitor = DeepOrchestratorMonitor(orchestrator)

        # Create display layout
        layout = create_display_layout()

        # Define the complex grading task
        task = """
        Analyze the student's short story from short_story.md and create a comprehensive grading report.
        
        The report should include:
        1. Grammar and spelling check with specific corrections
        2. Style analysis against APA guidelines (fetch from https://owl.purdue.edu/owl/research_and_citation/apa_style/apa_formatting_and_style_guide/general_format.html)
        3. Story structure and narrative flow assessment
        4. Factual consistency and logical coherence check
        5. Overall grade with detailed justification
        
        Save the complete grading report to graded_report.md in the same directory.
        
        Use a systematic approach: first understand the story, then analyze each aspect in detail,
        and finally synthesize all findings into a comprehensive report.
        """

        # Run with live display
        console.print("[yellow]Starting Deep Orchestrator workflow...[/yellow]\n")

        with Live(layout, console=console, refresh_per_second=2):
            # Update display in background
            async def update_loop():
                while True:
                    try:
                        update_display(layout, monitor)
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        logger.error(f"Display update error: {e}")
                        break

            # Start update loop
            update_task = asyncio.create_task(update_loop())

            try:
                # Run the orchestrator
                start_time = time.time()

                result = await orchestrator.generate_str(
                    message=task,
                    request_params=RequestParams(
                        model="gpt-4o", temperature=0.7, max_iterations=10
                    ),
                )

                execution_time = time.time() - start_time

                # Final update
                update_display(layout, monitor)

            finally:
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass

        # Display results
        console.print("\n[bold green]✨ Grading Complete![/bold green]\n")

        # Show the grading report
        console.print(
            Panel(
                result[:2000] + "..." if len(result) > 2000 else result,
                title="📝 Grading Report (Preview)",
                border_style="green",
            )
        )

        # Display final statistics
        console.print("\n[bold cyan]📊 Final Statistics[/bold cyan]\n")

        # Create summary table
        summary_table = Table(title="Execution Summary", box=box.DOUBLE_EDGE)
        summary_table.add_column("Metric", style="cyan", width=20)
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Time", f"{execution_time:.2f}s")
        summary_table.add_row("Iterations", str(orchestrator.iteration))
        summary_table.add_row("Replans", str(orchestrator.replan_count))
        summary_table.add_row(
            "Tasks Completed", str(len(orchestrator.queue.completed_task_ids))
        )
        summary_table.add_row(
            "Tasks Failed", str(len(orchestrator.queue.failed_task_ids))
        )
        summary_table.add_row(
            "Knowledge Items", str(len(orchestrator.memory.knowledge))
        )
        summary_table.add_row(
            "Artifacts Created", str(len(orchestrator.memory.artifacts))
        )
        summary_table.add_row("Agents Cached", str(len(orchestrator.agent_cache.cache)))
        summary_table.add_row(
            "Cache Hit Rate",
            f"{orchestrator.agent_cache.hits / max(1, orchestrator.agent_cache.hits + orchestrator.agent_cache.misses):.1%}",
        )

        console.print(summary_table)

        # Display budget summary
        budget_summary = orchestrator.budget.get_status_summary()
        console.print(f"\n[yellow]{budget_summary}[/yellow]")

        # Display knowledge learned
        if orchestrator.memory.knowledge:
            console.print("\n[bold cyan]🧠 Knowledge Extracted[/bold cyan]\n")

            knowledge_table = Table(box=box.SIMPLE)
            knowledge_table.add_column("Category", style="cyan")
            knowledge_table.add_column("Key", style="yellow")
            knowledge_table.add_column("Value", style="green", max_width=50)
            knowledge_table.add_column("Confidence", style="magenta")

            for item in orchestrator.memory.knowledge[:10]:  # Show first 10
                knowledge_table.add_row(
                    item.category,
                    item.key[:30] + "..." if len(item.key) > 30 else item.key,
                    str(item.value)[:50] + "..."
                    if len(str(item.value)) > 50
                    else str(item.value),
                    f"{item.confidence:.2f}",
                )

            console.print(knowledge_table)

        # Display token usage if available
        if context.token_counter:
            summary = context.token_counter.get_summary()
            console.print(
                f"\n[bold]Total Tokens:[/bold] {summary.usage.total_tokens:,}"
            )
            console.print(f"[bold]Total Cost:[/bold] ${summary.cost:.4f}")

        # Show workspace artifacts if any were created
        if orchestrator.memory.artifacts:
            console.print("\n[bold cyan]📁 Artifacts Created[/bold cyan]")
            for name in list(orchestrator.memory.artifacts.keys())[:5]:
                console.print(f"  • {name}")


if __name__ == "__main__":
    # Change to example directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create sample story file if it doesn't exist
    if not os.path.exists("short_story.md"):
        with open("short_story.md", "w") as f:
            f.write("""# The Enchanted Garden

Sarah pushed through the overgrown hedge and gasped. Before her lay a garden unlike any she had ever seen. Flowers glowed with an ethereal light, their petals shifting through colors that shouldn't exist in nature.

"You shouldn't be here," a voice said softly.

She spun around to find a boy about her age, with eyes that seemed to hold the same impossible colors as the flowers. He wasn't threatening, just... sad.

"I'm sorry, I was just exploring and—"

"No, it's okay," he interrupted. "I'm just surprised anyone could find this place. The garden usually keeps people out."

"The garden... keeps people out?" Sarah asked, her scientific mind immediately curious.

The boy nodded. "It's been in my family for generations. My grandmother says it chooses who can enter." He gestured to a particularly vibrant cluster of roses that seemed to pulse with inner light. "These flowers, they're not exactly... normal."

Sarah stepped closer to examine them. As she did, the roses seemed to lean toward her, as if greeting an old friend.

"They like you," the boy said, wonder in his voice. "They never respond to anyone except—" He stopped abruptly.

"Except who?"

"My grandmother. And me. We're the gardeners." He looked at her intently. "What's your name?"

"Sarah. Sarah Chen."

The boy's eyes widened. "Chen? Your grandmother wouldn't happen to be Rose Chen, would she?"

Sarah nodded, confused. "Yes, but she passed away last year. How did you—"

"She was my grandmother's best friend. They founded this garden together, decades ago, before they had a falling out." He extended his hand. "I'm Alex Morrison. And I think our grandmothers have been waiting for us to meet."

As their hands touched, every flower in the garden burst into brilliant bloom, filling the air with a harmony of scents and colors that spoke of old magic and new beginnings.
""")
        console.print("[yellow]Created sample short_story.md file[/yellow]")

    # Run the example
    asyncio.run(main())
