#!/usr/bin/env python
"""
Enhanced Assignment Grader with Granular Progress Updates
This example shows how to use the Adaptive Workflow with detailed progress tracking.
"""

import asyncio
import os
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.adaptive.adaptive_workflow import AdaptiveWorkflow
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

console = Console()


class EnhancedProgressTracker:
    """Enhanced progress tracker with granular operation tracking"""

    def __init__(self):
        self.operations = []  # Stack of current operations
        self.completed_operations = []
        self.start_time = time.time()
        self.current_phase = "Initializing"
        self.subtasks = {}
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_used = 0
        self.cost = 0.0

    def start_operation(self, name: str, description: str = "") -> str:
        """Start a new operation and return its ID"""
        op_id = f"op_{len(self.operations)}_{int(time.time() * 1000)}"
        operation = {
            "id": op_id,
            "name": name,
            "description": description,
            "start_time": time.time(),
            "status": "running",
            "children": [],
        }

        # Add as child to current operation if exists
        if self.operations:
            self.operations[-1]["children"].append(operation)

        self.operations.append(operation)
        return op_id

    def complete_operation(self, op_id: str, result: str = "completed"):
        """Complete an operation"""
        # Find and complete the operation
        for i, op in enumerate(self.operations):
            if op["id"] == op_id:
                op["end_time"] = time.time()
                op["duration"] = op["end_time"] - op["start_time"]
                op["status"] = result
                self.completed_operations.append(op)
                self.operations.pop(i)
                break

    def add_llm_call(self, model: str, tokens: int = 0, cost: float = 0.0):
        """Track an LLM call"""
        self.llm_calls += 1
        self.tokens_used += tokens
        self.cost += cost

    def add_tool_call(self, tool_name: str):
        """Track a tool call"""
        self.tool_calls += 1

    def get_operation_tree(self) -> Tree:
        """Get current operations as a Rich tree"""
        tree = Tree("📊 Operations")

        # Add completed operations
        for op in self.completed_operations[-5:]:  # Show last 5 completed
            status = "✅" if op["status"] == "completed" else "❌"
            duration = f"({op['duration']:.1f}s)" if "duration" in op else ""
            tree.add(f"{status} {op['name']} {duration}")

        # Add current operations
        for op in self.operations:
            elapsed = time.time() - op["start_time"]
            tree.add(f"⏳ {op['name']} ({elapsed:.1f}s)")

        return tree

    def get_stats_table(self) -> Table:
        """Get statistics as a Rich table"""
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        elapsed = time.time() - self.start_time
        table.add_row("Elapsed Time", f"{elapsed:.1f}s")
        table.add_row("Current Phase", self.current_phase)
        table.add_row("LLM Calls", str(self.llm_calls))
        table.add_row("Tool Calls", str(self.tool_calls))
        table.add_row("Tokens Used", f"{self.tokens_used:,}")
        table.add_row("Cost", f"${self.cost:.4f}")

        return table


class AdaptiveWorkflowMonitor:
    """Monitor adaptive workflow with detailed progress updates"""

    def __init__(self, workflow: AdaptiveWorkflow, tracker: EnhancedProgressTracker):
        self.workflow = workflow
        self.tracker = tracker
        self._original_methods = {}
        self.console = Console()

    async def __aenter__(self):
        self._patch_methods()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._unpatch_methods()

    def _patch_methods(self):
        """Patch workflow methods to track detailed progress"""

        # Patch _analyze_objective
        original_analyze = self.workflow._analyze_objective

        async def patched_analyze(*args, **kwargs):
            op_id = self.tracker.start_operation(
                "Analyze Objective", "Understanding task requirements"
            )
            self.tracker.current_phase = "Analyzing"
            try:
                result = await original_analyze(*args, **kwargs)
                self.tracker.complete_operation(op_id, "completed")
                return result
            except Exception:
                self.tracker.complete_operation(op_id, "failed")
                raise

        self._original_methods["_analyze_objective"] = original_analyze
        self.workflow._analyze_objective = patched_analyze

        # Patch _needs_decomposition
        original_decomp = self.workflow._needs_decomposition

        async def patched_decomp(subtask, *args, **kwargs):
            op_id = self.tracker.start_operation(
                f"Check Complexity: {subtask.aspect.name}"
            )
            try:
                result = await original_decomp(subtask, *args, **kwargs)
                self.tracker.complete_operation(op_id)
                return result
            except Exception:
                self.tracker.complete_operation(op_id, "failed")
                raise

        self._original_methods["_needs_decomposition"] = original_decomp
        self.workflow._needs_decomposition = patched_decomp

        # Patch _plan_research
        original_plan = self.workflow._plan_research

        async def patched_plan(*args, **kwargs):
            op_id = self.tracker.start_operation(
                "Plan Research", "Breaking down into subtasks"
            )
            self.tracker.current_phase = "Planning"
            try:
                result = await original_plan(*args, **kwargs)
                if result:
                    for aspect in result:
                        self.tracker.subtasks[aspect.name] = "pending"
                self.tracker.complete_operation(op_id)
                return result
            except Exception:
                self.tracker.complete_operation(op_id, "failed")
                raise

        self._original_methods["_plan_research"] = original_plan
        self.workflow._plan_research = patched_plan

        # Patch _execute_single_subtask
        original_execute = self.workflow._execute_single_subtask

        async def patched_execute(subtask, *args, **kwargs):
            aspect_name = (
                subtask.aspect.name if hasattr(subtask, "aspect") else "Unknown"
            )
            op_id = self.tracker.start_operation(f"Execute: {aspect_name}")
            self.tracker.current_phase = f"Processing {aspect_name}"

            if aspect_name in self.tracker.subtasks:
                self.tracker.subtasks[aspect_name] = "running"

            try:
                result = await original_execute(subtask, *args, **kwargs)
                if aspect_name in self.tracker.subtasks:
                    self.tracker.subtasks[aspect_name] = "completed"
                self.tracker.complete_operation(op_id)
                return result
            except Exception:
                if aspect_name in self.tracker.subtasks:
                    self.tracker.subtasks[aspect_name] = "failed"
                self.tracker.complete_operation(op_id, "failed")
                raise

        self._original_methods["_execute_single_subtask"] = original_execute
        self.workflow._execute_single_subtask = patched_execute

        # Patch _synthesize_results
        original_synth = self.workflow._synthesize_results

        async def patched_synth(*args, **kwargs):
            op_id = self.tracker.start_operation(
                "Synthesize Results", "Combining findings"
            )
            self.tracker.current_phase = "Synthesizing"
            try:
                result = await original_synth(*args, **kwargs)
                self.tracker.complete_operation(op_id)
                return result
            except Exception:
                self.tracker.complete_operation(op_id, "failed")
                raise

        self._original_methods["_synthesize_results"] = original_synth
        self.workflow._synthesize_results = patched_synth

        # Patch _generate_final_report
        original_report = self.workflow._generate_final_report

        async def patched_report(*args, **kwargs):
            op_id = self.tracker.start_operation(
                "Generate Report", "Creating final document"
            )
            self.tracker.current_phase = "Reporting"
            try:
                result = await original_report(*args, **kwargs)
                self.tracker.complete_operation(op_id)
                return result
            except Exception:
                self.tracker.complete_operation(op_id, "failed")
                raise

        self._original_methods["_generate_final_report"] = original_report
        self.workflow._generate_final_report = patched_report

    def _unpatch_methods(self):
        """Restore original workflow methods"""
        for method_name, original_method in self._original_methods.items():
            setattr(self.workflow, method_name, original_method)


def create_display_layout() -> Layout:
    """Create the display layout"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    layout["body"].split_row(
        Layout(name="operations", ratio=2), Layout(name="stats", ratio=1)
    )

    return layout


def update_display(layout: Layout, tracker: EnhancedProgressTracker, title: str):
    """Update the display with current progress"""
    # Header
    layout["header"].update(Panel(f"📚 {title}", style="bold blue"))

    # Operations tree
    layout["operations"].update(
        Panel(tracker.get_operation_tree(), title="Operations", border_style="green")
    )

    # Stats
    layout["stats"].update(
        Panel(tracker.get_stats_table(), title="Statistics", border_style="cyan")
    )

    # Footer with subtasks
    if tracker.subtasks:
        subtask_lines = []
        for name, status in tracker.subtasks.items():
            icon = {
                "pending": "⬜",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
            }.get(status, "❓")
            subtask_lines.append(f"{icon} {name}")
        footer_content = "\n".join(subtask_lines)
    else:
        footer_content = "Initializing..."

    layout["footer"].update(
        Panel(footer_content, title="Subtasks", border_style="yellow")
    )


async def main():
    """Run the enhanced assignment grader"""
    app = MCPApp(name="enhanced_assignment_grader")

    async with app.run() as mcp_app:
        context = mcp_app.context

        # Configure filesystem server
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        # Create predefined agents
        console.print("\n[bold cyan]Creating specialized agents...[/bold cyan]")

        _file_handler = Agent(
            name="FileHandler",
            instruction="I manage file operations for the grading process.",
            server_names=["filesystem"],
        )

        _style_expert = Agent(
            name="APAStyleExpert",
            instruction="I analyze text for APA style compliance.",
            server_names=["fetch", "filesystem"],
        )

        _proofreader = Agent(
            name="Proofreader",
            instruction="I check for grammar, spelling, and clarity issues.",
            server_names=["fetch", "filesystem"],
        )

        # Define the grading task
        task = """Load the student's short story from short_story.md,
        and generate a report with feedback across proofreading,
        factuality/logical consistency and style adherence. Use the style rules from
        https://owl.purdue.edu/owl/research_and_citation/apa_style/apa_formatting_and_style_guide/general_format.html.
        Write the graded report to graded_report.md in the same directory as short_story.md"""

        # Create workflow with predefined agents
        workflow = AdaptiveWorkflow(
            name="AssignmentGrader",
            max_iterations=10,
            time_budget_minutes=5,
            cost_budget_dollars=3.0,
            llm_factory=OpenAIAugmentedLLM,
            # predefined_agents={
            #     "FileHandler": file_handler,
            #     "APAStyleExpert": style_expert,
            #     "Proofreader": proofreader,
            # },
        )

        # Create tracker and monitor
        tracker = EnhancedProgressTracker()

        # Create display layout
        layout = create_display_layout()

        # Run with live display
        with Live(layout, console=console, refresh_per_second=4) as _live:
            async with AdaptiveWorkflowMonitor(workflow, tracker) as _monitor:
                # Update display in background
                async def update_loop():
                    while True:
                        update_display(
                            layout, tracker, "Assignment Grader - Enhanced Progress"
                        )
                        await asyncio.sleep(0.25)

                # Start update loop
                update_task = asyncio.create_task(update_loop())

                try:
                    # Run the workflow
                    result = await workflow.generate_str(
                        message=task, request_params=RequestParams(model="gpt-4o")
                    )

                    # Final update
                    tracker.current_phase = "Completed"
                    update_display(layout, tracker, "Assignment Grader - Completed")

                finally:
                    update_task.cancel()

        # Display results
        console.print("\n[bold green]✨ Grading Complete![/bold green]\n")
        console.print(Panel(result, title="Grading Report", border_style="green"))

        # Display token usage summary
        if context.token_counter:
            summary = context.token_counter.get_summary()
            console.print(
                f"\n[bold]Total Tokens:[/bold] {summary.usage.total_tokens:,}"
            )
            console.print(f"[bold]Total Cost:[/bold] ${summary.cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
