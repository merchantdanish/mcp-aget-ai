"""
Main entry point for Reliable Conversation Manager.
Implements REPL with conversation-as-workflow pattern.
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_agent.app import MCPApp
from workflows.conversation_workflow import ConversationWorkflow
from models.conversation_models import ConversationConfig, ConversationState
from utils.logging import get_rcm_logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Create app instance
app = MCPApp(name="reliable_conversation_manager")

# No task registration needed - we import functions directly in workflows

# Register the workflow with the app
from workflows.conversation_workflow import ConversationWorkflow

@app.workflow
class RegisteredConversationWorkflow(ConversationWorkflow):
    """Workflow registered with app"""
    pass

async def run_repl():
    """Run the RCM REPL interface"""

    async with app.run() as rcm_app:
        logger = get_rcm_logger("main")
        
        # Add current directory to filesystem server
        rcm_app.context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
        
        # Display welcome message
        console.print(Panel.fit(
            "[bold blue]Reliable Conversation Manager[/bold blue]\n\n"
            "Multi-turn chat with quality control based on 'LLMs Get Lost' research\n"
            f"Execution Engine: {rcm_app.context.config.execution_engine}\n"
            f"Phase: 1 (Basic Implementation)\n\n"
            "Commands: /stats, /requirements, /exit",
            border_style="blue"
        ))

        # Create workflow instance
        workflow = RegisteredConversationWorkflow(app)
        conversation_state = None

        logger.info("RCM REPL started")

        while True:
            # Get user input
            try:
                user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
            except (EOFError, KeyboardInterrupt):
                break

            # Handle commands
            if user_input.lower() == "/exit":
                break
            elif user_input.lower() == "/stats":
                _display_stats(conversation_state)
                continue
            elif user_input.lower() == "/requirements":
                _display_requirements(conversation_state)
                continue

            # Process turn through workflow
            with console.status("[bold green]Processing...[/bold green]"):
                try:
                    result = await workflow.run({
                        "user_input": user_input,
                        "state": conversation_state.to_dict() if conversation_state else None
                    })

                    # Extract response and state
                    response_data = result.value
                    conversation_state = ConversationState.from_dict(response_data["state"])

                    # Display response
                    console.print(f"\n[bold green]Assistant:[/bold green] {response_data['response']}")

                    # Display quality metrics if verbose
                    if rcm_app.context.config.get("rcm", {}).get("verbose_metrics", False):
                        _display_quality_metrics(response_data.get("metrics", {}))

                    logger.info("Turn completed", data={
                        "turn": response_data["turn_number"],
                        "response_length": len(response_data["response"])
                    })

                except Exception as e:
                    console.print(f"[red]Error processing turn: {str(e)}[/red]")
                    logger.error(f"Turn processing error: {str(e)}")

        # Display final summary
        if conversation_state and conversation_state.current_turn > 0:
            _display_final_summary(conversation_state)

        logger.info("RCM REPL ended")

def _display_quality_metrics(metrics: dict):
    """Display quality metrics in a table"""
    if not metrics:
        return

    table = Table(title="Response Quality Metrics", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")

    for key, value in metrics.items():
        if key not in ["issues", "overall_score"]:  # Skip nested objects
            display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
            table.add_row(key.replace("_", " ").title(), display_value)

    if "overall_score" in metrics:
        table.add_row("Overall Score", f"{metrics['overall_score']:.2f}")

    console.print(table)

def _display_stats(state: ConversationState):
    """Display conversation statistics"""
    if not state:
        console.print("[yellow]No conversation started yet[/yellow]")
        return

    table = Table(title="Conversation Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Turns", str(state.current_turn))
    table.add_row("Messages", str(len(state.messages)))
    table.add_row("Requirements Tracked", str(len(state.requirements)))

    if state.requirements:
        pending = len([r for r in state.requirements if r.status == "pending"])
        table.add_row("Pending Requirements", str(pending))

    if state.quality_history:
        avg_quality = sum(q.overall_score for q in state.quality_history) / len(state.quality_history)
        table.add_row("Average Quality Score", f"{avg_quality:.2f}")

    if state.answer_lengths:
        avg_length = sum(state.answer_lengths) / len(state.answer_lengths)
        table.add_row("Avg Response Length", f"{avg_length:.0f} chars")

        # Check for bloat
        if len(state.answer_lengths) > 2:
            bloat = state.answer_lengths[-1] / state.answer_lengths[0]
            color = "red" if bloat > 2.0 else "yellow" if bloat > 1.5 else "green"
            table.add_row("Response Bloat Ratio", f"[{color}]{bloat:.1f}x[/{color}]")

    console.print(table)

def _display_requirements(state: ConversationState):
    """Display tracked requirements"""
    if not state or not state.requirements:
        console.print("[yellow]No requirements tracked yet[/yellow]")
        return

    table = Table(title="Tracked Requirements")
    table.add_column("ID", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Status", style="green")
    table.add_column("Turn", style="blue")

    for req in state.requirements:
        status_color = {
            "pending": "yellow",
            "addressed": "blue", 
            "confirmed": "green"
        }.get(req.status, "white")
        
        table.add_row(
            req.id[:8],  # Show first 8 chars of ID
            req.description[:50] + "..." if len(req.description) > 50 else req.description,
            f"[{status_color}]{req.status}[/{status_color}]",
            str(req.source_turn)
        )

    console.print(table)

def _display_final_summary(state: ConversationState):
    """Display final conversation summary"""
    console.print(Panel.fit(
        f"[bold green]Conversation Summary[/bold green]\n\n"
        f"Total turns: {state.current_turn}\n"
        f"Messages exchanged: {len(state.messages)}\n"
        f"Requirements tracked: {len(state.requirements)}\n"
        f"Conversation ID: {state.conversation_id}",
        border_style="green"
    ))

if __name__ == "__main__":
    start = time.time()
    asyncio.run(run_repl())
    end = time.time()
    console.print(f"\nTotal runtime: {end - start:.2f}s")