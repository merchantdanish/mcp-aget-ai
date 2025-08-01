"""Token usage progress display using Rich Progress widget."""

from typing import Optional, Dict
from rich.console import Console
from rich.progress import Progress, TextColumn
from mcp_agent.console import console as default_console
from mcp_agent.tracing.token_counter import TokenNode, TokenUsage, TokenCounter
from contextlib import contextmanager


class TokenProgressDisplay:
    """Rich Progress-based display for token usage."""

    def __init__(self, token_counter: TokenCounter, console: Optional[Console] = None):
        """Initialize the token progress display."""
        self.console = console or default_console
        self.token_counter = token_counter
        self._taskmap: Dict[str, int] = {}
        self._watch_ids = []

        # Create progress display with custom columns
        self._progress = Progress(
            TextColumn("[bold cyan]Token Usage", justify="left"),
            TextColumn("{task.fields[node_info]:<30}", style="white"),
            TextColumn("{task.fields[tokens]:>10}", style="bold green"),
            TextColumn("{task.fields[cost]:>10}", style="bold yellow"),
            console=self.console,
            transient=False,
            refresh_per_second=10,
        )
        self._paused = False
        self._total_task_id = None

    def start(self):
        """Start the progress display and register watches."""
        self._progress.start()

        # Add a task for the total
        self._total_task_id = self._progress.add_task(
            "", total=None, node_info="[bold]TOTAL", tokens="0", cost="$0.0000"
        )

        # Register watch on app node for aggregate totals
        if hasattr(self.token_counter, "_root") and self.token_counter._root:
            watch_id = self.token_counter.watch(
                callback=self._on_token_update,
                node=self.token_counter._root,
                threshold=1,
                throttle_ms=100,
            )
            self._watch_ids.append(watch_id)

    def stop(self):
        """Stop the progress display and unregister watches."""
        # Unregister watches
        for watch_id in self._watch_ids:
            self.token_counter.unwatch(watch_id)
        self._watch_ids.clear()

        self._progress.stop()

    def pause(self):
        """Pause the progress display."""
        if not self._paused:
            self._paused = True
            for task in self._progress.tasks:
                task.visible = False
            self._progress.stop()

    def resume(self):
        """Resume the progress display."""
        if self._paused:
            for task in self._progress.tasks:
                task.visible = True
            self._paused = False
            self._progress.start()

    @contextmanager
    def paused(self):
        """Context manager for temporarily pausing the display."""
        self.pause()
        try:
            yield
        finally:
            self.resume()

    def _format_tokens(self, tokens: int) -> str:
        """Format token count with thousands separator."""
        return f"{tokens:,}"

    def _format_cost(self, cost: float) -> str:
        """Format cost in USD."""
        return f"${cost:.4f}"

    async def _on_token_update(self, node: TokenNode, usage: TokenUsage):
        """Handle token usage updates."""
        # Update total
        summary = self.token_counter.get_summary()
        self._progress.update(
            self._total_task_id,
            node_info="[bold]TOTAL",
            tokens=self._format_tokens(summary.usage.total_tokens),
            cost=self._format_cost(summary.cost),
        )

        # Get app-level cost
        cost = self.token_counter.get_node_cost(node.name, node.node_type)

        # Create task key
        task_key = f"{node.name}_{node.node_type}"

        # Update or create task for this node
        if task_key not in self._taskmap:
            task_id = self._progress.add_task(
                "",
                total=None,
                node_info=f"{node.name} ({node.node_type})",
                tokens=self._format_tokens(usage.total_tokens),
                cost=self._format_cost(cost),
            )
            self._taskmap[task_key] = task_id
        else:
            task_id = self._taskmap[task_key]
            self._progress.update(
                task_id,
                node_info=f"{node.name} ({node.node_type})",
                tokens=self._format_tokens(usage.total_tokens),
                cost=self._format_cost(cost),
            )

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
