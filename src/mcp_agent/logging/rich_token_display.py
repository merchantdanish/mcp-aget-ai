"""Rich-based live token usage display for MCP Agent."""

from typing import Dict, Optional, Tuple
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from threading import Lock
from datetime import datetime, timedelta

from mcp_agent.console import console as default_console
from mcp_agent.tracing.token_counter import TokenNode, TokenUsage, TokenCounter


class RichTokenDisplay:
    """Rich-based live display for token usage tracking."""

    def __init__(
        self,
        token_counter: TokenCounter,
        console: Optional[Console] = None,
        update_interval: float = 0.5,
    ):
        """Initialize the token display.

        Args:
            token_counter: The TokenCounter instance to monitor
            console: Rich console to use (defaults to mcp_agent console)
            update_interval: How often to update the display in seconds
        """
        self.console = console or default_console
        self.token_counter = token_counter
        self.update_interval = update_interval
        self._lock = Lock()
        self._node_data: Dict[str, Tuple[TokenNode, TokenUsage, datetime]] = {}
        self._total_cost = 0.0
        self._last_update = datetime.now()
        self._watch_ids = []
        self._live = None
        self._running = False

    def _format_tokens(self, tokens: int) -> str:
        """Format token count with thousands separator."""
        return f"{tokens:,}"

    def _format_cost(self, cost: float) -> str:
        """Format cost in USD."""
        if cost < 0.01:
            return f"${cost:.4f}"
        elif cost < 1.0:
            return f"${cost:.3f}"
        else:
            return f"${cost:.2f}"

    def _format_rate(self, tokens: int, elapsed: timedelta) -> str:
        """Format tokens per second rate."""
        seconds = elapsed.total_seconds()
        if seconds > 0:
            rate = tokens / seconds
            return f"{rate:.1f} tok/s"
        return "- tok/s"

    def _create_display(self) -> Layout:
        """Create the Rich layout for display."""
        layout = Layout()

        # Create header
        header = Panel(
            Align.center(
                Text("🔍 Live Token Usage Monitor", style="bold white on blue"),
                vertical="middle",
            ),
            height=3,
            border_style="blue",
        )

        # Create main table
        table = Table(
            title="Token Usage by Node",
            title_style="bold cyan",
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
            expand=True,
        )

        table.add_column("Node", style="cyan", width=20)
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Input", style="green", justify="right", width=12)
        table.add_column("Output", style="green", justify="right", width=12)
        table.add_column("Total", style="bold green", justify="right", width=12)
        table.add_column("Cost", style="bold yellow", justify="right", width=10)
        table.add_column("Rate", style="blue", justify="right", width=12)
        table.add_column("Last Update", style="dim", width=15)

        # Add rows for each tracked node
        with self._lock:
            sorted_nodes = sorted(
                self._node_data.items(),
                key=lambda x: x[1][1].total_tokens,
                reverse=True,
            )

            for node_key, (node, usage, last_update) in sorted_nodes:
                elapsed = datetime.now() - last_update
                cost = self.token_counter.get_node_cost(node.name, node.node_type)

                table.add_row(
                    node.name[:20],
                    node.node_type,
                    self._format_tokens(usage.input_tokens),
                    self._format_tokens(usage.output_tokens),
                    self._format_tokens(usage.total_tokens),
                    self._format_cost(cost),
                    self._format_rate(usage.total_tokens, elapsed),
                    f"{elapsed.total_seconds():.1f}s ago",
                )

        # Create summary panel
        summary_data = self.token_counter.get_summary()
        summary_text = Text()
        summary_text.append("Total Tokens: ", style="bold")
        summary_text.append(f"{self._format_tokens(summary_data.usage.total_tokens)}\n")
        summary_text.append("Total Cost: ", style="bold")
        summary_text.append(f"{self._format_cost(summary_data.cost)}\n")
        summary_text.append("Active Nodes: ", style="bold")
        summary_text.append(f"{len(self._node_data)}")

        summary = Panel(
            summary_text,
            title="Summary",
            title_align="left",
            border_style="green",
            padding=(1, 2),
        )

        # Arrange layout
        layout.split_column(
            header,
            Layout(table, name="table", ratio=3),
            Layout(summary, name="summary", ratio=1),
        )

        return layout

    def _on_token_update(self, node: TokenNode, usage: TokenUsage) -> None:
        """Callback for token updates."""
        with self._lock:
            node_key = f"{node.name}_{node.node_type}_{id(node)}"
            self._node_data[node_key] = (node, usage, datetime.now())

            # Clean up old entries (not updated in last 30 seconds)
            cutoff = datetime.now() - timedelta(seconds=30)
            self._node_data = {
                k: v for k, v in self._node_data.items() if v[2] > cutoff
            }

    def start(self) -> None:
        """Start the live display and token watching."""
        if self._running:
            return

        self._running = True

        # Watch all node types with low threshold for frequent updates
        watch_configs = [
            # Watch app level for overall stats
            {"node_type": "app", "threshold": 1},
            # Watch workflows for intermediate tracking
            {"node_type": "workflow", "threshold": 1},
            # Watch agents for detailed tracking
            {"node_type": "agent", "threshold": 1},
            # Watch LLMs for most granular tracking
            {"node_type": "llm", "threshold": 1},
        ]

        for config in watch_configs:
            watch_id = self.token_counter.watch(
                callback=self._on_token_update,
                throttle_ms=100,  # Update at most 10 times per second
                **config,
            )
            self._watch_ids.append(watch_id)

        # Start live display
        self._live = Live(
            self._create_display(),
            console=self.console,
            refresh_per_second=1 / self.update_interval,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display and clean up watches."""
        if not self._running:
            return

        self._running = False

        # Remove all watches
        for watch_id in self._watch_ids:
            self.token_counter.unwatch(watch_id)
        self._watch_ids.clear()

        # Stop live display
        if self._live:
            self._live.stop()
            self._live = None

    def update(self) -> None:
        """Manually update the display."""
        if self._live and self._running:
            self._live.update(self._create_display())

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
