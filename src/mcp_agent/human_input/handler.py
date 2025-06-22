import asyncio
import json
from typing import Any, Optional

from rich.panel import Panel
from mcp_agent.console import console
from mcp_agent.human_input.types import HumanInputRequest, HumanInputResponse
from mcp_agent.logging.progress_display import progress_display
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)

# Slash command constants
SLASH_COMMANDS = {
    "/decline": "Decline the human input request.",
    "/cancel": "Cancel the human input request.",
    "/help": "Show available commands",
}


class SlashCommandResult:
    def __init__(self, command: str, action: str):
        self.command = command
        self.action = action


def _process_slash_command(input_text: str) -> Optional[SlashCommandResult]:
    """Detect and map slash commands to actions."""
    if not input_text.startswith("/"):
        return None
    cmd = input_text.strip().lower()
    action = {
        "/decline": "decline",
        "/cancel": "cancel",
        "/help": "help",
    }.get(cmd, "unknown" if cmd != "/" else "help")

    if action == "unknown":
        console.print(f"\n[red]Unknown command: {cmd}[/red]")
        console.print("[dim]Type /help for available commands[/dim]\n")
    return SlashCommandResult(cmd, action)


def _print_slash_help() -> None:
    """Display available slash commands."""
    console.print("\n[cyan]Available commands:[/cyan]")
    for cmd, desc in SLASH_COMMANDS.items():
        console.print(f"  [green]{cmd}[/green] - {desc}")
    console.print()


def _create_panel(request: HumanInputRequest) -> Panel:
    """Generate styled panel for prompts."""
    title = (
        f"HUMAN INPUT NEEDED FROM: {request.server_name}"
        if request.server_name
        else "HUMAN INPUT NEEDED"
    )
    content = (
        request.description
        and f"[bold]{request.description}[/bold]\n\n{request.prompt}"
        or request.prompt
    )
    content += "\n\n[dim]Type / to see available commands[/dim]"
    return Panel(
        content, title=title, style="blue", border_style="bold white", padding=(1, 2)
    )


async def console_input_callback(request: HumanInputRequest) -> HumanInputResponse:
    """Entry point: handle both simple and schema-based input."""
    with progress_display.paused():
        console.print(_create_panel(request))
        if request.requested_schema and isinstance(request.requested_schema, dict):
            response = await _handle_schema_input(request)
        else:
            response = await _handle_simple_input(request)
    return HumanInputResponse(request_id=request.request_id, response=response)


async def _handle_simple_input(request: HumanInputRequest) -> str:
    """Handle free-text input."""
    while True:
        if request.timeout_seconds:
            try:
                user_input = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: console.input("> ")
                    ),
                    request.timeout_seconds,
                )
            except asyncio.TimeoutError:
                console.print("\n[red]Timeout waiting for input[/red]")
                raise TimeoutError("No response received within timeout period")
        else:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: console.input("> ")
            )

        user_input = user_input.strip()
        cmd_result = _process_slash_command(user_input)
        if not cmd_result:
            return user_input
        if cmd_result.action in ("decline", "cancel"):
            return cmd_result.action
        if cmd_result.action == "help":
            _print_slash_help()
            continue


async def _handle_schema_input(request: HumanInputRequest) -> str:
    """Prompt for structured input based on provided schema."""
    schema = request.requested_schema
    if not schema or "properties" not in schema:
        raise ValueError("Invalid schema: must contain 'properties'")

    result = {}
    for name, props in schema["properties"].items():
        prompt_text = f"Enter {name}"
        if desc := props.get("description"):
            prompt_text += f" - {desc}"
        default = props.get("default")
        loop_prompt = (
            f"{prompt_text}{f' [default: {default}]' if default is not None else ''}"
        )

        while True:
            console.print(f"\n{loop_prompt}", style="cyan", markup=False)
            console.print("[dim]Type / to see available commands[/dim]")
            # Show type-specific input hints
            field_type = props.get("type", "string")
            if field_type == "boolean":
                console.print("[dim]Enter: true/false, yes/no, y/n, or 1/0[/dim]")
            elif field_type == "number":
                console.print("[dim]Enter a decimal number[/dim]")
            elif field_type == "integer":
                console.print("[dim]Enter a whole number[/dim]")

            # Show optional hint when a default exists
            if default is not None:
                console.print(f"[dim]Press Enter to accept default [{default}][/dim]")

            value = console.input("> ").strip() or (
                str(default) if default is not None else ""
            )
            cmd_result = _process_slash_command(value)
            if cmd_result:
                if cmd_result.action in ("decline", "cancel"):
                    return cmd_result.action
                if cmd_result.action == "help":
                    _print_slash_help()
                    continue
            processed = _process_field_value(props.get("type", "string"), value)
            if processed is not None:
                result[name] = processed
                break
    return json.dumps(result)


def _process_field_value(field_type: str, value: str) -> Any:
    if field_type == "boolean":
        v = value.lower()
        if v in ("true", "yes", "y", "1"):
            return True
        if v in ("false", "no", "n", "0"):
            return False
        console.print(f"[red]Invalid boolean value: {value}[/red]")
        return None
    if field_type == "number":
        try:
            return float(value)
        except ValueError:
            console.print(f"[red]Invalid number: {value}[/red]")
            return None
    if field_type == "integer":
        try:
            return int(value)
        except ValueError:
            console.print(f"[red]Invalid integer: {value}[/red]")
            return None
    return value
