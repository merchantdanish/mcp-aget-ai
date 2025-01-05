# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rich",
#     "typer",
#     "pydantic>=2.10.4",
#     "pydantic-settings>=2.7.0"
# ]
# ///
"""
Generate JSON schema for MCP Agent configuration (mcp-agent.config.yaml).
"""

import json
import sys
from pathlib import Path
from typing import Any
import typer
from rich.console import Console
from pydantic import BaseModel
from pydantic_settings import BaseSettings

app = typer.Typer()
console = Console()


class MockModule:
    """Mock module that returns itself for any attribute access."""

    def __getattr__(self, _: str) -> Any:
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self


def create_mock_modules() -> None:
    """Create mock modules for imports we want to ignore."""
    mocked_modules = [
        "opentelemetry",
        "opentelemetry.sdk",
        "opentelemetry.sdk.trace",
        "opentelemetry.sdk.resources",
        "opentelemetry.exporter.otlp.proto.http",
        "opentelemetry.trace",
        "mcp_agent.logging",
        "mcp_agent.logging.logger",
        "yaml",
    ]

    for module_name in mocked_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = MockModule()


def load_settings_class(file_path: Path) -> type[BaseSettings]:
    """Load Settings class from a Python file."""
    # Add src directory to Python path
    src_dir = file_path.parent.parent.parent / "src"
    sys.path.insert(0, str(src_dir))

    # Mock required modules
    create_mock_modules()

    # Create namespace with required classes
    namespace = {
        "BaseModel": BaseModel,
        "BaseSettings": BaseSettings,
        "Path": Path,
        "Dict": dict,
        "List": list,
        "Literal": str,  # Simplified for schema
    }

    with open(file_path) as f:
        exec(f.read(), namespace)

    return namespace["Settings"]


@app.command()
def generate(
    config_py: Path = typer.Option(
        Path("src/mcp_agent/config.py"),
        "--config",
        "-c",
        help="Path to the config.py file",
    ),
    output: Path = typer.Option(
        Path("schema/mcp-agent.config.schema.json"),
        "--output",
        "-o",
        help="Output path for the schema file",
    ),
):
    """Generate JSON schema from Pydantic models in config.py"""
    if not config_py.exists():
        console.print(f"[red]Error:[/] File not found: {config_py}")
        raise typer.Exit(1)

    try:
        Settings = load_settings_class(config_py)
        schema = Settings.model_json_schema()

        # Add schema metadata
        schema.update(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": "MCP Agent Configuration Schema",
                "description": "Configuration schema for MCP Agent applications",
            }
        )

        # Ensure output directory exists
        output.parent.mkdir(parents=True, exist_ok=True)

        # Write schema
        with open(output, "w") as f:
            json.dump(schema, f, indent=2)

        console.print(f"[green]✓[/] Schema written to: {output}")

        # Print VS Code settings suggestion
        vscode_settings = {
            "yaml.schemas": {
                f"./{output}": [
                    "mcp-agent.config.yaml",
                    "mcp_agent.config.yaml",
                ]
            }
        }
        console.print("\n[yellow]VS Code Integration:[/]")
        console.print("Add this to .vscode/settings.json:")
        console.print(json.dumps(vscode_settings, indent=2))

    except Exception as e:
        console.print(f"[red]Error generating schema:[/] {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
