# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rich",
#     "typer",
# ]
# ///
"""
Convert the project directory structure and file contents into a single markdown file.
Really helpful for using as a prompt for LLM code generation tasks.
"""

import fnmatch
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.tree import Tree


def parse_gitignore(path: Path) -> List[str]:
    """Parse .gitignore file and return list of patterns."""
    gitigore_path = path / ".gitignore"
    if not gitigore_path.exists():
        return []

    with open(file=gitigore_path, mode="r", encoding="utf-8") as f:
        patterns = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
    return patterns


def should_ignore(
    path: Path, ignore_patterns: List[str], gitignore_patterns: List[str]
) -> bool:
    """Check if path should be ignored based on patterns."""
    str_path = str(path)

    # Check custom ignore patterns
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(str_path, pattern):
            return True

    # Check gitignore patterns
    for pattern in gitignore_patterns:
        if fnmatch.fnmatch(str_path, pattern):
            return True

    return False


def create_tree_structure(
    path: Path,
    include_patterns: List[str],
    ignore_patterns: List[str],
    gitignore_patterns: List[str],
) -> Tree:
    """Create a rich Tree representation of the directory structure."""
    tree = Tree(f"📁 {path.name}")

    def add_to_tree(current_path: Path, tree: Tree):
        for item in sorted(current_path.iterdir()):
            # Skip if should be ignored
            if should_ignore(item, ignore_patterns, gitignore_patterns):
                continue

            # Check if item matches include patterns (if any)
            if include_patterns and not any(
                fnmatch.fnmatch(str(item), p) for p in include_patterns
            ):
                continue

            if item.is_file():
                tree.add(f"📄 {item.name}")
            elif item.is_dir():
                branch = tree.add(f"📁 {item.name}")
                add_to_tree(item, branch)

    add_to_tree(path, tree)
    return tree


def package_project(
    path: Path,
    output_file: Path,
    include_patterns: List[str],
    ignore_patterns: List[str],
    gitignore_patterns: List[str],
) -> None:
    """Package project files into a single markdown file."""
    with open(output_file, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"# Project: {path.name}\n\n")

        # Write directory structure
        f.write("## Directory Structure\n\n")
        f.write("```\n")
        console = Console(file=None)
        with console.capture() as capture:
            console.print(
                create_tree_structure(
                    path, include_patterns, ignore_patterns, gitignore_patterns
                )
            )
        f.write(capture.get())
        f.write("```\n\n")

        # Write file contents
        f.write("## File Contents\n\n")

        def write_files(current_path: Path):
            for item in sorted(current_path.iterdir()):
                if should_ignore(item, ignore_patterns, gitignore_patterns):
                    continue

                if include_patterns and not any(
                    fnmatch.fnmatch(str(item), p) for p in include_patterns
                ):
                    continue

                if item.is_file():
                    try:
                        with open(item, "r", encoding="utf-8") as source_file:
                            content = source_file.read()
                            f.write(f"### {item.relative_to(path)}\n\n")
                            f.write("```")
                            # Add file extension for syntax highlighting if available
                            if item.suffix:
                                f.write(
                                    item.suffix[1:]
                                )  # Remove the dot from extension
                            f.write("\n")
                            f.write(content)
                            f.write("\n```\n\n")
                    except UnicodeDecodeError:
                        f.write(f"### {item.relative_to(path)}\n\n")
                        f.write("```\nBinary file not included\n```\n\n")
                elif item.is_dir():
                    write_files(item)

        write_files(path)


def main(
    path: str = typer.Argument(".", help="Path to the project directory"),
    output: str = typer.Option(
        "project_contents.md", "--output", "-o", help="Output file path"
    ),
    include: List[str] | None = typer.Option(
        None, "--include", "-i", help="Patterns to include (e.g. '*.py')"
    ),
    ignore: List[str] | None = typer.Option(
        None, "--ignore", "-x", help="Patterns to ignore"
    ),
    skip_gitignore: bool = typer.Option(
        False, "--skip-gitignore", help="Skip reading .gitignore patterns"
    ),
):
    """
    Package project files into a single markdown file with directory structure.
    """
    project_path = Path(path).resolve()
    output_path = Path(output).resolve()

    if not project_path.exists():
        typer.echo(f"Error: Project path '{path}' does not exist")
        raise typer.Exit(1)

    # Parse .gitignore if needed
    gitignore_patterns = [] if skip_gitignore else parse_gitignore(project_path)

    # Convert None to empty lists
    include_patterns = include or []
    ignore_patterns = ignore or []

    # Add some default ignore patterns
    # Default ignore patterns for Python development
    default_ignores = [
        "**/__pycache__/**",
        "**/.git/**",
        "**/.idea/**",
        "**/.vscode/**",
        "**/.ruff_cache/**",
        "**/.venv/**",
        "**/venv/**",
        "**/env/**",
        "**/uv.lock",
        "**/.pytest_cache/**",
        "**/*.pyc",
        "**/.coverage",
        "**/htmlcov/**",
    ]
    ignore_patterns.extend(default_ignores)

    try:
        package_project(
            project_path,
            output_path,
            include_patterns,
            ignore_patterns,
            gitignore_patterns,
        )
        typer.echo(f"Successfully packaged project to {output_path}")
    except Exception as e:
        typer.echo(f"Error packaging project: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
