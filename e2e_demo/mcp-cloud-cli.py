#!/usr/bin/env python3
"""
MCP Agent Cloud CLI

A command-line tool for managing MCP Agent Cloud deployments.
"""

import argparse
import os
import sys
import subprocess
import time
from typing import List, Dict, Any, Optional

# Initialize CLI with rich console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    has_rich = True
except ImportError:
    has_rich = False
    print("Rich library not found. Install with: pip install rich")
    print("Falling back to standard output.")
    class FakeConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = FakeConsole()

def run_command(command: List[str], cwd: Optional[str] = None) -> str:
    """Run a shell command and return the output."""
    try:
        process = subprocess.run(
            command,
            cwd=cwd or os.path.dirname(os.path.abspath(__file__)),
            check=True,
            text=True,
            capture_output=True
        )
        return process.stdout
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error running command: {' '.join(command)}[/bold red]")
        console.print(f"[red]{e.stderr}[/red]")
        return e.stderr

def deploy_command(args):
    """Deploy MCP Agent Cloud components."""
    console.print(f"[bold blue]Deploying {args.component}...[/bold blue]")
    
    if args.component == "all":
        # Deploy all components
        run_command(["docker-compose", "up", "-d"])
        console.print("[bold green]All components deployed successfully![/bold green]")
    elif args.component == "auth":
        # Deploy auth service
        run_command(["docker-compose", "up", "-d", "cloud-auth"])
        console.print("[bold green]Auth service deployed successfully![/bold green]")
    elif args.component == "servers":
        # Deploy MCP servers
        run_command(["docker-compose", "up", "-d", "filesystem-server", "fetch-server"])
        console.print("[bold green]MCP servers deployed successfully![/bold green]")
    elif args.component == "app":
        # Deploy MCP application
        run_command(["docker-compose", "up", "-d", "mcp-app"])
        console.print("[bold green]MCP application deployed successfully![/bold green]")
    elif args.component == "filesystem":
        # Deploy filesystem server
        run_command(["docker-compose", "up", "-d", "filesystem-server"])
        console.print("[bold green]Filesystem server deployed successfully![/bold green]")
    elif args.component == "fetch":
        # Deploy fetch server
        run_command(["docker-compose", "up", "-d", "fetch-server"])
        console.print("[bold green]Fetch server deployed successfully![/bold green]")
    else:
        console.print(f"[bold red]Unknown component: {args.component}[/bold red]")
        return 1
    
    return 0

def stop_command(args):
    """Stop MCP Agent Cloud components."""
    console.print(f"[bold blue]Stopping {args.component}...[/bold blue]")
    
    if args.component == "all":
        # Stop all components
        run_command(["docker-compose", "down"])
        console.print("[bold green]All components stopped![/bold green]")
    elif args.component == "auth":
        # Stop auth service
        run_command(["docker-compose", "stop", "cloud-auth"])
        console.print("[bold green]Auth service stopped![/bold green]")
    elif args.component == "servers":
        # Stop MCP servers
        run_command(["docker-compose", "stop", "filesystem-server", "fetch-server"])
        console.print("[bold green]MCP servers stopped![/bold green]")
    elif args.component == "app":
        # Stop MCP application
        run_command(["docker-compose", "stop", "mcp-app"])
        console.print("[bold green]MCP application stopped![/bold green]")
    elif args.component == "filesystem":
        # Stop filesystem server
        run_command(["docker-compose", "stop", "filesystem-server"])
        console.print("[bold green]Filesystem server stopped![/bold green]")
    elif args.component == "fetch":
        # Stop fetch server
        run_command(["docker-compose", "stop", "fetch-server"])
        console.print("[bold green]Fetch server stopped![/bold green]")
    else:
        console.print(f"[bold red]Unknown component: {args.component}[/bold red]")
        return 1
    
    return 0

def logs_command(args):
    """View logs for MCP Agent Cloud components."""
    console.print(f"[bold blue]Viewing logs for {args.component}...[/bold blue]")
    
    if args.component == "all":
        # View all logs
        subprocess.run(["docker-compose", "logs", "--tail=100", "-f"], 
                      cwd=os.path.dirname(os.path.abspath(__file__)))
    elif args.component == "auth":
        # View auth service logs
        subprocess.run(["docker-compose", "logs", "--tail=100", "-f", "cloud-auth"],
                      cwd=os.path.dirname(os.path.abspath(__file__)))
    elif args.component == "filesystem-server":
        # View filesystem server logs
        subprocess.run(["docker-compose", "logs", "--tail=100", "-f", "filesystem-server"],
                      cwd=os.path.dirname(os.path.abspath(__file__)))
    elif args.component == "fetch-server":
        # View fetch server logs
        subprocess.run(["docker-compose", "logs", "--tail=100", "-f", "fetch-server"],
                      cwd=os.path.dirname(os.path.abspath(__file__)))
    elif args.component == "app":
        # View MCP application logs
        subprocess.run(["docker-compose", "logs", "--tail=100", "-f", "mcp-app"],
                      cwd=os.path.dirname(os.path.abspath(__file__)))
    else:
        console.print(f"[bold red]Unknown component: {args.component}[/bold red]")
        return 1
    
    return 0

def status_command(args):
    """Check status of MCP Agent Cloud components."""
    console.print("[bold blue]Checking status of MCP Agent Cloud components...[/bold blue]")
    
    # Run docker-compose ps
    output = run_command(["docker-compose", "ps"])
    
    # Parse the output
    if has_rich:
        # Create a table for the status
        table = Table(title="MCP Agent Cloud Status")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Ports", style="yellow")
        
        # Skip header lines
        lines = output.strip().split("\n")[2:]
        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                continue
                
            service = parts[0].replace("e2e_demo_", "").replace("_1", "")
            status = "Running" if "Up" in line else "Stopped"
            ports = " ".join([p for p in parts if "->" in p])
            
            table.add_row(service, status, ports)
            
        console.print(table)
    else:
        console.print(output)
    
    return 0

def build_command(args):
    """Build MCP Agent Cloud components."""
    console.print(f"[bold blue]Building {args.component}...[/bold blue]")
    
    if args.component == "all":
        # Build all components
        run_command(["docker-compose", "build"])
        console.print("[bold green]All components built successfully![/bold green]")
    elif args.component == "auth":
        # Build auth service
        run_command(["docker-compose", "build", "cloud-auth"])
        console.print("[bold green]Auth service built successfully![/bold green]")
    elif args.component == "filesystem-server":
        # Build filesystem server
        run_command(["docker-compose", "build", "filesystem-server"])
        console.print("[bold green]Filesystem server built successfully![/bold green]")
    elif args.component == "fetch-server":
        # Build fetch server
        run_command(["docker-compose", "build", "fetch-server"])
        console.print("[bold green]Fetch server built successfully![/bold green]")
    elif args.component == "app":
        # Build MCP application
        run_command(["docker-compose", "build", "mcp-app"])
        console.print("[bold green]MCP application built successfully![/bold green]")
    else:
        console.print(f"[bold red]Unknown component: {args.component}[/bold red]")
        return 1
    
    return 0

def view_output_command(args):
    """View output files from the MCP application."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp-app", "output")
    
    if not os.path.exists(output_dir):
        console.print("[bold red]Output directory does not exist.[/bold red]")
        return 1
        
    if not os.listdir(output_dir):
        console.print("[bold yellow]No output files found.[/bold yellow]")
        return 1
        
    console.print("[bold blue]Available output files:[/bold blue]")
    for i, filename in enumerate(os.listdir(output_dir), 1):
        console.print(f"[cyan]{i}.[/cyan] [green]{filename}[/green]")
        
    if args.file:
        file_path = os.path.join(output_dir, args.file)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
                if has_rich:
                    console.print(f"\n[bold]{'-' * 80}[/bold]")
                    console.print(f"[bold]{args.file}:[/bold]")
                    console.print(f"[bold]{'-' * 80}[/bold]")
                    from rich.markdown import Markdown
                    console.print(Markdown(content))
                else:
                    print(f"\n{'-' * 80}")
                    print(f"{args.file}:")
                    print(f"{'-' * 80}")
                    print(content)
        else:
            console.print(f"[bold red]File not found: {args.file}[/bold red]")
    
    return 0

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="MCP Agent Cloud CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy MCP Agent Cloud components")
    deploy_parser.add_argument("component", choices=["all", "auth", "servers", "app", "filesystem", "fetch"], help="Component to deploy")
    deploy_parser.set_defaults(func=deploy_command)
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop MCP Agent Cloud components")
    stop_parser.add_argument("component", choices=["all", "auth", "servers", "app", "filesystem", "fetch"], help="Component to stop")
    stop_parser.set_defaults(func=stop_command)
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="View logs for MCP Agent Cloud components")
    logs_parser.add_argument("component", choices=["all", "auth", "filesystem-server", "fetch-server", "app"], help="Component to view logs for")
    logs_parser.set_defaults(func=logs_command)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check status of MCP Agent Cloud components")
    status_parser.set_defaults(func=status_command)
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build MCP Agent Cloud components")
    build_parser.add_argument("component", choices=["all", "auth", "filesystem-server", "fetch-server", "app"], help="Component to build")
    build_parser.set_defaults(func=build_command)
    
    # View output command
    output_parser = subparsers.add_parser("output", help="View output files from the MCP application")
    output_parser.add_argument("--file", help="Specific output file to view")
    output_parser.set_defaults(func=view_output_command)
    
    args = parser.parse_args()
    
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
        
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())