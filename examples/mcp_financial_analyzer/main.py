"""
Google Search Stock Analyzer
------------------------------------------------------------
A streamlined tool that uses g-search-mcp to find current stock prices
and earnings information for company analysis.
"""

import asyncio
import os
import time
import sys
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
# Removed the unused import: from mcp_agent.workflows.llm.augmented_llm import RequestParams

# Configuration values
OUTPUT_DIR = "company_reports"
COMPANY_NAME = "Apple" if len(sys.argv) <= 1 else sys.argv[1]
MAX_RETRIES = 2  # Number of times to retry on failure

# Initialize console and app
console = Console()
app = MCPApp(name="gsearch_stock_analyzer")

async def main():
    start_time = time.time()
    
    # Create output directory and set up file paths
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{COMPANY_NAME.lower().replace(' ', '_')}_report_{timestamp}.md"
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    console.print(Panel(f"[bold green]Analyzing [bold blue]{COMPANY_NAME}[/bold blue][/bold green]"))
    
    async with app.run() as orchestrator_app:
        context = orchestrator_app.context
        
        # Configure filesystem server
        if "filesystem" in context.config.mcp.servers:
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            console.print("[green]Filesystem server configured[/green]")
        else:
            console.print("[bold yellow]Warning: filesystem server not found[/bold yellow]")
        
        # Check for g-search server
        has_gsearch = "g-search" in context.config.mcp.servers
        if has_gsearch:
            console.print("[green]Google Search server available[/green]")
        else:
            console.print("[bold red]Google Search server not found! This script requires g-search-mcp[/bold red]")
            return False
        
        # 1. SEARCH FINDER AGENT: Specifically uses g-search
        finder_agent = Agent(
            name="search_finder",
            instruction=f"""Use Google Search to find information about {COMPANY_NAME} in the current month of May 2025:
            You are a world class research analyst.
            Execute these exact search queries:
            1. "{COMPANY_NAME} stock price today"
            2. "{COMPANY_NAME} latest quarterly earnings"
            3. "{COMPANY_NAME} financial news"
            
            Extract the most relevant information about:
            - Current stock price and recent movement
            - Latest earnings report data
            - Any significant recent news
            
            Keep responses short and focused on facts.""",
            server_names=["g-search", "fetch", "filesystem"],
        )
        
        # 2. ANALYST AGENT: Simple analysis
        analyst_agent = Agent(
            name="simple_analyst",
            instruction=f"""Analyze the key data for {COMPANY_NAME}:
            You are a world class financial analyst.
            1. Note if stock is up or down and by how much
            2. Check if earnings beat or missed expectations
            3. List 1-2 main strengths and concerns
            
            Be very concise.""",
            server_names=["fetch"],
        )
        
        # 3. REPORT WRITER AGENT: Creates a minimal report
        report_writer_agent = Agent(
            name="basic_writer",
            instruction=f"""Create a minimal stock report for {COMPANY_NAME}:
            You are a world class report writer.

            1. Start with a very brief company description (1 sentence)
            2. Include current stock price and performance
            3. Summarize latest earnings results
            4. List All of the news articles that are relevant to the company in the past month
            5. Add a 1-2 sentence outlook on how the company is doing
            
            Format as simple markdown and keep under 600 words total.""",
            server_names=["filesystem"],
        )
        
        # Create the agents list directly without using intermediate variables
        agents = [
            finder_agent,
            analyst_agent,
            report_writer_agent,
        ]
        
        # Try with multiple retries if needed
        success = False
        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                console.print(f"[yellow]Retry attempt {attempt}/{MAX_RETRIES}...[/yellow]")
                
            console.print("[bold yellow]Searching and analyzing...[/bold yellow]")
            try:
                # Execute search and analysis without intermediate variables
                await Orchestrator(
                    llm_factory=OpenAIAugmentedLLM,
                    available_agents=agents,
                    plan_type="full",
                ).generate_str(
                    message=f"""Use Google Search to analyze {COMPANY_NAME}:
        
                    1. Find current stock price and performance
                    2. Find latest earnings results
                    3. Find any major recent news
                    4. Create a brief analysis report
                    5. Save to "{output_file}" in "{OUTPUT_DIR}"
                    
                    Keep everything extremely simple and factual."""
                )
                
                # Check if output file exists to confirm success
                if os.path.exists(output_path):
                    success = True
                    console.print("[bold green]Analysis complete[/bold green]")
                    console.print(f"[bold green]Report saved:[/bold green] {output_path}")
                    
                    # Show preview
                    try:
                        with open(output_path, 'r') as f:
                            lines = f.readlines()
                            preview = ''.join(lines[:10] if len(lines) > 10 else lines)
                        console.print(Panel(f"[bold]Preview:[/bold]\n\n{preview}\n...", 
                                     title="[bold]Report[/bold]",
                                     expand=False))
                    except Exception as e:
                        console.print(f"[yellow]Could not preview: {e}[/yellow]")
                    
                    break  # Exit retry loop on success
                else:
                    console.print("[yellow]Output file not created, will retry...[/yellow]")
                    
            except Exception as e:
                console.print(f"[bold red]Error during attempt {attempt+1}: {e}[/bold red]")
                
                # On final attempt, create a basic report with minimal info
                if attempt == MAX_RETRIES:
                    console.print("[yellow]Creating minimal fallback report...[/yellow]")
                    try:
                        # Create a simple report directly
                        report_content = f"""# {COMPANY_NAME} Stock Report

## Basic Information
This is a minimal report generated due to search limitations.

## Stock Information
For current stock data, please check financial sites like:
- Yahoo Finance
- Google Finance
- MarketWatch

## Recent Earnings
For the latest earnings information, visit {COMPANY_NAME}'s investor relations page.

## Recommendation
For current financial information on {COMPANY_NAME}, we recommend:
1. Company's official investor relations website
2. Financial news sites (CNBC, Bloomberg, etc.)
3. Stock market tracking apps

*Report generated on {datetime.now().strftime("%Y-%m-%d")}*
"""
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        with open(output_path, 'w') as f:
                            f.write(report_content)
                        
                        console.print("[bold green]Fallback report created[/bold green]")
                        console.print(f"[bold green]Report saved:[/bold green] {output_path}")
                        success = True
                    except Exception as write_error:
                        console.print(f"[bold red]Failed to create fallback report: {write_error}[/bold red]")
        
        if not success:
            console.print("[bold red]All attempts failed to create report[/bold red]")
        
    console.print(f"[bold]⏱ Time:[/bold] {(time.time() - start_time):.2f}s")
    return success

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]🛑 Interrupted[/bold red]")
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")