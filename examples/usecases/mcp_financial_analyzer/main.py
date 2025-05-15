"""
Google Search Stock Analyzer with Smart Data Verification
------------------------------------------------------------
A streamlined tool that uses g-search-mcp to find current stock prices
with an improved verification process instead of blind retries.
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
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Configuration values
OUTPUT_DIR = "company_reports"
COMPANY_NAME = "Apple" if len(sys.argv) <= 1 else sys.argv[1]

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
        
        # 1. SEARCH FINDER AGENT: Enhanced to provide more complete data
        finder_agent = Agent(
            name="search_finder",
            instruction=f"""Use Google Search to find complete information about {COMPANY_NAME} in the current month of May 2025:
            You are a world class research analyst. Be thorough but precise.
            
            Execute these exact search queries:
            1. "{COMPANY_NAME} stock price today"
            2. "{COMPANY_NAME} latest quarterly earnings"
            3. "{COMPANY_NAME} financial news"
            
            Extract specifically:
            - Current stock price with exact $ figure and % change
            - Latest earnings results with revenue, EPS figures, and whether expectations were met
            - At least 2-3 recent significant news items with dates
            
            For each fact, include the source [Source: URL].
            Use precise numbers and dates wherever possible.""",
            server_names=["g-search", "fetch", "filesystem"],
        )
        
        # 2. VERIFIER AGENT: Checks if data is complete before proceeding
        verifier_agent = Agent(
            name="data_verifier",
            instruction=f"""Verify if the collected data for {COMPANY_NAME} is complete and reliable.
            
            Check specifically for:
            1. Current stock price (must have $ amount)
            2. Stock % change (must have percentage)
            3. Latest earnings data (must have date, revenue, and EPS figures)
            4. Recent news (must have at least 2 news items with dates)
            
            For each item, mark as PRESENT or MISSING.
            If any key item is MISSING, provide specific feedback.
            
            Conclude with a clear YES or NO on whether data is sufficient to proceed.""",
            server_names=["fetch"],
        )
        
        # 3. ANALYST AGENT: Simple analysis (unchanged)
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
        
        # 4. REPORT WRITER AGENT: Enhanced with better formatting
        report_writer_agent = Agent(
            name="basic_writer",
            instruction=f"""Create a professional stock report for {COMPANY_NAME}:
            You are a world class report writer.
            
            Structure your report with:
            1. Title with company name and current date
            2. Summary table showing current stock price and performance 
            3. Brief company description (1-2 sentences)
            4. Current stock performance section
            5. Latest earnings results section with key figures
            6. Recent news section with relevant articles
            7. Outlook section (2-3 sentences)
            
            Use proper markdown formatting with tables and headers.
            Keep under 1000 words total. Include sources for key data points.""",
            server_names=["filesystem"],
        )
        
        console.print("[bold yellow]Starting analysis process...[/bold yellow]")
        
        try:
            # Step 1: First gather the data with the finder agent
            finder_llm = await finder_agent.attach_llm(OpenAIAugmentedLLM)
            research_data = await finder_llm.generate_str(
                message=f"""Find comprehensive financial information about {COMPANY_NAME}:
                
                1. Current stock price and recent performance
                2. Latest earnings results with specific figures
                3. Major recent news items (at least 2-3)
                
                Include exact figures, dates, and proper citations."""
            )
            console.print("[green]Initial research data gathered[/green]")
            
            # Step 2: Verify the data quality
            verifier_llm = await verifier_agent.attach_llm(OpenAIAugmentedLLM)
            verification = await verifier_llm.generate_str(
                message=f"Verify if this research data for {COMPANY_NAME} is complete and reliable:\n\n{research_data}"
            )
            console.print("[green]Data verification completed[/green]")
            
            # Check if data is sufficient
            data_sufficient = "YES" in verification.upper()
            
            if not data_sufficient:
                console.print("[yellow]Initial data incomplete. Collecting additional information...[/yellow]")
                
                # Extract what's missing from verification
                missing_info = verification.upper().split("FEEDBACK:")[1].split("CONCLUSION:")[0].strip() if "FEEDBACK:" in verification.upper() else "More detailed information needed."
                
                # Targeted search for missing information
                improved_research = await finder_llm.generate_str(
                    message=f"""The initial search data is missing some information:
                    
                    {missing_info}
                    
                    Please conduct additional searches to fill these specific gaps for {COMPANY_NAME}.
                    Be very precise and thorough. Include proper citations."""
                )
                
                # Combine the results
                research_data = f"{research_data}\n\nADDITIONAL RESEARCH:\n{improved_research}"
                console.print("[green]Additional research data collected[/green]")
            
            # Step 3: Analyze the data
            console.print("[bold yellow]Analyzing financial data...[/bold yellow]")
            analyst_llm = await analyst_agent.attach_llm(OpenAIAugmentedLLM)
            analysis = await analyst_llm.generate_str(
                message=f"Analyze this financial data for {COMPANY_NAME}:\n\n{research_data}"
            )
            console.print("[green]Financial analysis completed[/green]")
            
            # Step 4: Generate the report
            console.print("[bold yellow]Generating final report...[/bold yellow]")
            writer_llm = await report_writer_agent.attach_llm(OpenAIAugmentedLLM)
            report_content = await writer_llm.generate_str(
                message=f"""Create a professional stock report for {COMPANY_NAME} using this data:
                
                RESEARCH DATA:
                {research_data}
                
                ANALYSIS:
                {analysis}
                
                Format as a professional report with tables, clear sections, and proper citations."""
            )
            
            # Save the report
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            console.print("[bold green]Report generation completed[/bold green]")
            console.print(f"[bold green]Report saved to:[/bold green] {output_path}")
            
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
                
        except Exception as e:
            console.print(f"[bold red]Error during execution: {e}[/bold red]")
            # Create fallback report
            try:
                fallback_report = f"""# {COMPANY_NAME} Stock Report

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
                    f.write(fallback_report)
                console.print("[yellow]Created fallback report due to error[/yellow]")
                console.print(f"[bold yellow]Fallback report saved to:[/bold yellow] {output_path}")
            except Exception as write_error:
                console.print(f"[bold red]Failed to create fallback report: {write_error}[/bold red]")
        
    console.print(f"[bold]⏱ Time:[/bold] {(time.time() - start_time):.2f}s")
    return True

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]🛑 Interrupted[/bold red]")
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")