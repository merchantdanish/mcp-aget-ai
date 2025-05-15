# Stock Analyzer with Google Search MCP

A streamlined financial analysis tool that orchestrates multiple specialized agents to generate company stock reports using Google Search.

## Overview

This application demonstrates the power of multi-agent orchestration using the MCP (Model Context Protocol) framework. It gathers real-time financial data through Google Search and produces concise, informative stock reports without requiring any local background data.

## Features

- **Google Search Integration**: Uses g-search-mcp to fetch current stock prices and financial news
- **Multi-Agent Architecture**: Employs specialized agents for data collection, analysis, and report generation
- **Fully Automated Workflow**: Handles the entire process from searching to report creation
- **Fallback Mechanisms**: Includes retry logic and failsafe report generation
- **Minimal Token Usage**: Optimized prompts reduce LLM token consumption

## Requirements

- Python 3.10+
- Node.js and npm (for MCP servers)
- OpenAI API key

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stock-analyzer
   ```

2. **Install Python dependencies**:
   ```bash
   pip install mcp-agent rich
   ```

3. **Install MCP servers**:
   ```bash
   npm install -g g-search-mcp
   ```

4. **Configure your API key**:
   Edit `mcp_agent.secrets.yaml` and add your OpenAI API key:
   ```yaml
   openai:
     api_key: "YOUR_OPENAI_API_KEY"
   ```

## Running the Application

Run the analyzer with a company name argument:

```bash
python main.py "Microsoft"
```

If no company name is provided, it defaults to analyzing Apple:

```bash
python main.py
```

The application will:
1. Create a `company_reports` directory if it doesn't exist
2. Use Google Search to find current financial information
3. Generate a report with the naming pattern: `{company_name}_report_{timestamp}.md`
4. Display a preview of the report in the console

## Understanding the Orchestrator Workflow

This application uses an orchestrated multi-agent approach for several key reasons:

### Workflow Architecture

The orchestrator coordinates three specialized agents:

1. **Search Finder Agent** (`search_finder`):
   - Uses Google Search to find current stock data and financial news
   - Performs targeted searches for specific financial information
   - Extracts relevant data points from search results

2. **Analyst Agent** (`simple_analyst`):
   - Processes the raw data gathered by the finder agent
   - Determines if earnings beat expectations
   - Identifies key strengths and concerns

3. **Report Writer Agent** (`basic_writer`):
   - Compiles the analyzed information into a structured report
   - Creates a well-formatted markdown document
   - Saves the report to the filesystem

### Why This Approach?

This orchestrated architecture is ideal for financial analysis because:

1. **Separation of Concerns**: Each agent focuses on a specific task, improving output quality
2. **Enhanced Reliability**: If one component fails, others can adapt or retry
3. **Modular Design**: Easy to update or replace individual components
4. **Parallel Processing**: The orchestrator can run compatible tasks simultaneously
5. **Complex Workflow Management**: Handles multi-step processes with dependencies automatically

## Customizing for Different Applications

You can adapt this framework for various applications by modifying:

### 1. Agent Instructions

Each agent has a focused instruction set that can be customized:

```python
finder_agent = Agent(
    name="search_finder",
    instruction=f"""Your custom instructions here...""",
    server_names=["g-search", "fetch", "filesystem"],
)
```

### 2. Search Queries

Modify the specific search queries in the finder agent:

```python
"""
Execute these exact search queries:
1. "{COMPANY_NAME} your custom query"
2. "{COMPANY_NAME} another custom query"
"""
```

### 3. Report Format

Change the report structure by updating the writer agent instructions:

```python
"""
Create a custom report format:
1. Section one...
2. Section two...
"""
```

### 4. Different Search Tools

You can add popular search tools such as Brave or Playwright for the search part. 

All you need to do is replace g-search-mcp with other MCP servers in `mcp_agent.config.yaml`:

```yaml
mcp:
  servers:
    alternative-search:
      command: "npx"
      args: ["-y", "alternative-search-mcp"]
```

## Advanced Configuration

### Adding More Agents

To add a specialized agent for deeper analysis:

```python
technical_analyst_agent = Agent(
    name="technical_analyst",
    instruction="""Analyze technical indicators and patterns...""",
    server_names=["fetch"],
)

# Add to orchestrator
orchestrator = Orchestrator(
    llm_factory=OpenAIAugmentedLLM,
    available_agents=[
        finder_agent,
        analyst_agent,
        technical_analyst_agent,  # New agent
        report_writer_agent,
    ],
    plan_type="full",
)
```

### Using Different LLM Providers

To use Anthropic instead of OpenAI:

1. Uncomment the Anthropic section in `mcp_agent.secrets.yaml`
2. Update the orchestrator to use Anthropic's LLM:

```python
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM

orchestrator = Orchestrator(
    llm_factory=AnthropicAugmentedLLM,  # Use Anthropic instead
    # rest of the configuration remains the same
)
```

## Troubleshooting

- **"Google Search server not found"**: Make sure g-search-mcp is installed globally
- **"Error during attempt"**: Check your API key in mcp_agent.secrets.yaml
- **Empty or missing reports**: Verify the filesystem server is configured correctly

## Why This Architecture Works Well

The MCP Agent Orchestrator framework is perfect for this application because:

1. **Real-time Data Needs**: Financial data changes constantly, requiring fresh information
2. **Multi-source Synthesis**: Combines data from search results, earnings reports, and news
3. **Specialized Analysis Steps**: Different stages require different expertise
4. **Resilience Requirements**: Financial analysis tools need fallback mechanisms
5. **Independent Components**: Separation of data gathering, analysis, and reporting improves maintainability
