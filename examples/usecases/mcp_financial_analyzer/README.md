# MCP Financial Analyzer with Google Search

This MCP Agent app uses an orchestrator to coordinate multiple specialized agents that work with the [g-search server](https://github.com/jae-jae/g-search-mcp), [fetch server](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch), and [filesystem server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) to generate comprehensive financial reports for companies.

```plaintext
┌──────────────┐      ┌──────────────┐
│              │      │  G-Search    │
│  Orchestrator│──┬──▶│  MCP Server  │
│  (manages    │  │   └──────────────┘
│   agents)    │  │   ┌──────────────┐
│              │  ├──▶│  Fetch       │
└──┬─────┬─────┘  │   │  MCP Server  │
   │     │        │   └──────────────┘
   │     │        │   ┌──────────────┐
   │     │        └──▶│  Filesystem  │
   │     │            │  MCP Server  │
   │     │            └──────────────┘
   ▼     ▼                    
┌─────┐ ┌─────┐ ┌─────────┐
│Find │ │Anal-│ │Report   │
│Agent│ │yst  │ │Writer   │
│     │ │Agent│ │Agent    │
└─────┘ └─────┘ └─────────┘
```

## `1` App set up

First, clone the repo and navigate to the financial analyzer example:

```bash
git clone https://github.com/lastmile-ai/mcp-agent.git
cd mcp-agent/examples/mcp_financial_analyzer
```

Install the UV tool (if you don't have it) to manage dependencies:

```bash
pip install uv
# inside the example:
uv pip install -r requirements.txt
```

Install the g-search-mcp server:

```bash
npm install -g g-search-mcp
```

## `2` Set up secrets and environment variables

Copy and configure your secrets:

```bash
cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml
```

Then open `mcp_agent.secrets.yaml` and add your API key for your preferred LLM (OpenAI):

```yaml
openai:
  api_key: "YOUR_OPENAI_API_KEY"
```

## `3` Run locally

Run your MCP Agent app with a company name:

```bash
uv run main.py "Apple"
```

Or run with a different company:

```bash
uv run main.py "Microsoft"
```

The program will:
1. Use Google Search to find current stock price and financial data
2. Analyze the company's performance and recent news
3. Generate a comprehensive report in the `company_reports` directory

## How It Works

The financial analyzer demonstrates the power of multi-agent orchestration:

1. **Finder Agent**: Uses Google Search to gather real-time financial data
2. **Analyst Agent**: Evaluates the financial metrics and identifies key trends
3. **Report Writer Agent**: Compiles findings into a well-structured markdown report

The orchestrator coordinates these agents, handling the workflow and error recovery.

## Customizing the Analysis

You can modify the agent instructions in `main.py` to focus on different aspects of financial analysis:

- Change search queries to target specific financial metrics
- Adjust analysis criteria for different industries
- Customize the report format and content focus

## Deploy your MCP Agent app

Coming soon
