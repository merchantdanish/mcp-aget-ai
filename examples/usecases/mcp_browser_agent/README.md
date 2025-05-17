# 🌐 Browser Console Agent Example

A command-line application that lets you interact with websites using natural language through the Model Context Protocol (MCP) with the use of the [Puppeteer MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer).

https://github.com/user-attachments/assets/195af0e7-1bd1-42bf-b77a-15ca28d36f1f

- **Natural Language Control**: Navigate and interact with websites using conversational commands
- **Continuous Browser Session**: Keep the same browser context across multiple queries
- **Real-time Website Analysis**: Extract information, analyze content, and take screenshots
- **Interactive Console Interface**: Simple terminal-based interface for browsing the web

```plaintext
┌─────────┐      ┌───────────┐      ┌──────────────┐
│ Console │─────▶│  Browser  │─────▶│  Puppeteer   │
└─────────┘      │  Agent    │      │  MCP Server  │
                 └───────────┘      └──────────────┘
```

## `1` App set up

First, clone the repo and navigate to the browser agent example:

```bash
git clone https://github.com/lastmile-ai/mcp-agent.git
cd mcp-agent/examples/usecases/mcp_browser_agent
```

Install the UV tool (if you don’t have it) to manage dependencies:

```bash
pip install uv

# inside the example:
uv pip install .
```

Make sure Node.js and npm are installed:

```bash
node --version
npm --version
```

## `2` Set up environment variables

Copy and configure your secrets and env variables:

```bash
cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml
```

Then open `mcp_agent.secrets.yaml` and add your api key for your preferred LLM.

## `3` Run locally

Run your MCP Agent app:

```bash
uv run console_agent.py [URL]
```

### Example Commands

- "Summarize the content on this page"
- "Click on the 'Documentation' link"
- "Fill out the contact form with this information..."
- "Find all links on this page"
- "Navigate to the pricing page"
- "Extract the main headings from this article"
- "Take a screenshot of the current page"
