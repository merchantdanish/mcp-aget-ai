# GitHub PRs to Slack Summary Agent

This application creates an MCP Agent that monitors GitHub pull requests and submits prioritized summaries to Slack. The agent uses a LLM to analyze PR information, prioritize issues, and create informative summaries.

## How It Works

1. The application connects to both GitHub and Slack via their respective MCP servers
2. The agent retrieves the latest pull requests from a specified GitHub repository
3. It analyzes each PR and prioritizes them based on importance factors:
   - PRs marked as high priority or urgent
   - PRs addressing security vulnerabilities
   - PRs fixing critical bugs
   - PRs blocking other work
   - PRs that have been open for a long time
4. The agent formats a professional summary of high-priority items
5. The summary is posted to the specified Slack channel

## Setup

### Prerequisites

- Python 3.10 or higher
- MCP Agent framework
- GitHub MCP Server
- Slack MCP Server
- Access to a GitHub repository
- Access to a Slack workspace

### Installation

1. Install dependencies:
```
uv sync --dev
```

2. Create a secrets file:
```
cp mcp_agent.secrets.yaml mcp_agent.secrets.yaml
```

3. Update the secrets file with your API keys

### Usage

Run the application with:
```
uv run main.py --owner <github-owner> --repo <repository-name> --channel <slack-channel>
```
