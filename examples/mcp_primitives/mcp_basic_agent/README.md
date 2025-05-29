# MCP Primitives Example: Using Resources

This example demonstrates how to use **MCP primitives**—specifically, **resources**—in an agent application. It shows how to connect to a custom MCP server that exposes structured resources, list and access those resources, and use them as context for an LLM-powered agent.

---

## What are MCP Primitives?

MCP (Model Context Protocol) primitives are standardized building blocks for agent applications. The two most important primitives are:

- **Resources**: Structured data (files, documents, datasets, status endpoints, etc.) exposed by an MCP server, accessible via URIs.
- **Prompts**: (Not demonstrated in this example) Standardized prompt templates that can be listed and invoked from an MCP server.

This example focuses on **resources**.

---

## Example Overview

- **resource_demo_server.py** implements a simple MCP server that exposes several resources:
  - `demo://docs/readme`: A sample README file (Markdown)
  - `demo://config/settings`: Example configuration settings (JSON)
  - `demo://data/users`: Example user data (JSON)
  - `demo://status/health`: Dynamic server health/status info (JSON)

- **main.py** shows how to:
  1. Connect an agent to the resource-demo MCP server
  2. List all available resources
  3. Use an LLM (OpenAI) to summarize the content of selected resources by passing their URIs as context

---

## Architecture

```plaintext
┌────────────────────┐
│  resource-demo     │
│  MCP Server        │
│  (resources)       │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Agent (Python)    │
│  + LLM (OpenAI)    │
└─────────┬──────────┘
          │
          ▼
   [User/Developer]
```

---

## 1. Setup

Clone the repo and navigate to this example:

```bash
git clone https://github.com/lastmile-ai/mcp-agent.git
cd mcp-agent/examples/mcp_primitives/mcp_basic_agent
```

Install dependencies (using [uv](https://docs.astral.sh/uv/) or pip):

```bash
pip install uv
uv pip install -r requirements.txt
```

---

## 2. Start the Resource Demo MCP Server

In one terminal, start the resource-demo server:

```bash
uv run resource_demo_server.py
```

This will launch an MCP server exposing the demo resources.

---

## 3. Configure API Keys

Edit `main.py` and set your OpenAI and/or Anthropic API keys in the `Settings` section.

---

## 4. Run the Agent Example

In a new terminal, run the agent script:

```bash
uv run main.py
```

You should see logs showing:
- The agent connecting to the resource-demo server
- Listing available resources
- Using the LLM to summarize the content of selected resources

---

## How it Works

- The agent connects to the resource-demo MCP server and calls `list_resources()` to discover available resources.
- It selects specific resource URIs (e.g., README and user data) and passes them as context to the LLM.
- The LLM receives the actual content of those resources and generates a summary.

---

## Extending

You can add your own resources to `resource_demo_server.py` using the `@mcp.resource` decorator. Any function can expose a resource—static or dynamic.

---

## References

- [Model Context Protocol (MCP) Introduction](https://modelcontextprotocol.io/introduction)
- [MCP Agent Framework](https://github.com/lastmile-ai/mcp-agent)
- [MCP Server Primitives](https://modelcontextprotocol.io/specification#primitives)

---

This example is a minimal, practical demonstration of how to use **MCP resources** as first-class context for agent applications.
