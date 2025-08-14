Factory examples

This folder demonstrates loading AgentSpecs from multiple sources and composing workflows using the `mcp_agent.workflows.factory` helpers.

What's included:

- agents.yaml: simple YAML agents
- mcp_agent.config.yaml: enables auto-loading subagents from inline definitions and directories
- mcp_agent.secrets.yaml.example: template for API keys
- load_and_route.py: loads from a file and routes via LLM
- auto_loaded_subagents.py: relies on app config to discover subagents (Claude-style markdown and others)
- orchestrator_demo.py: shows the Orchestrator workflow
- parallel_demo.py: shows a parallel fan-out/fan-in workflow

Quick start:

1. Copy `mcp_agent.secrets.yaml.example` to `mcp_agent.secrets.yaml` and fill in keys
2. Run an example, e.g.:

```bash
python examples/workflows/factory/load_and_route.py
```

3. To try auto-loaded subagents, add markdown agents to `.claude/agents` or `.mcp-agent/agents` (project or home) or use the inline examples in the config.

Notes:

- Claude-style agents store `tools:` in front matter. For now, these are mapped to `server_names`. TODO: represent tools distinctly from MCP servers.
- App agent loading deduplicates by name and enforces precedence: inline > earlier search_paths > later search_paths.
