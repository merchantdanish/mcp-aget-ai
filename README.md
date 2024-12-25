# silsila (mcp-agent)

An agent framework built on MCP and Temporal for distributed AI applications

## Introduction

This framework aims to make it straightforward to orchestrate complex AI workflows that rely on MCP (Model Context Protocol) servers for:

- **Tools**: Model-controlled functions like reading documents, calling APIs, or performing custom logic.
- **Prompts**: Templated interactions or question prompts exposed by servers.
- **Memory**: Persistent or semi-persistent data accessible through an MCP-compliant memory server.
- **LLM Completions**: Interfacing with large language models for planning or generating text.

We leverage **Temporal** to provide durable execution, parallelization, and the ability to pause or resume workflows. The code includes a developer-friendly dev mode for local iteration and a production path using Temporal for large-scale, resilient deployments.

## Key Principles

1. **MCP-First**  
   All memory, prompts, tools, and even workflows are accessed via MCP calls. This ensures consistent usage patterns and the ability to switch underlying servers without changing workflow logic.

2. **Agents as Aggregators**  
   Each **Agent** encapsulates multiple MCP servers and exposes typed methods for LLM completions, memory operations, and tool invocations. You never call raw MCP tools by name—instead, you invoke higher-level agent methods. Multiple agents can collaborate in a single workflow (e.g., one “planner” agent, one “worker” agent).

3. **Planner and Patterns**  
   A **Planner** class handles LLM-driven decision-making (planning next actions). It returns structured directives (e.g., _“Use tool X,” “Request human input,” “Done”_). Common workflow patterns—like a “swarm” approach that repeatedly calls the planner—are encapsulated in base workflow classes such as **SwarmWorkflowBase**.

4. **Temporal Integration**  
   By defining an **OrchestratorInterface**, the same workflows can run in dev mode (synchronously in your local environment) or production mode (as **Temporal** workflows). Human-in-the-loop signals and progress notifications are readily handled in both modes.

5. **Notifications and Human Input**  
   Workflows can send progress or message notifications back to the upstream MCP client with `notifications/progress` or `notifications/message`. They can also pause to wait for human input, which the user provides through an MCP tool call like `tools/call(provide_human_input)`.

6. **Developer-Focused**  
   The code aims to be readable and extensible, providing a straightforward mental model of how MCP orchestrations can be organized, tested, and extended for real-world AI pipelines.

## High-Level Architecture

1. **Agents (`MCPAggregator`)**

   - Wrap multiple MCP servers.
   - Provide typed methods (e.g., `agent.llm.complete_text()` or `agent.memory.get()`).

2. **Planner**

   - Uses an LLM-based approach to decide the next action.
   - Returns a structured `Directive` that might say “use tool”, “request human input,” or “done.”

3. **OrchestratorInterface**

   - Abstracts away the difference between dev (local) orchestration and production (Temporal) orchestration.
   - Methods like `run_workflow()`, `send_message()`, `send_progress()`, and `wait_for_human_input()` unify these environments.

4. **Base Workflow Classes**

   - E.g., `SwarmWorkflowBase` implements a loop calling the Planner repeatedly, following the returned directives.
   - Developers can subclass to introduce domain-specific logic.

5. **MCP Server**
   - Optionally, this framework can itself be exposed as an MCP server, allowing upstream clients to call `tools/call(run_workflow)` to start a workflow.
   - Human input is gathered by the upstream client calling `tools/call(provide_human_input)` on the orchestrator server.
