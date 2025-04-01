# mcp-agent Framework

<img src="https://github.com/user-attachments/assets/6f4e40c4-dc88-47b6-b965-5856b69416d2" alt="Logo" width="300" />

*Build effective agents with Model Context Protocol using simple, composable patterns.*

[![PyPI version](https://img.shields.io/pypi/v/mcp-agent?color=%2334D058&label=pypi)](https://pypi.org/project/mcp-agent/)
[![Discord](https://shields.io/discord/1089284610329952357)](https://lmai.link/discord/mcp-agent)
[![Downloads](https://img.shields.io/pepy/dt/mcp-agent?label=pypi%20%7C%20downloads)](https://img.shields.io/pepy/dt/mcp-agent?label=pypi%20%7C%20downloads)
[![License](https://img.shields.io/pypi/l/mcp-agent)](https://github.com/lastmile-ai/mcp-agent/blob/main/LICENSE)

## Overview

**`mcp-agent`** is a simple, composable framework to build agents using [Model Context Protocol](https://modelcontextprotocol.io/introduction).

**Inspiration**: Anthropic announced 2 foundational updates for AI application developers:

1. [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) - a standardized interface to let any software be accessible to AI assistants via MCP servers.
2. [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - a seminal writeup on simple, composable patterns for building production-ready AI agents.

`mcp-agent` puts these two foundational pieces into an AI application framework:

1. It handles the pesky business of managing the lifecycle of MCP server connections so you don't have to.
2. It implements every pattern described in Building Effective Agents, and does so in a _composable_ way, allowing you to chain these patterns together.
3. **Bonus**: It implements [OpenAI's Swarm](https://github.com/openai/swarm) pattern for multi-agent orchestration, but in a model-agnostic way.

Altogether, this is the simplest and easiest way to build robust agent applications.

## Core Components

The following are the building blocks of the mcp-agent framework:

- **MCPApp**: global state and app configuration
- **MCP server management**: `gen_client` and `MCPConnectionManager` to easily connect to MCP servers
- **Agent**: An Agent is an entity that has access to a set of MCP servers and exposes them to an LLM as tool calls. It has a name and purpose (instruction).
- **AugmentedLLM**: An LLM that is enhanced with tools provided from a collection of MCP servers. Every Workflow pattern is an `AugmentedLLM` itself, allowing you to compose and chain them together.

## Workflow Patterns

mcp-agent provides implementations for every pattern in Anthropic's [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents), as well as the OpenAI Swarm pattern. Each pattern is model-agnostic and exposed as an `AugmentedLLM`, making everything very composable.

### AugmentedLLM
An LLM that has access to MCP servers and functions via Agents. Provides `generate`, `generate_str`, and `generate_structured` methods, plus memory for tracking history.

### Parallel
Fan-out tasks to multiple sub-agents and fan-in the results. Each subtask is an AugmentedLLM, as is the overall Parallel workflow.

### Router
Given an input, route to the `top_k` most relevant categories. A category can be an Agent, an MCP server or a regular function.

### Intent-Classifier
Identifies the `top_k` Intents that most closely match a given input.

### Evaluator-Optimizer
One LLM (the "optimizer") refines a response, another (the "evaluator") critiques it until a response exceeds a quality criteria.

### Orchestrator-Workers
A higher-level LLM generates a plan, then assigns them to sub-agents, and synthesizes the results. Automatically parallelizes steps that can be done in parallel.

### Swarm
OpenAI's experimental multi-agent pattern for complex task decomposition and delegation.

## Example Usage

Here is a basic "finder" agent that uses the fetch and filesystem servers to look up a file, read a blog and write a tweet:

```python
import asyncio
import os

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

app = MCPApp(name="hello_world_agent")

async def example_usage():
    async with app.run() as mcp_agent_app:
        logger = mcp_agent_app.logger
        # This agent can read the filesystem or fetch URLs
        finder_agent = Agent(
            name="finder",
            instruction="""You can read local files or fetch URLs.
                Return the requested information when asked.""",
            server_names=["fetch", "filesystem"], # MCP servers this Agent can use
        )

        async with finder_agent:
            # Automatically initializes the MCP servers and adds their tools for LLM use
            tools = await finder_agent.list_tools()
            logger.info(f"Tools available:", data=tools)

            # Attach an OpenAI LLM to the agent (defaults to GPT-4o)
            llm = await finder_agent.attach_llm(OpenAIAugmentedLLM)

            # This will perform a file lookup and read using the filesystem server
            result = await llm.generate_str(
                message="Show me what's in README.md verbatim"
            )
            logger.info(f"README.md contents: {result}")

            # Uses the fetch server to fetch the content from URL
            result = await llm.generate_str(
                message="Print the first two paragraphs from https://www.anthropic.com/research/building-effective-agents"
            )
            logger.info(f"Blog intro: {result}")

            # Multi-turn interactions by default
            result = await llm.generate_str("Summarize that in a 128-char tweet")
            logger.info(f"Tweet: {result}")

if __name__ == "__main__":
    asyncio.run(example_usage())
```

Example configuration file (`mcp_agent.config.yaml`):

```yaml
execution_engine: asyncio
logger:
  transports: [console]  # You can use [file, console] for both
  level: debug
  path: "logs/mcp-agent.jsonl"  # Used for file transport

mcp:
  servers:
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]
    filesystem:
      command: "npx"
      args:
        [
          "-y",
          "@modelcontextprotocol/server-filesystem",
          "<add_your_directories>",
        ]

openai:
  # Secrets (API keys, etc.) are stored in an mcp_agent.secrets.yaml file which can be gitignored
  default_model: gpt-4o
```

## Advanced Features

- **Composability**: Chain multiple workflows together, such as using an Evaluator-Optimizer as the planner inside an Orchestrator
- **Signaling and Human Input**: Pause/resume tasks and handle user input mid-workflow
- **App Config**: Configuration via YAML files for logging, execution, LLM providers, and MCP servers
- **MCP Server Management**: Easy lifecycle management of MCP server connections

## Why use mcp-agent?

There are too many AI frameworks out there already. But `mcp-agent` is the only one that is purpose-built for a shared protocol - [MCP](https://modelcontextprotocol.io/introduction). It is also the most lightweight, and is closer to an agent pattern library than a framework.

As [more services become MCP-aware](https://github.com/punkpeye/awesome-mcp-servers), you can use mcp-agent to build robust and controllable AI agents that can leverage those services out-of-the-box.

## FAQs

### Do you need an MCP client to use mcp-agent?

No, you can use mcp-agent anywhere, since it handles MCPClient creation for you. This allows you to leverage MCP servers outside of MCP hosts like Claude Desktop.

Here's all the ways you can set up your mcp-agent application:

#### MCP-Agent Server

You can expose mcp-agent applications as MCP servers themselves, allowing MCP clients to interface with sophisticated AI workflows using the standard tools API of MCP servers. This is effectively a server-of-servers.

#### MCP Client or Host

You can embed mcp-agent in an MCP client directly to manage the orchestration across multiple MCP servers.

#### Standalone

You can use mcp-agent applications in a standalone fashion (i.e. they aren't part of an MCP client). Most examples are standalone applications.

# MCP Agent Cloud: Design Document

## Executive Summary

MCP Agent Cloud is a platform enabling deployment, management, and orchestration of AI agent applications leveraging the Model Context Protocol (MCP). It combines an enterprise workplace productivity platform (AI "App Store") with a developer-centric agent platform focused on MCP integration.

The platform builds on the success of the `mcp-agent` framework described above and aims to create a standardized, interoperable ecosystem for AI agents, workflows, and tools.

## Core Product

MCP Agent Cloud is a platform enabling deployment, management, and orchestration of AI agent applications leveraging the Model Context Protocol (MCP). The key capabilities include:

### Server Deployment and Endpoint Management
- Deploy secure MCP servers with authentication, accessible via HTTP endpoints.
- Deploy and manage MCPApps (built with mcp-agent), each comprising collections of AI agents and workflow tools exposed as MCP servers themselves.
- Platform-level APIs for managing deployed MCP servers:
  - `/servers/<server_name>/{call_tool, read_resource, …<other MCP APIs>}`
  - `/agents/<agent_name>/{run, pause, resume, stop}`
  - `/workflows/<workflow_name>/{run, pause, resume, stop}`

### Workflow Orchestration
- Execute workflows with robust control (run, pause, resume).
- Allow human-in-the-loop interactions (i.e. pause/resume with additional input) for workflow runs.
- State persistence for paused workflows using Temporal, enabling seamless resumption with user input.

### Observability and Tracing
- Tracing of MCPApp runs for observability, debugging and introspection.

### User Experience
- MCP Agent UX Console:
  - App Dashboard to see deployed MCP servers and applications.
  - App Store to discover and use MCP agents
  - Track workflow executions, view history, manage paused states, and resume workflows.

### Developer CLI Integration
- Quick commands for deploying (`mcp-agent deploy`) and managing (`mcp-agent list`) MCP applications and servers.

## mcp-agent Framework: Conceptual Foundation

MCP Agent Cloud builds on the powerful `mcp-agent` framework, which provides a streamlined way to build agents with LLMs and Model Context Protocol (MCP) servers. Understanding the key concepts of `mcp-agent` is essential for grasping the architecture of MCP Agent Cloud.

### Key Concepts in Hierarchical Order

#### MCPApp
The main encapsulation of an mcp-agent application, similar to how `app = FastAPI()` encapsulates a FastAPI server. An MCPApp is the top-level deployment unit that represents a complete AI application composed of agents, workflows, and MCP servers.

#### Agent
A configuration of a collection of MCP servers with a "name" and "instruction." An agent has:
- A purpose (e.g., "find files on filesystem")
- Access to specific MCP servers (e.g., "filesystem_mcp")
- An instruction/system prompt (e.g., "You are the Sherlock Holmes of finding files. Do your best")

#### AugmentedLLM
The base building block of agentic workflows. In its simplest form, it's an Agent connected to an LLM that runs in a loop until reaching max_iterations or satisfying an objective. "Augmented" refers to the LLM having access to tools and resources to do its job.

All workflow patterns in mcp-agent are derivatives of the AugmentedLLM type, allowing for maximum composability:
- **OpenAIAugmentedLLM**: Compatible with OpenAI API
- **AnthropicAugmentedLLM**: Works with Claude models
- **ParallelLLM**: Fan-out tasks to multiple sub-agents and fan-in results
- **RouterLLM**: Route to the most relevant categories based on input
- **EvaluatorOptimizerLLM**: Has two AugmentedLLMs to generate, critique, and refine responses
- **OrchestratorLLM**: Has planner and worker AugmentedLLMs

Importantly, workflows are not separate deployable components - they materialize as part of an MCPApp's configuration and execution.

#### Executor
The execution engine (asyncio or a workflow orchestrator like Temporal) that handles signal handling, pause/resume functionality, etc.

### The "Everything is an MCP Server" Principle

A fundamental principle of MCP Agent Cloud is that **everything is an MCP server**. This means:

1. All deployable components (MCPApps, utility MCP servers) are exposed as MCP servers
2. Components communicate with each other using the standardized MCP protocol
3. Deployment, management, and orchestration all operate on MCP servers

This principle drives our design decisions, including how we structure our CLI commands. Rather than having separate conceptual models for MCPApps and utility servers, we recognize that MCPApps deployed to MCP Agent Cloud are themselves implemented as MCP servers.

## 1. Vision and Core Offering

### 1.1 Core Vision

The vision for MCP Agent Cloud is to create an end-to-end platform that combines two complementary offerings:

1. **Enterprise Workplace Productivity Platform**
   - A centralized "AI App Store" within your organization
   - Enables employees to easily discover, deploy, and manage pre-built AI agents
   - Tailored for common productivity tasks (inspired by platforms like Writer, Contextual, and Cohere North)

2. **Developer-Centric Agent Platform**
   - Comprehensive tools and infrastructure for developers
   - Build, test, and deploy custom AI agents for internal use or external customers
   - Similar to platforms like Langgraph, but focused specifically on MCP integration

### 1.2 Key Capabilities

- **Server Deployment and Endpoint Management**
  - Deploy secure MCP servers with authentication
  - Accessible via HTTP endpoints
  - Deploy and manage MCPApps (built with mcp-agent)
  - APIs: `/servers/<server_name>/{call_tool, read_resource, …}`
  - APIs: `/agents/<agent_name>/{run, pause, resume, stop}`
  - APIs: `/workflows/<workflow_name>/{run, pause, resume, stop}`

- **Workflow Orchestration**
  - Execute workflows with robust control (run, pause, resume)
  - Allow human-in-the-loop interactions for workflow runs
  - Pause/resume workflows with additional input

- **Observability and Tracing**
  - Tracing of MCPApp runs for observability, debugging and introspection
  - Performance monitoring and error reporting

- **User Experience**
  - MCP Agent UX Console with App Dashboard
  - App Store to discover and use MCP agents
  - Track workflow executions, view history, manage paused states
  - Resume workflows with user input

- **Developer CLI Integration**
  - Commands for deploying (`mcp-agent deploy`)
  - Commands for managing (`mcp-agent list`) MCP applications and servers

### 1.3 Strategic Rationale

There are two key insights that shape this product direction:

1. **Timing**: 
   - MCP is rapidly emerging as an industry standard
   - Early mover advantage to establish the platform as the go-to solution
   - Making everything MCP-compatible creates differentiation in the market
   - Standardization helps with wider community acceptance

2. **Product Shape**:
   - Lessons from Langchain show selling a developer platform alone is difficult
   - Langgraph has more success, but still trails more vertical product suites
   - Code is no longer a bottleneck thanks to AI
   - Combining vertical (AI app store) with horizontal flexibility gives leverage
   - Strong organic pull for mcp-agent indicates market appetite

## 2. Core Technical Concepts

### 2.1 MCP-Agent Framework Overview

MCP-agent is a streamlined way of building agents with LLMs and Model Context Protocol (MCP) servers. The key components include:

- **MCPApp**: Global state and app configuration
- **MCP Server Management**: Tools to easily connect to MCP servers
- **Agent**: An entity with access to MCP servers that exposes them to an LLM
- **AugmentedLLM**: An LLM enhanced with tools from MCP servers

### 2.2 MCP Server Types

There are two distinct types of MCP servers that require different containerization approaches:

1. **Networked MCP Servers**
   - Expose HTTP endpoints natively
   - Follow MCP protocol over HTTP
   - Examples: cloud-based services, custom web services implementing MCP
   - Containerization: Package with dependencies, configure networking, set up authentication

2. **STDIO-based MCP Servers**
   - Command-line tools following MCP protocol
   - Communicate via stdin/stdout
   - Examples: local utilities, file system tools, simple MCP servers
   - Containerization: Require a wrapper service (HTTP-to-STDIO adapter)
   - Container runs both wrapper and STDIO-based server

### 2.3 Workflow Patterns Within MCPApps

Within MCPApps, the framework supports multiple workflow patterns, each implemented as a derivative of the AugmentedLLM type:

- **AugmentedLLM**: Base building block of agentic workflows - an LLM with access to MCP servers and functions
- **Parallel**: Fan-out tasks to multiple sub-agents and fan-in results, useful for tasks requiring multiple perspectives
- **Router**: Route to the most relevant categories based on input, allowing for targeted processing
- **Evaluator-Optimizer**: One LLM refines, another critiques until quality criteria are met
- **Orchestrator-Workers**: Higher-level LLM generates plan, assigns tasks, synthesizes results
- **Swarm**: Multi-agent pattern for complex task decomposition and delegation

These workflow patterns exist within the context of an MCPApp and are not separate deployable components. Each workflow type inherits from AugmentedLLM, allowing them to be composable (workflows can contain other workflows).

## 3. System Architecture

### 3.1 High-Level Architecture

The system is structured as a layered architecture:

1. **Foundation Layer**
   - Core mcp-agent framework
   - MCP server connections and lifecycle management
   - Agent configuration and LLM integration

2. **Orchestration Layer**
   - Deployment and management of MCP servers
   - Workflow execution and monitoring
   - State management and persistence

3. **Interface Layer**
   - Developer CLI for deployment and management
   - Web console for monitoring and management
   - API endpoints for programmatic access

### 3.2 Key Components

#### 3.2.1 MCPApp and Workflow Management

```python
class MCPAppManager:
    """Manages MCPApps and their workflow configurations."""
    
    def __init__(self, auth_service=None):
        self.auth_service = auth_service or AuthService()
        self.api_base_url = os.environ.get("MCP_AGENT_CLOUD_API_URL", "https://api.mcp-agent-cloud.example.com")
        self.manifest_file = "mcp_agent.config.yaml"
        
    async def deploy_mcpapp(self, app_dir, name=None, region="us-west"):
        """Deploy an MCPApp as an MCP server to the cloud."""
        # Ensure authentication
        is_auth, auth_error = await self.auth_service.ensure_authenticated()
        if not is_auth:
            return False, f"Authentication error: {auth_error}", None
        
        # Package the app and its workflows
        is_packaged, package_error, package_data = await self.package_app(app_dir)
        if not is_packaged:
            return False, f"Packaging error: {package_error}", None
        
        # Use directory name as default name
        if not name:
            name = app_dir.name
        
        # Deploy the MCPApp as an MCP server
        deployment_info = await self._upload_and_deploy(name, package_data, region)
        if not deployment_info:
            return False, "Deployment failed", None
        
        return True, None, deployment_info
```

#### 3.2.2 Server Deployment System

```python
class MCPServerDeploymentManager:
    def __init__(self, config=None):
        self.config = config
        self.container_service = ContainerService()
        self.auth_service = AuthService()
        self.registry = ServerRegistry()
    
    async def deploy_server(self, server_name, server_type, config=None):
        # Create containerized MCP server
        container_config = self._get_container_config(server_type, config)
        container_id, port = await self.container_service.create_container(container_config)
        
        # Configure authentication
        auth_config = await self.auth_service.create_server_auth(server_name)
        
        # Register server in registry
        server_record = {
            "id": f"srv-{uuid.uuid4().hex[:8]}",
            "name": server_name,
            "type": server_type,
            "container_id": container_id,
            "endpoint": f"/servers/{server_name}",
            "port": port,
            "url": f"https://{server_name}.mcp-agent-cloud.example.com",
            "local_url": f"http://localhost:{port}",
            "auth_config": auth_config,
            "created_at": datetime.now(timezone.utc),
            "status": "running"
        }
        await self.registry.register_server(server_record)
        
        return server_record
```

#### 3.2.3 STDIO-to-HTTP Adapter

```python
class StdioAdapter:
    def __init__(self, command, args):
        self.process = None
        self.command = command
        self.args = args
        self.id = str(uuid.uuid4())
        
    async def start(self):
        """Start the STDIO process"""
        self.process = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
    async def handle_request(self, request):
        """Handle HTTP request by passing to STDIO process"""
        if not self.process:
            await self.start()
            
        data = await request.json()
        request_json = json.dumps(data) + "\n"
        
        # Write to stdin
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        # Read from stdout
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode())
        
        return web.json_response(response)
```

#### 3.2.3 Container Service

```python
class ContainerService:
    def __init__(self):
        self.client = docker.from_env()
        
    async def build_stdio_container(self, server_name, command, args):
        """Build a container for STDIO-based MCP server"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create Dockerfile for STDIO adapter + server
            with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
                f.write(f"""
                FROM python:3.10-slim
                
                # Install server dependencies
                RUN apt-get update && apt-get install -y nodejs npm
                
                # Set up adapter
                WORKDIR /app
                COPY stdio_adapter.py .
                COPY server.py .
                
                # Install adapter requirements
                RUN pip install aiohttp
                
                # Set server command
                ENV SERVER_COMMAND="{command}"
                ENV SERVER_ARGS="{' '.join(args)}"
                
                EXPOSE 8000
                CMD ["python", "server.py"]
                """)
            
            # Copy adapter files
            with open(os.path.join(tmpdir, "stdio_adapter.py"), "w") as f:
                # Copy stdio_adapter.py content here
                pass
                
            with open(os.path.join(tmpdir, "server.py"), "w") as f:
                # Write simple aiohttp server that uses adapter
                pass
            
            # Build the container
            image, logs = self.client.images.build(
                path=tmpdir, 
                tag=f"mcp-server-{server_name}:latest"
            )
            
            return image.id
```

#### 3.2.4 Deployment Services

Following the "everything is an MCP server" principle, we have specialized deployment services:

##### 3.2.4.1 MCPApp Deployment Service

```python
class AppDeploymentService:
    """Service for deploying MCPApps as MCP servers to the cloud."""
    
    def __init__(self, auth_service=None):
        self.auth_service = auth_service or AuthService()
        self.api_base_url = os.environ.get("MCP_AGENT_CLOUD_API_URL", "https://api.mcp-agent-cloud.example.com")
        
    async def deploy_app(self, app_dir, name=None, region="us-west"):
        """Deploy an MCPApp as an MCP server to the cloud."""
        # Ensure authentication
        is_auth, auth_error = await self.auth_service.ensure_authenticated()
        if not is_auth:
            return False, f"Authentication error: {auth_error}", None
            
        # Package the app (includes all workflow configurations within the app)
        is_packaged, package_error, package_data = await self.package_app(app_dir)
        if not is_packaged:
            return False, f"Packaging error: {package_error}", None
            
        # Use directory name as default name
        if not name:
            name = app_dir.name
            
        # Deploy the MCPApp as a server
        deployment_info = await self._upload_and_deploy(name, package_data, region)
        if not deployment_info:
            return False, "Deployment failed", None
            
        return True, None, deployment_info
        
    async def package_app(self, app_dir):
        """Package an MCPApp for deployment, including all its workflows and configurations."""
        try:
            # Check for configuration files
            config_file = app_dir / "mcp_agent.config.yaml"
            if not config_file.exists():
                return False, f"Configuration file not found at {config_file}", None
                
            # Read the configuration to identify workflow patterns and agent configurations
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                
            # Package all the application code, including workflow implementations
            package_data = {
                "config": config,
                "code": await self._package_code(app_dir),
                "requirements": await self._get_requirements(app_dir),
                "workflow_patterns": self._extract_workflow_patterns(config)
            }
            
            return True, None, package_data
        except Exception as e:
            return False, f"Error packaging MCPApp: {str(e)}", None
            
    def _extract_workflow_patterns(self, config):
        """Extract workflow pattern configurations from the MCPApp config."""
        workflow_patterns = []
        
        # Look for AugmentedLLM derivatives in the configuration
        if "workflows" in config:
            for wf_name, wf_config in config["workflows"].items():
                if "type" in wf_config:
                    workflow_patterns.append({
                        "name": wf_name,
                        "type": wf_config["type"],  # e.g., "parallel", "router", "evaluator-optimizer"
                        "config": wf_config
                    })
        
        return workflow_patterns
```

#### 3.2.5 Workflow Orchestration

```python
class TemporalWorkflowExecutor:
    def __init__(self, temporal_client):
        self.client = temporal_client
    
    async def execute_workflow(self, workflow_id, workflow_def, input_data):
        # Create a Temporal workflow
        execution = await self.client.start_workflow(
            workflow_id=workflow_id,
            workflow_def=workflow_def,
            input=input_data,
            task_queue="mcp-agent-workflows"
        )
        
        return execution.id
    
    async def signal_workflow(self, execution_id, signal_name, payload=None):
        # Send a signal to a running workflow
        await self.client.signal_workflow(
            execution_id=execution_id,
            signal_name=signal_name,
            payload=payload
        )
    
    async def get_workflow_status(self, execution_id):
        # Get current status of workflow
        return await self.client.describe_workflow(execution_id)
```

#### 3.2.6 Registry Service

```python
class ServerRegistry:
    def __init__(self, db_path="registry.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize the database if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS servers (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE,
            type TEXT,
            container_id TEXT,
            endpoint TEXT,
            config TEXT,
            status TEXT,
            created_at TEXT
        )
        ''')
        conn.commit()
        conn.close()
        
    def register_server(self, server_data):
        """Register a new server in the registry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO servers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                server_data["id"],
                server_data["name"],
                server_data["type"],
                server_data["container_id"],
                server_data["endpoint"],
                json.dumps(server_data.get("config", {})),
                server_data["status"],
                server_data["created_at"]
            )
        )
        conn.commit()
        conn.close()
```

## 4. User Interfaces

### 4.1 CLI Implementation

The CLI provides commands for deploying and managing MCP servers, apps, and workflows, with a unified deployment approach that reflects the "everything is an MCP server" principle:

```python
import os
import sys
import json
import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
from typing import List, Optional

app = typer.Typer(help="MCP Agent Cloud CLI")
console = Console()

# Command groups
deploy_app = typer.Typer(help="Deploy MCP servers")
app.add_typer(deploy_app, name="deploy")

list_app = typer.Typer(help="List deployed resources")
app.add_typer(list_app, name="list")

# MCP Server deployment command
@deploy_app.command()
def server(
    name: str = typer.Argument(..., help="Name of the MCP server"),
    server_type: str = typer.Option(..., "--type", "-t", help="Type of MCP server (fetch, filesystem, etc.)"),
    region: str = typer.Option("us-west", "--region", "-r", help="Deployment region"),
    public: bool = typer.Option(False, "--public", help="Make the server publicly accessible"),
):
    """Deploy a utility MCP server to the cloud."""
    token = authenticate()
    config = load_config()
    
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]Deploying {server_type} MCP server..."),
        transient=True,
    ) as progress:
        progress.add_task("deploy", total=None)
        # Run the deployment process
        server_record = asyncio.run(deploy_mcp_server(name, server_type, region, public))
    
    # Display deployment result
    console.print(Panel.fit(
        f"✅ Successfully deployed [bold]{name}[/bold] MCP server!\n\n"
        f"Server URL: [bold]{server_record['url']}[/bold]\n"
        f"Server ID: {server_record['id']}\n"
        f"Status: [bold green]{server_record['status']}[/bold green]",
        title="Deployment Complete",
        subtitle="View in console: https://console.mcp-agent-cloud.example.com"
    ))

# MCPApp deployment command
@deploy_app.command()
def app(
    directory: Path = typer.Argument(
        ".", 
        help="Directory containing the MCPApp code",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Name for the deployed MCPApp (defaults to directory name)"
    ),
    region: str = typer.Option(
        "us-west",
        "--region",
        "-r",
        help="Region to deploy the MCPApp to"
    ),
):
    """Deploy an MCPApp as an MCP server to the cloud."""
    # Absolute path for the directory
    directory_path = directory.resolve()
    
    # Default to directory name if no name provided
    if not name:
        name = directory_path.name
    
    # Display deployment information
    console.print(f"Deploying MCPApp as MCP server: [bold cyan]{name}[/bold cyan] from [bold]{directory_path}[/bold]")
    
    # Authenticate and deploy
    ensure_authenticated()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Deploying MCPApp as MCP server..."),
        transient=True,
    ) as progress:
        progress.add_task("deploy", total=None)
        
        # Deploy the MCPApp as an MCP server
        result, error, deployment_info = asyncio.run(deploy_mcpapp_as_server(directory_path, name, region))
    
    if not result:
        console.print(f"[bold red]Deployment failed:[/bold red] {error}")
        sys.exit(1)
    
    # Display deployment result
    console.print(Panel.fit(
        f"✅ Successfully deployed [bold]{name}[/bold] as MCP server!\n\n"
        f"Server URL: [bold]{deployment_info['url']}[/bold]\n"
        f"Server ID: {deployment_info['id']}\n"
        f"Status: [bold green]{deployment_info['status']}[/bold green]\n\n"
        f"This MCPApp is now available as an MCP server and can be accessed by other agents and applications.",
        title="Deployment Complete",
        subtitle="View in console: https://console.mcp-agent-cloud.example.com"
    ))
```

The CLI structure follows our "everything is an MCP server" principle. The deployment commands are structured to reflect the actual deployable components in the MCP ecosystem:

1. `mcp-agent deploy server`: Deploys a utility MCP server (e.g., fetch, filesystem)
2. `mcp-agent deploy app`: Deploys an MCPApp as an MCP server

Note that there is no separate "workflow" deployment command, as workflows are not separate deployable entities but rather patterns that materialize within MCPApps. Workflows are configured as part of the MCPApp definition and are deployed alongside the MCPApp.

This unified approach ensures consistency and reinforces the core principle that all components in the MCP Agent ecosystem are exposed as MCP servers that communicate using the standardized protocol.

### 4.2 Web Dashboard

The web dashboard provides a user interface for managing and monitoring MCPApps and utility MCP servers:

- **Overview**: High-level metrics and recent activity
- **Servers**: List of deployed utility MCP servers with status and management options
  - Lifecycle controls (start, stop, delete)
  - Dependency graphs showing which apps use each server
  - Monitoring and usage metrics
  
- **Apps**: Deployed MCPApps with their status, configuration, and execution controls
  - Within each MCPApp view, users can see the agents and workflow patterns configured in the app
  - Users can monitor and manage running workflow executions, including pause/resume functionality
  - State inspection for paused workflows with user input capability
  - Deployment and versioning history
  
- **App Store**: Marketplace for discovering and deploying pre-built MCPApps and utility servers
  - Publishing workflow for making apps and servers public
  - Usage metrics and adoption statistics for published components
  - Versioning and dependency tracking

## 5. User Stories and Workflows

### 5.1 Target Personas

MCP Agent will eventually serve two main personas:

1. **Producer (P)**: Agent developer who develops agentic workflows for their own use or for public use
2. **Consumer (C)**: User looking to run an MCP Agent Server on ad-hoc basis, or a Producer who needs public agents to enhance their own agentic workflow

### 5.2 Key User Stories

#### 5.2.1 Producer (P) User Stories

**P0 Priority Stories:**

- As an agent developer, I want to access clear, clean, and concise documentation on getting started, building, and deploying MCP Agents so I can quickly understand the necessary steps and requirements to achieve my goal of building an agentic workflow.

- As an agent developer, I want to run my MCP Agent locally using a simple terminal command (e.g., mcp-agent run) so I can test its functionality and debug issues in my own environment to achieve confidence in the agent's behavior before deployment.

- As an agent developer, I want to deploy my MCP Agent using a secure terminal command (e.g., mcp-agent deploy) requiring authentication so I can push my validated agent code to the cloud to achieve making my agent operational in the remote managed environment.

- As an agent developer, I want to receive deployment status updates and a direct link to the management portal within my terminal so I can immediately know the outcome of the deployment and easily access the monitoring interface to achieve a seamless transition from deployment to management.

- As an agent developer, I want to connect my mcp-agent server to an external telemetry provider of my choice so I can monitor my servers performance in my preferred and approved environment to achieve centralized control and visibility over my agents.

**P1 Priority Stories:**

- As an agent developer, I want to view a marketplace of public MCP Servers with clear descriptions, usage metrics, and verifiable public URLs so I can evaluate and select suitable servers for integration to achieve a high quality bar for my agentic workflow.

- As an agent developer, I want to easily copy URLs of selected MCP Servers from the marketplace so I can obtain the necessary connection details without manual transcription to achieve accurate configuration in my mcp-agent code.

- As an agent developer, I want to log into a web portal using my credentials so I can access management and monitoring features for my deployments to achieve centralized control and visibility over my agents.

- As an agent developer, I want to view detailed health metrics, observability traces, and logs for my specific deployment within the portal so I can monitor performance, diagnose issues, and understand operational status to achieve reliable and performant agent operation.

**P2 Priority Stories:**

- As an agent developer, I want to easily copy my agent's deployment URL via dedicated buttons within the portal so I can quickly retrieve the endpoint needed for integration to achieve connecting client applications to my deployed agent.

- As an agent developer, I want to have a clear 'publish' option for my deployed agent within the portal and provide a name, description, optional icon, and example code when publishing my agent so I can initiate the process of making my agent publicly discoverable to achieve sharing my creation with the wider community and potentially get social credit and/or charge per request.

- As an agent developer, I want to see a clear visual indicator (e.g., a 'public' tag) next to my published agents in the deployment list so I can easily distinguish between my private and public deployments to achieve better management and overview of my agent portfolio.

- As an agent developer, I want to view usage metrics and trends specifically for my published public agent so I can understand its adoption and impact within the community to achieve insights for potential improvements or further development.

- As an agent developer, I want to be able to retire my public agents and delete them or bring them private with strict downstream dependency requirements so I can avoid damaging agents that depend on my servers and to achieve better resource management.

#### 5.2.2 Discovery Phase

**User Goal**: Get familiar with MCP agent framework and platform value

**Requirements:**
- Documentation Hub
- Document instructions to setup local dev environment
- Document instructions to install (Library, SDK, etc.)
- Document instructions for account creation/authentication
- Document getting started section with clear, concise code examples
- Document main concepts (MCP, MCP Agent, MCP Server Deployment, Publishing, etc.)
- Document CLI commands (list, run, deploy, etc.)
- Include CTA/URL to the MCP Marketplace
- Include CTA/URL to the cloud portal

#### 5.2.3 Local Development & Testing

**User Goal**: Build and test agentic workflows

**Requirements:**
- Local environment that mimics production environment
- CLI command to setup environment (mcp-agent build)
- CLI command to print logs (mcp-agent logs) - auto-print in failure modes
- CLI command to test/simulate the agentic workflow (mcp-agent run)
- Print metrics (time, tokens, cost, evals, etc.) with every agentic run in CLI
- Use https://charm.sh/ for Terminal App

#### 5.2.4 Deployment Phase

**User Goal**: Make workflows available in stable environment

**Requirements:**
- CLI deployment command (mcp-agent deploy)
- CLI prints deployment loading screen and status messages
- CLI prints success message and link to cloud portal
- Auto-print logs in failure modes

#### 5.2.5 Management & Monitoring

**User Goal**: Ensure health and performance of deployed agents

**Requirements:**
- Authentication for cloud portal
- Landing page with charts for high-level observability metrics across all deployments
- Clear indication between private and public deployments
- Detailed page for each deployment with observability and tracing views
- Display key health metrics (CPU Usage, Memory Usage, Request Count, Error Rate, Latency)
- Provide access to real-time or recent observability traces
- Provide access to deployment logs
- Display the Deployment URL and configuration details
- Workflow state inspection and resumption interface for paused workflows
- Clear dependency visualization showing relationships between components
- Clear CTA to publish a deployment to be public MCP server
- Path to decommission MCP agent servers (easy for private, stricter requirements for public)
- Namespacing indicators showing organization and server naming conventions
- Use https://ui.shadcn.com/ for Portal App

#### 5.2.6 Key User Story Implementation: MCPApp Deployment

**User Story**: As an agent developer, I want to deploy my MCPApp using a secure terminal command (e.g., `mcp-agent deploy app`) requiring authentication so I can push my validated agent code to the cloud, making it operational in the remote managed environment as an MCP server.

**Implementation**:
1. User authenticates using OAuth device authorization flow
2. User packages their MCPApp code using the CLI
   - This includes the application code, configuration, and any workflow patterns implemented within the app
   - The packaging process identifies AugmentedLLM derivatives and their configurations within the MCPApp
3. The packaged code is uploaded to the MCP Agent Cloud platform
4. The platform builds and deploys the MCPApp as an MCP server
5. The CLI displays deployment status and the server URL to the user

This implementation follows the "everything is an MCP server" principle, where MCPApps are deployed as MCP servers in our ecosystem. The workflows are not separate deployable components but rather patterns that materialize as part of the MCPApp's configuration and execution.

## 6. Implementation Plan

### 6.1 Initial MVP Approach (Weeks 1-2)

Focus on building a minimal viable product with core functionality:

1. **CLI Implementation**
   - Basic CLI skeleton with deploy and list commands
   - Configuration and authentication handling

2. **STDIO-to-HTTP Adapter**
   - Create adapter for STDIO-based MCP servers
   - Implement request/response handling

3. **Container Service**
   - Docker container building for MCP servers
   - Support for networked and STDIO-based servers

4. **Registry Service**
   - Simple database for tracking deployed servers
   - Basic querying capabilities

5. **Integration Testing**
   - End-to-end testing of deployment flow
   - Verification of server accessibility

### 6.2 Phase 2: Orchestration (Weeks 3-4)

1. **Workflow Orchestration**
   - Integration with Temporal for durable execution
   - Implementation of pause/resume functionality
   - State persistence for paused workflows

2. **Human-in-the-Loop Features**
   - Signal handling for workflow interruption
   - User input collection and integration
   - Secure storage of user-provided data during pause/resume cycles

3. **Observability Framework**
   - Logging and monitoring infrastructure
   - Trace collection for debugging
   - Structured event storage for workflow history

### 6.3 Phase 3: User Experience (Weeks 5-6)

1. **Web Console Development**
   - Dashboard for deployment management
   - Workflow visualization and control

2. **App Store Implementation**
   - Server and agent discovery
   - Publishing mechanism for developers

3. **Documentation System**
   - Comprehensive guides and references
   - Example applications and templates

## 7. Guiding Principles

The following principles should guide the development of MCP Agent Cloud:

1. **Minimalist design language**
   - Task-focused user interface that reduces noise
   - Beauty ingrained in simplicity

2. **Consistent language across surfaces**
   - Same terminology and simple commands from framework to SDK to portal
   - Smooth transition between different interfaces
   - Reduce confusion through consistency

3. **Progressive disclosure of information**
   - Don't throw all data at users at once
   - Show most critical information first
   - Let users take intended action to view more information as needed
   - Avoid landing users in views with too many options

4. **Transparency & control**
   - Let users control and configure how the system works
   - Provide clear messages on system status
   - Detailed error reporting and failure modes
   - Transparency about system transformations taken on behalf of users

## 8. Authentication and Security

### 8.1 Authentication and Authorization

MCP Agent Cloud implements a comprehensive authentication and authorization model addressing three fundamentally different authentication needs:

1. **Platform Authentication**: How developers authenticate with the MCP Agent Cloud platform to deploy and manage their agents
2. **Service Authentication**: How deployed agents authenticate with external services (like Gmail, Slack, etc.) on behalf of end users 
3. **Client-to-MCPApp Authentication**: How client applications authenticate with deployed MCPApps

These three authentication layers serve different purposes but work together seamlessly in a secure environment with consistent authentication patterns.

#### 8.1.1 Platform Authentication (Device Authorization Flow)

For the MCP Agent CLI, we implement the OAuth 2.0 Device Authorization Grant (RFC 8628) to provide a secure, headless authentication experience:

```python
class CLIAuthService:
    """Service for authenticating CLI commands with MCP Agent Cloud."""

    async def device_authorization_flow(self, device_code_callback=None):
        """Perform the OAuth 2.0 Device Authorization Grant flow."""
        # Request device code
        response = await client.post(
            self.device_code_endpoint,
            data={
                "client_id": "mcp-agent-cli",
                "scope": "deploy:server deploy:app deploy:workflow"
            }
        )
        
        # Extract device code data
        device_code = device_code_data["device_code"]
        user_code = device_code_data["user_code"]
        verification_uri = device_code_data["verification_uri"]
        
        # Display to user
        device_code_callback(user_code, verification_uri)
        
        # Poll for token
        while True:
            token_response = await client.post(
                self.token_endpoint,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": "mcp-agent-cli"
                }
            )
            
            if token_response.status_code == 200:
                # Authentication successful
                token_data = token_response.json()
                
                # Store tokens securely
                self.tokens = {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token", ""),
                    "expires_at": int(time.time()) + token_data.get("expires_in", 3600)
                }
                
                self._save_tokens()
                return True
```

This flow is particularly well-suited for CLI applications because:
1. No browser redirection is required on the device running the CLI
2. The user can authenticate on any device with a browser
3. The tokens are stored securely with appropriate permissions
4. Token refresh is handled automatically
5. The CLI can be used in headless environments

CLI Authentication Commands:
```
# Login to MCP Agent Cloud with device code flow
mcp-agent deploy auth login

# Check authentication status
mcp-agent deploy auth status 

# Log out and revoke tokens
mcp-agent deploy auth logout
```

#### 8.1.2 MCP Server Authentication

MCP Agent Cloud supports three authorization patterns for MCP servers:

1. **Self-Contained Authorization**: MCP server handles both authorization and authentication
   - Includes built-in identity provider
   - Simplified setup, ideal for development or internal-only deployments
   - Lower security boundary separation

2. **Third-Party OAuth Provider**: MCP server delegates to services like GitHub, Google, etc.
   - Complete separation of authentication concerns
   - Leverages established identity providers with strong security
   - Users don't need to create additional accounts

3. **Custom OAuth Provider**: Integration with enterprise identity solutions
   - Works with Stytch, Auth0, Okta, or custom enterprise IdP
   - Matches existing organizational security policies
   - Supports SSO and other enterprise auth patterns

#### 8.1.3 OAuth Provider Implementation

For MCP servers, we implement the OAuth flow with a structure like this:

```javascript
// MCP Server with GitHub OAuth
export default new OAuthProvider({
  apiRoute: "/sse",
  apiHandler: MCPServerRouter, 
  defaultHandler: GitHubOAuthHandler,
  authorizeEndpoint: "/authorize",
  tokenEndpoint: "/token",
  clientRegistrationEndpoint: "/register",
});
```

The standard MCP server authentication flow works as follows:

1. MCP client attempts to connect to MCP server
2. Server returns 401 Unauthorized response
3. Client opens browser with authorization URL
4. User authenticates with identity provider
5. Provider redirects back to MCP server with auth code
6. MCP client exchanges code for token
7. MCP client uses token for subsequent requests

#### 8.1.4 Secure Token Storage for Platform Authentication

For CLI platform authentication, tokens are stored securely on the local filesystem:

```python
def _save_tokens(self) -> None:
    """Save platform authentication tokens to storage."""
    try:
        with open(self.token_file, "w") as f:
            json.dump(self.tokens, f, indent=2)
            
        # Set secure permissions
        if os.name != 'nt':  # Not Windows
            os.chmod(self.token_file, 0o600)  # Read/write permissions for owner only
    except IOError as e:
        logger.error(f"Error saving tokens: {str(e)}")
```

This ensures:
1. Platform tokens are stored in a user-specific location (`~/.mcp-agent-cloud/auth_tokens.json`)
2. File permissions are set to read/write for owner only on Unix-like systems
3. Tokens are refreshed automatically before they expire
4. Revocation is properly handled during logout

#### 8.1.5 Service Authentication Management

MCP Agent Cloud provides a secure vault system for managing service authentication credentials:

```python
class ServiceCredentialManager:
    """Manages credentials for third-party services that agents need to access."""
    
    async def store_credential(self, agent_id: str, service_name: str, credentials: Dict[str, Any]) -> str:
        """Store service credentials securely in the credential vault."""
        # Generate a unique credential ID
        credential_id = f"cred-{uuid.uuid4().hex[:8]}"
        
        # Encrypt credentials before storage
        encrypted_creds = self._encrypt_credentials(credentials)
        
        # Store in secure credential store with proper access controls
        await self.credential_store.put(
            credential_id,
            {
                "agent_id": agent_id,
                "service_name": service_name,
                "encrypted_data": encrypted_creds,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
        return credential_id
        
    async def get_credential(self, credential_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve service credentials from the vault."""
        # Get encrypted credentials
        cred_data = await self.credential_store.get(credential_id)
        if not cred_data:
            return None
            
        # Decrypt credentials
        decrypted_creds = self._decrypt_credentials(cred_data["encrypted_data"])
        
        return decrypted_creds
```

This service credential system:
1. Securely stores credentials for third-party services (OAuth tokens, API keys, etc.)
2. Isolates credentials from agent code
3. Enforces access controls so agents can only access their own credentials
4. Supports credential rotation and revocation
5. Encrypts all sensitive data at rest

### 8.2 Container Security Model

MCP servers run user code and interact with external resources, creating several security concerns that MCP Agent Cloud must address:

#### 8.2.1 File System Security

MCP servers with filesystem access present security risks:

- **Implementation**: Use read-only mounts for most of the container
- **Access Control**: 
  - Define specific allowlisted directories for read/write access
  - Use volume mounts with strict permissions
  - Isolate file operations to prevent path traversal attacks

```yaml
# Configuration for filesystem access
filesystem:
  allowed_paths:
    - path: "/data"
      permissions: "rw"
    - path: "/templates"
      permissions: "r"
  denied_paths:
    - "/etc"
    - "/var"
    - "/home"
```

#### 8.2.2 Network Security

MCP servers may need network access to fulfill their purpose while maintaining security:

- **Egress Filtering**: Control which external services an MCP server can access
- **Network Policies**: Implement container-level network policies
- **Request Signing**: Require signed requests for privileged operations

```yaml
# Network security configuration
network:
  egress_allow:
    - domain: "api.github.com"
      ports: [443]
    - domain: "s3.amazonaws.com"
      ports: [443]
  egress_deny:
    - domain: "169.254.169.254"  # Block metadata services
    - domain: "internal.example.com"
```

#### 8.2.3 User Code Execution

When MCP servers execute user-provided code:

- **Process Isolation**: Run code in isolated sandbox environments
- **Resource Limits**: Enforce CPU, memory, disk, and network quotas
- **Timeout Controls**: Limit maximum execution time
- **Audit Logging**: Log all code execution for security review

```yaml
# User code execution constraints
execution:
  sandbox: "gvisor"  # or "firecracker", "docker"
  limits:
    cpu: "1.0"
    memory: "512Mi"
    disk: "1Gi"
    execution_time: "30s"
  allowed_languages:
    - "javascript"
    - "python"
```

### 8.3 Handling External Service Authentication

MCP servers often need to authenticate with external services (like Gmail or Slack) on behalf of users. This presents unique challenges:

#### 8.3.1 OAuth Token Management

For services using OAuth (most modern APIs):

- **Token Storage**: Securely store OAuth tokens using encryption at rest
- **Token Refresh**: Automatically handle token refresh flows
- **Scope Limitation**: Request minimal required scopes
- **Revocation Support**: Allow users to revoke access at any time

#### 8.3.2 Secret Isolation

Different approaches to isolate secrets:

1. **Per-User Secret Vaults**: 
   - Each user gets an isolated secret vault
   - Secrets never exposed to MCP server code directly
   - Access mediated through secure API calls

2. **Secret Proxying**:
   - External service calls proxied through secure intermediary
   - Original credentials never exposed to MCP server
   - Calls validated against allowed operations

#### 8.3.3 Implementation Example

The following pattern demonstrates securely handling Gmail authentication:

```python
class GmailSecureConnector:
    def __init__(self, secret_manager):
        self.secret_manager = secret_manager
        
    async def authenticate_user(self, user_id):
        # Direct user to Gmail OAuth consent screen
        auth_url = self._generate_auth_url(user_id)
        return auth_url
        
    async def handle_callback(self, code, user_id):
        # Exchange code for token
        token = await self._exchange_code(code)
        
        # Store encrypted token in user's secret vault
        await self.secret_manager.store(
            user_id=user_id,
            service="gmail",
            secret_type="oauth_token",
            value=token
        )
        
    async def make_authenticated_request(self, user_id, resource, method, data=None):
        # Retrieve token from secure storage
        token = await self.secret_manager.retrieve(
            user_id=user_id,
            service="gmail",
            secret_type="oauth_token"
        )
        
        # Make the authenticated request to Gmail API
        response = await self._make_api_call(
            token=token,
            resource=resource,
            method=method,
            data=data
        )
        
        return response
```

This approach ensures:
- User credentials are never directly exposed to MCP servers
- Tokens are securely stored and managed
- Access can be audited and revoked
- Service connections follow principle of least privilege

## 9. Technical Configuration

### 9.1 MCP-Agent Configuration

```yaml
# mcp_agent.config.yaml
execution_engine: asyncio  # or temporal for durable execution
logger:
  transports: [console]  # You can use [file, console] for both
  level: debug
  path: "logs/mcp-agent.jsonl"  # Used for file transport

mcp:
  servers:
    fetch:
      type: "networked"  # Uses HTTP endpoints directly
      command: "uvx"
      args: ["mcp-server-fetch"]
      
    filesystem:
      type: "stdio"  # Uses stdin/stdout
      command: "npx"
      args: ["@modelcontextprotocol/server-filesystem", "."]
      adapter: "stdio-http"  # Use the STDIO-to-HTTP adapter
```

### 9.2 Containerization Configuration

```dockerfile
# Example Dockerfile for networked MCP server
FROM node:18

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .

EXPOSE 3000
CMD ["node", "server.js"]
```

```dockerfile
# Example Dockerfile for STDIO-based MCP server with adapter with security enhancements
FROM python:3.10-slim

# Create non-root user for security
RUN useradd -m -s /bin/bash mcp-user

# Install server dependencies
RUN apt-get update && apt-get install -y nodejs npm

# Set up secure directory permissions
WORKDIR /app
COPY stdio_adapter.py .
COPY server.py .

# Install adapter requirements
RUN pip install aiohttp

# Set server command
ENV SERVER_COMMAND="npx"
ENV SERVER_ARGS="-y @modelcontextprotocol/server-filesystem /data"

# Create secured data directory 
RUN mkdir -p /data && chown -R mcp-user:mcp-user /data

# Drop privileges to non-root user
USER mcp-user

EXPOSE 8000
CMD ["python", "server.py"]
```

## 10. Conclusion and Next Steps

MCP Agent Cloud leverages the power of the Model Context Protocol and the flexibility of the mcp-agent framework to create a comprehensive platform for AI agent deployment, management, and orchestration. By combining enterprise productivity features with developer-centric tools, it provides a unique offering in the AI agent ecosystem.

### 10.1 Immediate Next Steps

1. Set up basic project structure and development environment - ✅ Completed
2. Implement CLI skeleton with core commands - ✅ Completed
3. Build STDIO-to-HTTP adapter for server compatibility - ✅ Completed
4. Develop container service for MCP server deployment - ✅ Completed
5. Create simple registry for tracking deployed resources - ✅ Completed
6. Implement OAuth authentication framework - ✅ Completed
7. Establish security boundaries for containerized MCP servers - ✅ Completed
8. Implement agent deployment functionality - ✅ Completed
9. Further develop monitoring and observability

### 10.2 Success Metrics

- CLI download counts and active usage
- Number of deployed MCP servers and agents
- Active workflows and completion rates
- User engagement with marketplace
- Community contributions to ecosystem
- Security incident metrics (should remain at zero)
- Authentication success rates

### 10.3 Technical Notes

- MCP Agent Cloud service should be runnable on localhost
- Workflow runs modeled off of Temporal with support for long-running tasks and automatic triggers
- Use websockets for streaming events, possibly leveraging MCP protocol's /notifications
- Wrap stdio MCP servers as dockerized containers with SSE or HTTP endpoints
- MCPApp should function as MCP servers
- Temporal Executor integration for workflow management
- Use oauth for authentication and push capability
- Implement key management systems

### 10.4 Product Updates

#### 10.4.1 Initial Implementation of MCP Agent Cloud Deployment

The first phase of MCP Agent Cloud has been implemented, focusing on the core functionality of deploying MCPApps and utility MCP servers. Key features include:

1. **Authentication System**: OAuth 2.0 device authorization flow for secure authentication
2. **Code Packaging**: Automated packaging of MCPApp code (including embedded workflow patterns) for deployment as MCP servers
3. **Deployment Service**: Cloud service for building and hosting MCP servers
4. **CLI Integration**: Seamless deployment via the MCP Agent CLI with unified commands
5. **Status Monitoring**: Real-time deployment status updates

This implementation follows the "everything is an MCP server" principle and addresses user story #5: "As an agent developer, I want to deploy my MCPApp using a secure terminal command requiring authentication so I can push my validated agent code to the cloud, making it operational in the remote managed environment as an MCP server."

The deployment architecture supports two primary types of MCP servers:

1. **Utility MCP Servers**: Basic servers like filesystem and fetch that provide core functionality
2. **MCPApp Servers**: Complete AI applications built with the mcp-agent framework that contain various AugmentedLLM workflow patterns

All of these are deployed using a consistent approach through the `mcp-agent deploy` command family, reinforcing our architectural principle that all components in our ecosystem are MCP servers at their core. Workflow patterns exist within MCPApps and materialize during execution rather than as separately deployable components.

#### 10.4.2 Lifecycle Management and Namespacing

To address resource management concerns and prevent naming collisions, we've implemented:

1. **Namespacing Strategy**: 
   - All server and app names are prefixed with the user's organization ID
   - Public servers use a verified namespace system with unique, immutable identifiers
   - Format: `{org_id}.{server_name}` for private, `@{org_id}/{server_name}` for public

2. **Resource Lifecycle Management**:
   - Deployed servers and apps can be stopped, started, and deleted via CLI or web console
   - Public servers follow strict dependency management before decommissioning
   - System provides dependency graphs showing which apps depend on utility servers
   - Automatic cleanup of associated resources on deletion

3. **Client Interaction with Deployed MCPApps**:
   - All deployed MCPApps expose standard MCP endpoints (`/call_tool`, `/read_resource`) 
   - Clients interact with MCPApps using standard MCP protocol
   - Platform-level APIs (`/agents/{name}/run`) control the MCPApp instances themselves
   - Each workflow within an MCPApp is accessible through standard MCP tools exposed by the app
