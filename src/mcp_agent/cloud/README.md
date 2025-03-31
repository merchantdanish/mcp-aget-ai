# MCP Agent Cloud

This module provides all functionality for the MCP Agent Cloud platform, which enables deployment, management, and orchestration of AI agent applications leveraging the Model Context Protocol (MCP).

## Overview

The MCP Agent Cloud platform combines:

1. **Enterprise Workplace Productivity Platform** - A centralized "AI App Store" within organizations
2. **Developer-Centric Agent Platform** - Comprehensive tools and infrastructure for MCP integration

The cloud module is the central implementation for:

1. **Server Deployment and Endpoint Management**
   - Deploying secure MCP servers with authentication
   - Managing API endpoints
   - Deploying and managing MCPApps (built with mcp-agent)

2. **Workflow Orchestration**
   - Executing workflows with robust control (run, pause, resume)
   - Supporting human-in-the-loop interactions
   - Managing workflow state and execution

3. **Observability and Tracing**
   - Tracing MCPApp runs for observability and debugging
   - Performance monitoring and error reporting

4. **Authentication and Security**
   - Implementing OAuth-based authentication
   - Managing server and client credentials
   - Securing API endpoints and data storage

## Authentication

Authentication with MCP Agent Cloud uses OAuth 2.0 device authorization flow:

1. The CLI initiates the authentication process
2. A device code and verification URL are displayed to the user
3. The user visits the URL and enters the code
4. The CLI exchanges the device code for an access token
5. The access token is used for subsequent requests

## Deployment

The deployment process works as follows:

1. The CLI validates the agent directory
2. The agent code is packaged into a tarball
3. A deployment request is created on the platform
4. The package is uploaded
5. The platform builds and deploys the agent
6. The CLI waits for the deployment to complete
7. Deployment information is displayed to the user

## Usage

The MCP Agent Cloud functionality is exposed through the CLI:

```bash
# Deploy an agent from the current directory
mcp-agent deploy agent

# Deploy an agent from a specific directory with a custom name
mcp-agent deploy agent /path/to/agent --name my-agent

# List all deployed agents
mcp-agent deploy agent list
```

## Configuration

The MCP Agent Cloud module uses the following configuration:

- Authentication data is stored in `~/.mcp-agent-cloud/auth.json`
- Session data is stored in `~/.mcp-agent-cloud/session.json`
- API endpoints can be configured via environment variables:
  - `MCP_AGENT_CLOUD_API_URL` - Base URL for the API
  - `MCP_AGENT_CLOUD_AUTH_URL` - Base URL for authentication

## Development

To contribute to the MCP Agent Cloud module:

1. Ensure you have the required dependencies installed
2. Run the tests: `pytest tests/cloud/`
3. Follow the code style of the existing codebase
4. Add tests for your changes