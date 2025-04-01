# MCP Agent Cloud End-to-End Demo

This demo showcases a complete MCP Agent Cloud deployment with authenticated MCP servers and example applications that use them. The demo demonstrates the core functionality of MCP Agent Cloud in a containerized environment.

## What's Included

The demo contains all the necessary components for a full end-to-end demonstration:

1. **Cloud Authentication Service**: OAuth 2.0 authentication for MCP servers
2. **STDIO-based Filesystem Server**: Filesystem access via the MCP protocol
3. **Networked Fetch Server**: Web content fetching via the MCP protocol
4. **MCP Application**: Example using both servers with different LLMs and workflow patterns
5. **Setup Script**: Unified tool for setting up and managing the demo environment

## Prerequisites

To run this demo, you will need:

- Docker and Docker Compose
- Python 3.8 or higher
- OpenAI API key
- Anthropic API key

## Running the Demo

### Step 1: Set Up Environment Variables

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Step 2: Run the Setup Script

Make the setup script executable and run it:

```bash
chmod +x setup.sh
./setup.sh start
```

This script will:
- Verify your API keys
- Create necessary directories
- Start the registry service
- Deploy the authentication service
- Deploy the filesystem and fetch MCP servers
- Deploy the MCP application
- Run health checks to ensure all services are running

### Step 3: Monitor the Demo

You can view the logs of any component:

```bash
# View logs for the MCP application
./setup.sh logs app

# View logs for the auth service
./setup.sh logs auth

# View logs for the filesystem server
./setup.sh logs filesystem

# View logs for the fetch server
./setup.sh logs fetch

# View all logs
./setup.sh logs all
```

### Step 4: Check the Results

When the application completes, view the output files:

```bash
# List all output files
ls -la mcp-app/output/

# View the research results
cat mcp-app/output/mcp_research.md

# View the parallel workflow results
cat mcp-app/output/parallel_workflow_result.md
```

### Step 5: Reset When Done

When you're finished with the demo:

```bash
./setup.sh reset
```

This will stop all services and clear output files.

## What to Expect

When you run the demo successfully:

1. The auth service will start and initialize
2. The MCP servers will connect to the auth service
3. The MCP application will authenticate with the auth service
4. The application will run multiple workflow patterns:
   - Basic research workflow using OpenAI LLM
   - Parallel workflow with multiple specialized agents
5. Results will be saved to the `mcp-app/output` directory

## Architecture

The components are organized as Docker containers that communicate with each other:

```
+----------------+     +------------------+     +---------------+
|                |     |                  |     |               |
| Cloud Auth     |<--->| MCP Application  |<--->| Redis Registry|
|    Service     |     |                  |     |               |
|                |     +------------------+     +---------------+
+----------------+            ^   ^
        ^                     |   |
        |                     |   |
        v                     v   v
+----------------+     +-------------+
|                |     |             |
|  Filesystem    |     |    Fetch    |
|    Server      |     |    Server   |
|                |     |             |
+----------------+     +-------------+
```

## Key Implementation Files

- **docker-compose.yml**: Defines all the services and their connections
- **mcp-app/main.py**: Shows how to register with the auth service and use MCP servers
- **mcp-app/mcp_agent.config.yaml**: Configured to use the deployed servers with auth
- **setup.sh**: Unified script for managing the demo environment

## Troubleshooting

If you encounter issues:

- **Build Errors**: Check Dockerfile configurations and ensure all required files exist
- **Connection Errors**: Verify container network settings in docker-compose.yml
- **Authentication Errors**: Check auth service logs with `./setup.sh logs auth`
- **Missing API Keys**: Verify environment variables are set correctly
- **Health Check Failures**: Run `./setup.sh healthcheck` to verify service status

You can also run `./setup.sh reset` and then `./setup.sh start` to restart the entire demo from scratch.

## Next Steps

After running this demo, explore:

1. Creating your own MCP servers using `mcp-agent deploy server`
2. Deploying your own MCPApps using `mcp-agent deploy app`
3. Modifying the existing MCP application to use different workflow patterns
4. Integrating with your own MCP-compatible services