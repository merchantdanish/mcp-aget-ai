# MCP Agent Cloud End-to-End Demo

This step-by-step guide demonstrates a complete MCP Agent Cloud deployment with authentication, servers, and applications interacting with LLMs. Follow these steps to test the functionality of the platform.

## Prerequisites

Before beginning:

1. Ensure Docker and Docker Compose are installed:
   ```bash
   which docker docker-compose
   ```
   If either is missing, install them according to your operating system's instructions.

2. Set environment variables for API keys:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

## 1. Setting Up the Environment

First, let's create the necessary directories and ensure permissions:

```bash
# Ensure output directory exists
mkdir -p mcp-app/output

# Make CLI executable
chmod +x mcp-cloud-cli.py
```

## 2. Installing Node.js Packages

The networked server and STDIO server require Node.js packages:

```bash
# Install the Node.js packages needed for MCP servers
npm install -g @modelcontextprotocol/server-filesystem
```

## 3. Building the Components

```bash
# Build all components
docker-compose build
```

If you encounter any issues, you can build individual components:

```bash
# Build auth service
docker-compose build cloud-auth

# Build filesystem server
docker-compose build filesystem-server

# Build fetch server
docker-compose build fetch-server

# Build MCP application
docker-compose build mcp-app
```

## 4. Starting the Platform

You can start the entire platform at once:

```bash
# Start all services
docker-compose up -d
```

Or start services incrementally to see how they interact:

```bash
# Start the registry and auth services first
docker-compose up -d registry cloud-auth

# After the auth service is ready, start the MCP servers
docker-compose up -d filesystem-server fetch-server

# Finally, start the MCP app
docker-compose up -d mcp-app
```

## 5. Monitoring the System

Monitor the logs to see the system in action:

```bash
# View logs for auth service
docker-compose logs -f cloud-auth

# View logs for filesystem server
docker-compose logs -f filesystem-server

# View logs for fetch server
docker-compose logs -f fetch-server

# View logs for MCP application
docker-compose logs -f mcp-app
```

## 6. Examining the Results

Once the application completes, view the output files:

```bash
# List all output files
ls -la mcp-app/output/

# View the research results
cat mcp-app/output/mcp_research.md

# View the parallel workflow results
cat mcp-app/output/parallel_workflow_result.md
```

## 7. API Testing

You can test the authentication and servers directly:

```bash
# Test auth service health endpoint
curl http://localhost:8000/health

# Register a new client
curl -X POST -H "Content-Type: application/json" \
  -d '{"client_name":"test-client","redirect_uris":["http://localhost:3000/callback"]}' \
  http://localhost:8000/register

# Test filesystem server health endpoint
curl http://localhost:8001/health

# Test fetch server health endpoint
curl http://localhost:8002/health
```

## 8. Using the MCP Agent Cloud CLI

For CLI-based management:

```bash
# Check status of all components
./mcp-cloud-cli.py status

# View logs for a specific component
./mcp-cloud-cli.py logs auth
./mcp-cloud-cli.py logs filesystem-server

# Deploy individual servers
./mcp-cloud-cli.py deploy filesystem
./mcp-cloud-cli.py deploy fetch
```

## 9. Cleaning Up

When you're done with the demo:

```bash
# Stop all components
docker-compose down
```

## Troubleshooting

If you encounter issues:

- **Build Errors**: Check Dockerfile configurations and ensure all required files exist
- **Connection Errors**: Verify container network settings in docker-compose.yml
- **Authentication Errors**: Check auth service logs and ensure proper client registration
- **Missing API Keys**: Verify environment variables are set correctly

## Integration with Your Own Project

To use MCP Agent Cloud in your own project:

1. Configure servers in your mcp_agent.config.yaml file
2. Deploy servers and connect to them:
   ```python
   from mcp_agent.app import MCPApp
   from mcp_agent.agents.agent import Agent
   
   app = MCPApp(name="your-app-name")
   
   async with app.run() as agent_app:
       agent = Agent(
           name="your-agent",
           instruction="Your instruction here",
           server_names=["filesystem", "fetch"]
       )
       # Use the agent with LLMs
   ```