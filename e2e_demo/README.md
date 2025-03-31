# MCP Agent Cloud End-to-End Demo

This demo showcases a complete MCP Agent Cloud deployment with authenticated MCP servers and an example application that uses them. The demo structure is set up and ready to run in an environment with Docker and Docker Compose.

## What's Included

The demo contains all the necessary components for a full end-to-end demonstration of MCP Agent Cloud:

1. **Cloud Authentication Service**: OAuth 2.0 authentication for MCP servers (from cloud/auth)
2. **STDIO-based Filesystem Server**: Filesystem access via the MCP protocol (from cloud/deployment)
3. **Networked Fetch Server**: Web content fetching via the MCP protocol (from cloud/deployment)
4. **MCP Application**: Example using both servers with different LLMs and workflow patterns
5. **Command-Line Interface**: Tool for managing the demo environment

## Directory Structure

```
e2e_demo/
├── DEMO.md                # Step-by-step instructions for running the demo
├── README.md              # This overview file
├── data/                  # Data directory mounted to the filesystem server
│   └── mcp_info.md        # Example data file with MCP information
├── docker-compose.yml     # Docker Compose configuration for all services
├── mcp-app/               # Example MCP application
│   ├── Dockerfile         # Container configuration for the application
│   ├── main.py            # Main application code using MCP Agent
│   ├── mcp_agent.config.yaml  # MCP Agent configuration
│   ├── output/            # Output directory for application results
│   │   ├── mcp_research.md       # Sample research output
│   │   └── parallel_workflow_result.md  # Sample parallel workflow output
│   └── requirements.txt   # Python dependencies
└── mcp-cloud-cli.py       # CLI tool for managing the demo
```

## Prerequisites

To run this demo, you will need:

- Docker and Docker Compose
- Python 3.8 or higher
- OpenAI API key
- Anthropic API key

## Running the Demo

Detailed step-by-step instructions for running the demo are provided in the `DEMO.md` file. Here's a quick overview:

1. Set environment variables for API keys:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

2. Build and deploy all components:
   ```bash
   chmod +x mcp-cloud-cli.py
   ./mcp-cloud-cli.py build all
   ./mcp-cloud-cli.py deploy all
   ```

3. Check the status and logs:
   ```bash
   ./mcp-cloud-cli.py status
   ./mcp-cloud-cli.py logs app
   ```

4. When finished, stop all components:
   ```bash
   ./mcp-cloud-cli.py stop all
   ```

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
- **mcp-cloud-cli.py**: CLI tool for managing the demo environment

## Cloud Module Integration

This demo uses the following components from the MCP Agent Cloud modules:

- **cloud/auth**: Authentication service with OAuth support
- **cloud/deployment**: Containerization for both STDIO and networked servers
- **cloud/deployment/adapters**: STDIO-to-HTTP adapter for filesystem server

## Troubleshooting

- If Docker isn't available, you can review the code and configuration to understand the architecture
- If services fail to start, check the logs for each service
- If authentication fails, ensure the auth service is running and accessible
- For any issues with the demo, review the detailed steps in DEMO.md