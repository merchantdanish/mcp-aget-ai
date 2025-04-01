#!/bin/bash
# MCP Agent Cloud E2E Demo Setup Script
# This script automates the setup and running of the MCP Agent Cloud E2E Demo
# using actual mcp-agent cloud CLI commands

set -e  # Exit on error

# Define color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}       MCP Agent Cloud End-to-End Demo Setup             ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check for required API keys
echo -e "\n${YELLOW}Checking for required API keys...${NC}"
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY environment variable is not set.${NC}"
    echo -e "Please set your OpenAI API key with: export OPENAI_API_KEY=\"your-key-here\""
    exit 1
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}Error: ANTHROPIC_API_KEY environment variable is not set.${NC}"
    echo -e "Please set your Anthropic API key with: export ANTHROPIC_API_KEY=\"your-key-here\""
    exit 1
fi
echo -e "${GREEN}✓ API keys verified${NC}"

# Create required directories
echo -e "\n${YELLOW}Creating required directories...${NC}"
mkdir -p mcp-app/output
chmod 777 mcp-app/output
echo -e "${GREEN}✓ Directories created${NC}"

# Check dependencies
echo -e "\n${YELLOW}Checking system dependencies...${NC}"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is required but not installed.${NC}"
    echo -e "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker is installed${NC}"

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is required but not installed.${NC}"
    echo -e "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose is installed${NC}"

# Path to mcp-agent CLI
MCP_AGENT="python -m mcp_agent.cli"

# Function to run health checks
run_health_checks() {
    echo -e "\n${YELLOW}Running health checks...${NC}"
    local max_attempts=10
    local attempt=1
    local services_ready=false

    while [ $attempt -le $max_attempts ] && [ "$services_ready" != "true" ]; do
        echo -e "Health check attempt $attempt of $max_attempts"
        
        # Check auth service
        if curl -s http://localhost:8000/health > /dev/null; then
            echo -e "${GREEN}✓ Auth service is healthy${NC}"
            auth_ready=true
        else
            echo -e "${YELLOW}⟳ Auth service is not ready yet${NC}"
            auth_ready=false
        fi
        
        # Check filesystem server
        if curl -s http://localhost:8001/health > /dev/null; then
            echo -e "${GREEN}✓ Filesystem server is healthy${NC}"
            fs_ready=true
        else
            echo -e "${YELLOW}⟳ Filesystem server is not ready yet${NC}"
            fs_ready=false
        fi
        
        # Check fetch server
        if curl -s http://localhost:8002/health > /dev/null; then
            echo -e "${GREEN}✓ Fetch server is healthy${NC}"
            fetch_ready=true
        else
            echo -e "${YELLOW}⟳ Fetch server is not ready yet${NC}"
            fetch_ready=false
        fi
        
        # Check if all services are ready
        if [ "$auth_ready" = true ] && [ "$fs_ready" = true ] && [ "$fetch_ready" = true ]; then
            services_ready=true
            echo -e "${GREEN}All services are healthy and ready!${NC}"
        else
            echo -e "${YELLOW}Waiting for services to become ready (attempt $attempt of $max_attempts)...${NC}"
            sleep 5
            attempt=$((attempt + 1))
        fi
    done

    if [ "$services_ready" != "true" ]; then
        echo -e "${RED}Health checks failed after $max_attempts attempts.${NC}"
        echo -e "${YELLOW}You may need to check the individual service logs.${NC}"
        return 1
    fi
    
    return 0
}

# Function to run the registry service (which isn't part of mcp-agent cloud commands)
start_registry() {
    echo -e "\n${YELLOW}Starting registry service...${NC}"
    docker-compose up -d registry
    echo -e "${GREEN}✓ Registry service started${NC}"
}

# Function to deploy all components using MCP Agent Cloud CLI
deploy_components() {
    echo -e "\n${YELLOW}Deploying auth service...${NC}"
    # This uses the auth service defined in the original docker-compose
    docker-compose up -d cloud-auth
    echo -e "${GREEN}✓ Auth service deployed${NC}"
    
    echo -e "\n${YELLOW}Deploying filesystem MCP server...${NC}"
    $MCP_AGENT deploy server deploy filesystem --type filesystem
    echo -e "${GREEN}✓ Filesystem server deployed${NC}"
    
    echo -e "\n${YELLOW}Deploying fetch MCP server...${NC}"
    $MCP_AGENT deploy server deploy fetch --type fetch
    echo -e "${GREEN}✓ Fetch server deployed${NC}"
    
    echo -e "\n${YELLOW}Deploying MCP application...${NC}"
    $MCP_AGENT deploy app deploy ./mcp-app
    echo -e "${GREEN}✓ MCP application deployed${NC}"
}

# Function to list all deployed components
list_components() {
    echo -e "\n${YELLOW}Listing deployed MCP servers...${NC}"
    $MCP_AGENT deploy server list
    
    echo -e "\n${YELLOW}Listing deployed MCPApps...${NC}"
    $MCP_AGENT deploy app list
}

# Function to start the environment
start_environment() {
    echo -e "\n${YELLOW}Starting demo environment...${NC}"
    
    # Start registry first (needed by other services)
    start_registry
    
    # Deploy all components using the MCP Agent Cloud CLI
    deploy_components
    
    # Run health checks
    if run_health_checks; then
        echo -e "${GREEN}✓ Environment is ready for use!${NC}"
        
        # List all components
        list_components
        
        # Show output command
        echo -e "\n${YELLOW}To view outputs when they're ready:${NC}"
        echo -e "ls -la mcp-app/output/"
        echo -e "cat mcp-app/output/mcp_research.md"
        echo -e "cat mcp-app/output/parallel_workflow_result.md"
    else
        echo -e "${YELLOW}Environment started but health checks failed. Check the logs for more information.${NC}"
    fi
}

# Function to reset the environment
reset_environment() {
    echo -e "\n${YELLOW}Resetting the demo environment...${NC}"
    
    # Stop all servers using MCP Agent commands
    echo -e "${YELLOW}Stopping servers...${NC}"
    $MCP_AGENT deploy server stop filesystem
    $MCP_AGENT deploy server stop fetch
    
    # Stop the auth service (using docker-compose as it wasn't deployed via MCP Agent)
    echo -e "${YELLOW}Stopping auth service...${NC}"
    docker-compose stop cloud-auth
    
    # Stop the registry service
    echo -e "${YELLOW}Stopping registry service...${NC}"
    docker-compose stop registry
    
    # Clear output directory
    echo -e "${YELLOW}Clearing output files...${NC}"
    rm -f mcp-app/output/*
    echo -e "${GREEN}✓ Output files cleared${NC}"
    
    echo -e "\n${GREEN}Environment has been reset. Run './setup.sh start' to start a fresh demo.${NC}"
}

# Function to display logs for a component
view_logs() {
    component=$1
    
    case "$component" in
        auth)
            docker-compose logs -f cloud-auth
            ;;
        filesystem)
            docker-compose logs -f filesystem-server
            ;;
        fetch)
            docker-compose logs -f fetch-server
            ;;
        app)
            docker-compose logs -f mcp-app
            ;;
        all)
            docker-compose logs -f
            ;;
        *)
            echo -e "${RED}Unknown component: $component${NC}"
            echo -e "Available components: auth, filesystem, fetch, app, all"
            exit 1
            ;;
    esac
}

# Main command handling
case "$1" in
    start)
        start_environment
        ;;
    reset)
        reset_environment
        ;;
    healthcheck)
        run_health_checks
        ;;
    logs)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Missing component name${NC}"
            echo -e "Usage: $0 logs [auth|filesystem|fetch|app|all]"
            exit 1
        fi
        view_logs $2
        ;;
    *)
        echo -e "\n${BLUE}Available commands:${NC}"
        echo -e "  ${GREEN}start${NC}               - Build and start all components"
        echo -e "  ${GREEN}reset${NC}               - Stop all components and clear output files"
        echo -e "  ${GREEN}healthcheck${NC}         - Run health checks on all services"
        echo -e "  ${GREEN}logs [component]${NC}    - View logs for a specific component"
        echo
        echo -e "${BLUE}Example usage:${NC}"
        echo -e "  ${GREEN}./setup.sh start${NC}    - Start the demo environment"
        echo -e "  ${GREEN}./setup.sh logs app${NC} - View logs from the MCP application"
        echo
        echo -e "${YELLOW}To start the demo, run: ./setup.sh start${NC}"
        ;;
esac

echo -e "\n${BLUE}=========================================================${NC}"
echo -e "${BLUE}For more information, see README.md${NC}"
echo -e "${BLUE}=========================================================${NC}"