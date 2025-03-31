"""MCP Agent Cloud Demo Application

This application demonstrates using authenticated MCP servers with MCP Agent Cloud.
It creates multiple agents that use both a STDIO-based filesystem server and a
networked fetch server, with different LLM providers.
"""

import asyncio
import logging
import os
import time
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams, ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mcp_app.log"),
    ],
)

logger = logging.getLogger(__name__)

# Initialize the MCP app
app = MCPApp(name="mcp_cloud_demo")

async def register_with_auth_service():
    """Register the client with the auth service."""
    import requests
    auth_service_url = os.environ.get("AUTH_SERVICE_URL")
    
    if not auth_service_url:
        logger.warning("No AUTH_SERVICE_URL provided, skipping registration")
        return
    
    try:
        response = requests.post(
            f"{auth_service_url}/register",
            json={
                "client_name": "MCP Cloud Demo App",
                "redirect_uris": ["http://localhost:3000/callback"],
            },
        )
        
        if response.status_code == 200:
            client_data = response.json()
            logger.info(f"Registered client: {client_data.get('client_id')}")
            
            # Update config with client credentials
            context = app.context
            context.config.mcp.servers["filesystem"].auth.client_id = client_data.get("client_id")
            context.config.mcp.servers["filesystem"].auth.client_secret = client_data.get("client_secret")
            context.config.mcp.servers["fetch"].auth.client_id = client_data.get("client_id")
            context.config.mcp.servers["fetch"].auth.client_secret = client_data.get("client_secret")
        else:
            logger.error(f"Failed to register client: {response.text}")
    except Exception as e:
        logger.error(f"Error registering client: {str(e)}")

async def demo_workflows():
    """Run demo workflows with MCP Agents."""
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context
        
        # Log the current configuration
        logger.info("Current config:", data=context.config.model_dump())
        
        # Create agent using both servers
        research_agent = Agent(
            name="researcher",
            instruction="""You are a research agent with access to the filesystem and the ability to 
            fetch web content. Your job is to gather information from both local files and web sources, 
            and synthesize the results into a comprehensive report.""",
            server_names=["filesystem", "fetch"],
        )
        
        # Create a documentation agent
        docs_agent = Agent(
            name="documenter",
            instruction="""You are a documentation agent that specializes in creating clear, 
            concise documentation. When given research findings, you create a well-structured 
            document with proper formatting, headings, and organization.""",
            server_names=["filesystem"],
        )
        
        # Create a data analysis agent
        analysis_agent = Agent(
            name="analyst", 
            instruction="""You are a data analysis agent that specializes in interpreting 
            research findings. When given research information, you identify key insights, 
            patterns, and implications.""",
            server_names=["fetch"],
        )
        
        async with research_agent:
            logger.info("Researcher: Connected to servers, checking tools...")
            tools_result = await research_agent.list_tools()
            logger.info("Tools available:", data=tools_result.model_dump())
            
            # Create OpenAI LLM
            openai_llm = await research_agent.attach_llm(OpenAIAugmentedLLM)
            
            # Test with a basic query
            logger.info("Running OpenAI LLM research query...")
            result = await openai_llm.generate_str(
                message="Create a brief overview of the Model Context Protocol (MCP) by researching online sources.",
            )
            logger.info("Research result:", data=result)
            
            # Save the research output to a file
            with open("/app/output/mcp_research.md", "w") as f:
                f.write(result)
                
            logger.info("Research result saved to /app/output/mcp_research.md")
            
            # Now create a parallel workflow with all three agents
            logger.info("Running parallel workflow with multiple agents...")
            
            # Create parallel workflow
            parallel = ParallelLLM(
                fan_in_agent=docs_agent,
                fan_out_agents=[research_agent, analysis_agent],
                llm_factory=AnthropicAugmentedLLM,
            )
            
            result = await parallel.generate_str(
                """Create a comprehensive report on MCP (Model Context Protocol).
                Consider both what it is and its implications for LLM applications."""
            )
            
            logger.info("Parallel workflow result:", data=result)
            
            # Save the parallel workflow output to a file
            with open("/app/output/parallel_workflow_result.md", "w") as f:
                f.write(result)
                
            logger.info("Parallel workflow result saved to /app/output/parallel_workflow_result.md")

async def main():
    """Main entry point for the MCP Cloud demo."""
    try:
        # Wait for auth service to be ready
        await asyncio.sleep(5)
        
        # Register with auth service
        await register_with_auth_service()
        
        # Run demo workflows
        await demo_workflows()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    t = end - start
    
    logger.info(f"Total run time: {t:.2f}s")