#!/usr/bin/env python3
import asyncio
import os
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM


class AgentEvaluator:
    """Simple evaluator for MCP agents."""
    
    def __init__(self, agent_path: str, output_dir: str):
        """Initialize the evaluator with the agent path."""
        self.agent_path = agent_path
        self.agent_name = os.path.basename(agent_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up visualization directory
        self.vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Test cases - simple queries to evaluate the agent
        self.test_cases = [
            {
                "id": "basic_query_1",
                "query": "Hello, can you help me today?",
                "expected_contains": ["help", "assist", "support"]
            },
            {
                "id": "complex_query",
                "query": "What can you do for me?",
                "expected_contains": ["capabilities", "assist", "help"]
            },
            {
                "id": "follow_up",
                "query": "Tell me more about your capabilities.",
                "expected_contains": ["feature", "function", "ability"]
            }
        ]
        
        # Metrics to track
        self.metrics = {
            "response_times": [],
            "response_lengths": [],
            "success_rate": 0.0,
            "tool_usage_count": 0
        }
        
        # Results for each test case
        self.results = []
    
    async def run_evaluation(self) -> Dict[str, Any]:
        """Run the evaluation on the agent and return metrics."""
        app = MCPApp(name=f"eval_{self.agent_name}")
        
        async with app.run() as agent_app:
            logger = agent_app.logger
            context = agent_app.context
            
            logger.info(f"Starting evaluation for {self.agent_name}")
            
            # Create a generic agent for testing
            test_agent = Agent(
                name="test_agent",
                instruction="You are an agent being evaluated for quality and performance.",
                server_names=[]
            )
            
            async with test_agent:
                # Determine which LLM to use based on config
                if hasattr(context.config, 'anthropic') and context.config.anthropic and context.config.anthropic.api_key:
                    llm = await test_agent.attach_llm(AnthropicAugmentedLLM)
                    logger.info("Using Anthropic model for evaluation")
                else:
                    llm = await test_agent.attach_llm(OpenAIAugmentedLLM)
                    logger.info("Using OpenAI model for evaluation")
                
                # Run each test case
                for i, test_case in enumerate(self.test_cases):
                    logger.info(f"Running test case {i+1}/{len(self.test_cases)}: {test_case['id']}")
                    
                    # Measure response time
                    start_time = time.time()
                    response = await llm.generate_str(
                        message=test_case["query"],
                        request_params=RequestParams(temperature=0.7)
                    )
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    # Add metrics
                    self.metrics["response_times"].append(response_time)
                    self.metrics["response_lengths"].append(len(response))
                    
                    # Check if response contains expected keywords
                    success = any(word.lower() in response.lower() for word in test_case["expected_contains"])
                    
                    # Record result
                    result = {
                        "test_id": test_case["id"],
                        "query": test_case["query"],
                        "response": response,
                        "response_time": response_time,
                        "success": success
                    }
                    self.results.append(result)
                    
                    logger.info(f"Test case {test_case['id']} completed in {response_time:.2f}s")
                
                # Calculate overall metrics
                successful_tests = sum(1 for r in self.results if r["success"])
                self.metrics["success_rate"] = (successful_tests / len(self.test_cases)) * 100
                self.metrics["avg_response_time"] = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
                
                # Save results
                self.save_results()
                
                return {
                    "agent": self.agent_name,
                    "metrics": self.metrics,
                    "results": self.results
                }
    
    def save_results(self) -> None:
        """Save evaluation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(self.output_dir, f"evaluation_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump({
                "agent": self.agent_name,
                "timestamp": timestamp,
                "metrics": self.metrics,
                "results": self.results
            }, f, indent=2)
        
        # Save visualization data
        vis_file = os.path.join(self.vis_dir, f"metrics_{timestamp}.json")
        with open(vis_file, "w") as f:
            json.dump({
                "agent": self.agent_name,
                "response_times": self.metrics["response_times"],
                "success_rate": self.metrics["success_rate"],
                "avg_response_time": self.metrics["avg_response_time"]
            }, f, indent=2)
        
        print(f"Evaluation results saved to {results_file}")
        print(f"Visualization data saved to {vis_file}")


async def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate an MCP agent")
    parser.add_argument("--agent", type=str, default="mcp_basic_slack_agent",
                      help="Name of the agent to evaluate")
    args = parser.parse_args()
    
    # Set up paths
    agent_path = os.path.join("examples", args.agent)
    output_dir = os.path.join(agent_path, "eval_results")
    
    # Run evaluation
    evaluator = AgentEvaluator(agent_path, output_dir)
    results = await evaluator.run_evaluation()
    
    # Print summary
    print(f"\n===== Evaluation Summary for {args.agent} =====")
    print(f"Success Rate: {results['metrics']['success_rate']:.2f}%")
    print(f"Average Response Time: {results['metrics']['avg_response_time']:.2f}s")
    print(f"Test Cases Run: {len(results['results'])}")


if __name__ == "__main__":
    asyncio.run(main())