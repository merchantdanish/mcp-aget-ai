"""
Evaluation runner for MCP Agents.

This module runs a set of predefined tasks for different scenarios
and evaluates the agent's performance using the metrics defined in the metrics module.
"""

import asyncio
import json
import time
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from .metrics import (
    Task, ToolAction, 
    GroundingMetrics, MCPSpecificMetrics, TaskCompletionMetrics,
    Visualization
)
from .scenario_tasks import get_scenario_tasks

# Dictionary of scenarios with their configurations
SCENARIOS = {
    "education": {
        "agent_name": "education_tutor",
    },
    "airline": {
        "agent_name": "airline_assistant",
    },
}

class AgentEvaluator:
    """Evaluator for MCP Agents across different scenarios."""
    
    def __init__(
        self, 
        agent_app: MCPApp, 
        scenario_agent: Agent, 
        llm, 
        scenario_name: str,
        results_dir: str
    ):
        """Initialize the evaluator."""
        self.app = agent_app
        self.agent = scenario_agent
        self.llm = llm
        self.scenario_name = scenario_name
        self.logger = agent_app.logger
        self.tasks = get_scenario_tasks(scenario_name)
        self.results_dir = results_dir
        
        # Storage for evaluation data
        self.responses: Dict[str, List[str]] = {}
        self.timestamps: Dict[str, List[float]] = {}
        self.final_states: Dict[str, str] = {}
        self.tool_actions: Dict[str, List[ToolAction]] = {}
        self.turns_per_task: Dict[str, int] = {}
        
    async def run_task(self, task: Task) -> None:
        """Run a single task and record data for evaluation."""
        task_id = task.id
        self.responses[task_id] = []
        self.timestamps[task_id] = []
        self.tool_actions[task_id] = []
        
        # Get the appropriate prompt for this task type
        prompt = self._get_task_prompt(task)
        
        self.logger.info(f"Running task: {task.name} (ID: {task.id})")
        
        start_time = time.time()
        self.timestamps[task_id].append(start_time)
        
        # Call the LLM to handle the task
        result = await self.llm.generate_str(
            message=prompt,
            request_params=RequestParams(
                modelPreferences=ModelPreferences(
                    intelligencePriority=0.8,
                    speedPriority=0.2,
                ),
            ),
        )
        
        # Record the response and timestamp
        self.responses[task_id].append(result)
        end_time = time.time()
        self.timestamps[task_id].append(end_time)
        
        # Store final state for completion evaluation
        self.final_states[task_id] = result
        
        # Record turn count (1 for single-turn evaluation)
        self.turns_per_task[task_id] = 1
        
        # Log completion
        self.logger.info(f"Task {task.id} completed in {end_time - start_time:.2f}s")
    
    async def run_all_tasks(self) -> Dict[str, Any]:
        """Run all tasks and return evaluation results."""
        for task in self.tasks:
            await self.run_task(task)
        
        return self.evaluate_results()
    
    def evaluate_results(self) -> Dict[str, Any]:
        """Evaluate the performance based on recorded data."""
        results = {
            "scenario": self.scenario_name,
            "tasks": {},
            "overall": {}
        }
        
        # Process task-specific evaluations
        for task in self.tasks:
            task_id = task.id
            
            # Skip tasks that weren't run
            if task_id not in self.responses:
                continue
            
            # Calculate progress rate for this task
            progress_trajectory = task.progress_rate(self.responses[task_id])
            
            # Generate visualization data
            viz_data = Visualization.generate_progress_trajectory(
                task=task,
                states=self.responses[task_id],
                timestamps=self.timestamps[task_id]
            )
            
            # Count matched subgoals
            subgoals_matched = sum(1 for sg in task.subgoals if sg.is_matched(self.final_states[task_id]))
            
            # Save task results
            results["tasks"][task_id] = {
                "name": task.name,
                "difficulty": task.difficulty,
                "progress_trajectory": progress_trajectory,
                "completion_time": self.timestamps[task_id][-1] - self.timestamps[task_id][0],
                "subgoals_matched": subgoals_matched,
                "total_subgoals": len(task.subgoals),
                "subgoal_match_rate": (subgoals_matched / len(task.subgoals)) * 100,
                "is_completed": task.is_completed(self.final_states[task_id]),
                "turns_taken": self.turns_per_task.get(task_id, 0)
            }
            
            # Save visualization data
            os.makedirs(os.path.join(self.results_dir, "visualizations"), exist_ok=True)
            Visualization.export_visualization_data(
                viz_data,
                os.path.join(self.results_dir, "visualizations", f"{task_id}_trajectory.json")
            )
        
        # Calculate overall metrics
        completion_results = {
            task_id: data["is_completed"]
            for task_id, data in results["tasks"].items()
        }
        
        # Breakdown by difficulty
        difficulty_breakdown = TaskCompletionMetrics.breakdown_by_difficulty(
            self.tasks, completion_results
        )
        
        # Calculate turn efficiency
        turn_efficiency = TaskCompletionMetrics.calculate_turn_efficiency(
            self.tasks, self.turns_per_task, completion_results
        )
        
        # Overall results
        results["overall"] = {
            "total_tasks": len(self.tasks),
            "tasks_completed": sum(1 for status in completion_results.values() if status),
            "completion_rate": sum(1 for status in completion_results.values() if status) / len(completion_results) * 100,
            "difficulty_breakdown": {str(k): v for k, v in difficulty_breakdown.items()},
            "average_subgoal_match_rate": sum(task["subgoal_match_rate"] for task in results["tasks"].values()) / len(results["tasks"]),
            "turn_efficiency": {task_id: efficiency for task_id, efficiency in turn_efficiency.items()}
        }
        
        # Export the full evaluation results
        self._export_results(results)
        
        return results
    
    def _get_task_prompt(self, task: Task) -> str:
        """Get the appropriate prompt for a task based on scenario and task ID."""
        # Education scenario prompts
        education_prompts = {
            "math_problem": "I'm stuck on this equation: 2x + 5 = 15. Can you help?",
            "science_explanation": "Why does the moon change shape?",
            "writing_assistance": "I need to write a story about dinosaurs."
        }
        
        # Airline scenario prompts
        airline_prompts = {
            "flight_change": "I need to change my flight from New York to Los Angeles tomorrow.",
            "baggage_policy": "What's your baggage policy for international flights?",
            "delay_compensation": "My flight was delayed by 3 hours. Am I eligible for compensation?"
        }
        
        # Select the appropriate prompt based on scenario
        if self.scenario_name == "education":
            return education_prompts.get(task.id, f"I need help with {task.description}")
        elif self.scenario_name == "airline":
            return airline_prompts.get(task.id, f"I need help with {task.description}")
        else:
            return f"I need help with {task.description}"
    
    def _export_results(self, results: Dict[str, Any]) -> None:
        """Export evaluation results to a JSON file."""
        timestamp = int(time.time())
        filename = os.path.join(self.results_dir, f"{self.scenario_name}_evaluation_{timestamp}.json")
        
        # Convert to JSON serializable format
        serializable_results = json.loads(json.dumps(results, default=lambda o: str(o)))
        
        with open(filename, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Evaluation results exported to {filename}")


class EvaluationRunner:
    """Runner for evaluating MCP agents on specific scenarios."""
    
    def __init__(self, agent_path: str, scenario_name: str):
        """Initialize the evaluation runner."""
        self.agent_path = agent_path
        self.scenario_name = scenario_name
        self.results_dir = os.path.join("eval", "results", os.path.basename(agent_path))
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_evaluation(self, tasks: List[Task]) -> Dict[str, Any]:
        """Run evaluation for agent on scenario tasks."""
        print(f"Evaluating agent in {self.agent_path} on {self.scenario_name} scenario...")
        
        # Prepare results dictionary
        results = {
            "scenario": self.scenario_name,
            "agent": os.path.basename(self.agent_path),
            "tasks": {},
            "overall": {
                "total_tasks": len(tasks),
                "tasks_completed": 0,
                "completion_rate": 0.0,
                "average_subgoal_match_rate": 0.0,
                "difficulty_breakdown": {},
            }
        }
        
        # Process each task
        for task in tasks:
            task_id = task.id
            
            # Simulate task execution (this is a stub - replace with real execution)
            # In a real implementation, you would:
            # 1. Start the agent
            # 2. Send the task prompt
            # 3. Record interactions and responses
            # 4. Evaluate based on responses
            
            # Simulate task completion data for now
            completion_state = "This is a simulated response that would be evaluated."
            subgoals_matched = len(task.subgoals) // 2  # Simulate partial matching
            
            # Add task results
            results["tasks"][task_id] = {
                "name": task.name,
                "difficulty": task.difficulty,
                "progress_trajectory": [0.0, 0.5, 1.0],  # Simulated progress
                "completion_time": 5.0,  # Simulated time in seconds
                "subgoals_matched": subgoals_matched,
                "total_subgoals": len(task.subgoals),
                "subgoal_match_rate": (subgoals_matched / len(task.subgoals)) * 100,
                "is_completed": subgoals_matched == len(task.subgoals),
                "turns_taken": 3  # Simulated turn count
            }
            
            # Update completion count if task was completed
            if results["tasks"][task_id]["is_completed"]:
                results["overall"]["tasks_completed"] += 1
        
        # Calculate overall metrics
        if results["tasks"]:
            results["overall"]["completion_rate"] = (
                results["overall"]["tasks_completed"] / results["overall"]["total_tasks"] * 100
            )
            results["overall"]["average_subgoal_match_rate"] = (
                sum(task["subgoal_match_rate"] for task in results["tasks"].values()) / len(results["tasks"])
            )
        
        # Breakdown by difficulty
        difficulty_counts = {}
        difficulty_completions = {}
        
        for task in tasks:
            diff = task.difficulty
            if diff not in difficulty_counts:
                difficulty_counts[diff] = 0
                difficulty_completions[diff] = 0
            
            difficulty_counts[diff] += 1
            if results["tasks"][task.id]["is_completed"]:
                difficulty_completions[diff] += 1
        
        # Calculate completion rates by difficulty
        for diff in difficulty_counts:
            if difficulty_counts[diff] > 0:
                results["overall"]["difficulty_breakdown"][str(diff)] = (
                    difficulty_completions[diff] / difficulty_counts[diff] * 100
                )
        
        # Print summary
        print(f"\n=== {self.scenario_name.title()} Agent Evaluation Results ({results['agent']}) ===")
        print(f"Total Tasks: {results['overall']['total_tasks']}")
        print(f"Tasks Completed: {results['overall']['tasks_completed']}")
        print(f"Completion Rate: {results['overall']['completion_rate']:.2f}%")
        print(f"Average Subgoal Match Rate: {results['overall']['average_subgoal_match_rate']:.2f}%")
        print("\nDifficulty Breakdown:")
        for diff, rate in results["overall"]["difficulty_breakdown"].items():
            print(f"  {diff}: {rate:.2f}%")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation for a specific agent and scenario")
    parser.add_argument(
        "--agent", 
        type=str, 
        required=True,
        help="Path to the agent to evaluate"
    )
    parser.add_argument(
        "--scenario", 
        type=str, 
        default="airline",
        choices=SCENARIOS.keys(),
        help=f"Scenario to evaluate. Available options: {', '.join(SCENARIOS.keys())}"
    )
    parser.add_argument(
        "--results-dir", 
        type=str, 
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    # Get tasks for scenario
    tasks = get_scenario_tasks(args.scenario)
    if not tasks:
        print(f"Error: No tasks found for scenario '{args.scenario}'")
        sys.exit(1)
    
    # Run evaluation
    runner = EvaluationRunner(args.agent, args.scenario)
    results = runner.run_evaluation(tasks)