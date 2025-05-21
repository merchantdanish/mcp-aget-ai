"""
Test script for running evaluation functions.

This script provides a way to test the various evaluation functions in the MCP Agent eval module.
It serves as an example of how to use the evaluation functions programmatically.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
import argparse

from .core import read_trace_file, calculate_chat_cost
from .trace_analyzer import analyze_trace_file
from .tool_metrics import (
    measure_agent_tool_calls,
    analyze_tasks_tool_usage,
    extract_tool_call_details,
    generate_comprehensive_report
)
from .agent_analyzer import (
    get_detailed_trace_analysis,
    generate_trace_report
)
from .workflow_eval import evaluate_workflow


async def test_trace_analysis(trace_file: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Test trace analysis functionality"""
    print(f"Running trace analysis on {trace_file}")
    
    # Analyze the trace file
    analysis = analyze_trace_file(trace_file)
    
    # Save analysis to file if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(trace_file).stem}_trace_analysis.json")
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {output_path}")
    
    return analysis


async def test_tool_metrics(trace_file: str, agent_name: Optional[str] = None, output_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Test tool metrics measurement"""
    print(f"Measuring tool metrics for {trace_file}" + (f" (agent: {agent_name})" if agent_name else ""))
    
    # Measure tool call metrics
    metrics = measure_agent_tool_calls(trace_file, agent_name)
    
    # Save metrics to file if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(trace_file).stem}_tool_metrics.json")
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Tool metrics saved to {output_path}")
    
    return metrics


async def test_task_analysis(trace_file: str, agent_name: Optional[str] = None, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Test task analysis"""
    print(f"Analyzing tasks for {trace_file}" + (f" (agent: {agent_name})" if agent_name else ""))
    
    # Analyze tasks
    tasks = analyze_tasks_tool_usage(trace_file, agent_name)
    
    # Generate summary metrics
    summary = {
        "total_tasks": len(tasks),
        "total_tool_calls": sum(task["tool_call_count"] for task in tasks),
        "avg_tool_calls_per_task": sum(task["tool_call_count"] for task in tasks) / len(tasks) if tasks else 0,
        "max_tool_calls": max(task["tool_call_count"] for task in tasks) if tasks else 0,
        "min_tool_calls": min(task["tool_call_count"] for task in tasks) if tasks else 0,
        "tasks_with_tool_calls": sum(1 for task in tasks if task["tool_call_count"] > 0),
        "tasks_without_tool_calls": sum(1 for task in tasks if task["tool_call_count"] == 0),
        "avg_task_duration_ms": sum(task["duration_ms"] for task in tasks) / len(tasks) if tasks else 0,
        "avg_tool_time_percentage": sum(task["tool_time_percentage"] for task in tasks) / len(tasks) if tasks else 0
    }
    
    # Save tasks to file if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(trace_file).stem}_task_analysis.json")
        with open(output_path, 'w') as f:
            json.dump({"summary": summary, "tasks": tasks}, f, indent=2)
        print(f"Task analysis saved to {output_path}")
    
    return tasks


async def test_tool_details(trace_file: str, agent_name: Optional[str] = None, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Test extracting tool call details"""
    print(f"Extracting tool details for {trace_file}" + (f" (agent: {agent_name})" if agent_name else ""))
    
    # Extract tool call details
    tool_details = extract_tool_call_details(trace_file, agent_name)
    
    # Save tool details to file if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(trace_file).stem}_tool_details.json")
        with open(output_path, 'w') as f:
            json.dump(tool_details, f, indent=2)
        print(f"Tool details saved to {output_path}")
    
    return tool_details


async def test_comprehensive_report(trace_file: str, agent_name: Optional[str] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Test generating a comprehensive report"""
    print(f"Generating comprehensive report for {trace_file}" + (f" (agent: {agent_name})" if agent_name else ""))
    
    # Generate comprehensive report
    report = generate_comprehensive_report(trace_file, agent_name)
    
    # Save report to file if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(trace_file).stem}_comprehensive_report.json")
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Comprehensive report saved to {output_path}")
    
    # Print summary of report
    summary = report['summary']
    print(f"Summary: {len(summary['total_tasks'])} tasks, {summary['total_tool_calls']} tool calls")
    
    return report


async def test_agent_analysis(trace_file: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Test agent analysis"""
    print(f"Analyzing agent trace for {trace_file}")
    
    # Get detailed analysis
    analysis = get_detailed_trace_analysis(trace_file)
    
    # Generate report
    output_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(trace_file).stem}_agent_report.md")
    
    report = generate_trace_report(analysis, output_path)
    
    if output_path:
        print(f"Agent report saved to {output_path}")
    
    return analysis


async def test_workflow_eval(trace_file: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Test workflow evaluation"""
    print(f"Evaluating workflow for {trace_file}")
    
    # Evaluate workflow
    evaluation = evaluate_workflow(trace_file)
    
    # Save evaluation to file if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(trace_file).stem}_workflow_eval.json")
        with open(output_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        print(f"Workflow evaluation saved to {output_path}")
    
    return evaluation


async def test_all_evaluations(trace_file: str, agent_name: Optional[str] = None, output_dir: Optional[str] = None):
    """Run all evaluation types on a trace file"""
    print(f"Running all evaluations on {trace_file}")
    
    # Create specific output directory for this trace file if output_dir provided
    trace_output_dir = None
    if output_dir:
        trace_name = Path(trace_file).stem
        trace_output_dir = os.path.join(output_dir, trace_name)
        os.makedirs(trace_output_dir, exist_ok=True)
    
    # Run all evaluations concurrently
    tasks = [
        test_trace_analysis(trace_file, trace_output_dir),
        test_tool_metrics(trace_file, agent_name, trace_output_dir),
        test_task_analysis(trace_file, agent_name, trace_output_dir),
        test_tool_details(trace_file, agent_name, trace_output_dir),
        test_comprehensive_report(trace_file, agent_name, trace_output_dir),
        test_agent_analysis(trace_file, trace_output_dir),
        test_workflow_eval(trace_file, trace_output_dir)
    ]
    
    # Await all tasks to complete
    results = await asyncio.gather(*tasks)
    
    print(f"All evaluations completed for {trace_file}")
    return results


async def main_async():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test MCP Agent evaluation functions")
    parser.add_argument("--trace-files", nargs="+", help="Trace files to analyze")
    parser.add_argument("--trace-dir", help="Directory containing trace files to analyze")
    parser.add_argument("--output-dir", help="Directory to save output files")
    parser.add_argument("--agent", help="Optional agent name to filter by")
    args = parser.parse_args()
    
    # Get trace files from command line arguments or default directory
    trace_files = []
    if args.trace_files:
        trace_files = [Path(file) for file in args.trace_files]
    elif args.trace_dir:
        trace_dir = Path(args.trace_dir)
        trace_files = list(trace_dir.glob("*.jsonl"))
    else:
        # Default to 'traces' directory in same directory as this script
        trace_dir = Path(__file__).resolve().parent / "traces"
        trace_files = list(trace_dir.glob("*.jsonl"))
    
    if not trace_files:
        print("No trace files found")
        return
    
    print(f"Found {len(trace_files)} trace files: {[f.name for f in trace_files]}")
    
    # Process each trace file sequentially
    for trace_file in trace_files:
        await test_all_evaluations(str(trace_file), args.agent, args.output_dir)


def main():
    """Command-line entry point"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()