"""
Unified evaluation entry point for MCP Agent tracing and evaluation.

This module provides a unified command-line interface for various evaluation tools:
- trace file analysis
- agent trace analysis 
- tool metrics measurement
- task analysis
- workflow evaluation

Instead of having separate main functions in each module, this centralizes all functionality
into a single entry point with subcommands.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from mcp_agent.eval.core import read_trace_file, separate_spans_by_id, organize_spans_by_parent, calculate_chat_cost
from mcp_agent.eval.trace_analyzer import analyze_trace_file, get_spans_by_category
from mcp_agent.eval.agent_analyzer import (
    analyze_trace_by_component, 
    get_trace_summary,
    get_detailed_trace_analysis, 
    generate_trace_report,
    extract_spans_by_feature,
    print_feature_spans_tree
)
from mcp_agent.eval.agent_trace_analysis import extract_agent_events
from mcp_agent.eval.tool_metrics import (
    extract_tool_call_details, 
    measure_agent_tool_calls, 
    analyze_tasks_tool_usage, 
    extract_user_requests,
    generate_comprehensive_report
)
from mcp_agent.eval.test_trace_analysis import (
    test_read_trace_file,
    test_separate_spans_by_id,
    test_organize_spans_by_parent,
    test_analyze_trace_file,
    test_get_spans_by_category,
    test_agent_trace_analysis,
    test_agent_measurements,
    test_calculate_chat_cost
)
# Uncomment if workflow_eval.py exists
# from mcp_agent.eval.workflow_eval import evaluate_workflow


def cmd_trace_analysis(args):
    """Run trace analysis command"""
    trace_file_path = args.trace_file
    if not Path(trace_file_path).exists():
        print(f"Error: File {trace_file_path} not found")
        sys.exit(1)
    
    print(f"Analyzing trace file: {trace_file_path}")
    
    analysis = analyze_trace_file(trace_file_path)
    
    # Print summary
    print(f"Total spans: {analysis['total_spans']}")
    print(f"Distinct traces: {analysis['trace_count']}")
    
    # Print details for each trace
    for trace_id, trace_info in analysis["traces"].items():
        print(f"\nTrace ID: {trace_id}")
        print(f"  Spans: {trace_info['span_count']}")
        print(f"  Start time: {trace_info['start_time']}")
        print(f"  End time: {trace_info['end_time']}")
        print(f"  Agent names: {', '.join(trace_info['agent_names'])}")
        
        print("  Span types:")
        for span_type, count in trace_info['span_types'].items():
            print(f"    {span_type}: {count}")
        
        print(f"  Root spans: {trace_info['root_span_count']}")
        for i, root_span in enumerate(trace_info['root_spans'][:5]):  # Show only first 5
            print(f"    {i+1}. {root_span['name']} ({root_span['span_id']})")
        
        if len(trace_info['root_spans']) > 5:
            print(f"    ... and {len(trace_info['root_spans']) - 5} more")


def cmd_agent_analysis(args):
    """Run agent analysis command"""
    if args.extract_agent_events:
        # Extract agent events
        agent_events = extract_agent_events(args.trace_file)
        
        # Write to output file or print summary
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(agent_events, f, indent=2)
            print(f"Agent events extracted. Saved to {args.output_file}")
        else:
            for agent_name, events in agent_events.items():
                print(f"\nAgent: {agent_name} - {len(events)} events")
                event_types = {}
                for event in events:
                    event_type = event.get('event_type', event.get('type', 'unknown'))
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                for event_type, count in event_types.items():
                    print(f"  {event_type}: {count} events")
    elif args.feature_tree:
        # Generate the feature spans tree
        spans = read_trace_file(args.trace_file)
        feature_spans = extract_spans_by_feature(spans)
        tree_output = print_feature_spans_tree(feature_spans)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(tree_output)
            print(f"Feature tree analysis complete. Report saved to {args.output_file}")
        else:
            print(tree_output)
    else:
        # Get detailed analysis
        analysis = get_detailed_trace_analysis(args.trace_file)
        
        # Generate report
        report = generate_trace_report(analysis, args.output_file)
        
        if not args.output_file:
            print(report)
        else:
            print(f"Analysis complete. Report saved to {args.output_file}")


def cmd_tool_metrics(args):
    """Run tool metrics command"""
    if args.extract_requests:
        # Extract user requests
        requests = extract_user_requests(args.trace_file, args.agent, args.filter)
        
        # Write to output file or print summary
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(requests, f, indent=2)
            print(f"User requests extracted. Saved to {args.output_file}")
        else:
            print(f"\nFound {len(requests)} user requests:")
            for i, req in enumerate(requests):
                print(f"\nRequest {i+1} - Agent: {req['agent_name']}")
                request_preview = str(req['request'])
                if len(request_preview) > 80:
                    request_preview = request_preview[:77] + "..."
                print(f"  Request: {request_preview}")
                print(f"  Tool calls: {req['tool_call_count']}")
                print(f"  Duration: {req['duration_ms']:.2f} ms")
                if req['tool_servers_used']:
                    print(f"  Servers used: {', '.join(req['tool_servers_used'])}")
                else:
                    print("  No servers used")
    
    elif args.measure_tools:
        # Measure tool call metrics
        metrics = measure_agent_tool_calls(args.trace_file, args.agent)
        
        # Write to output file or print summary
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Tool call metrics saved to {args.output_file}")
        else:
            for agent_name, agent_metrics in metrics.items():
                print(f"\nAgent: {agent_name}")
                print(f"  Total tool calls: {agent_metrics['total_tool_calls']}")
                print(f"  Success rate: {agent_metrics['success_rate']:.2f}%")
                print(f"  Average duration: {agent_metrics['avg_duration']:.2f} ms")
                
                print("\n  Tool calls by server:")
                for server, count in agent_metrics['tool_calls_by_server'].items():
                    print(f"    {server}: {count} calls")
                
                print("\n  Tool calls by method:")
                for method, count in agent_metrics['tool_calls_by_method'].items():
                    print(f"    {method}: {count} calls")
    
    elif args.comprehensive_report:
        # Generate comprehensive report
        report = generate_comprehensive_report(args.trace_file, args.agent)
        
        # Write to output file or print summary
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Comprehensive report generated. Saved to {args.output_file}")
        else:
            # Print summary metrics
            summary = report['summary']
            print("\nComprehensive Agent Analysis Report")
            print(f"Trace file: {report['trace_file']}")
            print(f"Total tasks: {summary['total_tasks']}")
            print(f"Total tool calls: {summary['total_tool_calls']}")
            print(f"Tasks with tool calls: {summary['tasks_with_tools']} ({(summary['tasks_with_tools']/summary['total_tasks'])*100:.1f}%)")
            print(f"Average tools per task: {summary['avg_tools_per_task']:.2f}")
            print(f"Average task duration: {summary['avg_task_duration_ms']:.2f} ms")
            
            # Print task and tool details
            print("\nTasks and Tool Calls:")
            for task_id, task in report['tasks'].items():
                print(f"\nTask {task_id} - Agent: {task['agent_name']}")
                
                # Show request
                request_preview = str(task['request'])
                if len(request_preview) > 80:
                    request_preview = request_preview[:77] + "..."
                print(f"  Request: {request_preview}")
                
                # Show task metrics
                print(f"  Duration: {task['duration_ms']:.2f} ms")
                print(f"  Tool calls: {task['total_tool_calls']} ({task['tool_time_percentage']:.1f}% of task time)")
                
                # Show tool details
                if task['tool_calls']:
                    print("\n  Tool Details:")
                    for i, tool in enumerate(task['tool_calls']):
                        print(f"    {i+1}. {tool['span_name']} - Server: {tool['server_name']}, Method: {tool['method_name']}")
                        print(f"       Duration: {tool['duration_ms']:.2f} ms")
                        
                        # Show arguments if available
                        if tool['arguments']:
                            args_preview = str(tool['arguments'])
                            if len(args_preview) > 80:
                                args_preview = args_preview[:77] + "..."
                            print(f"       Arguments: {args_preview}")
                            
                        # Show result preview for detailed output
                        if args.detailed and tool['result']:
                            result_preview = str(tool['result'])
                            if len(result_preview) > 100:
                                result_preview = result_preview[:97] + "..."
                            print(f"       Result: {result_preview}")
                else:
                    print("  No tool calls in this task")
                    
    elif args.extract_tool_details:
        # Extract detailed tool call information
        tool_details = extract_tool_call_details(args.trace_file, args.agent)
        
        # Write to output file or print summary
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(tool_details, f, indent=2)
            print(f"Tool call details extracted. Saved to {args.output_file}")
        else:
            print(f"\nFound {len(tool_details)} tool calls:")
            
            for i, tool in enumerate(tool_details):
                print(f"\nTool Call {i+1} - Agent: {tool['agent_name']}")
                print(f"  Tool name: {tool['tool_name'] or 'Unknown'}")
                print(f"  Server: {tool['server_name']}, Method: {tool['method_name']}")
                print(f"  Duration: {tool['duration_ms']:.2f} ms, Status: {tool['status']}")
                
                # Show arguments if available
                if tool['tool_arguments']:
                    args_preview = str(tool['tool_arguments'])
                    if len(args_preview) > 80:
                        args_preview = args_preview[:77] + "..."
                    print(f"  Arguments: {args_preview}")
                    
                # Show result if available with detailed option
                if args.detailed and tool['tool_result']:
                    result_preview = str(tool['tool_result'])
                    if len(result_preview) > 200:  # Allow longer preview for results
                        result_preview = result_preview[:197] + "..."
                    print(f"  Result: {result_preview}")
    
    elif args.analyze_tasks:
        # Analyze tool usage per task
        tasks = analyze_tasks_tool_usage(args.trace_file, args.agent)
        
        # Generate summary metrics across all tasks
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
        
        # Write to output file or print summary
        if args.output_file:
            output = {
                "summary": summary,
                "tasks": tasks
            }
            with open(args.output_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Task analysis saved to {args.output_file}")
        else:
            # Print summary
            print(f"\nTask Analysis Summary:")
            print(f"  Total tasks: {summary['total_tasks']}")
            print(f"  Total tool calls: {summary['total_tool_calls']}")
            print(f"  Average tool calls per task: {summary['avg_tool_calls_per_task']:.2f}")
            print(f"  Tasks with tool calls: {summary['tasks_with_tool_calls']} ({summary['tasks_with_tool_calls']/summary['total_tasks']*100:.1f}%)")
            print(f"  Average task duration: {summary['avg_task_duration_ms']:.2f} ms")
            print(f"  Average time spent in tools: {summary['avg_tool_time_percentage']:.1f}%")
            
            # Print individual task details
            print(f"\nFound {len(tasks)} tasks")
            for task in tasks:
                print(f"\nTask {task['task_id']} - Agent: {task['agent_name']}")
                print(f"  Tool calls: {task['tool_call_count']}")
                print(f"  Unique servers: {task['unique_servers_used']}")
                print(f"  Duration: {task['duration_ms']:.2f} ms")
                print(f"  Time in tools: {task['total_tool_time_ms']:.2f} ms ({task['tool_time_percentage']:.1f}%)")
                
                # Show request if available
                if task['task_request']:
                    request_preview = str(task['task_request'])
                    if len(request_preview) > 80:
                        request_preview = request_preview[:77] + "..."
                    print(f"  Request: {request_preview}")
                
                # Show full input if detailed output is requested
                elif args.detailed and task['task_input']:
                    input_preview = str(task['task_input'])
                    if len(input_preview) > 50:
                        input_preview = input_preview[:47] + "..."
                    print(f"  Input: {input_preview}")
                
                # Show tool calls
                if task['tool_calls']:
                    print("\n  Tool call details:")
                    for i, tc in enumerate(task['tool_calls']):
                        print(f"    {i+1}. {tc['name']} - Server: {tc['server_name']}, Method: {tc['method_name']}")
                        print(f"       Duration: {tc['duration_ms']:.2f} ms, Status: {tc['status']}")
                else:
                    print("  No tool calls in this task")


def cmd_workflow_eval(args):
    """Run workflow evaluation command"""
    print("Workflow evaluation is not implemented yet.")
    # Placeholder for future implementation
    evaluation = {"status": "Not implemented", "message": "Workflow evaluation is not available yet."}
    
    # Output results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(evaluation, f, indent=2)
        print(f"Workflow evaluation placeholder saved to {args.output_file}")
    else:
        print(json.dumps(evaluation, indent=2))


def cmd_test_trace_analysis(args):
    """Run test trace analysis"""
    # Get trace files
    if args.trace_files:
        trace_files = [Path(file_path) for file_path in args.trace_files]
    else:
        trace_dir = Path(__file__).resolve().parent / "traces"
        trace_files = list(trace_dir.glob("*.jsonl"))
    
    if not trace_files:
        print(f"No trace files found")
        return
    
    print(f"Found {len(trace_files)} trace files: {[f.name for f in trace_files]}")
    
    # Test with each trace file
    for trace_file in trace_files:
        print(f"\n\n{'=' * 80}")
        print(f"Testing with trace file: {trace_file}")
        print(f"{'=' * 80}")
        spans = test_read_trace_file(str(trace_file))
        test_separate_spans_by_id(spans)
        test_organize_spans_by_parent(spans)
        test_analyze_trace_file(str(trace_file))
        test_get_spans_by_category(spans)
        test_agent_trace_analysis(str(trace_file))
        test_agent_measurements(str(trace_file))
        test_calculate_chat_cost(spans)


def main():
    """Unified command-line entry point for evaluation tools"""
    parser = argparse.ArgumentParser(description="MCP Agent Evaluation and Trace Analysis Tools")
    subparsers = parser.add_subparsers(dest="command", help="Evaluation command", required=True)
    
    # Trace Analysis command
    trace_parser = subparsers.add_parser("trace-analysis", help="Basic trace file analysis")
    trace_parser.add_argument("trace_file", help="Path to the trace file")
    trace_parser.set_defaults(func=cmd_trace_analysis)
    
    # Agent Analysis command
    agent_parser = subparsers.add_parser("agent-analysis", help="Agent trace analysis by component type")
    agent_parser.add_argument("--trace-file", required=True, help="Path to the trace file")
    agent_parser.add_argument("--output-file", help="Path to save the analysis report")
    agent_parser.add_argument("--feature-tree", action="store_true", 
                          help="Generate a feature tree visualization of spans")
    agent_parser.add_argument("--extract-agent-events", action="store_true",
                          help="Extract all events for each agent")
    agent_parser.set_defaults(func=cmd_agent_analysis)
    
    # Tool Metrics command
    metrics_parser = subparsers.add_parser("tool-metrics", help="Agent tool performance metrics")
    metrics_parser.add_argument("--trace-file", required=True, help="Path to the trace file")
    metrics_parser.add_argument("--agent", help="Optional agent name to filter by")
    metrics_parser.add_argument("--output-file", help="Path to save the analysis report")
    metrics_parser.add_argument("--detailed", action="store_true", help="Generate more detailed output")
    metrics_parser.add_argument("--filter", help="Text to filter by (for requests)")
    metrics_parser.add_argument("--measure-tools", action="store_true", help="Measure agent tool call metrics")
    metrics_parser.add_argument("--analyze-tasks", action="store_true", help="Analyze tool calls per task")
    metrics_parser.add_argument("--extract-requests", action="store_true", help="Extract all user requests")
    metrics_parser.add_argument("--extract-tool-details", action="store_true", help="Extract detailed tool call information")
    metrics_parser.add_argument("--comprehensive-report", action="store_true", help="Generate a comprehensive report of tasks and tool calls")
    metrics_parser.set_defaults(func=cmd_tool_metrics)
    
    # Workflow Evaluation command
    workflow_parser = subparsers.add_parser("workflow-eval", help="Evaluate agent workflows")
    workflow_parser.add_argument("--trace-file", required=True, help="Path to the trace file")
    workflow_parser.add_argument("--output-file", help="Path to save evaluation results")
    workflow_parser.set_defaults(func=cmd_workflow_eval)
    
    # Test Trace Analysis command
    test_parser = subparsers.add_parser("test", help="Run test trace analysis")
    test_parser.add_argument("--trace-files", nargs="*", help="Optional list of trace files to test (if not provided, uses traces directory)")
    test_parser.set_defaults(func=cmd_test_trace_analysis)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()