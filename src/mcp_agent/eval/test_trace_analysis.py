#!/usr/bin/env python3
"""
Test script for trace analysis functionality.
This script reads trace files from the 'traces' directory and tests the evaluation functions.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from pprint import pprint

# Use absolute imports
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
    measure_agent_tool_calls, 
    analyze_tasks_tool_usage, 
    extract_user_requests,
    extract_tool_call_details,
    generate_comprehensive_report
)


def test_read_trace_file(trace_file_path: str) -> List[Dict[str, Any]]:
    """Test reading a trace file"""
    print(f"\n=== Testing read_trace_file with {trace_file_path} ===")
    spans = read_trace_file(trace_file_path)
    print(f"Read {len(spans)} spans from {trace_file_path}")
    
    # Print a sample span (first one)
    if spans:
        print("\nSample span structure:")
        sample_span = spans[0]
        print(f"Name: {sample_span.get('name')}")
        print(f"Trace ID: {sample_span.get('context', {}).get('trace_id')}")
        print(f"Start time: {sample_span.get('start_time')}")
        print(f"End time: {sample_span.get('end_time')}")
        print(f"Attributes count: {len(sample_span.get('attributes', {}))}")
    
    return spans


def test_separate_spans_by_id(spans: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Test separating spans by trace ID"""
    print("\n=== Testing separate_spans_by_id ===")
    trace_groups = separate_spans_by_id(spans)
    print(f"Found {len(trace_groups)} distinct trace groups")
    
    for trace_id, group_spans in trace_groups.items():
        print(f"Trace ID: {trace_id} - {len(group_spans)} spans")
    
    return trace_groups


def test_organize_spans_by_parent(spans: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Test organizing spans by parent ID"""
    print("\n=== Testing organize_spans_by_parent ===")
    parent_groups = organize_spans_by_parent(spans)
    print(f"Found {len(parent_groups)} distinct parent groups")
    
    for parent_id, child_spans in parent_groups.items():
        print(f"Parent ID: {parent_id} - {len(child_spans)} child spans")
    
    return parent_groups


def test_analyze_trace_file(trace_file_path: str) -> Dict[str, Any]:
    """Test analyzing a trace file"""
    print(f"\n=== Testing analyze_trace_file with {trace_file_path} ===")
    analysis = analyze_trace_file(trace_file_path)
    
    print(f"Total spans: {analysis.get('total_spans')}")
    print(f"Trace count: {analysis.get('trace_count')}")
    
    for trace_id, trace_info in analysis.get("traces", {}).items():
        print(f"\nTrace ID: {trace_id}")
        print(f"  Spans: {trace_info.get('span_count')}")
        print(f"  Root spans: {trace_info.get('root_span_count')}")
        print(f"  Agent names: {', '.join(trace_info.get('agent_names', []))}")
        
        # Show span types
        print("  Span types:")
        for span_type, count in trace_info.get('span_types', {}).items():
            print(f"    {span_type}: {count}")
    
    return analysis


def test_get_spans_by_category(spans: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Test categorizing spans"""
    print("\n=== Testing get_spans_by_category ===")
    categories = get_spans_by_category(spans)
    
    for category, category_spans in categories.items():
        print(f"Category: {category} - {len(category_spans)} spans")
        
        # Show sample names for each category
        if category_spans:
            sample_names = [span.get('name') for span in category_spans[:3]]
            print(f"  Sample names: {', '.join(sample_names)}")
    
    return categories


def test_agent_trace_analysis(trace_file_path: str) -> None:
    """Test agent trace analysis functionality"""
    print(f"\n=== Testing agent trace analysis with {trace_file_path} ===")
    
    # Test analyze_trace_by_component
    print("\n--- Testing analyze_trace_by_component ---")
    component_traces = analyze_trace_by_component(trace_file_path)
    for component_type, traces in component_traces.items():
        print(f"Component: {component_type} - {len(traces)} traces")
    
    # Test get_trace_summary
    print("\n--- Testing get_trace_summary ---")
    summary = get_trace_summary(trace_file_path)
    print(f"Components: {summary.get('components')}")
    print(f"Component counts: {summary.get('component_counts')}")
    print(f"Execution time: {summary.get('execution_time') / 1e9 if summary.get('execution_time') else 'N/A'} seconds")
    
    # Test get_detailed_trace_analysis
    print("\n--- Testing get_detailed_trace_analysis ---")
    detailed_analysis = get_detailed_trace_analysis(trace_file_path)
    print(f"Workflow steps: {len(detailed_analysis.get('workflow_steps', []))}")
    print(f"LLM interactions: {len(detailed_analysis.get('llm_interactions', []))}")
    print(f"MCP operations: {len(detailed_analysis.get('mcp_operations', []))}")
    
    # Test report generation
    print("\n--- Testing generate_trace_report ---")
    report = generate_trace_report(detailed_analysis)
    report_summary = '\n'.join(report.split('\n')[:10]) + "\n... (report truncated)"
    print(f"Report preview:\n{report_summary}")
    
    # Create a temporary report file
    test_report_path = f"{trace_file_path}.report.md"
    with open(test_report_path, "w") as f:
        f.write(report)
    print(f"Full report saved to: {test_report_path}")
    
    # Test feature span extraction
    print("\n--- Testing feature span extraction ---")
    spans = read_trace_file(trace_file_path)
    feature_spans = extract_spans_by_feature(spans)
    for feature, feature_spans_list in feature_spans.items():
        print(f"Feature: {feature} - {len(feature_spans_list)} spans")
    
    # Test feature spans tree visualization
    print("\n--- Testing feature spans tree visualization ---")
    tree_output = print_feature_spans_tree(feature_spans)
    tree_preview = '\n'.join(tree_output.split('\n')[:10]) + "\n... (tree output truncated)"
    print(f"Tree preview:\n{tree_preview}")
    
    # Create a temporary tree output file
    tree_output_path = f"{trace_file_path}.feature_tree.txt"
    with open(tree_output_path, "w") as f:
        f.write(tree_output)
    print(f"Full feature tree saved to: {tree_output_path}")


def test_agent_measurements(trace_file_path: str) -> None:
    """Test agent measurement functionality"""
    print(f"\n=== Testing agent measurements with {trace_file_path} ===")
    
    # Test extract_agent_events
    print("\n--- Testing extract_agent_events ---")
    agent_events = extract_agent_events(trace_file_path)
    for agent_name, events in agent_events.items():
        print(f"Agent: {agent_name} - {len(events)} events")
        
        # Count event types
        event_types = {}
        for event in events:
            event_type = event.get('event_type', event.get('type', 'unknown'))
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
        # Display counts by event type
        for event_type, count in event_types.items():
            print(f"  {event_type}: {count} events")
            
        # Show a preview of events (first 3)
        print("\n  Event samples:")
        for i, event in enumerate(events[:3]):
            event_name = event.get('name', 'Unnamed')
            event_type = event.get('event_type', event.get('type', 'unknown'))
            print(f"    {i+1}. [{event_type}] {event_name}")
            
            # Print a few key attributes if present
            if event.get('model'):
                print(f"       Model: {event.get('model')}")
            if event.get('input_tokens') and event.get('output_tokens'):
                print(f"       Tokens: {event.get('input_tokens')} in, {event.get('output_tokens')} out")
            if event.get('server_name'):
                print(f"       Server: {event.get('server_name')}")
            if event.get('method'):
                print(f"       Method: {event.get('method')}")
            if event.get('duration'):
                print(f"       Duration: {event.get('duration') / 1e6:.2f} ms")
                
        print("    ...")
    
    # Save agent events to a file for inspection
    agent_events_path = f"{trace_file_path}.agent_events.json"
    with open(agent_events_path, "w") as f:
        json.dump(agent_events, f, indent=2)
    print(f"Full agent events saved to: {agent_events_path}")
    
    # Test measure_agent_tool_calls
    print("\n--- Testing measure_agent_tool_calls ---")
    metrics = measure_agent_tool_calls(trace_file_path)
    for agent_name, agent_metrics in metrics.items():
        print(f"Agent: {agent_name}")
        print(f"  Total tool calls: {agent_metrics['total_tool_calls']}")
        print(f"  Success rate: {agent_metrics['success_rate']:.2f}%")
        if agent_metrics['avg_duration']:
            print(f"  Average duration: {agent_metrics['avg_duration']:.2f} ms")
            print(f"  Min duration: {agent_metrics['min_duration']:.2f} ms")
            print(f"  Max duration: {agent_metrics['max_duration']:.2f} ms")
        
        # Display server-specific metrics
        print("\n  Tool calls by server:")
        for server, count in agent_metrics['tool_calls_by_server'].items():
            print(f"    {server}: {count} calls")
        
        # Display method-specific metrics
        print("\n  Tool calls by method:")
        for method, count in agent_metrics['tool_calls_by_method'].items():
            print(f"    {method}: {count} calls")
    
    # Save tool metrics to a file for inspection
    tool_metrics_path = f"{trace_file_path}.tool_metrics.json"
    with open(tool_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Tool call metrics saved to: {tool_metrics_path}")
    
    # Test analyze_tasks_tool_usage
    print("\n--- Testing analyze_tasks_tool_usage ---")
    tasks = analyze_tasks_tool_usage(trace_file_path)
    print(f"Found {len(tasks)} tasks")
    
    for task in tasks:
        print(f"\nTask {task['task_id']} - Agent: {task['agent_name']}")
        print(f"  Tool calls: {task['tool_call_count']}")
        print(f"  Duration: {task['duration_ms']:.2f} ms")
        print(f"  Time in tools: {task['total_tool_time_ms']:.2f} ms ({task['tool_time_percentage']:.1f}%)")
        
        # Show request if available
        if task['task_request']:
            request_preview = str(task['task_request'])
            if len(request_preview) > 80:
                request_preview = request_preview[:77] + "..."
            print(f"  Request: {request_preview}")
        # Show input preview if available as fallback
        elif task['task_input']:
            input_preview = str(task['task_input'])
            if len(input_preview) > 50:
                input_preview = input_preview[:47] + "..."
            print(f"  Input preview: {input_preview}")
            
        # Show brief summary of tool calls
        if task['tool_calls']:
            tool_servers = {}
            for tc in task['tool_calls']:
                server = tc['server_name']
                tool_servers[server] = tool_servers.get(server, 0) + 1
                
            print("  Tool calls by server:")
            for server, count in tool_servers.items():
                print(f"    {server}: {count}")
    
    # Save task analysis to a file for inspection
    tasks_path = f"{trace_file_path}.tasks_analysis.json"
    with open(tasks_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Task analysis saved to: {tasks_path}")
    
    # Test extract_user_requests
    print("\n--- Testing extract_user_requests ---")
    requests = extract_user_requests(trace_file_path)
    
    print(f"Found {len(requests)} user requests:")
    for i, req in enumerate(requests):
        print(f"\nRequest {i+1} - Agent: {req['agent_name']}")
        request_preview = str(req['request'])
        if len(request_preview) > 80:
            request_preview = request_preview[:77] + "..."
        print(f"  Request: {request_preview}")
        print(f"  Tool calls: {req['tool_call_count']}")
        print(f"  Duration: {req['duration_ms']:.2f} ms")
    
    # Test request filtering
    if requests:
        # Find a word to filter that should match at least one request
        filter_word = None
        for req in requests:
            words = req['request'].split()
            if words:
                # Find a word longer than 4 characters
                for word in words:
                    if len(word) > 4:
                        filter_word = word
                        break
            if filter_word:
                break
                
        if filter_word:
            print(f"\n--- Testing request filtering with term: '{filter_word}' ---")
            filtered_requests = extract_user_requests(trace_file_path, filter_text=filter_word)
            print(f"Found {len(filtered_requests)} filtered requests")
            
            for i, req in enumerate(filtered_requests):
                request_preview = str(req['request'])
                if len(request_preview) > 80:
                    request_preview = request_preview[:77] + "..."
                print(f"  {i+1}. {request_preview}")
        
    # Save requests to a file for inspection
    requests_path = f"{trace_file_path}.user_requests.json"
    with open(requests_path, "w") as f:
        json.dump(requests, f, indent=2)
    print(f"User requests saved to: {requests_path}")
    
    # Test extract_tool_call_details
    print("\n--- Testing extract_tool_call_details ---")
    tool_details = extract_tool_call_details(trace_file_path)
    
    print(f"Found {len(tool_details)} tool calls:")
    for i, tool in enumerate(tool_details[:5]):  # Show first 5 for brevity
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
            
        # Show result preview
        if tool['tool_result']:
            result_preview = str(tool['tool_result'])
            if len(result_preview) > 100:
                result_preview = result_preview[:97] + "..."
            print(f"  Result: {result_preview}")
            
    if len(tool_details) > 5:
        print(f"  ... and {len(tool_details) - 5} more tool calls")
        
    # Save tool details to a file for inspection
    tool_details_path = f"{trace_file_path}.tool_details.json"
    with open(tool_details_path, "w") as f:
        json.dump(tool_details, f, indent=2)
    print(f"Tool call details saved to: {tool_details_path}")
    
    # Test comprehensive report generation
    print("\n--- Testing comprehensive report generation ---")
    report = generate_comprehensive_report(trace_file_path)
    
    # Print report summary
    summary = report['summary']
    print(f"Generated comprehensive report:")
    print(f"  Total tasks: {summary['total_tasks']}")
    print(f"  Total tool calls: {summary['total_tool_calls']}")
    print(f"  Tasks with tools: {summary['tasks_with_tools']} ({(summary['tasks_with_tools']/summary['total_tasks'])*100 if summary['total_tasks'] > 0 else 0:.1f}%)")
    
    # Sample a few tasks from the report
    print("\n  Sample tasks:")
    task_ids = list(report['tasks'].keys())[:3]  # Show first 3 tasks
    for task_id in task_ids:
        task = report['tasks'][task_id]
        print(f"    Task {task_id} - Agent: {task['agent_name']}")
        print(f"      Tool calls: {task['total_tool_calls']}")
        
        # Show request if available
        if task['request'] and task['request'] != 'Unknown':
            request_preview = str(task['request'])
            if len(request_preview) > 50:
                request_preview = request_preview[:47] + "..."
            print(f"      Request: {request_preview}")
    
    # Save comprehensive report to a file
    report_path = f"{trace_file_path}.comprehensive_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Comprehensive report saved to: {report_path}")


def test_calculate_chat_cost(spans: List[Dict[str, Any]]) -> None:
    """Test calculating chat cost"""
    print("\n=== Testing calculate_chat_cost ===")
    
    llm_spans = []
    for span in spans:
        name = span.get('name', '')
        if name.startswith('llm.') or name.startswith('openai.') or name.startswith('anthropic.'):
            llm_spans.append(span)
    
    print(f"Found {len(llm_spans)} LLM spans")
    
    if llm_spans:
        # Add test data for cost calculation
        test_span = llm_spans[0]
        
        # Make a copy with test data
        test_span_copy = test_span.copy()
        test_span_copy['data'] = {
            'model': 'gpt-4o-mini',
            'usage': {
                'prompt_tokens': 1000,
                'completion_tokens': 500
            }
        }
        
        try:
            cost = calculate_chat_cost(test_span_copy)
            print(f"Calculated cost for test span: ${cost:.6f}")
        except ValueError as e:
            print(f"Cost calculation error: {str(e)}")
            print("You may need to add pricing information for the model in the calculate_chat_cost function")



# Main function removed and placed in eval_runner.py
# This module now provides test utility functions used by eval_runner.py


if __name__ == "__main__":
    # For backward compatibility, import and run the unified entry point
    from mcp_agent.eval.eval_runner import main
    main()