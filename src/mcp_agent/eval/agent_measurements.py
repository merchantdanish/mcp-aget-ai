"""
Agent measurement module.

This module provides tools to measure and analyze agent performance metrics:
1. measure_agent_tool_calls - Extracts metrics about tool calls made by agents
2. extract_agent_events - Obtains all related events from spans for each agent
3. analyze_tasks_tool_usage - Analyzes how many tool calls were needed per task
4. extract_user_requests - Extracts all user requests/prompts from the trace
5. extract_tool_call_details - Extracts detailed information about tool calls including arguments and results
6. generate_comprehensive_report - Creates a unified report of tasks, requests, and tool calls

Example usage:
    # Get tool call metrics for all agents
    metrics = measure_agent_tool_calls(trace_file_path)
    
    # Get all events for a specific agent
    events = extract_agent_events(trace_file_path)
    
    # Analyze tool usage per task
    task_metrics = analyze_tasks_tool_usage(trace_file_path)
    
    # Extract user requests with corresponding metrics
    requests = extract_user_requests(trace_file_path)
    
    # Extract detailed tool call information
    tool_details = extract_tool_call_details(trace_file_path)
    
    # Generate comprehensive report combining tasks and tools
    report = generate_comprehensive_report(trace_file_path)
    
    # Run from command line
    python agent_measurements.py --trace-file=path/to/trace.jsonl --measure-tools
    python agent_measurements.py --trace-file=path/to/trace.jsonl --extract-events
    python agent_measurements.py --trace-file=path/to/trace.jsonl --analyze-tasks
    python agent_measurements.py --trace-file=path/to/trace.jsonl --extract-requests
    python agent_measurements.py --trace-file=path/to/trace.jsonl --extract-tool-details
    python agent_measurements.py --trace-file=path/to/trace.jsonl --comprehensive-report
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timezone

import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(abspath(__file__)))

from evals import read_trace_file


def extract_tool_call_details(trace_file_path: str, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract detailed information about tool calls including arguments and results.
    
    Args:
        trace_file_path: Path to the trace JSONL file
        agent_name: Optional name of the agent to filter by
        
    Returns:
        A list of dictionaries containing detailed tool call information
    """
    # Read all spans from the trace file
    spans = read_trace_file(trace_file_path)
    
    # Sort spans by timestamp to ensure chronological order
    def parse_timestamp(timestamp):
        if isinstance(timestamp, str):
            try:
                from datetime import datetime
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp() * 1e9
            except (ValueError, TypeError):
                return 0
        return timestamp or 0
    
    # Sort spans chronologically
    sorted_spans = sorted(
        spans,
        key=lambda x: parse_timestamp(x.get('start_time'))
    )
    
    # Extract tool call details
    tool_calls = []
    for span in sorted_spans:
        attributes = span.get('attributes', {})
        span_name = span.get('name', '')
        span_agent_name = attributes.get('gen_ai.agent.name', 'unknown')
        
        # Apply agent filter if specified
        if agent_name and span_agent_name != agent_name:
            continue
            
        # Identify tool calls
        is_tool_call = (
            "tool" in span_name.lower() or 
            "call_tool" in span_name.lower() or
            (attributes.get('mcp.method.name') == 'tools/call')
        )
        
        if is_tool_call:
            # Extract timestamps
            start_time = parse_timestamp(span.get('start_time'))
            end_time = parse_timestamp(span.get('end_time'))
            
            # Calculate duration in milliseconds
            duration_ms = (end_time - start_time) / 1e6 if end_time and start_time else None
            
            # Extract tool details
            tool_name = None
            tool_args = None
            tool_result = None
            
            # Look for tool name in various attributes
            for key in ['tool.name', 'mcp.tools.name', 'name', 'tool', 'tools.name']:
                if key in attributes:
                    tool_name = attributes[key]
                    break
                    
            # Check for tool names in tool description attributes (format: tool.{server}_{tool})
            if not tool_name:
                for key in attributes:
                    if key.startswith('tool.') and '_' in key and '.description' not in key and '.inputSchema' not in key:
                        parts = key.split('.')
                        if len(parts) == 2:
                            tool_parts = parts[1].split('_', 1)
                            if len(tool_parts) == 2:
                                server_name, actual_tool = tool_parts
                                tool_name = f"{server_name}/{actual_tool}"
                                break
            
            # Look for tool arguments in various attributes
            for key in ['mcp.tools.arguments', 'tool.arguments', 'arguments', 'args', 'params', 'parameters']:
                if key in attributes:
                    tool_args = attributes[key]
                    break
                    
            # Check for mcp.request.argument pattern which might contain tool arguments
            if not tool_args:
                for key in attributes:
                    if key.startswith('mcp.request.argument.'):
                        # Collect all mcp.request.argument.* keys as a potential arguments object
                        if not tool_args:
                            tool_args = {}
                        param_name = key.replace('mcp.request.argument.', '')
                        tool_args[param_name] = attributes[key]
            
            # Look for tool results in various attributes
            for key in ['mcp.tools.result', 'tool.result', 'result', 'output', 'response']:
                if key in attributes:
                    tool_result = attributes[key]
                    break
                    
            # Check for result.* pattern which might contain tool results
            if not tool_result:
                result_keys = {}
                for key in attributes:
                    if key.startswith('result.') and key != 'result_type':
                        # Collect all result.* keys as a potential result object
                        param_name = key.replace('result.', '')
                        result_keys[param_name] = attributes[key]
                if result_keys:
                    tool_result = result_keys
                    
            # Check events for additional details
            for event in span.get('events', []):
                event_name = event.get('name', '')
                event_attrs = event.get('attributes', {})
                
                # Look for tool execution events
                if any(term in event_name.lower() for term in ['tool', 'execute', 'call', 'invoke']):
                    # Extract tool info from event if not already found
                    if not tool_name:
                        for key in ['tool.name', 'mcp.tools.name', 'name', 'tool']:
                            if key in event_attrs:
                                tool_name = event_attrs[key]
                                break
                                
                    if not tool_args:
                        for key in ['mcp.tools.arguments', 'tool.arguments', 'arguments', 'args', 'params']:
                            if key in event_attrs:
                                tool_args = event_attrs[key]
                                break
                                
                    if not tool_result:
                        for key in ['mcp.tools.result', 'tool.result', 'result', 'output', 'response']:
                            if key in event_attrs:
                                tool_result = event_attrs[key]
                                break
            
            # Create tool call record
            tool_call = {
                'agent_name': span_agent_name,
                'span_id': span.get('span_id'),
                'span_name': span_name,
                'timestamp': start_time,
                'duration_ms': duration_ms,
                'status': span.get('status', {}).get('status_code', 'UNKNOWN'),
                'server_name': attributes.get('server_name', 'unknown'),
                'method_name': attributes.get('mcp.method.name', 'unknown'),
                'tool_name': tool_name,
                'tool_arguments': tool_args,
                'tool_result': tool_result,
                'raw_attributes': attributes  # Include all attributes for further analysis
            }
            
            tool_calls.append(tool_call)
    
    # Sort by timestamp
    tool_calls.sort(key=lambda x: x['timestamp'])
    
    return tool_calls


def measure_agent_tool_calls(trace_file_path: str, agent_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Measure tool call metrics for a specific agent or all agents.
    
    Args:
        trace_file_path: Path to the trace JSONL file
        agent_name: Optional name of the agent to measure (if None, measure all agents)
        
    Returns:
        A dictionary mapping agent names to their tool call metrics
    """
    # Read all spans from the trace file
    spans = read_trace_file(trace_file_path)
    
    # Initialize metrics dictionary
    agent_metrics = {}
    
    # Process each span to extract tool call information
    for span in spans:
        attributes = span.get('attributes', {})
        span_name = span.get('name', '')
        span_agent_name = attributes.get('gen_ai.agent.name', 'unknown')
        
        # Filter by agent name if specified
        if agent_name and span_agent_name != agent_name:
            continue
            
        # Initialize metrics for this agent if not already present
        if span_agent_name not in agent_metrics:
            agent_metrics[span_agent_name] = {
                'total_tool_calls': 0,
                'tool_calls_by_server': {},
                'tool_calls_by_method': {},
                'tool_call_durations': [],
                'successful_calls': 0,
                'failed_calls': 0,
                'total_duration': 0
            }
            
        # Look for tool call spans
        is_tool_call = (
            "tool" in span_name.lower() or 
            "call_tool" in span_name.lower() or
            (attributes.get('mcp.method.name') == 'tools/call')
        )
        
        if is_tool_call:
            # Extract tool call details
            server_name = attributes.get('server_name', 'unknown')
            method_name = attributes.get('mcp.method.name', 'unknown')
            
            # Extract timestamps and calculate duration
            start_time = span.get('start_time')
            end_time = span.get('end_time')
            
            # Parse timestamps if they're strings
            if isinstance(start_time, str):
                try:
                    from datetime import datetime
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp() * 1e9
                except (ValueError, TypeError):
                    start_time = None
                    
            if isinstance(end_time, str):
                try:
                    from datetime import datetime
                    end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00')).timestamp() * 1e9
                except (ValueError, TypeError):
                    end_time = None
            
            # Calculate duration in milliseconds
            duration = None
            if start_time is not None and end_time is not None:
                try:
                    duration = (end_time - start_time) / 1e6  # Convert ns to ms
                except (TypeError, ValueError):
                    duration = None
            
            # Check if call was successful
            status = span.get('status', {})
            is_success = status.get('status_code') != 'ERROR'
            
            # Increment counts
            agent_metrics[span_agent_name]['total_tool_calls'] += 1
            
            # Update server-specific counts
            if server_name not in agent_metrics[span_agent_name]['tool_calls_by_server']:
                agent_metrics[span_agent_name]['tool_calls_by_server'][server_name] = 0
            agent_metrics[span_agent_name]['tool_calls_by_server'][server_name] += 1
            
            # Update method-specific counts
            if method_name not in agent_metrics[span_agent_name]['tool_calls_by_method']:
                agent_metrics[span_agent_name]['tool_calls_by_method'][method_name] = 0
            agent_metrics[span_agent_name]['tool_calls_by_method'][method_name] += 1
            
            # Update success/failure counts
            if is_success:
                agent_metrics[span_agent_name]['successful_calls'] += 1
            else:
                agent_metrics[span_agent_name]['failed_calls'] += 1
                
            # Add duration data
            if duration is not None:
                agent_metrics[span_agent_name]['tool_call_durations'].append(duration)
                agent_metrics[span_agent_name]['total_duration'] += duration
    
    # Calculate average durations and success rates
    for agent_name, metrics in agent_metrics.items():
        if metrics['tool_call_durations']:
            metrics['avg_duration'] = sum(metrics['tool_call_durations']) / len(metrics['tool_call_durations'])
            metrics['min_duration'] = min(metrics['tool_call_durations'])
            metrics['max_duration'] = max(metrics['tool_call_durations'])
        else:
            metrics['avg_duration'] = 0
            metrics['min_duration'] = 0
            metrics['max_duration'] = 0
            
        # Calculate success rate
        total_calls = metrics['total_tool_calls']
        if total_calls > 0:
            metrics['success_rate'] = (metrics['successful_calls'] / total_calls) * 100
        else:
            metrics['success_rate'] = 0
    
    return agent_metrics


def extract_agent_events(trace_file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract all events from spans in the trace for each agent.
    
    Args:
        trace_file_path: Path to the trace JSONL file
        
    Returns:
        A dictionary mapping agent names to lists of events
    """
    # Read all spans from the trace file
    spans = read_trace_file(trace_file_path)
    
    # Extract all agent names from spans
    agent_names = set()
    for span in spans:
        attributes = span.get('attributes', {})
        agent_name = attributes.get('gen_ai.agent.name')
        if agent_name:
            agent_names.add(agent_name)
    
    # If no agent names found, create a default "unknown" agent
    if not agent_names:
        agent_names = {"unknown"}
    
    # Initialize events dictionary for each agent
    agent_events = {agent_name: [] for agent_name in agent_names}
    
    # Function to format timestamps consistently
    def parse_timestamp(timestamp):
        if isinstance(timestamp, str):
            try:
                from datetime import datetime
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp() * 1e9
            except (ValueError, TypeError):
                return 0
        return timestamp or 0
    
    # Process each span to extract relevant events
    for span in spans:
        # Get span attributes and name
        attributes = span.get('attributes', {})
        span_name = span.get('name', '')
        agent_name = attributes.get('gen_ai.agent.name', 'unknown')
        
        # Make sure the agent exists in our dictionary
        if agent_name not in agent_events:
            agent_events[agent_name] = []
        
        # Extract span timestamps
        start_time = parse_timestamp(span.get('start_time'))
        end_time = parse_timestamp(span.get('end_time'))
        
        # Determine event type based on span name
        event_type = "other"
        if any(prefix in span_name.lower() for prefix in ["llm", "openai", "anthropic"]):
            event_type = "llm_call"
        elif "tool" in span_name.lower() or "call_tool" in span_name.lower():
            event_type = "tool_call"
        elif "MCPAgentClientSession" in span_name:
            event_type = "mcp_session"
        elif "agent" in span_name.lower():
            event_type = "agent_operation"
        elif "workflow" in span_name.lower() or "orchestrator" in span_name.lower() or "parallel" in span_name.lower():
            event_type = "workflow"
        
        # Create a span event
        span_event = {
            "type": "span",
            "event_type": event_type,
            "name": span_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time if end_time and start_time else None,
            "attributes": attributes.copy()
        }
        
        # Add detailed information based on event type
        if event_type == "llm_call":
            # Extract LLM-specific details
            if "gen_ai.request.model" in attributes:
                span_event["model"] = attributes["gen_ai.request.model"]
            if "gen_ai.usage.input_tokens" in attributes:
                span_event["input_tokens"] = attributes["gen_ai.usage.input_tokens"]
            if "gen_ai.usage.output_tokens" in attributes:
                span_event["output_tokens"] = attributes["gen_ai.usage.output_tokens"]
                
        elif event_type == "tool_call":
            # Extract tool-specific details
            if "mcp.method.name" in attributes:
                span_event["method"] = attributes["mcp.method.name"]
            if "server_name" in attributes:
                span_event["server_name"] = attributes["server_name"]
                
        # Add the span event to the agent's events list
        agent_events[agent_name].append(span_event)
        
        # Process sub-events within the span
        for event in span.get('events', []):
            # Extract event timestamp and name
            event_timestamp = parse_timestamp(event.get('timestamp'))
            event_name = event.get('name', '')
            event_attributes = event.get('attributes', {})
            
            # Create sub-event record
            sub_event = {
                "type": "sub_event",
                "name": event_name,
                "timestamp": event_timestamp,
                "attributes": event_attributes,
                "parent_span_name": span_name,
                "parent_span_id": span.get('span_id'),
                "agent_name": agent_name
            }
            
            # Add sub-event to agent's events list
            agent_events[agent_name].append(sub_event)
    
    # Sort each agent's events by timestamp
    for agent_name in agent_events:
        agent_events[agent_name] = sorted(
            agent_events[agent_name],
            key=lambda x: x.get('start_time') or x.get('timestamp') or 0
        )
    
    return agent_events


def analyze_tasks_tool_usage(trace_file_path: str, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Analyze how many tool calls were needed to complete each task.
    
    Args:
        trace_file_path: Path to the trace JSONL file
        agent_name: Optional name of the agent to analyze (if None, analyze all agents)
        
    Returns:
        A list of task metrics, each containing tool usage information
    """
    # Read all spans from the trace file
    spans = read_trace_file(trace_file_path)
    
    # Sort spans by timestamp to ensure chronological order
    def parse_timestamp(timestamp):
        if isinstance(timestamp, str):
            try:
                from datetime import datetime
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp() * 1e9
            except (ValueError, TypeError):
                return 0
        return timestamp or 0
    
    # Sort spans chronologically
    sorted_spans = sorted(
        spans,
        key=lambda x: parse_timestamp(x.get('start_time'))
    )
    
    # Extract LLM generation spans (which likely represent task boundaries)
    llm_spans = []
    for span in sorted_spans:
        span_name = span.get('name', '')
        span_agent_name = span.get('attributes', {}).get('gen_ai.agent.name')
        
        # Apply agent filter if specified
        if agent_name and span_agent_name != agent_name:
            continue
            
        # Identify LLM generate calls - use broader patterns to capture various LLM interactions
        is_llm_span = (
            any(prefix in span_name.lower() for prefix in ['llm.generate', 'openai.generate', 'anthropic.generate']) or
            (span_name.startswith('llm') and 'generate' in span_name.lower()) or
            ('openai' in span_name.lower() and not 'tool' in span_name.lower()) or
            ('anthropic' in span_name.lower() and not 'tool' in span_name.lower()) or
            # Look for generation-related attributes
            any(key in span.get('attributes', {}) for key in [
                'gen_ai.usage.input_tokens', 'gen_ai.completion.text', 
                'gen_ai.request.model', 'gen_ai.request.messages'
            ])
        )
        
        if is_llm_span:
            llm_spans.append(span)
    
    # Group tool calls between LLM generations (each LLM generation likely represents a task)
    tasks = []
    
    # Loop through LLM spans to identify task boundaries
    for i in range(len(llm_spans)):
        current_llm_span = llm_spans[i]
        
        # Calculate task start time 
        start_time = parse_timestamp(current_llm_span.get('start_time'))
        
        # Calculate task end time (either next LLM generation or end of trace)
        if i < len(llm_spans) - 1:
            next_llm_span = llm_spans[i + 1]
            end_time = parse_timestamp(next_llm_span.get('start_time'))
        else:
            # For the last task, find the latest span end time
            end_time = max([parse_timestamp(span.get('end_time')) for span in sorted_spans])
        
        # Get spans belonging to this task (between current and next LLM call)
        task_spans = []
        for span in sorted_spans:
            span_start = parse_timestamp(span.get('start_time'))
            if start_time <= span_start < end_time:
                # Skip the LLM generation span itself
                if span is not current_llm_span:
                    task_spans.append(span)
        
        # Extract tool calls within this task
        tool_calls = []
        for span in task_spans:
            attributes = span.get('attributes', {})
            span_name = span.get('name', '')
            span_agent_name = attributes.get('gen_ai.agent.name')
            
            # Apply agent filter if specified
            if agent_name and span_agent_name != agent_name:
                continue
                
            # Identify tool calls
            is_tool_call = (
                "tool" in span_name.lower() or 
                "call_tool" in span_name.lower() or
                (attributes.get('mcp.method.name') == 'tools/call')
            )
            
            if is_tool_call:
                # Extract tool call details
                server_name = attributes.get('server_name', 'unknown')
                method_name = attributes.get('mcp.method.name', 'unknown')
                
                # Calculate duration
                duration = parse_timestamp(span.get('end_time')) - parse_timestamp(span.get('start_time'))
                duration_ms = duration / 1e6 if duration else None
                
                # Tool call information
                tool_call = {
                    'span_id': span.get('span_id'),
                    'name': span_name,
                    'agent_name': span_agent_name,
                    'server_name': server_name,
                    'method_name': method_name,
                    'start_time': parse_timestamp(span.get('start_time')),
                    'end_time': parse_timestamp(span.get('end_time')),
                    'duration_ms': duration_ms,
                    'status': span.get('status', {}).get('status_code', 'UNKNOWN')
                }
                
                tool_calls.append(tool_call)
        
        # Extract task input/message if available
        task_input = None
        task_request = None
        if 'attributes' in current_llm_span:
            attributes = current_llm_span.get('attributes', {})
            
            # Look for common patterns in attributes that might contain user message
            for key in ['gen_ai.request.messages', 'gen_ai.prompt.text', 'request.messages', 'messages']:
                if key in attributes:
                    task_input = attributes[key]
                    
                    # Try to extract the user message/request
                    if isinstance(task_input, list):
                        # For message arrays (OpenAI/Claude format)
                        for msg in task_input:
                            if isinstance(msg, dict) and msg.get('role') == 'user':
                                task_request = msg.get('content')
                                break
                    elif isinstance(task_input, str):
                        # For direct string prompts
                        task_request = task_input
                    
                    # If it's a model name, don't use it as a request
                    if task_request and any(model_name in task_request for model_name in 
                        ['gpt-', 'claude-', 'llama', 'mistral', 'gemini']):
                        if len(task_request) < 30:  # Simple check to detect model names
                            task_request = None
                            
                    break
                    
            # If we didn't find a structured message, try other attributes
            if task_request is None:
                # First, try to find user message in common patterns
                message_keys = [
                    'gen_ai.request.user_message', 'prompt.text', 'input.query',
                    'input.message', 'user.input', 'request.text', 'query'
                ]
                
                for key in message_keys:
                    if key in attributes:
                        value = attributes[key]
                        if isinstance(value, str) and len(value) > 5:
                            task_request = value
                            break
                
                # If still no message, try to guess from any attribute containing message-like terms
                if task_request is None:
                    for key in attributes:
                        if any(term in key.lower() for term in 
                               ['prompt', 'query', 'question', 'input', 'request', 'message', 'instruction']):
                            value = attributes[key]
                            if isinstance(value, str) and len(value) > 10:  # Require longer text to avoid model names
                                # Skip if it looks like a model name
                                if not any(model_name in value for model_name in 
                                      ['gpt-', 'claude-', 'llama', 'mistral', 'gemini']):
                                    task_request = value
                                    break
            
            # Check events for possible prompt information
            if task_request is None:
                for event in current_llm_span.get('events', []):
                    event_name = event.get('name', '')
                    # Look for events that might contain user prompts
                    if any(term in event_name.lower() for term in ['prompt', 'input', 'request', 'message']):
                        event_attrs = event.get('attributes', {})
                        for key, value in event_attrs.items():
                            if isinstance(value, str) and len(value) > 10:
                                # Skip if it looks like a model name
                                if not any(model_name in value for model_name in 
                                          ['gpt-', 'claude-', 'llama', 'mistral', 'gemini']):
                                    task_request = value
                                    break
        
        # Extract task output if available
        task_output = None
        if 'attributes' in current_llm_span:
            attributes = current_llm_span.get('attributes', {})
            # Look for common patterns in attributes that might contain LLM response
            for key in ['gen_ai.response.message', 'gen_ai.completion.text', 'response.message']:
                if key in attributes:
                    task_output = attributes[key]
                    break
        
        # Create task record
        task = {
            'task_id': i + 1,  # 1-based task ID for readability
            'llm_span_id': current_llm_span.get('span_id'),
            'agent_name': current_llm_span.get('attributes', {}).get('gen_ai.agent.name', 'unknown'),
            'start_time': start_time,
            'end_time': end_time,
            'duration_ms': (end_time - start_time) / 1e6,
            'task_input': task_input,
            'task_request': task_request,  # Add extracted user request
            'task_output': task_output,
            'tool_calls': tool_calls,
            'tool_call_count': len(tool_calls),
            'unique_servers_used': len(set(tc['server_name'] for tc in tool_calls)),
            'unique_methods_used': len(set(tc['method_name'] for tc in tool_calls)),
            'total_tool_time_ms': sum(tc['duration_ms'] for tc in tool_calls if tc['duration_ms']) if tool_calls else 0
        }
        
        # Calculate percentage of time spent in tools
        if task['duration_ms'] > 0:
            task['tool_time_percentage'] = (task['total_tool_time_ms'] / task['duration_ms']) * 100
        else:
            task['tool_time_percentage'] = 0
        
        tasks.append(task)
    
    return tasks


def extract_user_requests(
    trace_file_path: str, 
    agent_name: Optional[str] = None,
    filter_text: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract all user requests from a trace file.
    
    Args:
        trace_file_path: Path to the trace JSONL file
        agent_name: Optional name of the agent to filter by
        filter_text: Optional text to filter requests (case-insensitive substring match)
        
    Returns:
        A list of dictionaries containing user requests and related information
    """
    # Use the task analysis function to get tasks with requests
    tasks = analyze_tasks_tool_usage(trace_file_path, agent_name)
    
    # Extract requests from each task
    requests = []
    for task in tasks:
        # Skip tasks without a request
        if not task.get('task_request'):
            continue
            
        # Apply text filter if specified
        if filter_text and filter_text.lower() not in task['task_request'].lower():
            continue
            
        request_info = {
            'task_id': task['task_id'],
            'agent_name': task['agent_name'],
            'timestamp': task['start_time'],
            'request': task['task_request'],
            'tool_call_count': task['tool_call_count'],
            'duration_ms': task['duration_ms'],
            'tool_servers_used': list(set(tc['server_name'] for tc in task['tool_calls'])) if task['tool_calls'] else []
        }
        
        requests.append(request_info)
    
    # Sort by timestamp
    requests.sort(key=lambda x: x['timestamp'])
    
    return requests


def generate_comprehensive_report(trace_file_path: str, agent_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive report combining tasks, requests, and tool calls.
    
    Args:
        trace_file_path: Path to the trace JSONL file
        agent_name: Optional name of the agent to filter by
        
    Returns:
        A dictionary containing the comprehensive report
    """
    # Get tasks and tool calls
    tasks = analyze_tasks_tool_usage(trace_file_path, agent_name)
    tool_details = extract_tool_call_details(trace_file_path, agent_name)
    
    # Map tool calls to their tasks based on timestamp ranges
    for tool in tool_details:
        tool_time = tool['timestamp']
        assigned_task = None
        
        # Find the task this tool call belongs to
        for task in tasks:
            if task['start_time'] <= tool_time < task['end_time']:
                assigned_task = task['task_id']
                break
                
        tool['assigned_task'] = assigned_task
    
    # Create a summary of tools per task
    task_tool_summary = {}
    for task in tasks:
        task_id = task['task_id']
        task_tool_summary[task_id] = {
            'request': task.get('task_request', 'Unknown'),
            'agent_name': task['agent_name'],
            'duration_ms': task['duration_ms'],
            'tool_calls': [],
            'total_tool_calls': task['tool_call_count'],
            'tool_time_percentage': task['tool_time_percentage']
        }
    
    # Add detailed tool information to each task
    for tool in tool_details:
        task_id = tool.get('assigned_task')
        if task_id and task_id in task_tool_summary:
            # Add this tool to the task's tools list
            task_tool_summary[task_id]['tool_calls'].append({
                'tool_name': tool.get('tool_name'),
                'server_name': tool.get('server_name'),
                'method_name': tool.get('method_name'),
                'span_name': tool.get('span_name'),
                'duration_ms': tool.get('duration_ms'),
                'arguments': tool.get('tool_arguments'),
                'result': tool.get('tool_result')
            })
    
    # Generate overall metrics
    report = {
        'trace_file': trace_file_path,
        'summary': {
            'total_tasks': len(tasks),
            'total_tool_calls': len(tool_details),
            'tasks_with_tools': sum(1 for task in tasks if task['tool_call_count'] > 0),
            'avg_tools_per_task': sum(task['tool_call_count'] for task in tasks) / len(tasks) if tasks else 0,
            'avg_task_duration_ms': sum(task['duration_ms'] for task in tasks) / len(tasks) if tasks else 0
        },
        'tasks': task_tool_summary
    }
    
    return report


# Main function removed and placed in eval_runner.py


if __name__ == "__main__":
    # For backward compatibility, import and run the unified entry point
    from mcp_agent.eval.eval_runner import main
    main()