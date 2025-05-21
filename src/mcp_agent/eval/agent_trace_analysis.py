"""
Agent trace analysis module.

This module provides tools to analyze agent traces by component type.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timezone

from mcp_agent.eval.core import read_trace_file
from mcp_agent.eval.tool_metrics import measure_agent_tool_calls


def analyze_trace_by_component(trace_file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze a trace file and separate traces by component type (agent, llm, mcp_aggregator).
    
    Args:
        trace_file_path: Path to the trace JSONL file
        
    Returns:
        A dictionary mapping component types to their traces
    """
    # Read spans from trace file using the existing function
    spans = read_trace_file(trace_file_path)
    
    # Dictionary to store traces by component type
    component_traces = defaultdict(list)
    
    for span in spans:
        # Extract component type from span name
        span_name = span.get('name', '')
        component_type = extract_component_type(span_name)
        
        if component_type:
            component_traces[component_type].append(span)
    
    return component_traces


def extract_component_type(span_name: str) -> Optional[str]:
    """
    Extract the component type from a span name.
    
    Args:
        span_name: Name of the span
        
    Returns:
        Component type or None if not identifiable
    """
    # Define component type prefixes to look for
    component_prefixes = {
        'agent': 'agent',
        'mcp_aggregator': 'mcp_aggregator',
        'llm': 'llm',
        'openai': 'llm_openai',
        'anthropic': 'llm_anthropic',
        'llm_openai': 'llm_openai',
        'llm_anthropic': 'llm_anthropic'
    }
    
    for prefix, component_type in component_prefixes.items():
        if span_name.startswith(prefix + '.') or span_name == prefix:
            return component_type
    
    return None


def get_events_timeline(component_traces: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Create a chronological timeline of all events across components.
    
    Args:
        component_traces: Dictionary mapping component types to their traces
        
    Returns:
        A list of events in chronological order
    """
    all_events = []
    
    for component_type, traces in component_traces.items():
        for trace in traces:
            # Parse timestamps if they're in string format
            start_time = trace.get('start_time')
            end_time = trace.get('end_time')
            
            # Convert string timestamps to nanoseconds since epoch
            if isinstance(start_time, str):
                try:
                    from datetime import datetime
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp() * 1e9
                except (ValueError, TypeError):
                    start_time = 0
                    
            if isinstance(end_time, str):
                try:
                    from datetime import datetime
                    end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00')).timestamp() * 1e9
                except (ValueError, TypeError):
                    end_time = 0
            
            # Calculate duration
            duration = None
            if start_time and end_time:
                try:
                    duration = end_time - start_time
                except TypeError:
                    duration = None
            
            # Add the main span as an event
            event = {
                'component_type': component_type,
                'type': 'span',
                'name': trace.get('name', ''),
                'start_time': start_time,
                'end_time': end_time,
                'attributes': trace.get('attributes', {}),
                'duration': duration
            }
            all_events.append(event)
            
            # Add all events within the span
            for span_event in trace.get('events', []):
                # Parse event timestamp
                timestamp = span_event.get('timestamp')
                if isinstance(timestamp, str):
                    try:
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp() * 1e9
                    except (ValueError, TypeError):
                        timestamp = 0
                
                event = {
                    'component_type': component_type,
                    'type': 'event',
                    'name': span_event.get('name', ''),
                    'timestamp': timestamp,
                    'attributes': span_event.get('attributes', {})
                }
                all_events.append(event)
    
    # Sort events by timestamp
    sorted_events = sorted(
        all_events, 
        key=lambda x: x.get('start_time') or x.get('timestamp') or 0
    )
    
    return sorted_events


def analyze_component_interactions(events_timeline: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Analyze interactions between components.
    
    Args:
        events_timeline: Chronological list of events
        
    Returns:
        A dictionary mapping component types to sets of components they interact with
    """
    interactions = defaultdict(set)
    
    for event in events_timeline:
        component_type = event.get('component_type')
        attributes = event.get('attributes', {})
        
        # Look for attributes that reference other components
        for key, value in attributes.items():
            if isinstance(value, str):
                for component in ['agent', 'llm', 'mcp_aggregator']:
                    if component in key and component != component_type:
                        interactions[component_type].add(component)
    
    return interactions


def get_component_metrics(component_traces: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate metrics for each component type.
    
    Args:
        component_traces: Dictionary mapping component types to their traces
        
    Returns:
        A dictionary mapping component types to metrics
    """
    metrics = {}
    
    for component_type, traces in component_traces.items():
        # Initialize metrics
        total_duration = 0
        valid_duration_count = 0
        
        for trace in traces:
            # Parse timestamps if they're in string format
            start_time = trace.get('start_time')
            end_time = trace.get('end_time')
            
            # Convert string timestamps to nanoseconds since epoch
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
            
            # Calculate duration
            if start_time is not None and end_time is not None:
                try:
                    duration = end_time - start_time
                    total_duration += duration
                    valid_duration_count += 1
                except (TypeError, ValueError):
                    pass
        
        component_metrics = {
            'count': len(traces),
            'total_duration': total_duration,
            'avg_duration': total_duration / valid_duration_count if valid_duration_count > 0 else None,
            'operation_counts': defaultdict(int)
        }
        
        # Count occurrences of each operation
        for trace in traces:
            operation = trace.get('name', '').split('.')[-1] if '.' in trace.get('name', '') else trace.get('name', '')
            component_metrics['operation_counts'][operation] += 1
        
        metrics[component_type] = component_metrics
    
    return metrics


def analyze_error_patterns(component_traces: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Identify error patterns in traces.
    
    Args:
        component_traces: Dictionary mapping component types to their traces
        
    Returns:
        A list of error events
    """
    errors = []
    
    for component_type, traces in component_traces.items():
        for trace in traces:
            # Check status code
            status = trace.get('status', {})
            if status.get('status_code') == 'ERROR':
                errors.append({
                    'component_type': component_type,
                    'span_name': trace.get('name', ''),
                    'error_info': status,
                    'attributes': trace.get('attributes', {})
                })
            
            # Check events for exceptions
            for event in trace.get('events', []):
                if event.get('name') == 'exception':
                    errors.append({
                        'component_type': component_type,
                        'span_name': trace.get('name', ''),
                        'event_name': 'exception',
                        'attributes': event.get('attributes', {})
                    })
    
    return errors


def get_trace_summary(trace_file: str) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of a trace file, organized by component type.
    
    Args:
        trace_file: Path to the trace file
        
    Returns:
        A dictionary with summary information
    """
    # Analyze the trace file
    component_traces = analyze_trace_by_component(trace_file)
    
    # Get events timeline
    events_timeline = get_events_timeline(component_traces)
    
    # Analyze component interactions
    interactions = analyze_component_interactions(events_timeline)
    
    # Get component metrics
    metrics = get_component_metrics(component_traces)
    
    # Analyze error patterns
    errors = analyze_error_patterns(component_traces)
    
    # Calculate overall metrics - using the parsed numeric timestamps from get_events_timeline
    execution_time = None
    if events_timeline:
        # Get valid timestamps
        valid_timestamps = []
        for event in events_timeline:
            if event.get('type') == 'span':
                if event.get('start_time') is not None:
                    valid_timestamps.append(event.get('start_time'))
                if event.get('end_time') is not None:
                    valid_timestamps.append(event.get('end_time'))
            elif event.get('timestamp') is not None:
                valid_timestamps.append(event.get('timestamp'))
        
        if valid_timestamps:
            first_event_time = min(valid_timestamps)
            last_event_time = max(valid_timestamps)
            execution_time = last_event_time - first_event_time
    
    return {
        'trace_file': trace_file,
        'component_counts': {component: len(traces) for component, traces in component_traces.items()},
        'component_metrics': metrics,
        'component_interactions': {component: list(interactions) for component, interactions in interactions.items()},
        'errors': errors,
        'total_events': len(events_timeline),
        'execution_time': execution_time,
        'components': list(component_traces.keys()),
        'component_traces': component_traces  # Include the actual traces for further analysis
    }


def extract_agent_workflow(component_traces: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Extract the agent workflow steps in chronological order.
    
    Args:
        component_traces: Dictionary mapping component types to their traces
        
    Returns:
        List of workflow steps with their details
    """
    # Get agent traces
    agent_traces = component_traces.get('agent', [])
    
    # Process timestamps and create a list to sort
    traces_with_parsed_times = []
    for trace in agent_traces:
        # Parse timestamps if they're in string format
        start_time = trace.get('start_time')
        end_time = trace.get('end_time')
        
        # Convert string timestamps to nanoseconds since epoch
        if isinstance(start_time, str):
            try:
                from datetime import datetime
                start_time_parsed = datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp() * 1e9
            except (ValueError, TypeError):
                start_time_parsed = 0
        else:
            start_time_parsed = start_time or 0
                
        if isinstance(end_time, str):
            try:
                from datetime import datetime
                end_time_parsed = datetime.fromisoformat(end_time.replace('Z', '+00:00')).timestamp() * 1e9
            except (ValueError, TypeError):
                end_time_parsed = 0
        else:
            end_time_parsed = end_time or 0
        
        # Store original trace with parsed times for sorting
        traces_with_parsed_times.append({
            'trace': trace,
            'start_time_parsed': start_time_parsed
        })
    
    # Sort by parsed start time
    sorted_traces = sorted(
        traces_with_parsed_times,
        key=lambda x: x['start_time_parsed']
    )
    
    # Extract workflow steps
    workflow_steps = []
    
    for item in sorted_traces:
        trace = item['trace']
        
        # Parse timestamps again for the step
        start_time = trace.get('start_time')
        end_time = trace.get('end_time')
        
        # Convert string timestamps to nanoseconds since epoch
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
        
        # Calculate duration
        duration = None
        if start_time is not None and end_time is not None:
            try:
                duration = end_time - start_time
            except (TypeError, ValueError):
                duration = None
        
        step = {
            'name': trace.get('name', ''),
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'attributes': trace.get('attributes', {}),
            'events': trace.get('events', [])
        }
        workflow_steps.append(step)
    
    return workflow_steps


def extract_llm_interactions(component_traces: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Extract LLM interactions from traces.
    
    Args:
        component_traces: Dictionary mapping component types to their traces
        
    Returns:
        List of LLM interactions with details
    """
    # Get LLM traces
    llm_traces = []
    for component_type, traces in component_traces.items():
        if component_type.startswith('llm'):
            llm_traces.extend(traces)
    
    # Process timestamps and create a list to sort
    traces_with_parsed_times = []
    for trace in llm_traces:
        # Parse timestamps if they're in string format
        start_time = trace.get('start_time')
        
        # Convert string timestamps to nanoseconds since epoch
        if isinstance(start_time, str):
            try:
                from datetime import datetime
                start_time_parsed = datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp() * 1e9
            except (ValueError, TypeError):
                start_time_parsed = 0
        else:
            start_time_parsed = start_time or 0
        
        # Store original trace with parsed times for sorting
        traces_with_parsed_times.append({
            'trace': trace,
            'start_time_parsed': start_time_parsed
        })
    
    # Sort by parsed start time
    sorted_traces = sorted(
        traces_with_parsed_times,
        key=lambda x: x['start_time_parsed']
    )
    
    # Extract LLM interactions
    interactions = []
    
    for item in sorted_traces:
        trace = item['trace']
        
        # Parse timestamps again for the interaction
        start_time = trace.get('start_time')
        end_time = trace.get('end_time')
        
        # Convert string timestamps to nanoseconds since epoch
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
        
        # Calculate duration
        duration = None
        if start_time is not None and end_time is not None:
            try:
                duration = end_time - start_time
            except (TypeError, ValueError):
                duration = None
        
        interaction = {
            'name': trace.get('name', ''),
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'attributes': trace.get('attributes', {}),
            'events': trace.get('events', [])
        }
        
        # Extract input/output tokens if available
        attributes = trace.get('attributes', {})
        if 'gen_ai.usage.input_tokens' in attributes:
            interaction['input_tokens'] = attributes['gen_ai.usage.input_tokens']
        if 'gen_ai.usage.output_tokens' in attributes:
            interaction['output_tokens'] = attributes['gen_ai.usage.output_tokens']
        
        # Extract model information
        if 'gen_ai.request.model' in attributes:
            interaction['model'] = attributes['gen_ai.request.model']
        
        interactions.append(interaction)
    
    return interactions


def extract_mcp_aggregator_operations(component_traces: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Extract MCP aggregator operations from traces.
    
    Args:
        component_traces: Dictionary mapping component types to their traces
        
    Returns:
        List of MCP aggregator operations with details
    """
    # Get MCP aggregator traces
    mcp_traces = component_traces.get('mcp_aggregator', [])
    
    # Process timestamps and create a list to sort
    traces_with_parsed_times = []
    for trace in mcp_traces:
        # Parse timestamps if they're in string format
        start_time = trace.get('start_time')
        
        # Convert string timestamps to nanoseconds since epoch
        if isinstance(start_time, str):
            try:
                from datetime import datetime
                start_time_parsed = datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp() * 1e9
            except (ValueError, TypeError):
                start_time_parsed = 0
        else:
            start_time_parsed = start_time or 0
        
        # Store original trace with parsed times for sorting
        traces_with_parsed_times.append({
            'trace': trace,
            'start_time_parsed': start_time_parsed
        })
    
    # Sort by parsed start time
    sorted_traces = sorted(
        traces_with_parsed_times,
        key=lambda x: x['start_time_parsed']
    )
    
    # Extract MCP operations
    operations = []
    
    for item in sorted_traces:
        trace = item['trace']
        
        # Parse timestamps again for the operation
        start_time = trace.get('start_time')
        end_time = trace.get('end_time')
        
        # Convert string timestamps to nanoseconds since epoch
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
        
        # Calculate duration
        duration = None
        if start_time is not None and end_time is not None:
            try:
                duration = end_time - start_time
            except (TypeError, ValueError):
                duration = None
        
        operation = {
            'name': trace.get('name', ''),
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'attributes': trace.get('attributes', {}),
            'events': trace.get('events', [])
        }
        
        # Extract server name if available
        attributes = trace.get('attributes', {})
        if 'server_name' in attributes:
            operation['server_name'] = attributes['server_name']
        
        operations.append(operation)
    
    return operations


def extract_spans_by_feature(spans: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract spans by feature/functionality based on their name patterns.
    
    Args:
        spans: List of span dictionaries from a trace file
        
    Returns:
        Dictionary mapping feature categories to lists of spans
    """
    feature_spans = {
        "initialization": [],         # Initialization operations
        "connection": [],             # Connection management
        "mcp_session": [],            # MCP session operations
        "agent_lifecycle": [],        # Agent lifecycle operations
        "llm_operations": [],         # LLM operations
        "tool_operations": [],        # Tool operations
        "workflow": [],               # Workflow management
        "error_handling": [],         # Error handling
        "other": []                   # Uncategorized spans
    }
    
    for span in spans:
        name = span.get("name", "")
        attributes = span.get("attributes", {})
        status = span.get("status", {})
        
        # Parse the spans based on name patterns and attributes
        if name.startswith("init") or "initialize" in name.lower() or "setup" in name.lower():
            feature_spans["initialization"].append(span)
        elif "connection" in name.lower() or "connect" in name.lower():
            feature_spans["connection"].append(span)
        elif "MCPAgentClientSession" in name or "send_request" in name or "send_notification" in name:
            feature_spans["mcp_session"].append(span)
        elif "agent" in name.lower() and any(x in name.lower() for x in ["create", "setup", "init", "start", "stop"]):
            feature_spans["agent_lifecycle"].append(span)
        elif name.startswith("llm.") or name.startswith("openai.") or name.startswith("anthropic.") or "generate" in name.lower():
            feature_spans["llm_operations"].append(span)
        elif "tool" in name.lower() or "call_tool" in name.lower() or "tools/list" in name:
            feature_spans["tool_operations"].append(span)
        elif "workflow" in name.lower() or "orchestrator" in name.lower() or "parallel" in name.lower() or "executor" in name.lower():
            feature_spans["workflow"].append(span)
        elif status.get("status_code") == "ERROR" or "exception" in str(span.get("events", [])):
            feature_spans["error_handling"].append(span)
        else:
            feature_spans["other"].append(span)
    
    return feature_spans


def print_feature_spans_tree(feature_spans: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Generate a tree-like representation of spans organized by feature.
    
    Args:
        feature_spans: Dictionary mapping feature categories to lists of spans
        
    Returns:
        Formatted string with tree-like report
    """
    output_lines = ["Feature Spans Tree:"]
    
    # Format timestamps
    def format_time(timestamp):
        if timestamp:
            # Handle string timestamps
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    return dt.strftime('%H:%M:%S.%f')[:-3]
                except (ValueError, TypeError):
                    return "N/A"
            else:
                # Handle numeric timestamps (nanoseconds since epoch)
                try:
                    dt = datetime.fromtimestamp(timestamp / 1e9, tz=timezone.utc)
                    return dt.strftime('%H:%M:%S.%f')[:-3]
                except (ValueError, TypeError, OverflowError):
                    return "N/A"
        return "N/A"
    
    # Calculate duration in milliseconds
    def calculate_duration_ms(start, end):
        if isinstance(start, str) and isinstance(end, str):
            try:
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                return (end_dt - start_dt).total_seconds() * 1000
            except (ValueError, TypeError):
                return None
        elif start is not None and end is not None:
            try:
                return (end - start) / 1e6  # Convert ns to ms
            except (TypeError, ValueError):
                return None
        return None
    
    # Process each feature category
    for feature, spans in feature_spans.items():
        # Skip empty categories
        if not spans:
            continue
            
        output_lines.append(f"└── {feature.replace('_', ' ').title()} ({len(spans)} spans)")
        
        # Sort spans by start time
        sorted_spans = sorted(
            spans,
            key=lambda x: (
                datetime.fromisoformat(x.get('start_time', '').replace('Z', '+00:00')).timestamp() 
                if isinstance(x.get('start_time', ''), str) 
                else (x.get('start_time', 0) / 1e9 if x.get('start_time') else 0)
            )
        )
        
        # Process each span in the category
        for i, span in enumerate(sorted_spans):
            name = span.get('name', 'Unnamed Span')
            start_time = span.get('start_time')
            end_time = span.get('end_time')
            duration_ms = calculate_duration_ms(start_time, end_time)
            
            # Last item in the list gets a different branch symbol
            if i == len(sorted_spans) - 1:
                prefix = "    └── "
                child_prefix = "        "
            else:
                prefix = "    ├── "
                child_prefix = "    │   "
            
            # Format span with duration (if available)
            span_line = f"{name}"
            if duration_ms:
                span_line += f" ({duration_ms:.2f}ms)"
                
            output_lines.append(f"{prefix}{span_line}")
            
            # Add time information
            output_lines.append(f"{child_prefix}Start: {format_time(start_time)}, End: {format_time(end_time)}")
            
            # Add all span attributes with detailed view
            attributes = span.get('attributes', {})
            
            # First, extract and display key attributes we're interested in at the top level
            key_attrs = []
            if 'server_name' in attributes:
                key_attrs.append(f"Server: {attributes['server_name']}")
            if 'gen_ai.agent.name' in attributes:
                key_attrs.append(f"Agent: {attributes['gen_ai.agent.name']}")
            if 'gen_ai.request.model' in attributes:
                key_attrs.append(f"Model: {attributes['gen_ai.request.model']}")
            if 'gen_ai.usage.input_tokens' in attributes and 'gen_ai.usage.output_tokens' in attributes:
                key_attrs.append(f"Tokens: {attributes['gen_ai.usage.input_tokens']} in, {attributes['gen_ai.usage.output_tokens']} out")
            if 'mcp.method.name' in attributes:
                key_attrs.append(f"Method: {attributes['mcp.method.name']}")
                
            if key_attrs:
                output_lines.append(f"{child_prefix}{', '.join(key_attrs)}")
            
            # Show all attributes in detail
            if attributes:
                output_lines.append(f"{child_prefix}Attributes:")
                
                # Sort attributes by key for consistent display
                sorted_attrs = sorted(attributes.items())
                for j, (key, value) in enumerate(sorted_attrs):
                    # Skip attributes already shown above
                    if key in ['server_name', 'gen_ai.agent.name', 'gen_ai.request.model', 
                              'gen_ai.usage.input_tokens', 'gen_ai.usage.output_tokens', 
                              'mcp.method.name']:
                        continue
                        
                    # Use different branch symbols for last item
                    if j == len(sorted_attrs) - 1:
                        attr_prefix = f"{child_prefix}└── "
                    else:
                        attr_prefix = f"{child_prefix}├── "
                        
                    # Format the value appropriately
                    if isinstance(value, str) and len(value) > 50:
                        # Truncate long string values
                        value_str = f"{value[:50]}..."
                    else:
                        value_str = str(value)
                        
                    output_lines.append(f"{attr_prefix}{key}: {value_str}")
            
            # Add events as children with full details
            events = span.get('events', [])
            if events:
                output_lines.append(f"{child_prefix}Events:")
                
                for j, event in enumerate(events):
                    event_name = event.get('name', 'Unnamed Event')
                    event_time = event.get('timestamp')
                    
                    # Last event gets a different branch symbol
                    if j == len(events) - 1:
                        event_prefix = f"{child_prefix}└── "
                        event_child_prefix = f"{child_prefix}    "
                    else:
                        event_prefix = f"{child_prefix}├── "
                        event_child_prefix = f"{child_prefix}│   "
                    
                    output_lines.append(f"{event_prefix}{event_name} at {format_time(event_time)}")
                    
                    # Add ALL event attributes
                    event_attributes = event.get('attributes', {})
                    if event_attributes:
                        output_lines.append(f"{event_child_prefix}Event Attributes:")
                        
                        # Sort attributes by key for consistent display
                        sorted_event_attrs = sorted(event_attributes.items())
                        for k, (key, value) in enumerate(sorted_event_attrs):
                            # Use different branch symbols for last item
                            if k == len(sorted_event_attrs) - 1:
                                event_attr_prefix = f"{event_child_prefix}└── "
                            else:
                                event_attr_prefix = f"{event_child_prefix}├── "
                                
                            # Format the value appropriately
                            if isinstance(value, str) and len(value) > 50:
                                # Truncate long string values
                                value_str = f"{value[:50]}..."
                            else:
                                value_str = str(value)
                                
                            output_lines.append(f"{event_attr_prefix}{key}: {value_str}")
            
            # Add context information if available
            context = span.get('context', {})
            if context:
                output_lines.append(f"{child_prefix}Context:")
                
                # Sort context items by key for consistent display
                sorted_context = sorted(context.items())
                for j, (key, value) in enumerate(sorted_context):
                    # Use different branch symbols for last item
                    if j == len(sorted_context) - 1:
                        context_prefix = f"{child_prefix}└── "
                    else:
                        context_prefix = f"{child_prefix}├── "
                        
                    output_lines.append(f"{context_prefix}{key}: {value}")
            
            # Add status information if available
            status = span.get('status', {})
            if status and status.get('status_code') != "UNSET":
                output_lines.append(f"{child_prefix}Status: {status.get('status_code', 'Unknown')}")
                if 'description' in status:
                    output_lines.append(f"{child_prefix}Status Description: {status.get('description')}")
        
        # Add an empty line between features for better readability
        output_lines.append("")
    
    return "\n".join(output_lines)


def get_detailed_trace_analysis(trace_file: str) -> Dict[str, Any]:
    """
    Get a detailed analysis of a trace file.
    
    Args:
        trace_file: Path to the trace file
        
    Returns:
        Dictionary with detailed analysis information
    """
    # Get summary
    summary = get_trace_summary(trace_file)
    
    # Extract component traces
    component_traces = summary.get('component_traces', {})
    
    # Extract workflow steps
    workflow_steps = extract_agent_workflow(component_traces)
    
    # Extract LLM interactions
    llm_interactions = extract_llm_interactions(component_traces)
    
    # Extract MCP aggregator operations
    mcp_operations = extract_mcp_aggregator_operations(component_traces)
    
    return {
        'summary': summary,
        'workflow_steps': workflow_steps,
        'llm_interactions': llm_interactions,
        'mcp_operations': mcp_operations
    }


def generate_trace_report(analysis: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """
    Generate a human-readable report from the trace analysis.
    
    Args:
        analysis: Dictionary with trace analysis information
        output_file: Path to save the report (optional)
        
    Returns:
        Report text
    """
    summary = analysis.get('summary', {})
    workflow_steps = analysis.get('workflow_steps', [])
    llm_interactions = analysis.get('llm_interactions', [])
    mcp_operations = analysis.get('mcp_operations', [])
    
    # Format timestamps
    def format_time(timestamp):
        if timestamp:
            # Handle string timestamps
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    return dt.strftime('%H:%M:%S.%f')[:-3]
                except (ValueError, TypeError):
                    return "N/A"
            else:
                # Handle numeric timestamps (nanoseconds since epoch)
                try:
                    dt = datetime.fromtimestamp(timestamp / 1e9, tz=timezone.utc)
                    return dt.strftime('%H:%M:%S.%f')[:-3]
                except (ValueError, TypeError, OverflowError):
                    return "N/A"
        return "N/A"
    
    # Generate report sections
    report = [
        "# Agent Trace Analysis Report",
        f"Trace file: {summary.get('trace_file', 'Unknown')}",
        f"Execution time: {summary.get('execution_time') / 1e9:.2f} seconds" if summary.get('execution_time') else "Execution time: N/A",
        f"Total events: {summary.get('total_events', 0)}",
        "",
        "## Component Summary",
    ]
    
    # Add component counts
    component_counts = summary.get('component_counts', {})
    for component, count in component_counts.items():
        report.append(f"- **{component}**: {count} traces")
    
    # Add component metrics
    report.append("")
    report.append("## Component Metrics")
    component_metrics = summary.get('component_metrics', {})
    for component, metrics in component_metrics.items():
        report.append(f"### {component}")
        report.append(f"- Count: {metrics.get('count', 0)}")
        if metrics.get('total_duration') is not None:
            report.append(f"- Total duration: {metrics.get('total_duration') / 1e9:.2f} seconds")
        if metrics.get('avg_duration') is not None:
            report.append(f"- Average duration: {metrics.get('avg_duration') / 1e9:.2f} seconds")
        
        report.append("- Operation counts:")
        for operation, count in metrics.get('operation_counts', {}).items():
            report.append(f"  - {operation}: {count}")
            
        report.append("")
    
    # Add agent workflow steps
    report.append("## Agent Workflow Steps")
    for i, step in enumerate(workflow_steps):
        report.append(f"### Step {i+1}: {step.get('name', 'Unknown')}")
        report.append(f"- Start time: {format_time(step.get('start_time'))}")
        report.append(f"- End time: {format_time(step.get('end_time'))}")
        if step.get('duration') is not None:
            report.append(f"- Duration: {step.get('duration') / 1e9:.2f} seconds")
        
        # Add events
        if step.get('events'):
            report.append("- Events:")
            for event in step.get('events'):
                report.append(f"  - {event.get('name', 'Unknown')} at {format_time(event.get('timestamp'))}")
        
        report.append("")
    
    # Add LLM interactions
    report.append("## LLM Interactions")
    for i, interaction in enumerate(llm_interactions):
        report.append(f"### Interaction {i+1}: {interaction.get('name', 'Unknown')}")
        report.append(f"- Start time: {format_time(interaction.get('start_time'))}")
        report.append(f"- End time: {format_time(interaction.get('end_time'))}")
        if interaction.get('duration') is not None:
            report.append(f"- Duration: {interaction.get('duration') / 1e9:.2f} seconds")
        
        if 'model' in interaction:
            report.append(f"- Model: {interaction.get('model')}")
        if 'input_tokens' in interaction:
            report.append(f"- Input tokens: {interaction.get('input_tokens')}")
        if 'output_tokens' in interaction:
            report.append(f"- Output tokens: {interaction.get('output_tokens')}")
        
        report.append("")
    
    # Add MCP aggregator operations
    report.append("## MCP Aggregator Operations")
    for i, operation in enumerate(mcp_operations):
        report.append(f"### Operation {i+1}: {operation.get('name', 'Unknown')}")
        report.append(f"- Start time: {format_time(operation.get('start_time'))}")
        report.append(f"- End time: {format_time(operation.get('end_time'))}")
        if operation.get('duration') is not None:
            report.append(f"- Duration: {operation.get('duration') / 1e9:.2f} seconds")
        
        if 'server_name' in operation:
            report.append(f"- Server: {operation.get('server_name')}")
        
        report.append("")
    
    # Add errors if any
    errors = summary.get('errors', [])
    if errors:
        report.append("## Errors")
        for i, error in enumerate(errors):
            report.append(f"### Error {i+1}")
            report.append(f"- Component: {error.get('component_type', 'Unknown')}")
            report.append(f"- Span: {error.get('span_name', 'Unknown')}")
            if 'event_name' in error:
                report.append(f"- Event: {error.get('event_name')}")
            if 'error_info' in error:
                report.append(f"- Info: {error.get('error_info')}")
            
            report.append("")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
    
    return report_text


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


# Main function removed and placed in eval_runner.py