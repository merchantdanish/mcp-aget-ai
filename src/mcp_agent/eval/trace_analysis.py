#!/usr/bin/env python3
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(abspath(__file__)))

from evals import read_trace_file, separate_spans_by_id, organize_spans_by_parent


def analyze_trace_file(trace_file_path: str) -> Dict[str, Any]:
    """
    Analyze a trace file and return information about its structure.
    
    Args:
        trace_file_path: Path to the trace file
        
    Returns:
        Dictionary with analysis results
    """
    # Read trace file
    spans = read_trace_file(trace_file_path)
    if not spans:
        return {"error": "No spans found in trace file"}
    
    # Separate spans by trace ID
    trace_groups = separate_spans_by_id(spans)
    
    results = {
        "trace_file": trace_file_path,
        "total_spans": len(spans),
        "trace_count": len(trace_groups),
        "traces": {}
    }
    
    # For each trace, organize spans by parent ID
    for trace_id, trace_spans in trace_groups.items():
        if trace_id == "unknown":
            continue
            
        # Get trace timing information
        start_times = [span.get("start_time") for span in trace_spans if span.get("start_time")]
        end_times = [span.get("end_time") for span in trace_spans if span.get("end_time")]
        
        trace_info = {
            "span_count": len(trace_spans),
            "start_time": min(start_times) if start_times else None,
            "end_time": max(end_times) if end_times else None,
        }
        
        # Organize by parent ID
        parent_groups = organize_spans_by_parent(trace_spans)
        root_spans = parent_groups.get("root", [])
        
        trace_info["root_span_count"] = len(root_spans)
        trace_info["root_spans"] = [
            {
                "name": span.get("name"),
                "span_id": span.get("span_id"),
                "start_time": span.get("start_time"),
                "end_time": span.get("end_time")
            }
            for span in root_spans
        ]
        
        # Count span types by extracting prefix from span name
        span_types = {}
        for span in trace_spans:
            span_name = span.get("name", "")
            span_type = span_name.split(".")[0] if "." in span_name else span_name
            span_types[span_type] = span_types.get(span_type, 0) + 1
        
        trace_info["span_types"] = span_types
        
        # Extract agent names from span attributes
        agent_names = set()
        for span in trace_spans:
            attributes = span.get("attributes", {})
            agent_name = attributes.get("gen_ai.agent.name")
            if agent_name:
                agent_names.add(agent_name)
        
        trace_info["agent_names"] = list(agent_names)
        
        results["traces"][trace_id] = trace_info
    
    return results


def get_spans_by_category(spans: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize spans by their type based on name prefix.
    
    Args:
        spans: List of span dictionaries
        
    Returns:
        Dictionary mapping categories to spans
    """
    categories = {
        "agent": [],
        "llm": [],
        "mcp": [],
        "other": []
    }
    
    for span in spans:
        name = span.get("name", "")
        
        if name.startswith("agent."):
            categories["agent"].append(span)
        elif (name.startswith("llm.") or 
              name.startswith("llm_") or 
              name.startswith("openai.") or 
              name.startswith("anthropic.")):
            categories["llm"].append(span)
        elif name.startswith("mcp_"):
            categories["mcp"].append(span)
        else:
            categories["other"].append(span)
    
    return categories


# Main function removed and placed in eval_runner.py