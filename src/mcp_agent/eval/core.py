"""
Core utilities for trace evaluation.

This module provides foundational functions for loading and working with trace files.
"""

import json
from typing import Dict, List, Any
from collections import defaultdict


def read_trace_file(trace_file_path: str) -> List[Dict[str, Any]]:
    """
    Read a trace file and return a list of span dictionaries.
    
    Args:
        trace_file_path: Path to the trace file (JSONL format)
        
    Returns:
        List of span dictionaries from the trace file
    """
    spans = []
    with open(trace_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    span = json.loads(line)
                    spans.append(span)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
    return spans


def separate_spans_by_id(spans: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group spans by their trace_id.
    
    Args:
        spans: List of span dictionaries
        
    Returns:
        Dictionary mapping trace_ids to lists of spans
    """
    trace_groups = defaultdict(list)
    
    for span in spans:
        context = span.get("context", {})
        trace_id = context.get("trace_id")
        
        if trace_id:
            trace_groups[trace_id].append(span)
        else:
            # For spans without a trace_id (should be rare)
            trace_groups["unknown"].append(span)
            
    return dict(trace_groups)


def organize_spans_by_parent(spans: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Organize spans by their parent_id.
    
    Args:
        spans: List of span dictionaries
        
    Returns:
        Dictionary mapping parent_ids to lists of child spans
    """
    parent_groups = defaultdict(list)
    
    for span in spans:
        parent_id = span.get("parent_id")
        if parent_id:
            parent_groups[parent_id].append(span)
        else:
            # Root spans (no parent)
            parent_groups["root"].append(span)
            
    return dict(parent_groups)


def calculate_chat_cost(span: dict) -> float:
    """
    Calculate the estimated cost of an LLM chat interaction based on token usage.
    
    Args:
        span: The span dictionary containing LLM usage data
        
    Returns:
        Estimated cost in dollars
    """
    usage = span.get("data", {}).get("usage", {})
    model = span.get("data", {}).get("model", "")

    # Define rates per 1K tokens for supported models
    model_pricing = {
        "gpt-4o-mini": {
            "prompt_rate": 0.0005,        # $ per 1K tokens
            "completion_rate": 0.0015
        },
        "gpt-4o": {
            "prompt_rate": 0.005,         # $ per 1K tokens
            "completion_rate": 0.015
        },
        "gpt-4": {
            "prompt_rate": 0.03,          # $ per 1K tokens
            "completion_rate": 0.06
        },
        "gpt-3.5-turbo": {
            "prompt_rate": 0.0005,        # $ per 1K tokens
            "completion_rate": 0.0015
        },
        "claude-3-opus-20240229": {
            "prompt_rate": 0.015,         # $ per 1K tokens
            "completion_rate": 0.075
        },
        "claude-3-sonnet-20240229": {
            "prompt_rate": 0.003,         # $ per 1K tokens
            "completion_rate": 0.015
        },
        "claude-3-haiku-20240307": {
            "prompt_rate": 0.00025,       # $ per 1K tokens
            "completion_rate": 0.00125
        },
        # Add more models here if needed
    }

    # If no model specified, return zero cost
    if not model:
        return 0.0

    # Strip date suffix (e.g., "gpt-4o-mini-2024-07-18" -> "gpt-4o-mini")
    base_model = None
    
    # Handle OpenAI models
    if "gpt" in model.lower():
        parts = model.split("-")
        if len(parts) >= 2:
            if len(parts) >= 3 and parts[1] == "3" and parts[2] == "5":
                base_model = "gpt-3.5-turbo"
            elif parts[1] == "4":
                if len(parts) >= 3 and parts[2] == "o":
                    if len(parts) >= 4 and parts[3] == "mini":
                        base_model = "gpt-4o-mini"
                    else:
                        base_model = "gpt-4o"
                else:
                    base_model = "gpt-4"
    
    # Handle Anthropic models
    elif "claude" in model.lower():
        if "opus" in model.lower():
            base_model = "claude-3-opus-20240229"
        elif "sonnet" in model.lower():
            base_model = "claude-3-sonnet-20240229"
        elif "haiku" in model.lower():
            base_model = "claude-3-haiku-20240307"
    
    # If we couldn't determine the base model
    if base_model not in model_pricing:
        raise ValueError(f"Unknown model for pricing: {model}")

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    rates = model_pricing[base_model]
    cost = ((prompt_tokens * rates["prompt_rate"]) +
            (completion_tokens * rates["completion_rate"])) / 1000

    return round(cost, 8)