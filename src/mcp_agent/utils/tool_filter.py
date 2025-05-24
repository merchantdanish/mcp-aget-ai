"""
Lightweight tool filtering utilities for mcp-agent.

This module provides a non-invasive way to filter MCP tools at the LLM level,
allowing you to control which tools are available without modifying the core code.
"""

from typing import List, Set, Dict, Optional, Callable, Union
from mcp.types import Tool

from mcp_agent.logging.logger import get_logger

# Use the project's logger system
logger = get_logger(__name__)


class ToolFilter:
    """
    A simple tool filter that can be applied to any LLM instance.
    
    Usage:
        # Create a filter
        filter = ToolFilter(allowed=["read_file", "list_directory"])
        
        # Apply to an LLM
        filtered_llm = apply_tool_filter(llm, filter)
    """
    
    def __init__(
        self,
        allowed: Optional[List[str]] = None,
        excluded: Optional[List[str]] = None,
        server_filters: Optional[Dict[str, Dict[str, List[str]]]] = None,
        custom_filter: Optional[Callable[[Tool], bool]] = None
    ):
        """
        Initialize a tool filter.
        
        Args:
            allowed: Global list of allowed tool names (whitelist)
            excluded: Global list of excluded tool names (blacklist)
            server_filters: Server-specific filters, e.g.:
                {
                    "filesystem": {"allowed": ["read_file"], "excluded": ["delete_file"]},
                    "github": {"allowed": ["search_repositories"]}
                }
            custom_filter: Custom filter function that takes a Tool and returns bool
        
        Priority:
            1. custom_filter (if provided)
            2. allowed list (if specified)
            3. excluded list (if specified)
            4. Default: allow all
        """
        self.allowed_global = set(allowed) if allowed else None
        self.excluded_global = set(excluded) if excluded else None
        self.server_filters = server_filters or {}
        self.custom_filter = custom_filter
    
    def should_include_tool(self, tool: Tool) -> bool:
        """
        Determine if a tool should be included.
        
        Args:
            tool: The tool to check
            
        Returns:
            True if the tool should be included, False otherwise
        """
        # Custom filter takes precedence
        if self.custom_filter:
            return self.custom_filter(tool)
        
        tool_name = tool.name
        server_name = None
        
        # Extract server name from namespaced tools (format: "server_toolname")
        if "_" in tool_name:
            # First, try to match against known server filters
            if self.server_filters:
                # Check all configured server names, preferring longer matches
                # This handles cases where server names might contain underscores
                for srv_name in sorted(self.server_filters.keys(), key=len, reverse=True):
                    prefix = srv_name + "_"
                    if tool_name.startswith(prefix):
                        server_name = srv_name
                        tool_name = tool_name[len(prefix):]
                        break
            
            # If no server filter matched, try simple split for global filters
            # This assumes the first part before "_" is the server name
            if server_name is None:
                parts = tool_name.split("_", 1)
                if len(parts) == 2:
                    # Keep the original tool.name for full name matching
                    # but also prepare the extracted tool name
                    server_name = parts[0]
                    tool_name = parts[1]
        
        # Check server-specific filters first
        if server_name and server_name in self.server_filters:
            server_filter = self.server_filters[server_name]
            
            # Server-specific allowed list
            if "allowed" in server_filter:
                return tool_name in server_filter["allowed"]
            
            # Server-specific excluded list
            if "excluded" in server_filter:
                return tool_name not in server_filter["excluded"]
        
        # Check global allowed list
        if self.allowed_global is not None:
            return tool.name in self.allowed_global or tool_name in self.allowed_global
        
        # Check global excluded list
        if self.excluded_global is not None:
            return tool.name not in self.excluded_global and tool_name not in self.excluded_global
        
        # Default: include all tools
        return True
    
    def filter_tools(self, tools: List[Tool]) -> List[Tool]:
        """Filter a list of tools based on the configured rules."""
        filtered_tools = [tool for tool in tools if self.should_include_tool(tool)]
        
        # Log filtering summary
        if len(filtered_tools) != len(tools):
            logger.info(f"Tool filtering applied: {len(filtered_tools)}/{len(tools)} tools retained")
            
        return filtered_tools


def apply_tool_filter(llm_instance, tool_filter: Optional[ToolFilter]):
    """
    Apply a tool filter to an LLM instance without modifying its source code.
    
    This function wraps the LLM's generate methods to filter tools during execution.
    
    Args:
        llm_instance: An instance of AugmentedLLM (e.g., OpenAIAugmentedLLM)
        tool_filter: The ToolFilter to apply, or None to remove filtering
        
    Returns:
        The same LLM instance with filtering applied
        
    Example:
        llm = await agent.attach_llm(OpenAIAugmentedLLM)
        filter = ToolFilter(allowed=["read_file", "list_directory"])
        apply_tool_filter(llm, filter)
    """
    # Store original method
    if not hasattr(llm_instance, '_original_generate'):
        llm_instance._original_generate = llm_instance.generate
    
    # If no filter, restore original method
    if tool_filter is None:
        if hasattr(llm_instance, '_original_generate'):
            logger.info("Tool filter removed from LLM instance")
            llm_instance.generate = llm_instance._original_generate
        return llm_instance
    
    # Log filter configuration
    filter_info = []
    if tool_filter.allowed_global:
        filter_info.append(f"allowed: {list(tool_filter.allowed_global)}")
    if tool_filter.excluded_global:
        filter_info.append(f"excluded: {list(tool_filter.excluded_global)}")
    if tool_filter.server_filters:
        filter_info.append(f"server-specific: {tool_filter.server_filters}")
    if tool_filter.custom_filter:
        filter_info.append("custom filter function")
    
    logger.info(f"Tool filter applied to LLM instance with: {', '.join(filter_info) if filter_info else 'no constraints'}")
    
    # Create wrapper function that applies filtering
    async def filtered_generate(message, request_params=None):
        # Temporarily wrap the agent's list_tools method
        original_list_tools = llm_instance.agent.list_tools
        
        async def filtered_list_tools(server_name=None):
            result = await original_list_tools(server_name)
            if tool_filter:
                result.tools = tool_filter.filter_tools(result.tools)
            return result
        
        llm_instance.agent.list_tools = filtered_list_tools
        try:
            return await llm_instance._original_generate(message, request_params)
        finally:
            llm_instance.agent.list_tools = original_list_tools
    
    # Apply the wrapped method
    llm_instance.generate = filtered_generate
    
    return llm_instance