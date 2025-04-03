"""
Utility runner for MCP Agent Cloud.

This script is the entry point for utility containers. It reads the transport type
from the environment and runs either the STDIO adapter or the native server command.
"""

import json
import logging
import os
import sys

# Configure logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s [utility_runner] %(levelname)-8s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger("utility_runner")


def parse_command(command_str: str):
    """
    Parse a command string into a list of command parts.
    
    Args:
        command_str: Command string in JSON format
        
    Returns:
        List of command parts
    """
    try:
        cmd = json.loads(command_str)
        if not isinstance(cmd, list):
            raise ValueError("Command must be a list of strings")
        
        return cmd
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in command: {e}")


def main():
    """
    Main entry point for the utility runner.
    
    Reads configuration from environment variables and runs the appropriate command.
    """
    # Read configuration from environment variables
    transport_type = os.environ.get("TRANSPORT_TYPE")
    server_command_str = os.environ.get("SERVER_COMMAND")
    
    # Validate configuration
    if not transport_type:
        logger.critical("TRANSPORT_TYPE environment variable not set")
        return 1
    
    if not server_command_str:
        logger.critical("SERVER_COMMAND environment variable not set")
        return 1
    
    try:
        server_command = parse_command(server_command_str)
    except ValueError as e:
        logger.critical(f"Invalid SERVER_COMMAND: {e}")
        return 1
    
    # Handle different transport types
    if transport_type.lower() == "stdio":
        # Run STDIO adapter
        logger.info("Starting STDIO adapter")
        
        # Use execvpe to replace the current process with the adapter
        # This way, the adapter gets all of our environment variables
        adapter_command = [sys.executable, "stdio_adapter.py"]
        
        try:
            logger.info(f"Executing: {' '.join(adapter_command)}")
            os.execvpe(adapter_command[0], adapter_command, os.environ)
        except Exception as e:
            logger.critical(f"Failed to start STDIO adapter: {e}")
            return 1
    elif transport_type.lower() == "sse":
        # Run native server command
        logger.info(f"Starting native server: {' '.join(server_command)}")
        
        try:
            # Use execvpe to replace the current process with the server
            # This way, the server gets all of our environment variables
            os.execvpe(server_command[0], server_command, os.environ)
        except Exception as e:
            logger.critical(f"Failed to start server: {e}")
            return 1
    else:
        logger.critical(f"Unknown transport type: {transport_type}")
        return 1
    
    # This should never be reached due to execvpe
    return 0


if __name__ == "__main__":
    sys.exit(main())