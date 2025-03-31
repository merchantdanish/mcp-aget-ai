"""STDIO adapter for MCP servers in MCP Agent Cloud.

This module provides an adapter for STDIO-based MCP servers, allowing them
to be accessed via HTTP endpoints.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class StdioAdapter:
    """Adapter for STDIO-based MCP servers.
    
    This adapter handles the communication between HTTP requests and
    STDIO-based MCP servers. It starts a subprocess running the MCP server
    and communicates with it via stdin/stdout.
    """
    
    def __init__(self, command: str, args: List[str]):
        """Initialize the STDIO adapter.
        
        Args:
            command: Command to run
            args: Arguments for the command
        """
        self.process = None
        self.command = command
        self.args = args
        self.id = str(uuid.uuid4())
        self.lock = asyncio.Lock()
        
    async def start(self):
        """Start the STDIO process."""
        logger.debug(f"Starting process: {self.command} {' '.join(self.args)}")
        self.process = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Create a task to monitor stderr and log it
        asyncio.create_task(self._log_stderr())
        
    async def _log_stderr(self):
        """Log stderr output from the process."""
        while self.process and not self.process.stderr.at_eof():
            try:
                line = await self.process.stderr.readline()
                if line:
                    logger.debug(f"STDIO stderr: {line.decode().strip()}")
                else:
                    break
            except:
                break
        
    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request by passing it to the STDIO process.
        
        Args:
            request_data: Request data as JSON object
            
        Returns:
            Response from the STDIO process
        """
        async with self.lock:  # Ensure one request at a time
            if not self.process:
                await self.start()
                
            request_json = json.dumps(request_data) + "\n"
            logger.debug(f"Sending request to STDIO process: {request_json}")
            
            try:
                # Write to stdin
                self.process.stdin.write(request_json.encode())
                await self.process.stdin.drain()
                
                # Read from stdout
                response_line = await self.process.stdout.readline()
                if not response_line:
                    # Process probably crashed, restart it
                    logger.error("STDIO process exited unexpectedly, restarting...")
                    await self.start()
                    self.process.stdin.write(request_json.encode())
                    await self.process.stdin.drain()
                    response_line = await self.process.stdout.readline()
                    
                logger.debug(f"Received response from STDIO process: {response_line.decode()}")
                response = json.loads(response_line.decode())
                
                return response
            except Exception as e:
                logger.error(f"Error communicating with STDIO process: {str(e)}")
                # Try to restart the process
                if self.process:
                    try:
                        self.process.terminate()
                    except:
                        pass
                await self.start()
                return {
                    "error": {
                        "message": f"Error communicating with STDIO process: {str(e)}"
                    }
                }
        
    async def stop(self):
        """Stop the STDIO process."""
        logger.debug("Stopping STDIO process")
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except:
                pass
            self.process = None


class StdioServer:
    """HTTP server that adapts STDIO-based MCP servers.
    
    This server exposes HTTP endpoints for interacting with STDIO-based
    MCP servers. It handles the conversion between HTTP requests and
    STDIO communication.
    """
    
    def __init__(self, adapters: Dict[str, StdioAdapter] = None):
        """Initialize the STDIO server.
        
        Args:
            adapters: Dictionary of server_name -> StdioAdapter
        """
        self.adapters = adapters or {}
        
    def add_adapter(self, server_name: str, command: str, args: List[str]) -> None:
        """Add a new adapter for a server.
        
        Args:
            server_name: Name of the server
            command: Command to run
            args: Arguments for the command
        """
        self.adapters[server_name] = StdioAdapter(command, args)
        
    def remove_adapter(self, server_name: str) -> None:
        """Remove an adapter.
        
        Args:
            server_name: Name of the server
        """
        if server_name in self.adapters:
            adapter = self.adapters.pop(server_name)
            asyncio.create_task(adapter.stop())
            
    async def handle_request(self, server_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request for a specific server.
        
        Args:
            server_name: Name of the server
            request_data: Request data as JSON object
            
        Returns:
            Response from the STDIO process
        """
        adapter = self.adapters.get(server_name)
        if not adapter:
            return {
                "error": {
                    "message": f"Server '{server_name}' not found"
                }
            }
            
        return await adapter.handle_request(request_data)
    
    async def stop_all(self) -> None:
        """Stop all adapters."""
        for adapter in self.adapters.values():
            await adapter.stop()
        self.adapters = {}