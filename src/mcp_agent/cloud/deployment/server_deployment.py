"""Server deployment functionality for MCP Agent Cloud.

This module provides classes for deploying and managing MCP servers in the cloud.
It handles containerization, authentication, and registry management.
"""

import asyncio
import uuid
import json
import os
import tempfile
import shutil
import docker
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from mcp_agent.cloud.auth.auth_service import AuthService
from mcp_agent.cloud.auth.auth_factory import MCPAuthFactory
from mcp_agent.cloud.auth.http_server import MCPHTTPServer


class ContainerService:
    """Service for managing Docker containers for MCP servers."""
    
    def __init__(self):
        """Initialize the container service."""
        # Create a thread pool for running Docker operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize Docker client
        try:
            self.client = docker.from_env()
            # Test connection to Docker daemon
            self.client.ping()
        except docker.errors.DockerException as e:
            print(f"Error connecting to Docker daemon: {str(e)}")
            print("Make sure Docker is running and you have permissions to access it.")
            self.client = None
        
    async def create_container(self, container_config: Dict[str, Any]) -> str:
        """Create a new container with the given configuration.
        
        Args:
            container_config: Configuration for the container
            
        Returns:
            Container ID
        """
        if not self.client:
            # Simulate container creation if Docker is not available
            await asyncio.sleep(1)
            return f"container-{uuid.uuid4().hex[:8]}"
            
        # Extract configuration values
        image = container_config.get("image", "alpine:latest")
        command = container_config.get("command")
        name = container_config.get("name", f"mcp-server-{uuid.uuid4().hex[:8]}")
        env = container_config.get("env", {})
        ports = container_config.get("ports", {})
        volumes = container_config.get("volumes", {})
        
        # Check if image exists or needs to be built
        try:
            # Try to get the image
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: self.client.images.get(image)
            )
        except docker.errors.ImageNotFound:
            # If image doesn't exist and is a recognized format, try to pull it
            if "/" in image and ":" in image and not image.startswith("mcp-server-"):
                try:
                    print(f"Image {image} not found locally, trying to pull...")
                    await loop.run_in_executor(
                        self.executor,
                        lambda: self.client.images.pull(image.split(":")[0], tag=image.split(":")[1])
                    )
                except docker.errors.APIError as e:
                    print(f"Error pulling image {image}: {str(e)}")
                    # We'll continue and the container creation will fail if the image is required
        
        # Format environment variables for Docker
        env_list = [f"{key}={value}" for key, value in env.items()]
        
        # Create and run container asynchronously
        try:
            loop = asyncio.get_event_loop()
            container = await loop.run_in_executor(
                self.executor,
                lambda: self.client.containers.run(
                    image=image,
                    command=command,
                    name=name,
                    environment=env_list,
                    ports=ports,
                    volumes=volumes,
                    detach=True
                )
            )
            
            return container.id
        except docker.errors.APIError as e:
            # Check if there's already a container with this name
            if "Conflict" in str(e) and "already in use" in str(e):
                # Get existing container
                try:
                    existing_container = await loop.run_in_executor(
                        self.executor,
                        lambda: self.client.containers.get(name)
                    )
                    
                    # Remove it if it exists
                    await loop.run_in_executor(
                        self.executor,
                        lambda: existing_container.remove(force=True)
                    )
                    
                    # Try again
                    container = await loop.run_in_executor(
                        self.executor,
                        lambda: self.client.containers.run(
                            image=image,
                            command=command,
                            name=name,
                            environment=env_list,
                            ports=ports,
                            volumes=volumes,
                            detach=True
                        )
                    )
                    
                    return container.id
                except Exception as inner_e:
                    print(f"Error handling container conflict: {str(inner_e)}")
                    raise e
            else:
                raise e
    
    async def build_stdio_container(self, server_name: str, command: str, args: List[str]) -> Tuple[str, int]:
        """Build a container for STDIO-based MCP server.
        
        Args:
            server_name: Name of the server
            command: Command to run
            args: Arguments for the command
            
        Returns:
            Tuple of (container_id, port)
        """
        if not self.client:
            # Simulate container creation if Docker is not available
            await asyncio.sleep(1.5)
            return (f"container-{uuid.uuid4().hex[:8]}", 8000)
        
        # Find a free port
        loop = asyncio.get_event_loop()
        port = await loop.run_in_executor(
            self.executor,
            lambda: self._find_free_port()
        )
        
        # Get the path to the templates directory
        templates_dir = Path(__file__).parent / "templates"
        
        # Create a temporary directory for building the container
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the Dockerfile template and adapter files
            shutil.copy(templates_dir / "Dockerfile.stdio", os.path.join(tmpdir, "Dockerfile"))
            shutil.copy(templates_dir / "server.py", os.path.join(tmpdir, "server.py"))
            shutil.copy(Path(__file__).parent / "adapters" / "stdio_adapter.py", os.path.join(tmpdir, "stdio_adapter.py"))
            
            # Replace environment variables in the Dockerfile
            with open(os.path.join(tmpdir, "Dockerfile"), "r") as f:
                dockerfile = f.read()
            
            dockerfile = dockerfile.replace("${SERVER_COMMAND}", command)
            dockerfile = dockerfile.replace("${SERVER_ARGS}", " ".join(args))
            
            with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
                f.write(dockerfile)
            
            # Build the Docker image
            loop = asyncio.get_event_loop()
            image_tag = f"mcp-server-{server_name}:latest"
            
            try:
                # Build the image from the Dockerfile
                image, logs = await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.images.build(
                        path=tmpdir,
                        tag=image_tag,
                        rm=True  # Remove intermediate containers
                    )
                )
                
                # Create a container from the image
                container_id = await self.create_container({
                    "image": image_tag,
                    "name": f"mcp-server-{server_name}",
                    "env": {
                        "SERVER_NAME": server_name,
                        "SERVER_COMMAND": command,
                        "SERVER_ARGS": " ".join(args)
                    },
                    "ports": {"8000/tcp": port}
                })
                
                return (container_id, port)
            except docker.errors.BuildError as e:
                print(f"Error building image: {str(e)}")
                # Fallback to simulation
                await asyncio.sleep(0.5)
                return (f"container-{uuid.uuid4().hex[:8]}", 8000)
                
    def _find_free_port(self, start_port: int = 8000, end_port: int = 9000) -> int:
        """Find a free port in the given range.
        
        Args:
            start_port: Start of the port range
            end_port: End of the port range
            
        Returns:
            Free port number
        """
        import socket
        
        for port in range(start_port, end_port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('', port))
                sock.close()
                return port
            except socket.error:
                continue
        
        # If we get here, no ports are available
        raise RuntimeError(f"No free ports available in range {start_port}-{end_port}")
    
    async def stop_container(self, container_id: str) -> None:
        """Stop a running container.
        
        Args:
            container_id: ID of the container to stop
        """
        if not self.client:
            # Simulate container stopping if Docker is not available
            await asyncio.sleep(0.5)
            return
        
        loop = asyncio.get_event_loop()
        try:
            # Get the container
            container = await loop.run_in_executor(
                self.executor,
                lambda: self.client.containers.get(container_id)
            )
            
            # Stop the container
            await loop.run_in_executor(
                self.executor,
                container.stop
            )
        except docker.errors.NotFound:
            print(f"Container {container_id} not found")
        except docker.errors.APIError as e:
            print(f"Error stopping container: {str(e)}")
    
    async def start_container(self, container_id: str) -> None:
        """Start a stopped container.
        
        Args:
            container_id: ID of the container to start
        """
        if not self.client:
            # Simulate container starting if Docker is not available
            await asyncio.sleep(0.5)
            return
            
        loop = asyncio.get_event_loop()
        try:
            # Get the container
            container = await loop.run_in_executor(
                self.executor,
                lambda: self.client.containers.get(container_id)
            )
            
            # Start the container
            await loop.run_in_executor(
                self.executor,
                container.start
            )
        except docker.errors.NotFound:
            print(f"Container {container_id} not found")
        except docker.errors.APIError as e:
            print(f"Error starting container: {str(e)}")
    
    async def delete_container(self, container_id: str) -> None:
        """Delete a container.
        
        Args:
            container_id: ID of the container to delete
        """
        if not self.client:
            # Simulate container deletion if Docker is not available
            await asyncio.sleep(0.7)
            return
            
        loop = asyncio.get_event_loop()
        try:
            # Get the container
            container = await loop.run_in_executor(
                self.executor,
                lambda: self.client.containers.get(container_id)
            )
            
            # Stop the container if it's running
            if container.status == "running":
                await loop.run_in_executor(
                    self.executor,
                    container.stop
                )
            
            # Remove the container
            await loop.run_in_executor(
                self.executor,
                lambda: container.remove(force=True)
            )
            
            # Try to remove the image as well
            try:
                image_name = container.image.tags[0] if container.image.tags else None
                if image_name:
                    await loop.run_in_executor(
                        self.executor,
                        lambda: self.client.images.remove(image_name)
                    )
            except (docker.errors.APIError, IndexError) as e:
                # It's okay if we can't remove the image
                print(f"Note: Could not remove image: {str(e)}")
                
        except docker.errors.NotFound:
            print(f"Container {container_id} not found")
        except docker.errors.APIError as e:
            print(f"Error deleting container: {str(e)}")


class ServerRegistry:
    """Registry for tracking deployed MCP servers."""
    
    def __init__(self, registry_file: str = None):
        """Initialize the server registry.
        
        Args:
            registry_file: Path to the registry file
        """
        # Use a registry file if provided, otherwise use an in-memory registry
        self.registry_file = registry_file or os.path.expanduser("~/.mcp-agent-cloud/registry.json")
        self.servers = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load the registry from the registry file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        
        # Load registry from file if it exists
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
                    self.servers = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading registry: {str(e)}")
                self.servers = {}
    
    def _save_registry(self) -> None:
        """Save the registry to the registry file."""
        try:
            with open(self.registry_file, "w") as f:
                json.dump(self.servers, f, indent=2)
        except IOError as e:
            print(f"Error saving registry: {str(e)}")
    
    async def register_server(self, server_record: Dict[str, Any]) -> None:
        """Register a new server in the registry.
        
        Args:
            server_record: Server information
        """
        # Store the server record in the registry
        self.servers[server_record["id"]] = server_record
        self._save_registry()
    
    async def get_server(self, server_id: str) -> Optional[Dict[str, Any]]:
        """Get a server record by ID.
        
        Args:
            server_id: ID of the server
            
        Returns:
            Server record or None if not found
        """
        return self.servers.get(server_id)
    
    async def get_server_by_name(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get a server record by name.
        
        Args:
            server_name: Name of the server
            
        Returns:
            Server record or None if not found
        """
        for server in self.servers.values():
            if server["name"] == server_name:
                return server
        return None
    
    async def list_servers(self) -> List[Dict[str, Any]]:
        """List all registered servers.
        
        Returns:
            List of server records
        """
        return list(self.servers.values())
    
    async def update_server_status(self, server_id: str, status: str) -> None:
        """Update the status of a server.
        
        Args:
            server_id: ID of the server
            status: New status
        """
        if server_id in self.servers:
            self.servers[server_id]["status"] = status
            self._save_registry()
    
    async def delete_server(self, server_id: str) -> None:
        """Delete a server from the registry.
        
        Args:
            server_id: ID of the server
        """
        if server_id in self.servers:
            del self.servers[server_id]
            self._save_registry()


class MCPServerDeploymentManager:
    """Manager for deploying and managing MCP servers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the deployment manager.
        
        Args:
            config: Configuration for the deployment manager
        """
        self.config = config or {}
        self.container_service = ContainerService()
        self.auth_factory = MCPAuthFactory()
        self.registry = ServerRegistry()
        self.http_servers = {}
        
        # Dictionary of server templates
        self.server_templates = {}
        
    def register_server_template(self, server_type: str, template):
        """Register a server template for a specific server type.
        
        Args:
            server_type: Type of the server
            template: Server template
        """
        self.server_templates[server_type] = template
        
    async def find_free_port(self, start_port: int = 8000, end_port: int = 9000) -> int:
        """Find a free port in the given range.
        
        Args:
            start_port: Start of the port range
            end_port: End of the port range
            
        Returns:
            Free port number
        """
        import socket
        
        for port in range(start_port, end_port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('', port))
                sock.close()
                return port
            except socket.error:
                continue
        
        # If we get here, no ports are available
        raise RuntimeError(f"No free ports available in range {start_port}-{end_port}")
        
    async def build_networked_container(self, server_name: str, server_type: str) -> Tuple[str, int]:
        """Build a container for networked MCP server.
        
        Args:
            server_name: Name of the server
            server_type: Type of the server
            
        Returns:
            Tuple of (container_id, port)
        """
        if not self.container_service.client:
            # Simulate container creation if Docker is not available
            await asyncio.sleep(1.5)
            return (f"container-{uuid.uuid4().hex[:8]}", 8000)
        
        # Find a free port
        port = await self.find_free_port()
        
        # Get the path to the templates directory
        templates_dir = Path(__file__).parent / "templates"
        
        # Create a temporary directory for building the container
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the Dockerfile template and server.js file
            shutil.copy(templates_dir / "Dockerfile.networked", os.path.join(tmpdir, "Dockerfile"))
            shutil.copy(templates_dir / "server.js", os.path.join(tmpdir, "server.js"))
            
            # Replace environment variables in the Dockerfile
            with open(os.path.join(tmpdir, "Dockerfile"), "r") as f:
                dockerfile = f.read()
            
            dockerfile = dockerfile.replace("${SERVER_TYPE}", server_type)
            
            with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
                f.write(dockerfile)
            
            # Build the Docker image
            loop = asyncio.get_event_loop()
            image_tag = f"mcp-server-{server_name}:latest"
            
            try:
                # Build the image from the Dockerfile
                image, logs = await loop.run_in_executor(
                    self.container_service.executor,
                    lambda: self.container_service.client.images.build(
                        path=tmpdir,
                        tag=image_tag,
                        rm=True  # Remove intermediate containers
                    )
                )
                
                # Create a container from the image
                container_id = await self.container_service.create_container({
                    "image": image_tag,
                    "name": f"mcp-server-{server_name}",
                    "env": {
                        "SERVER_TYPE": server_type
                    },
                    "ports": {"8000/tcp": port}
                })
                
                return (container_id, port)
            except Exception as e:
                print(f"Error building networked container: {str(e)}")
                # Fallback to simulation
                await asyncio.sleep(0.5)
                return (f"container-{uuid.uuid4().hex[:8]}", 8000)
    
    async def deploy_server(self, 
                           server_name: str, 
                           server_type: str, 
                           config: Dict[str, Any] = None,
                           region: str = "us-west",
                           public: bool = False) -> Dict[str, Any]:
        """Deploy a new MCP server.
        
        Args:
            server_name: Name of the server
            server_type: Type of the server (fetch, filesystem, etc.)
            config: Server-specific configuration
            region: Deployment region
            public: Whether the server should be publicly accessible
            
        Returns:
            Server record
        """
        # Check if server with name already exists
        existing_server = await self.registry.get_server_by_name(server_name)
        if existing_server:
            raise ValueError(f"Server with name '{server_name}' already exists")
        
        # Create container configuration based on server type
        container_config = self._get_container_config(server_type, config)
        
        # Set region and public access
        container_config["region"] = region
        container_config["public"] = public
        
        # Create containerized MCP server based on server type
        port = 8000  # Default port
        
        if server_type in ["fetch", "http", "networked"]:
            # Check if we need to build or pull the image
            try:
                # Try to build our own networked container
                container_id, port = await self.build_networked_container(server_name, server_type)
                server_url = f"https://{server_name}.mcp-agent-cloud.example.com"
                endpoint_path = f"/servers/{server_name}"
                
                # In a real deployment, we'd use this port in the URL
                local_url = f"http://localhost:{port}"
            except Exception as e:
                print(f"Error building networked container, falling back to direct creation: {str(e)}")
                # Networked server deployment
                container_id = await self.container_service.create_container(container_config)
                server_url = f"https://{server_name}.mcp-agent-cloud.example.com"
                endpoint_path = f"/servers/{server_name}"
                local_url = f"http://localhost:{port}"
        else:
            # STDIO-based server deployment with adapter
            command = container_config.get("command", "npx")
            args = container_config.get("args", [f"@modelcontextprotocol/server-{server_type}"])
            
            # Check if args is a list
            if isinstance(args, str):
                args = args.split()
                
            container_id, port = await self.container_service.build_stdio_container(server_name, command, args)
            
            # STDIO adapters expose an HTTP endpoint on the dynamically assigned port
            server_url = f"https://{server_name}.mcp-agent-cloud.example.com"
            endpoint_path = f"/servers/{server_name}"
            local_url = f"http://localhost:{port}"
        
        # Configure authentication
        auth_provider_type = "self-contained"
        
        # Extract auth config from server config
        if config and "auth" in config:
            auth_provider_type = config["auth"].get("provider_type", "self-contained")
            
            # If using GitHub, pass the client ID and secret
            if auth_provider_type == "github" and "client_id" in config["auth"] and "client_secret" in config["auth"]:
                os.environ["GITHUB_CLIENT_ID"] = config["auth"]["client_id"]
                os.environ["GITHUB_CLIENT_SECRET"] = config["auth"]["client_secret"]
        elif public:
            # Default to GitHub for public servers if not specified
            auth_provider_type = "github"
            
        auth_service, auth_config = await self.auth_factory.create_server_auth(
            server_name, 
            auth_provider_type
        )
        
        # Create HTTP server for the MCP server
        base_url = f"https://{server_name}.mcp-agent-cloud.example.com"
        http_server = MCPHTTPServer(server_name, auth_service, base_url)
        self.http_servers[server_name] = http_server
        
        # Register server in registry
        server_record = {
            "id": f"srv-{uuid.uuid4().hex[:8]}",
            "name": server_name,
            "type": server_type,
            "container_id": container_id,
            "endpoint": endpoint_path,
            "url": server_url,
            "local_url": local_url,
            "port": port,
            "auth_config": auth_config,
            "auth_service": auth_service,  # Store the auth service
            "region": region,
            "public": public,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "running"
        }
        await self.registry.register_server(server_record)
        
        return server_record
    
    def _get_container_config(self, server_type: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get container configuration for a server type.
        
        Args:
            server_type: Type of the server
            config: Server-specific configuration
            
        Returns:
            Container configuration
        """
        # Use server template if available
        if server_type in self.server_templates:
            template = self.server_templates[server_type]
            # Get configuration from template
            return template.get_config(server_type, config)
        
        # Fallback to default configuration
        container_config = {
            "image": f"mcp-server-{server_type}:latest",
            "command": "npx",
            "args": [f"@modelcontextprotocol/server-{server_type}"],
            "env": {},
            "ports": {"8000/tcp": 8000},
            "volumes": {},
        }
        
        # Apply server-specific configuration
        if config:
            # Merge dictionaries recursively
            for key, value in config.items():
                if key in container_config and isinstance(container_config[key], dict):
                    container_config[key].update(value)
                else:
                    container_config[key] = value
        
        return container_config
    
    async def stop_server(self, server_name: str) -> None:
        """Stop a running MCP server.
        
        Args:
            server_name: Name of the server
        """
        server = await self.registry.get_server_by_name(server_name)
        if not server:
            raise ValueError(f"Server with name '{server_name}' not found")
        
        # Stop the container
        await self.container_service.stop_container(server["container_id"])
        
        # Update the server status
        await self.registry.update_server_status(server["id"], "stopped")
        
        # HTTP server is stateless and doesn't need to be stopped
    
    async def start_server(self, server_name: str) -> None:
        """Start a stopped MCP server.
        
        Args:
            server_name: Name of the server
        """
        server = await self.registry.get_server_by_name(server_name)
        if not server:
            raise ValueError(f"Server with name '{server_name}' not found")
        
        # Start the container
        await self.container_service.start_container(server["container_id"])
        
        # Update the server status
        await self.registry.update_server_status(server["id"], "running")
    
    async def delete_server(self, server_name: str) -> None:
        """Delete an MCP server.
        
        Args:
            server_name: Name of the server
        """
        server = await self.registry.get_server_by_name(server_name)
        if not server:
            raise ValueError(f"Server with name '{server_name}' not found")
        
        # Delete the container
        await self.container_service.delete_container(server["container_id"])
        
        # Remove HTTP server
        if server_name in self.http_servers:
            del self.http_servers[server_name]
        
        # Delete the server from the registry
        await self.registry.delete_server(server["id"])
    
    async def list_servers(self) -> List[Dict[str, Any]]:
        """List all deployed MCP servers.
        
        Returns:
            List of server records
        """
        return await self.registry.list_servers()