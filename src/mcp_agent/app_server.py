"""
MCPAgentServer - Exposes mcp-agent workflows and agents as MCP tools.
"""

import asyncio
import inspect
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Type

from mcp.server.fastmcp import Context as MCPContext, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.session import BaseSession
from mcp.types import ToolListChangedNotification

from mcp_agent.app import MCPApp
from mcp_agent.app_server_types import (
    MCPMessageParam,
    MCPMessageResult,
    create_model_from_schema,
)
from mcp_agent.agents.agent import Agent
from mcp_agent.config import MCPServerSettings
from mcp_agent.context_dependent import ContextDependent
from mcp_agent.executor.workflow import Workflow
from mcp_agent.executor.workflow_signal import Signal
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp_server_registry import ServerRegistry
from mcp_agent.workflows.llm.augmented_llm import MessageParamT, RequestParams

logger = get_logger(__name__)


class ServerContext(ContextDependent):
    """Context object for the MCP App server."""

    def __init__(self, mcp: FastMCP, context=None, **kwargs):
        super().__init__(context=context, **kwargs)
        self.mcp = mcp
        self.active_workflows: Dict[str, Any] = {}
        self.active_agents: Dict[str, Agent] = {}

        # Register existing workflows from the app
        for workflow_id, workflow_cls in self.context.app.workflows.items():
            self.register_workflow(workflow_id, workflow_cls)

    def register_workflow(self, workflow_id: str, workflow_cls: Type[Workflow]):
        """Register a workflow class."""
        if workflow_id not in self.context.app.workflows:
            self.context.app.workflows[workflow_id] = workflow_cls
            # Create tools for this workflow
            create_workflow_specific_tools(self.mcp, workflow_id, workflow_cls)

    def register_agent(self, agent: Agent):
        """Register an agent instance."""
        if agent.name not in self.active_agents:
            self.active_agents[agent.name] = agent
            # Create tools for this agent
            create_agent_specific_tools(self.mcp, agent.name, agent)
            return agent
        return self.active_agents[
            agent.name
        ]  # Return existing agent if already registered


def create_mcp_server_for_app(app: MCPApp) -> FastMCP:
    """
    Create an MCP server for a given MCPApp instance.

    Args:
        app: The MCPApp instance to create a server for

    Returns:
        A configured FastMCP server instance
    """

    # Create a lifespan function specific to this app
    @asynccontextmanager
    async def app_specific_lifespan(mcp: FastMCP) -> AsyncIterator[ServerContext]:
        """Initialize and manage MCPApp lifecycle."""
        # Initialize the app if it's not already initialized
        await app.initialize()

        # Create the server context which is available during the lifespan of the server
        server_context = ServerContext(mcp=mcp, context=app.context)

        # Register initial agent and workflow tools
        create_agent_tools(mcp, server_context)
        create_workflow_tools(mcp, server_context)

        try:
            yield server_context
        finally:
            # Don't clean up the MCPApp here - let the caller handle that
            pass

    # Create FastMCP server with the app's name
    mcp = FastMCP(
        name=app.name or "mcp_agent_server",
        # TODO: saqadri (MAC) - create a much more detailed description based on all the available agents and workflows,
        # or use the MCPApp's description if available.
        instructions=f"MCP server exposing {app.name} workflows and agents",
        lifespan=app_specific_lifespan,
    )

    # region Server Tools

    @mcp.tool(name="servers/list")
    def list_servers(ctx: MCPContext) -> List[MCPServerSettings]:
        """
        List all available MCP servers packaged with this MCP App server, along with their detailed information.

        Returns information about each server including its name, description,
        and configuration. This helps in understanding what each server is capable of,
        and consequently what this MCP App server can accomplish.
        """
        server_context: ServerContext = ctx.request_context.lifespan_context
        server_registry = server_context.context.server_registry

        if not server_registry:
            raise ToolError("Server registry not found for MCP App Server.")

        result: List[MCPServerSettings] = []
        for _, server_settings in server_registry.registry.items():
            # Remove sensitive information from the server settings
            safe_server_settings = server_settings.model_dump(exclude={"auth", "env"})
            result.append(MCPServerSettings(**safe_server_settings))

        return result

    # region Agent Tools

    @mcp.tool(name="agents/list")
    def list_agents(ctx: MCPContext) -> Dict[str, Dict[str, Any]]:
        """
        List all available agents with their detailed information.

        Returns information about each agent including their name, instruction,
        and the MCP servers they have access to. This helps with understanding
        what each agent is designed to do before calling it.
        """
        server_context: ServerContext = ctx.request_context.lifespan_context
        server_registry = server_context.context.server_registry
        result = {}
        for name, agent in server_context.active_agents.items():
            # Format instruction - handle callable instructions
            instruction = agent.instruction
            if callable(instruction):
                instruction = instruction({})

            servers = _get_server_descriptions(server_registry, agent.server_names)

            # Build detailed agent info
            result[name] = {
                "name": name,
                "instruction": instruction,
                "servers": servers,
                "capabilities": ["generate", "generate_str", "generate_structured"],
                "tool_endpoints": [
                    f"agents/{name}/generate",
                    f"agents/{name}/generate_str",
                    f"agents/{name}/generate_structured",
                ],
            }

        return result

    @mcp.tool(name="agents/create")
    async def create_agent(
        ctx: MCPContext,
        name: str,
        instruction: str,
        server_names: List[str],
        llm: Literal["openai", "anthropic"] = "openai",
    ) -> Dict[str, Any]:
        """
        Create a new agent with given name, instruction and list of MCP servers it is allowed to access.

        Args:
            name: The name of the agent to create. It must be a unique name not already in agents/list.
            instruction: Instructions for the agent (i.e. system prompt).
            server_names: List of MCP server names the agent should be able to access.
                These MUST be one of the names retrieved using servers/list tool endpoint.

        Returns:
            Detailed information about the created agent.
        """
        server_context: ServerContext = ctx.request_context.lifespan_context

        agent = Agent(
            name=name,
            instruction=instruction,
            server_names=server_names,
            context=server_context.context,
        )

        # TODO: saqadri (MAC) - Add better support for multiple LLMs.
        if llm == "openai":
            from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM  # pylint: disable=C0415

            await agent.attach_llm(OpenAIAugmentedLLM)
        elif llm == "anthropic":
            from mcp_agent.workflows.llm.augmented_llm_anthropic import (  # pylint: disable=C0415
                AnthropicAugmentedLLM,
            )

            await agent.attach_llm(AnthropicAugmentedLLM)
        else:
            raise ToolError(
                f"Unsupported LLM type: {llm}. Only 'openai' and 'anthropic' are presently supported."
            )

        await agent.initialize()
        server_context.register_agent(agent)

        # Notify that tools have changed
        session: BaseSession = ctx.session
        session.send_notification(
            ToolListChangedNotification(method="notifications/tools/list_changed")
        )

        server_registry = server_context.context.server_registry
        servers = _get_server_descriptions(server_registry, agent.server_names)

        # Return detailed agent info
        return {
            "name": name,
            "instruction": instruction,
            "servers": servers,
            "capabilities": ["generate", "generate_str", "generate_structured"],
            "tool_endpoints": [
                f"agents/{name}/generate",
                f"agents/{name}/generate_structured",
            ],
        }

    @mcp.tool(name="agents/generate")
    async def agent_generate(
        ctx: MCPContext,
        agent_name: str,
        message: str | MCPMessageParam | List[MCPMessageParam],
        request_params: RequestParams | None = None,
    ) -> List[MCPMessageResult]:
        """
        Run an agent using the given message.
        This is similar to generating an LLM completion.

        Args:
            agent_name: Name of the agent to use. This must be one of the names retrieved using agents/list tool endpoint.
            message: The prompt to send to the agent.
            request_params: Optional parameters for the request, such as max_tokens and model/model preferences.

        Returns:
            The generated response from the agent.
        """
        return await _agent_generate(ctx, agent_name, message, request_params)

    @mcp.tool(name="agents/generate_str")
    async def agent_generate_str(
        ctx: MCPContext,
        agent_name: str,
        message: str | MCPMessageParam | List[MCPMessageParam],
        request_params: RequestParams | None = None,
    ) -> str:
        """
        Run an agent using the given message and return the response as a string.
        Use agents/generate for results in the original format, and
        use agents/generate_structured for results conforming to a specific schema.

        Args:
            agent_name: Name of the agent to use. This must be one of the names retrieved using agents/list tool endpoint.
            message: The prompt to send to the agent.
            request_params: Optional parameters for the request, such as max_tokens and model/model preferences.

        Returns:
            The generated response from the agent.
        """
        return await _agent_generate_str(ctx, agent_name, message, request_params)

    @mcp.tool(name="agents/generate_structured")
    async def agent_generate_structured(
        ctx: MCPContext,
        agent_name: str,
        message: str | MCPMessageParam | List[MCPMessageParam],
        response_schema: Dict[str, Any],
        request_params: RequestParams | None = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured response from an agent that matches the given schema.

        Args:
            agent_name: Name of the agent to use. This must be one of the names retrieved using agents/list tool endpoint.
            message: The prompt to send to the agent.
            response_schema: The JSON schema that defines the shape to generate the response in.
                This schema can be generated using type.schema_json() for a Pydantic model.
            request_params: Optional parameters for the request, such as max_tokens and model/model preferences.

        Returns:
            A dictionary representation of the structured response.

        Example:
            response_schema:
            {
                "title": "UserProfile",
                "type": "object",
                "properties": {
                    "name": {
                        "title": "Name",
                        "type": "string"
                    },
                    "age": {
                        "title": "Age",
                        "type": "integer",
                        "minimum": 0
                    },
                    "email": {
                        "title": "Email",
                        "type": "string",
                        "format": "email",
                        "pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
                    }
                },
                "required": [
                    "name",
                    "age",
                    "email"
                ]
            }
        """
        return await _agent_generate_structured(
            ctx, agent_name, message, response_schema, request_params
        )

    # endregion

    # region Workflow Tools

    @mcp.tool(name="workflows/list")
    def list_workflows(ctx: MCPContext) -> Dict[str, Dict[str, Any]]:
        """
        List all available workflows exposed with their detailed information.

        Returns information about each workflow including name, description, and parameters.
        This helps in making an informed decision about which workflow to run.
        """
        server_config: ServerContext = ctx.request_context.lifespan_context

        result = {}
        for workflow_id, workflow_cls in server_config.context.app.workflows.items():
            # Get workflow documentation
            doc = workflow_cls.__doc__ or "No description available"

            # Get workflow run method parameters using inspection
            parameters = {}
            if hasattr(workflow_cls, "run"):
                sig = inspect.signature(workflow_cls.run)
                for param_name, param in sig.parameters.items():
                    if param_name != "self":
                        param_info = {
                            "type": str(param.annotation)
                            .replace("<class '", "")
                            .replace("'>", ""),
                            "required": param.default == inspect.Parameter.empty,
                        }
                        if param.default != inspect.Parameter.empty:
                            param_info["default"] = param.default
                        parameters[param_name] = param_info

            result[workflow_id] = {
                "name": workflow_id,
                "description": doc.strip(),
                "parameters": parameters,
                "capabilities": ["run", "pause", "resume", "cancel", "get_status"],
                "tool_endpoints": [
                    f"workflows/{workflow_id}/run",
                    f"workflows/{workflow_id}/get_status",
                    f"workflows/{workflow_id}/pause",
                    f"workflows/{workflow_id}/resume",
                    f"workflows/{workflow_id}/cancel",
                ],
            }

        return result

    @mcp.tool(name="workflows/list_running")
    def list_running_workflows(ctx: MCPContext) -> Dict[str, Dict[str, Any]]:
        """
        List all running workflow instances with their detailed status information.

        For each running workflow, returns its ID, name, current state, and available operations.
        This helps in identifying and managing active workflow instances.

        Returns:
            A dictionary mapping workflow IDs to their detailed status information.
        """
        server_config: ServerContext = ctx.request_context.lifespan_context

        result = {}
        for workflow_id, workflow in server_config.active_workflows.items():
            # Skip task entries
            if workflow_id.endswith("_task"):
                continue

            task = server_config.active_workflows.get(workflow_id + "_task")

            # Get workflow information
            workflow_info = {
                "id": workflow_id,
                "name": workflow.name,
                "running": task is not None and not task.done() if task else False,
                "state": workflow.state.model_dump()
                if hasattr(workflow, "state")
                else {},
                "tool_endpoints": [
                    f"workflows/{workflow.name}/get_status",
                    f"workflows/{workflow.name}/pause",
                    f"workflows/{workflow.name}/resume",
                    f"workflows/{workflow.name}/cancel",
                ],
            }

            if task and task.done():
                try:
                    task_result = task.result()
                    workflow_info["result"] = (
                        task_result.model_dump()
                        if hasattr(task_result, "model_dump")
                        else str(task_result)
                    )
                    workflow_info["completed"] = True
                    workflow_info["error"] = None
                except Exception as e:
                    workflow_info["result"] = None
                    workflow_info["completed"] = False
                    workflow_info["error"] = str(e)

            result[workflow_id] = workflow_info

        return result

    @mcp.tool(name="workflows/run")
    async def run_workflow(
        ctx: MCPContext,
        workflow_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a workflow with the given name.

        Args:
            workflow_name: The name of the workflow to run.
            args: Optional arguments to pass to the workflow.

        Returns:
            Information about the running workflow including its ID and metadata.
        """
        server_config: ServerContext = ctx.request_context.lifespan_context
        app = server_config.context.app

        if workflow_name not in app.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found.")

        # Create a workflow instance
        workflow_cls = app.workflows[workflow_name]
        workflow = workflow_cls(executor=app.executor, name=workflow_name)

        # Generate a unique ID for this workflow instance
        workflow_id = str(uuid.uuid4())

        # Store the workflow instance
        server_config.active_workflows[workflow_id] = workflow

        # Run the workflow in a separate task
        args = args or {}
        run_task = asyncio.create_task(workflow.run(**args))

        # Store the task to check status later
        server_config.active_workflows[workflow_id + "_task"] = run_task

        # Return information about the workflow
        return {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "status": "running",
            "args": args,
            "tool_endpoints": [
                f"workflows/{workflow_name}/get_status",
                f"workflows/{workflow_name}/pause",
                f"workflows/{workflow_name}/resume",
                f"workflows/{workflow_name}/cancel",
            ],
            "message": f"Workflow {workflow_name} started with ID {workflow_id}. Use the returned workflow_id with other workflow tools.",
        }

    @mcp.tool(name="workflows/get_status")
    def get_workflow_status(ctx: MCPContext, workflow_id: str) -> Dict[str, Any]:
        """
        Get the status of a running workflow.

        Provides detailed information about a workflow instance including its current state,
        whether it's running or completed, and any results or errors encountered.

        Args:
            workflow_id: The ID of the workflow to check.

        Returns:
            A dictionary with comprehensive information about the workflow status.
        """
        server_config: ServerContext = ctx.request_context.lifespan_context

        if workflow_id not in server_config.active_workflows:
            raise ValueError(f"Workflow with ID '{workflow_id}' not found.")

        workflow = server_config.active_workflows[workflow_id]
        task = server_config.active_workflows.get(workflow_id + "_task")

        status = {
            "id": workflow_id,
            "name": workflow.name,
            "running": task is not None and not task.done() if task else False,
            "state": workflow.state.model_dump() if hasattr(workflow, "state") else {},
            "available_actions": ["pause", "resume", "cancel"]
            if task and not task.done()
            else [],
            "tool_endpoints": [
                f"workflows/{workflow.name}/get_status",
            ],
        }

        # Add appropriate action endpoints based on status
        if task and not task.done():
            status["tool_endpoints"].extend(
                [
                    f"workflows/{workflow.name}/pause",
                    f"workflows/{workflow.name}/resume",
                    f"workflows/{workflow.name}/cancel",
                ]
            )

        if task and task.done():
            try:
                result = task.result()

                # Convert result to a useful format
                if hasattr(result, "model_dump"):
                    result_data = result.model_dump()
                elif hasattr(result, "__dict__"):
                    result_data = result.__dict__
                else:
                    result_data = str(result)

                status["result"] = result_data
                status["completed"] = True
                status["error"] = None
            except Exception as e:
                status["result"] = None
                status["completed"] = False
                status["error"] = str(e)
                status["exception_type"] = type(e).__name__

        return status

    @mcp.tool(name="workflows/pause")
    async def pause_workflow(ctx: MCPContext, workflow_id: str) -> bool:
        """
        Pause a running workflow.

        Args:
            workflow_id: The ID of the workflow to pause.

        Returns:
            True if the workflow was paused, False otherwise.
        """
        server_config: ServerContext = ctx.request_context.lifespan_context

        if workflow_id not in server_config.active_workflows:
            raise ValueError(f"Workflow with ID '{workflow_id}' not found.")

        _workflow = server_config.active_workflows[workflow_id]

        # Signal the workflow to pause
        try:
            await server_config.context.app.executor.signal(
                "pause", workflow_id=workflow_id
            )
            return True
        except Exception as e:
            logger.error(f"Error pausing workflow {workflow_id}: {e}")
            return False

    @mcp.tool(name="workflows/resume")
    async def resume_workflow(
        ctx: MCPContext, workflow_id: str, input_data: Optional[str] = None
    ) -> bool:
        """
        Resume a paused workflow.

        Args:
            workflow_id: The ID of the workflow to resume.
            input_data: Optional input data to provide to the workflow.

        Returns:
            True if the workflow was resumed, False otherwise.
        """
        server_config: ServerContext = ctx.request_context.lifespan_context

        if workflow_id not in server_config.active_workflows:
            raise ValueError(f"Workflow with ID '{workflow_id}' not found.")

        # Signal the workflow to resume
        try:
            signal = Signal(name="resume", workflow_id=workflow_id, payload=input_data)
            await server_config.context.app.executor.signal_bus.signal(signal)
            return True
        except Exception as e:
            logger.error(f"Error resuming workflow {workflow_id}: {e}")
            return False

    @mcp.tool(name="workflows/cancel")
    async def cancel_workflow(ctx: MCPContext, workflow_id: str) -> bool:
        """
        Cancel a running workflow.

        Args:
            workflow_id: The ID of the workflow to cancel.

        Returns:
            True if the workflow was cancelled, False otherwise.
        """
        server_config: ServerContext = ctx.request_context.lifespan_context

        if workflow_id not in server_config.active_workflows:
            raise ValueError(f"Workflow with ID '{workflow_id}' not found.")

        task = server_config.active_workflows.get(workflow_id + "_task")

        if task and not task.done():
            # Cancel the task
            task.cancel()

            # Signal the workflow to cancel
            try:
                await server_config.context.app.executor.signal(
                    "cancel", workflow_id=workflow_id
                )

                # Remove from active workflows
                server_config.active_workflows.pop(workflow_id, None)
                server_config.active_workflows.pop(workflow_id + "_task", None)

                return True
            except Exception as e:
                logger.error(f"Error cancelling workflow {workflow_id}: {e}")
                return False

        return False

    @mcp.tool(name="workflow_signal/wait_for_signal")
    async def wait_for_signal(
        ctx: MCPContext,
        signal_name: str,
        workflow_id: str = None,
        description: str = None,
        timeout_seconds: int = None,
    ) -> Dict[str, Any]:
        """
        Provides information about a signal that a workflow is waiting for.

        This tool doesn't actually make the workflow wait (that's handled internally),
        but it provides information about what signal is being waited for and how to
        respond to it.

        Args:
            signal_name: The name of the signal to wait for.
            workflow_id: Optional workflow ID to associate with the signal.
            description: Optional description of what the signal is for.
            timeout_seconds: Optional timeout in seconds.

        Returns:
            Information about the signal and how to respond to it.
        """
        _server_context: ServerContext = ctx.request_context.lifespan_context

        # Inform about how to send the signal
        return {
            "signal_name": signal_name,
            "workflow_id": workflow_id,
            "description": description or f"Waiting for signal '{signal_name}'",
            "status": "waiting_for_signal",
            "timeout_seconds": timeout_seconds,
            "instructions": "To respond to this signal, use the workflow_signal/send tool with the same signal_name and workflow_id.",
            "related_tools": ["workflow_signal/send"],
        }

    @mcp.tool(name="workflow_signal/send")
    async def send_signal(
        ctx: MCPContext,
        signal_name: str,
        workflow_id: str = None,
        payload: Any = None,
    ) -> Dict[str, bool]:
        """
        Send a signal to a workflow.

        This can be used to respond to a workflow that is waiting for input or
        to send a signal to control workflow execution.

        Args:
            signal_name: The name of the signal to send.
            workflow_id: Optional workflow ID to associate with the signal.
            payload: Optional data to include with the signal.

        Returns:
            Confirmation that the signal was sent.
        """
        server_config: ServerContext = ctx.request_context.lifespan_context
        executor = server_config.context.app.executor

        # Create and send the signal
        signal = Signal(name=signal_name, workflow_id=workflow_id, payload=payload)

        try:
            await executor.signal_bus.signal(signal)
            return {
                "success": True,
                "message": f"Signal '{signal_name}' sent successfully",
            }
        except Exception as e:
            logger.error(f"Error sending signal {signal_name}: {e}")
            return {"success": False, "message": f"Error sending signal: {str(e)}"}

    @mcp.tool(name="workflows/wait_for_input")
    async def workflow_wait_for_input(
        ctx: MCPContext, workflow_id: str, description: str = "Provide input"
    ) -> Dict[str, Any]:
        """
        Get information about a workflow that is waiting for human input.

        This tool helps coordinate when a workflow is waiting for human input by
        providing clear instructions on how to provide that input.

        Args:
            workflow_id: The ID of the workflow.
            description: Description of what input is needed.

        Returns:
            Instructions on how to provide input to the waiting workflow.
        """
        server_config: ServerContext = ctx.request_context.lifespan_context

        if workflow_id not in server_config.active_workflows:
            raise ValueError(f"Workflow with ID '{workflow_id}' not found.")

        workflow = server_config.active_workflows[workflow_id]

        # Provide more helpful information about how to send the input
        return {
            "workflow_id": workflow_id,
            "workflow_name": workflow.name,
            "description": description,
            "status": "waiting_for_input",
            "instructions": "To provide input, use workflows/resume with the workflow_id and input_data parameters.",
            "example": {
                "tool": "workflows/resume",
                "args": {
                    "workflow_id": workflow_id,
                    "input_data": "Example input data",
                },
            },
            "tool_endpoints": [f"workflows/{workflow.name}/resume"],
        }

    # endregion

    return mcp


# region per-Agent Tools


def create_agent_tools(mcp: FastMCP, server_context: ServerContext):
    """
    Create agent-specific tools for existing agents.
    This is called at server start to register specific endpoints for each agent.
    """
    if not server_context:
        logger.warning("Server config not available for creating agent tools")
        return

    for _, agent in server_context.active_agents.items():
        create_agent_specific_tools(mcp, server_context, agent)


def create_agent_specific_tools(
    mcp: FastMCP, server_context: ServerContext, agent: Agent
):
    """Create specific tools for a given agent."""

    # Format instruction - handle callable instructions
    instruction = agent.instruction
    if callable(instruction):
        instruction = instruction({})

    server_registry = server_context.context.server_registry

    # Add generate* tools for this agent
    @mcp.tool(
        name=f"agents/{agent.name}/generate",
        description=f"""
    Run the '{agent.name}' agent using the given message.
    This is similar to generating an LLM completion.

    Agent Description: {instruction}
    Connected Servers: {_get_server_descriptions_as_string(server_registry, agent.server_names)}

    Args:
        message: The prompt to send to the agent.
        request_params: Optional parameters for the request, such as max_tokens and model/model preferences.

    Returns:
        The generated response from the agent.
    """,
    )
    async def generate(
        ctx: MCPContext,
        message: str | MCPMessageParam | List[MCPMessageParam],
        request_params: RequestParams | None = None,
    ) -> List[MCPMessageResult]:
        return await _agent_generate(ctx, agent.name, message, request_params)

    @mcp.tool(
        name=f"agents/{agent.name}/generate_str",
        description=f"""
    Run the '{agent.name}' agent using the given message and return the response as a string.
    Use agents/{agent.name}/generate for results in the original format, and
    use agents/{agent.name}/generate_structured for results conforming to a specific schema.

    Agent Description: {instruction}
    Connected Servers: {_get_server_descriptions_as_string(server_registry, agent.server_names)}

    Args:
        message: The prompt to send to the agent.
        request_params: Optional parameters for the request, such as max_tokens and model/model preferences.

    Returns:
        The generated response from the agent.
    """,
    )
    async def generate_str(
        ctx: MCPContext,
        message: str | MCPMessageParam | List[MCPMessageParam],
        request_params: RequestParams | None = None,
    ) -> str:
        return await _agent_generate_str(ctx, agent.name, message, request_params)

    # Add structured generation tool for this agent
    @mcp.tool(
        name=f"agents/{agent.name}/generate_structured",
        description=f"""
    Run the '{agent.name}' agent using the given message and return a response that matches the given schema.

    Use agents/{agent.name}/generate for results in the original format, and
    use agents/{agent.name}/generate_str for string result.

    Agent Description: {instruction}
    Connected Servers: {_get_server_descriptions_as_string(server_registry, agent.server_names)}

    Args:
        message: The prompt to send to the agent.
        response_schema: The JSON schema that defines the shape to generate the response in.
            This schema can be generated using type.schema_json() for a Pydantic model.
        request_params: Optional parameters for the request, such as max_tokens and model/model preferences.

    Returns:
        A dictionary representation of the structured response.

    Example:
        response_schema:
        {{
            "title": "UserProfile",
            "type": "object",
            "properties": {{
                "name": {{
                    "title": "Name",
                    "type": "string"
                }},
                "age": {{
                    "title": "Age",
                    "type": "integer",
                    "minimum": 0
                }},
                "email": {{
                    "title": "Email",
                    "type": "string",
                    "format": "email",
                    "pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
                }}
            }},
            "required": [
                "name",
                "age",
                "email"
            ]
        }}
    """,
    )
    async def generate_structured(
        ctx: MCPContext,
        message: str,
        response_schema: Dict[str, Any],
        request_params: RequestParams | None = None,
    ) -> Dict[str, Any]:
        return await _agent_generate_structured(
            ctx, agent.name, message, response_schema, request_params
        )


# endregion

# region per-Workflow Tools


def create_workflow_tools(mcp: FastMCP, server_config: ServerContext):
    """
    Create workflow-specific tools for registered workflows.
    This is called at server start to register specific endpoints for each workflow.
    """
    if not server_config:
        logger.warning("Server config not available for creating workflow tools")
        return

    for workflow_id, workflow_cls in server_config.context.app.workflows.items():
        create_workflow_specific_tools(mcp, workflow_id, workflow_cls)


def create_workflow_specific_tools(mcp: FastMCP, workflow_id: str, workflow_cls: Type):
    """Create specific tools for a given workflow."""

    # Get workflow documentation
    doc = workflow_cls.__doc__ or "No description available"
    doc = doc.strip()

    # Get workflow run method parameters using inspection
    parameters = {}
    if hasattr(workflow_cls, "run"):
        sig = inspect.signature(workflow_cls.run)
        for param_name, param in sig.parameters.items():
            if param_name != "self":
                param_info = {
                    "type": str(param.annotation)
                    .replace("<class '", "")
                    .replace("'>", ""),
                    "required": param.default == inspect.Parameter.empty,
                }
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                parameters[param_name] = param_info

    # Create a run tool for this workflow
    @mcp.tool(name=f"workflows/{workflow_id}/run")
    async def workflow_specific_run(
        ctx: MCPContext,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the workflow with the given arguments."""
        server_config: ServerContext = ctx.request_context.lifespan_context
        app = server_config.context.app

        if workflow_id not in app.workflows:
            raise ValueError(f"Workflow '{workflow_id}' not found.")

        # Create workflow instance
        workflow = workflow_cls(executor=app.executor, name=workflow_id)

        # Generate workflow instance ID
        instance_id = str(uuid.uuid4())

        # Store workflow instance
        server_config.active_workflows[instance_id] = workflow

        # Run workflow in separate task
        run_args = args or {}
        run_task = asyncio.create_task(workflow.run(**run_args))

        # Store task
        server_config.active_workflows[instance_id + "_task"] = run_task

        # Return information about the workflow
        return {
            "workflow_id": instance_id,
            "workflow_name": workflow_id,
            "status": "running",
            "args": args,
            "tool_endpoints": [
                f"workflows/{workflow_id}/get_status",
                f"workflows/{workflow_id}/pause",
                f"workflows/{workflow_id}/resume",
                f"workflows/{workflow_id}/cancel",
            ],
            "message": f"Workflow {workflow_id} started with ID {instance_id}. Use the returned workflow_id with other workflow tools.",
        }

    # Format parameter documentation
    param_docs = []
    for param_name, param_info in parameters.items():
        default_info = (
            f" (default: {param_info.get('default', 'required')})"
            if not param_info.get("required", True)
            else ""
        )
        param_docs.append(
            f"- {param_name}: {param_info.get('type', 'Any')}{default_info}"
        )

    param_doc_str = "\n".join(param_docs) if param_docs else "- No parameters required"

    # Update the docstring
    workflow_specific_run.__doc__ = f"""
    Run the {workflow_id} workflow.
    
    Description: {doc}
    
    Parameters:
    {param_doc_str}
    
    Args:
        args: Dictionary containing the parameters for the workflow.
        
    Returns:
        Information about the running workflow including its ID and metadata.
    """

    # Create a status tool for this workflow
    @mcp.tool(name=f"workflows/{workflow_id}/get_status")
    def workflow_specific_status(
        ctx: MCPContext, workflow_instance_id: str
    ) -> Dict[str, Any]:
        """Get the status of a running workflow instance."""
        server_config: ServerContext = ctx.request_context.lifespan_context

        if workflow_instance_id not in server_config.active_workflows:
            raise ValueError(f"Workflow instance '{workflow_instance_id}' not found.")

        workflow = server_config.active_workflows[workflow_instance_id]
        if workflow_id != workflow.name:
            raise ValueError(
                f"Workflow instance '{workflow_instance_id}' is not a {workflow_id} workflow."
            )

        task = server_config.active_workflows.get(workflow_instance_id + "_task")

        status = {
            "id": workflow_instance_id,
            "name": workflow.name,
            "running": task is not None and not task.done() if task else False,
            "state": workflow.state.model_dump() if hasattr(workflow, "state") else {},
            "available_actions": ["pause", "resume", "cancel"]
            if task and not task.done()
            else [],
            "tool_endpoints": [
                f"workflows/{workflow_id}/get_status",
            ],
        }

        # Add appropriate action endpoints based on status
        if task and not task.done():
            status["tool_endpoints"].extend(
                [
                    f"workflows/{workflow_id}/pause",
                    f"workflows/{workflow_id}/resume",
                    f"workflows/{workflow_id}/cancel",
                ]
            )

        if task and task.done():
            try:
                result = task.result()

                # Convert result to a useful format
                if hasattr(result, "model_dump"):
                    result_data = result.model_dump()
                elif hasattr(result, "__dict__"):
                    result_data = result.__dict__
                else:
                    result_data = str(result)

                status["result"] = result_data
                status["completed"] = True
                status["error"] = None
            except Exception as e:
                status["result"] = None
                status["completed"] = False
                status["error"] = str(e)
                status["exception_type"] = type(e).__name__

        return status

    # Update the docstring
    workflow_specific_status.__doc__ = f"""
    Get the status of a running {workflow_id} workflow instance.
    
    Description: {doc}
    
    Args:
        workflow_instance_id: The ID of the workflow instance to check.
        
    Returns:
        A dictionary with detailed information about the workflow status.
    """

    # Create a pause tool for this workflow
    @mcp.tool(name=f"workflows/{workflow_id}/pause")
    async def workflow_specific_pause(
        ctx: MCPContext, workflow_instance_id: str
    ) -> bool:
        """Pause a running workflow instance."""
        server_config: ServerContext = ctx.request_context.lifespan_context

        if workflow_instance_id not in server_config.active_workflows:
            raise ValueError(f"Workflow instance '{workflow_instance_id}' not found.")

        workflow = server_config.active_workflows[workflow_instance_id]
        if workflow_id != workflow.name:
            raise ValueError(
                f"Workflow instance '{workflow_instance_id}' is not a {workflow_id} workflow."
            )

        # Signal workflow to pause
        try:
            await server_config.context.app.executor.signal(
                "pause", workflow_id=workflow_instance_id
            )
            return True
        except Exception as e:
            logger.error(f"Error pausing workflow {workflow_instance_id}: {e}")
            return False

    # Update the docstring
    workflow_specific_pause.__doc__ = f"""
    Pause a running {workflow_id} workflow instance.
    
    Description: {doc}
    
    Args:
        workflow_instance_id: The ID of the workflow instance to pause.
        
    Returns:
        True if the workflow was paused, False otherwise.
    """

    # Create a resume tool for this workflow
    @mcp.tool(name=f"workflows/{workflow_id}/resume")
    async def workflow_specific_resume(
        ctx: MCPContext, workflow_instance_id: str, input_data: Optional[str] = None
    ) -> bool:
        """Resume a paused workflow instance."""
        server_config: ServerContext = ctx.request_context.lifespan_context

        if workflow_instance_id not in server_config.active_workflows:
            raise ValueError(f"Workflow instance '{workflow_instance_id}' not found.")

        workflow = server_config.active_workflows[workflow_instance_id]
        if workflow_id != workflow.name:
            raise ValueError(
                f"Workflow instance '{workflow_instance_id}' is not a {workflow_id} workflow."
            )

        # Signal workflow to resume
        try:
            signal = Signal(
                name="resume", workflow_id=workflow_instance_id, payload=input_data
            )
            await server_config.context.app.executor.signal_bus.signal(signal)
            return True
        except Exception as e:
            logger.error(f"Error resuming workflow {workflow_instance_id}: {e}")
            return False

    # Update the docstring
    workflow_specific_resume.__doc__ = f"""
    Resume a paused {workflow_id} workflow instance.
    
    Description: {doc}
    
    Args:
        workflow_instance_id: The ID of the workflow instance to resume.
        input_data: Optional input data to provide to the workflow.
        
    Returns:
        True if the workflow was resumed, False otherwise.
    """

    # Create a cancel tool for this workflow
    @mcp.tool(name=f"workflows/{workflow_id}/cancel")
    async def workflow_specific_cancel(
        ctx: MCPContext, workflow_instance_id: str
    ) -> bool:
        """Cancel a running workflow instance."""
        server_config: ServerContext = ctx.request_context.lifespan_context

        if workflow_instance_id not in server_config.active_workflows:
            raise ValueError(f"Workflow instance '{workflow_instance_id}' not found.")

        workflow = server_config.active_workflows[workflow_instance_id]
        if workflow_id != workflow.name:
            raise ValueError(
                f"Workflow instance '{workflow_instance_id}' is not a {workflow_id} workflow."
            )

        task = server_config.active_workflows.get(workflow_instance_id + "_task")

        if task and not task.done():
            # Cancel task
            task.cancel()

            # Signal workflow to cancel
            try:
                await server_config.context.app.executor.signal(
                    "cancel", workflow_id=workflow_instance_id
                )

                # Remove from active workflows
                server_config.active_workflows.pop(workflow_instance_id, None)
                server_config.active_workflows.pop(workflow_instance_id + "_task", None)

                return True
            except Exception as e:
                logger.error(f"Error cancelling workflow {workflow_instance_id}: {e}")
                return False

        return False

    # Update the docstring
    workflow_specific_cancel.__doc__ = f"""
    Cancel a running {workflow_id} workflow instance.
    
    Description: {doc}
    
    Args:
        workflow_instance_id: The ID of the workflow instance to cancel.
        
    Returns:
        True if the workflow was cancelled, False otherwise.
    """


# endregion


def _get_server_descriptions(
    server_registry: ServerRegistry | None, server_names: List[str]
) -> List:
    servers: List[dict[str, str]] = []
    if server_registry:
        for server_name in server_names:
            config = server_registry.get_server_config(server_name)
            if config:
                servers.append(
                    {
                        "name": config.name,
                        "description": config.description,
                    }
                )
            else:
                servers.append({"name": server_name})
    else:
        servers = [{"name": server_name} for server_name in server_names]

    return servers


def _get_server_descriptions_as_string(
    server_registry: ServerRegistry | None, server_names: List[str]
) -> str:
    servers = _get_server_descriptions(server_registry, server_names)

    # Format each server's information as a string
    server_strings = []
    for server in servers:
        if "description" in server:
            server_strings.append(f"{server['name']}: {server['description']}")
        else:
            server_strings.append(f"{server['name']}")

    # Join all server strings with a newline
    return "\n".join(server_strings)


async def _agent_generate(
    ctx: MCPContext,
    agent_name: str,
    message: str | MCPMessageParam | List[MCPMessageParam],
    request_params: RequestParams | None = None,
) -> List[MCPMessageResult]:
    """
    Run an agent using the given message.
    This is similar to generating an LLM completion.

    Args:
        agent_name: Name of the agent to use. This must be one of the names retrieved using agents/list tool endpoint.
        message: The prompt to send to the agent.
        request_params: Optional parameters for the request, such as max_tokens and model/model preferences.

    Returns:
        The generated response from the agent.
    """
    server_context: ServerContext = ctx.request_context.lifespan_context

    if agent_name not in server_context.active_agents:
        raise ToolError(f"Agent not found: {agent_name}. Make sure the agent ")

    agent = server_context.active_agents[agent_name]
    if not agent:
        raise ToolError(f"Agent not found: {agent_name}")
    elif not agent.llm:
        raise ToolError(
            f"Agent {agent_name} does not have an LLM attached. Make sure to call the attach_llm method where the agent is created."
        )

    # Convert the input message to the appropriate format
    input_message: str | MessageParamT | List[MessageParamT]
    if isinstance(message, str):
        input_message = message
    elif isinstance(message, list):
        input_message = [agent.llm.from_mcp_message_param(msg) for msg in message]
    else:
        input_message = agent.llm.from_mcp_message_param(message)

    # Check if the agent is already initialized
    async with agent:
        result = await agent.llm.generate(
            message=input_message, request_params=request_params
        )
        return result


async def _agent_generate_str(
    ctx: MCPContext,
    agent_name: str,
    message: str | MCPMessageParam | List[MCPMessageParam],
    request_params: RequestParams | None = None,
) -> str:
    """
    Run an agent using the given message and return the response as a string.
    Use agents/generate for results in the original format, and
    use agents/generate_structured for results conforming to a specific schema.

    Args:
        agent_name: Name of the agent to use. This must be one of the names retrieved using agents/list tool endpoint.
        message: The prompt to send to the agent.
        request_params: Optional parameters for the request, such as max_tokens and model/model preferences.

    Returns:
        The generated response from the agent.
    """
    server_context: ServerContext = ctx.request_context.lifespan_context

    if agent_name not in server_context.active_agents:
        raise ToolError(f"Agent not found: {agent_name}. Make sure the agent ")

    agent = server_context.active_agents[agent_name]
    if not agent:
        raise ToolError(f"Agent not found: {agent_name}")
    elif not agent.llm:
        raise ToolError(
            f"Agent {agent_name} does not have an LLM attached. Make sure to call the attach_llm method where the agent is created."
        )

    # Convert the input message to the appropriate format
    input_message: str | MessageParamT | List[MessageParamT]
    if isinstance(message, str):
        input_message = message
    elif isinstance(message, list):
        input_message = [agent.llm.from_mcp_message_param(msg) for msg in message]
    else:
        input_message = agent.llm.from_mcp_message_param(message)

    # Check if the agent is already initialized
    async with agent:
        result = await agent.llm.generate_str(
            message=input_message, request_params=request_params
        )
        return result


async def _agent_generate_structured(
    ctx: MCPContext,
    agent_name: str,
    message: str | MCPMessageParam | List[MCPMessageParam],
    response_schema: Dict[str, Any],
    request_params: RequestParams | None = None,
) -> Dict[str, Any]:
    """
    Generate a structured response from an agent that matches the given schema.

    Args:
        agent_name: Name of the agent to use. This must be one of the names retrieved using agents/list tool endpoint.
        message: The prompt to send to the agent.
        response_schema: The JSON schema that defines the shape to generate the response in.
            This schema can be generated using type.schema_json() for a Pydantic model.
        request_params: Optional parameters for the request, such as max_tokens and model/model preferences.

    Returns:
        A dictionary representation of the structured response.

    Example:
        response_schema:
        {
            "title": "UserProfile",
            "type": "object",
            "properties": {
                "name": {
                    "title": "Name",
                    "type": "string"
                },
                "age": {
                    "title": "Age",
                    "type": "integer",
                    "minimum": 0
                },
                "email": {
                    "title": "Email",
                    "type": "string",
                    "format": "email",
                    "pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
                }
            },
            "required": [
                "name",
                "age",
                "email"
            ]
        }
    """
    server_context: ServerContext = ctx.request_context.lifespan_context

    if agent_name not in server_context.active_agents:
        raise ToolError(f"Agent not found: {agent_name}. Make sure the agent ")

    agent = server_context.active_agents[agent_name]
    if not agent:
        raise ToolError(f"Agent not found: {agent_name}")
    elif not agent.llm:
        raise ToolError(
            f"Agent {agent_name} does not have an LLM attached. Make sure to call the attach_llm method where the agent is created."
        )

    # Convert the input message to the appropriate format
    input_message: str | MessageParamT | List[MessageParamT]
    if isinstance(message, str):
        input_message = message
    elif isinstance(message, list):
        input_message = [agent.llm.from_mcp_message_param(msg) for msg in message]
    else:
        input_message = agent.llm.from_mcp_message_param(message)

    response_model = create_model_from_schema(response_schema)

    # Check if the agent is already initialized
    async with agent:
        result = await agent.llm.generate_structured(
            message=input_message,
            response_model=response_model,
            request_params=request_params,
        )
        # Convert to dictionary for JSON serialization
        return result.model_dump(mode="json")
