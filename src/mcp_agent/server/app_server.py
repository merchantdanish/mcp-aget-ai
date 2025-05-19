"""
MCPAgentServer - Exposes MCPApp as MCP server, and
mcp-agent workflows and agents as MCP tools.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Type, TYPE_CHECKING

from mcp.server.fastmcp import Context as MCPContext, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.tools import Tool as FastTool

from mcp_agent.app import MCPApp
from mcp_agent.server.app_server_types import (
    MCPMessageParam,
    MCPMessageResult,
    create_model_from_schema,
)
from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_config import AgentConfig, create_agent
from mcp_agent.config import MCPServerSettings
from mcp_agent.core.context_dependent import ContextDependent
from mcp_agent.executor.workflow import Workflow
from mcp_agent.executor.workflow_registry import (
    WorkflowRegistry,
    InMemoryWorkflowRegistry,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.mcp_server_registry import ServerRegistry
from mcp_agent.workflows.llm.augmented_llm import MessageParamT, RequestParams

if TYPE_CHECKING:
    from mcp_agent.core.context import Context

logger = get_logger(__name__)


class ServerContext(ContextDependent):
    """Context object for the MCP App server."""

    def __init__(self, mcp: FastMCP, context: "Context", **kwargs):
        super().__init__(context=context, **kwargs)
        self.mcp = mcp
        self.active_agents: Dict[str, Agent] = {}

        # Initialize workflow registry if not already present
        if not self.context.workflow_registry:
            if self.context.config.execution_engine == "asyncio":
                self.context.workflow_registry = InMemoryWorkflowRegistry()
            elif self.context.config.execution_engine == "temporal":
                from mcp_agent.executor.temporal.workflow_registry import (
                    TemporalWorkflowRegistry,
                )

                self.context.workflow_registry = TemporalWorkflowRegistry(
                    executor=self.context.executor
                )
            else:
                raise ValueError(
                    f"Unsupported execution engine: {self.context.config.execution_engine}"
                )

        # TODO: saqadri (MAC) - This shouldn't be needed here because
        # in app_specific_lifespan we'll call create_agent_specific_tools
        # and create_workflow_specific_tools respectively.
        # # Register existing workflows from the app
        # # Use the MCPApp's workflow registry as the source of truth for available workflows
        # logger.info(f"Registering {len(self.context.app.workflows)} workflows")
        # for workflow_name, workflow_cls in self.context.app.workflows.items():
        #     logger.info(f"Registering workflow: {workflow_name}")
        #     self.register_workflow(workflow_name, workflow_cls)

        # # Register existing agent configurations from the app
        # for name, config in self.context.app._agent_configs.items():
        #     logger.info(f"Registered agent config: {name}")
        #     # Use the same tools for agent configs as for agent instances
        #     # When the tools are called, we'll create the agent as needed
        #     create_agent_specific_tools(self.mcp, self, config)

        # TODO: saqadri (MAC) - Do we need to notify the client that tools list changed?
        # Since this is at initialization time, we may not need to
        # (depends on when the server reports that it's intialized/ready)

    def register_workflow(self, workflow_name: str, workflow_cls: Type[Workflow]):
        """Register a workflow class."""
        if workflow_name not in self.context.workflows:
            self.workflows[workflow_name] = workflow_cls
            # Create tools for this workflow
            create_workflow_specific_tools(self.mcp, workflow_name, workflow_cls)

    def register_agent(self, agent: Agent):
        """
        Register an agent instance and create tools for it.
        This is used for runtime agent instances.

        Args:
            agent: The agent instance to register

        Returns:
            The registered agent (may be an existing instance if already registered)
        """
        if agent.name not in self.active_agents:
            self.active_agents[agent.name] = agent
            # Create tools for this agent
            create_agent_specific_tools(self.mcp, self, agent)
            return agent
        return self.active_agents[
            agent.name
        ]  # Return existing agent if already registered

    async def get_or_create_agent(self, name: str) -> Agent:
        """
        Get an existing Agent or create it from a registered AgentConfig.

        Args:
            name: The name of the Agent/AgentConfig.

        Returns:
            The Agent instance (existing or newly created)

        Raises:
            ToolError: If no agent or configuration with that name exists
        """
        # Check if the agent is already active
        if name in self.active_agents:
            return self.active_agents[name]

        # Check if there's a configuration for this agent
        agent_config = self.agent_configs.get(name)
        if agent_config:
            try:
                agent = await create_agent(name=agent_config.name, context=self.context)
                self.register_agent(agent)
                return agent
            except Exception as e:
                logger.error(f"Error creating agent {name}: {str(e)}")
                raise ToolError(f"Failed to create agent {name}: {str(e)}") from e

        # Neither active nor configured
        raise ToolError(
            f"Agent not found: {name}. No active agent or configuration with this name exists."
        )

    @property
    def app(self) -> MCPApp:
        """Get the MCPApp instance associated with this server context."""
        return self.context.app

    @property
    def workflows(self) -> Dict[str, Type[Workflow]]:
        """Get the workflows registered in this server context."""
        return self.app.workflows

    @property
    def workflow_registry(self) -> WorkflowRegistry:
        """Get the workflow registry for this server context."""
        return self.context.workflow_registry

    @property
    def agent_configs(self) -> Dict[str, AgentConfig]:
        """Get the agent configurations for this server context."""
        return self.app._agent_configs


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
        # TODO: saqadri (MAC) - create a much more detailed description
        # based on all the available agents and workflows,
        # or use the MCPApp's description if available.
        instructions=f"MCP server exposing {app.name} workflows and agents. Description: {app.description}",
        lifespan=app_specific_lifespan,
    )

    @mcp.tool(name="servers-list")
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

    @mcp.tool(name="agents-list")
    def list_agents(ctx: MCPContext) -> Dict[str, Dict[str, Any]]:
        """
        List all available agents with their detailed information.

        Returns information about each agent including their name, instruction,
        the MCP servers they have access to.
        This helps with understanding what each agent is designed to do before calling it.
        """
        server_context: ServerContext = ctx.request_context.lifespan_context
        server_registry = server_context.context.server_registry
        result = {}

        # Add active agents
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
                    f"agents-{name}-generate",
                    f"agents-{name}-generate_str",
                    f"agents-{name}-generate_structured",
                ],
            }

        return result

    @mcp.tool(name="agents-generate")
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
            agent_name: Name of the agent to use.
                This must be one of the names retrieved using agents/list tool endpoint.
            message: The prompt to send to the agent.
            request_params: Optional parameters to configure the LLM generation.

        Returns:
            The generated response from the agent.
        """
        return await _agent_generate(ctx, agent_name, message, request_params)

    @mcp.tool(name="agents-generate_str")
    async def agent_generate_str(
        ctx: MCPContext,
        agent_name: str,
        message: str | MCPMessageParam | List[MCPMessageParam],
        request_params: RequestParams | None = None,
    ) -> str:
        """
        Run an agent using the given message and return the response as a string.
        Use agents-generate for results in the original format, and
        use agents-generate_structured for results conforming to a specific schema.

        Args:
            agent_name: Name of the agent to use.
                This must be one of the names retrieved using agents/list tool endpoint.
            message: The prompt to send to the agent.
            request_params: Optional parameters to configure the LLM generation.

        Returns:
            The generated response from the agent.
        """
        return await _agent_generate_str(ctx, agent_name, message, request_params)

    @mcp.tool(name="agents-generate_structured")
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
            agent_name: Name of the agent to use.
                This must be one of the names retrieved using agents/list tool endpoint.
            message: The prompt to send to the agent.
            response_schema: The JSON schema that defines the shape to generate the response in.
                This schema can be generated using type.schema_json() for a Pydantic model.
            request_params: Optional parameters to configure the LLM generation.

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

    @mcp.tool(name="workflows-list")
    def list_workflows(ctx: MCPContext) -> Dict[str, Dict[str, Any]]:
        """
        List all available workflow types with their detailed information.
        Returns information about each workflow type including name, description, and parameters.
        This helps in making an informed decision about which workflow to run.
        """
        server_context: ServerContext = ctx.request_context.lifespan_context

        result = {}
        for workflow_name, workflow_cls in server_context.workflows.items():
            # Get workflow documentation
            run_fn_tool = FastTool.from_function(workflow_cls.run)

            # Define common endpoints for all workflows
            endpoints = [
                f"workflows-{workflow_name}-run",
                f"workflows-{workflow_name}-get_status",
            ]

            result[workflow_name] = {
                "name": workflow_name,
                "description": workflow_cls.__doc__ or run_fn_tool.description,
                "capabilities": ["run", "resume", "cancel", "get_status"],
                "tool_endpoints": endpoints,
                "run_parameters": run_fn_tool.parameters,
            }

        return result

    @mcp.tool(name="workflows-runs-list")
    async def list_workflow_runs(ctx: MCPContext) -> List[Dict[str, Any]]:
        """
        List all workflow instances (runs) with their detailed status information.

        This returns information about actual workflow instances (runs), not workflow types.
        For each running workflow, returns its ID, name, current state, and available operations.
        This helps in identifying and managing active workflow instances.

        Returns:
            A dictionary mapping workflow instance IDs to their detailed status information.
        """
        server_context: ServerContext = ctx.request_context.lifespan_context

        # Get all workflow statuses from the registry
        workflow_statuses = (
            await server_context.workflow_registry.list_workflow_statuses()
        )
        return workflow_statuses

    @mcp.tool(name="workflows-run")
    async def run_workflow(
        ctx: MCPContext,
        workflow_name: str,
        run_parameters: Dict[str, Any] | None = None,
    ) -> str:
        """
        Run a workflow with the given name.

        Args:
            workflow_name: The name of the workflow to run.
            run_parameters: Arguments to pass to the workflow run.
                workflows/list method will return the run_parameters schema for each workflow.

        Returns:
            The run ID of the started workflow run, which can be passed to
            workflows/get_status, workflows/resume, and workflows/cancel.
        """
        return await _workflow_run(ctx, workflow_name, run_parameters)

    @mcp.tool(name="workflows-get_status")
    async def get_workflow_status(
        ctx: MCPContext, workflow_name: str, run_id: str
    ) -> Dict[str, Any]:
        """
        Get the status of a running workflow.

        Provides detailed information about a workflow instance including its current state,
        whether it's running or completed, and any results or errors encountered.

        Args:
            workflow_name: The name of the workflow to check.
            run_id: The ID of the workflow instance to check,
                received from workflows/run or workflows/runs/list.

        Returns:
            A dictionary with comprehensive information about the workflow status.
        """
        return await _workflow_status(ctx, run_id, workflow_name)

    @mcp.tool(name="workflows-resume")
    async def resume_workflow(
        ctx: MCPContext,
        run_id: str,
        workflow_name: str | None = None,
        signal_name: str | None = "resume",
        payload: str | None = None,
    ) -> bool:
        """
        Resume a paused workflow.

        Args:
            run_id: The ID of the workflow to resume,
                received from workflows/run or workflows/runs/list.
            workflow_name: The name of the workflow to resume.
            signal_name: Optional name of the signal to send to resume the workflow.
                This will default to "resume", but can be a custom signal name
                if the workflow was paused on a specific signal.
            payload: Optional payload to provide the workflow upon resumption.
                For example, if a workflow is waiting for human input,
                this can be the human input.

        Returns:
            True if the workflow was resumed, False otherwise.
        """
        server_context: ServerContext = ctx.request_context.lifespan_context
        workflow_registry = server_context.workflow_registry

        if not workflow_registry:
            raise ToolError("Workflow registry not found for MCPApp Server.")

        # Get the workflow instance from the registry
        workflow = await workflow_registry.get_workflow(
            run_id=run_id, workflow_id=workflow_name
        )
        if not workflow:
            raise ToolError(
                f"Workflow '{workflow_name}' with run ID '{run_id}' not found."
            )

        logger.info(
            f"Resuming workflow {workflow_name} with ID {run_id} with signal '{signal_name}' and payload '{payload}'"
        )

        # Resume the workflow directly
        return await workflow.resume(signal_name, payload)

    @mcp.tool(name="workflows-cancel")
    async def cancel_workflow(
        ctx: MCPContext, run_id: str, workflow_name: str | None = None
    ) -> bool:
        """
        Cancel a running workflow.

        Args:
            run_id: The ID of the workflow instance to cancel,
                received from workflows/run or workflows/runs/list.
            workflow_name: The name of the workflow to cancel.

        Returns:
            True if the workflow was cancelled, False otherwise.
        """
        server_context: ServerContext = ctx.request_context.lifespan_context
        workflow_registry = server_context.workflow_registry

        # Get the workflow instance from the registry
        workflow = await workflow_registry.get_workflow(
            run_id=run_id, workflow_id=workflow_name
        )
        if not workflow:
            raise ValueError(
                f"Workflow '{workflow_name}' with ID '{run_id}' not found."
            )

        # Cancel the workflow directly
        return await workflow.cancel()

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

    for _, agent_config in server_context.agent_configs.items():
        agent = server_context.get_or_create_agent(agent_config.name)


def create_agent_specific_tools(
    mcp: FastMCP, server_context: ServerContext, agent: Agent | AgentConfig
):
    """
    Create specific tools for a given agent instance or configuration.

    Args:
        mcp: The FastMCP server
        server_context: The server context
        agent_or_config: Either an Agent instance or an AgentConfig
    """
    # Extract common properties based on whether we have an Agent or AgentConfig
    if isinstance(agent, Agent):
        name = agent.name
        instruction = agent.instruction
        server_names = agent.server_names
    else:  # AgentConfig
        name = agent.name
        instruction = agent.instruction
        server_names = agent.server_names

    # Format instruction - handle callable instructions
    if callable(instruction):
        instruction = instruction({})

    server_registry = server_context.context.server_registry

    # Add generate* tools for this agent
    @mcp.tool(
        name=f"agents-{name}-generate",
        description=f"""
    Run the '{name}' agent using the given message.
    This is similar to generating an LLM completion.

    Agent Description: {instruction}
    Connected Servers: {_get_server_descriptions_as_string(server_registry, server_names)}

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
        return await _agent_generate(ctx, name, message, request_params)

    @mcp.tool(
        name=f"agents-{name}-generate_str",
        description=f"""
    Run the '{name}' agent using the given message and return the response as a string.
    Use agents/{name}/generate for results in the original format, and
    use agents/{name}/generate_structured for results conforming to a specific schema.

    Agent Description: {instruction}
    Connected Servers: {_get_server_descriptions_as_string(server_registry, server_names)}

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
        return await _agent_generate_str(ctx, name, message, request_params)

    # Add structured generation tool for this agent
    @mcp.tool(
        name=f"agents-{name}-generate_structured",
        description=f"""
    Run the '{name}' agent using the given message and return a response that matches the given schema.

    Use agents/{name}/generate for results in the original format, and
    use agents/{name}/generate_str for string result.

    Agent Description: {instruction}
    Connected Servers: {_get_server_descriptions_as_string(server_registry, server_names)}

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
            ctx, name, message, response_schema, request_params
        )


# endregion

# region per-Workflow Tools


def create_workflow_tools(mcp: FastMCP, server_context: ServerContext):
    """
    Create workflow-specific tools for registered workflows.
    This is called at server start to register specific endpoints for each workflow.
    """
    if not server_context:
        logger.warning("Server config not available for creating workflow tools")
        return

    for workflow_name, workflow_cls in server_context.workflows.items():
        create_workflow_specific_tools(mcp, workflow_name, workflow_cls)


def create_workflow_specific_tools(
    mcp: FastMCP, workflow_name: str, workflow_cls: Type["Workflow"]
):
    """Create specific tools for a given workflow."""

    run_fn_tool = FastTool.from_function(workflow_cls.run)
    run_fn_tool_params = json.dumps(run_fn_tool.parameters, indent=2)

    @mcp.tool(
        name=f"workflows-{workflow_name}-run",
        description=f"""
        Run the '{workflow_name}' workflow and get a run ID back.
        Workflow Description: {workflow_cls.__doc__}

        {run_fn_tool.description}

        Args:
            run_parameters: Dictionary of parameters for the workflow run.
            The schema for these parameters is as follows:
            {run_fn_tool_params}
        """,
    )
    async def run(
        ctx: MCPContext,
        run_parameters: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return await _workflow_run(ctx, workflow_name, run_parameters)

    @mcp.tool(
        name=f"workflows-{workflow_name}-get_status",
        description=f"""
        Get the status of a running {workflow_name} workflow.
        
        Args:
            run_id: The run ID of the running workflow, received from workflows/{workflow_name}/run.
        """,
    )
    def get_status(ctx: MCPContext, workflow_id: str) -> Dict[str, Any]:
        return _workflow_status(ctx, workflow_id, workflow_name)


# endregion


def _get_server_descriptions(
    server_registry: ServerRegistry | None, server_names: List[str]
) -> List:
    servers: List[dict[str, str]] = []
    if server_registry:
        for server_name in server_names:
            config = server_registry.get_server_context(server_name)
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


# region Agent Utils


async def _agent_generate(
    ctx: MCPContext,
    agent_name: str,
    message: str | MCPMessageParam | List[MCPMessageParam],
    request_params: RequestParams | None = None,
) -> List[MCPMessageResult]:
    server_context: ServerContext = ctx.request_context.lifespan_context

    # Get or create the agent - this will automatically create agent from config if needed
    try:
        agent = await server_context.get_or_create_agent(agent_name)
    except ToolError as e:
        raise e

    # Check if the agent has an LLM attached
    if not agent.llm:
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

    # Use the agent as a context manager to ensure proper initialization/cleanup
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
    server_context: ServerContext = ctx.request_context.lifespan_context

    # Get or create the agent - this will automatically create agent from config if needed
    try:
        agent = await server_context.get_or_create_agent(agent_name)
    except ToolError as e:
        raise e

    # Check if the agent has an LLM attached
    if not agent.llm:
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

    # Use the agent as a context manager to ensure proper initialization/cleanup
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
    server_context: ServerContext = ctx.request_context.lifespan_context

    # Get or create the agent - this will automatically create agent from config if needed
    try:
        agent = await server_context.get_or_create_agent(agent_name)
    except ToolError as e:
        raise e

    # Check if the agent has an LLM attached
    if not agent.llm:
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

    # Create a Pydantic model from the schema
    response_model = create_model_from_schema(response_schema)

    # Use the agent as a context manager to ensure proper initialization/cleanup
    async with agent:
        result = await agent.llm.generate_structured(
            message=input_message,
            response_model=response_model,
            request_params=request_params,
        )
        # Convert to dictionary for JSON serialization
        return result.model_dump(mode="json")


# endregion

# region Workflow Utils


async def _workflow_run(
    ctx: MCPContext,
    workflow_id: str,
    run_parameters: Dict[str, Any] | None = None,
) -> str:
    server_context: ServerContext = ctx.request_context.lifespan_context

    if workflow_id not in server_context.workflows:
        raise ToolError(f"Workflow '{workflow_id}' not found.")

    # Get the workflow class
    workflow_cls = server_context.workflows[workflow_id]

    # Create and initialize the workflow instance using the factory method
    try:
        # Create workflow instance
        workflow = await workflow_cls.create(
            name=workflow_id, context=server_context.context
        )

        run_parameters = run_parameters or {}

        # Run the workflow asynchronously and get its ID
        run_id = await workflow.run_async(**run_parameters)
        return run_id

    except Exception as e:
        logger.error(f"Error creating workflow {workflow_id}: {str(e)}")
        raise ToolError(f"Error creating workflow {workflow_id}: {str(e)}") from e


async def _workflow_status(
    ctx: MCPContext, run_id: str, workflow_id: str | None = None
) -> Dict[str, Any]:
    server_context: ServerContext = ctx.request_context.lifespan_context
    workflow_registry: WorkflowRegistry = server_context.workflow_registry

    if not workflow_registry:
        raise ToolError("Workflow registry not found for MCPApp Server.")

    # Get the workflow instance from the registry
    workflow = await workflow_registry.get_workflow(
        run_id=run_id, workflow_id=workflow_id
    )
    if not workflow:
        raise ToolError(f"Workflow with ID '{workflow_id}' not found.")

    # Get the status directly from the workflow instance
    status = await workflow.get_status()

    return status


# endregion
