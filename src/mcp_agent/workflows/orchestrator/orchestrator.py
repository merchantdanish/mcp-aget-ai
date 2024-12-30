import functools
from typing import Any, Callable, Coroutine, List, Literal, Type

from mcp_agent.agents.agent import Agent
from mcp_agent.context import get_current_context
from mcp_agent.executor.executor import Executor
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
)
from mcp_agent.mcp_server_registry import ServerRegistry
from mcp_agent.workflows.orchestrator.orchestrator_models import (
    format_plan_result,
    format_step_result,
    AgentTask,
    NextStep,
    Plan,
    PlanResult,
    ServerTask,
    Step,
    StepResult,
    TaskWithResult,
)
from mcp_agent.workflows.orchestrator.orchestrator_prompts import (
    FULL_PLAN_PROMPT_TEMPLATE,
    ITERATIVE_PLAN_PROMPT_TEMPLATE,
    SYNTHESIZE_PLAN_PROMPT_TEMPLATE,
    SYNTHESIZE_STEP_PROMPT_TEMPLATE,
    TASK_PROMPT_TEMPLATE,
)


class Orchestrator(AugmentedLLM[MessageParamT, MessageT]):
    """
    In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks,
    delegates them to worker LLMs, and synthesizes their results. It does this
    in a loop until the task is complete.

    When to use this workflow:
        - This workflow is well-suited for complex tasks where you can’t predict the
        subtasks needed (in coding, for example, the number of files that need to be
        changed and the nature of the change in each file likely depend on the task).

    Example where orchestrator-workers is useful:
        - Coding products that make complex changes to multiple files each time.
        - Search tasks that involve gathering and analyzing information from multiple sources
        for possible relevant information.
    """

    def __init__(
        self,
        llm_factory: Callable[[Agent], AugmentedLLM[MessageParamT, MessageT]],
        planner: AugmentedLLM | None = None,
        available_servers: List[str] | None = None,
        available_agents: List[Agent | AugmentedLLM] | None = None,
        executor: Executor | None = None,
        plan_type: Literal["full", "iterative"] = "full",
        server_registry: ServerRegistry | None = None,
    ):
        """
        Args:
            llm_factory: Factory function to create an LLM for a given agent
            planner: LLM to use for planning steps (if not provided, a default planner will be used)
            plan_type: "full" planning generates the full plan first, then executes. "iterative" plans the next step, and loops until success.
            available_servers: List of server names available to tasks executed by this orchestrator
            available_agents: List of agents available to tasks executed by this orchestrator
            executor: Executor to use for parallel task execution (defaults to asyncio)
        """
        super().__init__(executor=executor)

        self.llm_factory = llm_factory

        self.planner = planner or llm_factory(
            Agent(
                name="LLM Orchestration Planner",
                instruction="""
                You are an expert planner. Given an objective task and a list of MCP servers (which are collections of tools)
                or Agents (which are collections of servers), your job is to break down the objective into a series of steps,
                which can be performed by LLMs with access to the servers or agents.
                """,
            )
        )

        self.plan_type: Literal["full", "iterative"] = plan_type
        self.server_names = available_servers or []
        self.server_registry = server_registry or get_current_context().server_registry
        self.server_metadata = {
            server_name: self.server_registry.get_server_config(server_name)
            for server_name in self.server_names
        }
        self.agents = {agent.name: agent for agent in available_agents or []}

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = False,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> List[MessageT]:
        """Request an LLM generation, which may run multiple iterations, and return the result"""

        # TODO: saqadri - history tracking is complicated in this multi-step workflow, so we will ignore it for now
        if use_history:
            raise NotImplementedError(
                "History tracking is not yet supported for orchestrator workflows"
            )

        objective = str(message)
        plan_result = await self.execute(
            objective=objective,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
        )

        return [plan_result.result]

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> str:
        """Request an LLM generation and return the string representation of the result"""
        result = await self.generate(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        return str(result[0])

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> ModelT:
        """Request a structured LLM generation and return the result as a Pydantic model."""
        result_str = await self.generate(
            message=message,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        llm = self.llm_factory(
            Agent(
                name="Structured Output",
                instruction="Produce a structured output given a message",
            )
        )

        structured_result = await llm.generate_structured(
            message=result_str,
            response_model=response_model,
            use_history=use_history,
            max_iterations=max_iterations,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
        )

        return structured_result

    async def execute(
        self,
        objective: str,
        max_iterations: int = 30,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
    ) -> PlanResult:
        """Execute task with result chaining between steps"""
        iterations = 0

        plan_result = PlanResult(objective=objective, steps=[])

        while iterations < max_iterations:
            if self.plan_type == "iterative":
                # Get next plan/step
                next_step = await self._get_next_step(
                    objective=objective, plan_result=plan_result
                )
                plan = Plan(steps=[next_step], is_complete=next_step.is_complete)
            elif self.plan_type == "full":
                plan = await self._get_full_plan(
                    objective=objective, plan_result=plan_result
                )
            else:
                raise ValueError(f"Invalid plan type {self.plan_type}")

            plan_result.plan = plan

            if plan.is_complete:
                plan_result.is_complete = True

                # Synthesize final result into a single message
                synthesis_prompt = SYNTHESIZE_PLAN_PROMPT_TEMPLATE.format(
                    plan_result=format_plan_result(plan_result)
                )

                plan_result.result = await self.planner.generate_str(
                    message=synthesis_prompt,
                    max_iterations=1,
                    model=model,
                    stop_sequences=stop_sequences,
                    max_tokens=max_tokens,
                )

                return plan_result

            # Execute each step, collecting results
            # Note that in iterative mode this will only be a single step
            for step in plan.steps:
                step_result = await self._execute_step(
                    step=step,
                    previous_result=plan_result,
                    model=model,
                    max_iterations=max_iterations,
                    stop_sequences=stop_sequences,
                    max_tokens=max_tokens,
                )

                plan_result.add_step_result(step_result)

            iterations += 1

        raise RuntimeError(f"Task failed to complete in {max_iterations} iterations")

    async def _execute_step(
        self,
        step: Step,
        previous_result: PlanResult,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
    ) -> StepResult:
        """Execute a step's subtasks in parallel and synthesize results"""
        step_result = StepResult(step=step, task_results=[])

        # Format previous results
        context = (
            format_plan_result(previous_result)
            if previous_result
            else "No results so far"
        )

        # Execute subtasks in parallel
        futures: List[Coroutine[Any, Any, str]] = []
        for task in step.tasks:
            llm = self._get_llm_for_subtask(task)

            task_description = TASK_PROMPT_TEMPLATE.format(
                objective=previous_result.objective,
                task=task.description,
                context=context,
            )

            generate_str_bound = functools.partial(
                llm.generate_str,
                message=task_description,
                max_iterations=max_iterations,
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
            )

            futures.append(generate_str_bound)

        # Wait for all tasks to complete
        results = await self.executor.execute(*futures)

        # Store task results
        for task, result in zip(step.tasks, results):
            step_result.add_task_result(
                TaskWithResult(**task.model_dump(), result=str(result))
            )

        # Synthesize overall step result
        synthesis_prompt = SYNTHESIZE_STEP_PROMPT_TEMPLATE.format(
            step_result=format_step_result(step_result)
        )
        step_result.result = await self.planner.generate_str(
            message=synthesis_prompt,
            max_iterations=1,
            model=model,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
        )

        return step_result

    def _get_llm_for_subtask(self, task: ServerTask | AgentTask) -> AugmentedLLM:
        """Create appropriately configured LLM for a task"""
        if isinstance(task, ServerTask):
            # For server tasks, create LLM with access to specified servers
            return self.llm_factory(
                Agent(
                    name=f"Task: {task.description}",
                    server_names=task.servers,
                    instruction=task.description,
                )
            )
        elif isinstance(task, AgentTask):
            # For agent tasks, find the agent and create LLM
            agent = self.agents.get(task.agent)
            if not agent:
                # TODO: saqadri - this should be handled so we don't crash the orchestrator
                raise ValueError(f"No agent found matching {task.agent}")
            elif isinstance(agent, AugmentedLLM):
                return agent
            else:
                return self.llm_factory(agent)
        else:
            # TODO: saqadri - this should be handled so we don't crash the orchestrator
            raise ValueError(f"Unknown task type: {type(task)}")

    async def _get_full_plan(
        self,
        objective: str,
        plan_result: PlanResult,
        max_iterations: int = 30,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
    ) -> Plan:
        """Generate full plan considering previous results"""

        servers = "\n".join(
            [
                f"{idx}. {self._format_server_info(server)}"
                for idx, server in enumerate(self.server_names, 1)
            ]
        )

        agents = "\n".join(
            [
                f"{idx}. {self._format_server_info(server)}"
                for idx, server in enumerate(self.server_names, 1)
            ]
        )

        prompt = FULL_PLAN_PROMPT_TEMPLATE.format(
            objective=objective,
            plan_result=format_plan_result(plan_result),
            servers=servers,
            agents=agents,
        )

        return await self.planner.generate_structured(
            message=prompt,
            response_model=Plan,
            max_iterations=max_iterations,
            model=model,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
        )

    async def _get_next_step(self, objective: str, plan_result: PlanResult) -> NextStep:
        """Generate just the next needed step"""

        servers = "\n".join(
            [
                f"{idx}. {self._format_server_info(server)}"
                for idx, server in enumerate(self.server_names, 1)
            ]
        )

        agents = "\n".join(
            [
                f"{idx}. {self._format_server_info(server)}"
                for idx, server in enumerate(self.server_names, 1)
            ]
        )

        prompt = ITERATIVE_PLAN_PROMPT_TEMPLATE.format(
            objective=objective,
            plan_result=format_plan_result(plan_result),
            servers=servers,
            agents=agents,
        )

        next_step = await self.planner.generate_structured(prompt, NextStep)
        return next_step

    def _format_server_info(self, server_name: str) -> str:
        """Format server information for display to planners"""
        server_config = self.server_metadata.get(server_name)
        server_str = f"Server Name: {server_name}"
        if not server_config:
            return server_str

        description = server_config.description
        if description:
            server_str = f"{server_str}\nDescription: {description}"

        return server_str

    def _format_agent_info(self, agent_name: str) -> str:
        """Format Agent information for display to planners"""
        agent = self.agents.get(agent_name)
        if not agent:
            return ""

        servers = "\n".join(
            [
                f"- {self._format_server_info(server_name)}"
                for server_name in agent.server_names
            ]
        )

        return f"Agent Name: {agent.name}\nDescription: {agent.instruction}\nServers in Agent: {servers}"
