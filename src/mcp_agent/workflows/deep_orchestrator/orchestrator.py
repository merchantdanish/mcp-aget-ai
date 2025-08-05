"""
Deep Orchestrator - Production-ready adaptive workflow orchestration.

This module implements the main AdaptiveOrchestrator class with comprehensive
planning, execution, knowledge management, and synthesis capabilities.
"""

import asyncio
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING

from mcp_agent.agents.agent import Agent
from mcp_agent.logging.logger import get_logger
from mcp_agent.tracing.telemetry import get_tracer
from mcp_agent.tracing.token_tracking_decorator import track_tokens
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
    RequestParams,
)

from mcp_agent.workflows.deep_orchestrator.budget import SimpleBudget
from mcp_agent.workflows.deep_orchestrator.cache import AgentCache
from mcp_agent.workflows.deep_orchestrator.knowledge import (
    KnowledgeExtractor,
    KnowledgeItem,
)
from mcp_agent.workflows.deep_orchestrator.memory import WorkspaceMemory
from mcp_agent.workflows.deep_orchestrator.models import (
    AgentDesign,
    Plan,
    PlanVerificationResult,
    PolicyAction,
    Step,
    Task,
    TaskResult,
    TaskStatus,
    VerificationResult,
)
from mcp_agent.workflows.deep_orchestrator.policy import PolicyEngine
from mcp_agent.workflows.deep_orchestrator.prompts import (
    AGENT_DESIGNER_INSTRUCTION,
    EMERGENCY_RESPONDER_INSTRUCTION,
    ORCHESTRATOR_SYSTEM_INSTRUCTION,
    PLANNER_INSTRUCTION,
    SYNTHESIZER_INSTRUCTION,
    VERIFIER_INSTRUCTION,
    build_agent_instruction,
    get_agent_design_prompt,
    get_emergency_context,
    get_emergency_prompt,
    get_full_plan_prompt,
    get_planning_context,
    get_synthesis_context,
    get_synthesis_prompt,
    get_task_context,
    get_verification_context,
    get_verification_prompt,
)
from mcp_agent.workflows.deep_orchestrator.queue import TodoQueue
from mcp_agent.workflows.deep_orchestrator.utils import retry_with_backoff

if TYPE_CHECKING:
    from opentelemetry.trace.span import Span
    from mcp_agent.core.context import Context

logger = get_logger(__name__)


class AdaptiveOrchestrator(AugmentedLLM[MessageParamT, MessageT]):
    """
    Production-ready adaptive orchestrator with all recommended enhancements.

    Features:
    - Comprehensive upfront planning with dependency management
    - Dynamic agent creation optimized for each task
    - Knowledge extraction and accumulation
    - Smart replanning based on results and verification
    - Budget management (tokens, cost, time)
    - Policy-driven execution control
    - Parallel task execution with dependency resolution
    - Agent caching to reduce costs
    - Context management to prevent overflow
    """

    def __init__(
        self,
        llm_factory: Callable[[Agent], AugmentedLLM[MessageParamT, MessageT]],
        name: str = "AdaptiveOrchestrator",
        available_agents: List[Agent | AugmentedLLM] | None = None,
        available_servers: Optional[List[str]] = None,
        max_iterations: int = 20,
        max_replans: int = 3,
        enable_filesystem: bool = True,
        enable_parallel: bool = True,
        max_task_retries: int = 3,
        context: Optional["Context"] = None,
        task_context_budget: int = 50000,
        context_relevance_threshold: float = 0.7,
        context_compression_ratio: float = 0.8,
        enable_full_context_propagation: bool = True,
        context_window_limit: int = 100000,
        **kwargs,
    ):
        """
        Initialize the adaptive orchestrator with production features.

        Args:
            llm_factory: Factory function to create LLMs
            name: Name of the orchestrator
            available_agents: List of pre-defined Agent or AugmentedLLM instances
            available_servers: List of available MCP servers
            max_iterations: Maximum workflow iterations
            max_replans: Maximum number of replanning attempts
            enable_filesystem: Enable filesystem workspace
            enable_parallel: Enable parallel task execution
            max_task_retries: Maximum retries per task
            context: Application context
            task_context_budget: Maximum tokens for task context (default: 50000)
            context_relevance_threshold: Minimum relevance score to include context (default: 0.7)
            context_compression_ratio: When to start compressing context (default: 0.8)
            enable_full_context_propagation: Whether to propagate full context to tasks (default: True)
            context_window_limit: Model's context window limit (default: 100000)
            **kwargs: Additional arguments for AugmentedLLM
        """
        super().__init__(
            name=name,
            instruction=ORCHESTRATOR_SYSTEM_INSTRUCTION,
            context=context,
            **kwargs,
        )

        self.llm_factory = llm_factory
        self.agents = {agent.name: agent for agent in available_agents or []}
        self.max_iterations = max_iterations
        self.max_replans = max_replans
        self.enable_parallel = enable_parallel
        self.max_task_retries = max_task_retries

        # Get available servers
        if available_servers:
            self.available_servers = available_servers
        elif context and hasattr(context, "server_registry"):
            self.available_servers = list(context.server_registry.registry.keys())
            logger.info(
                f"Detected {len(self.available_servers)} MCP servers from registry"
            )
        else:
            self.available_servers = []
            logger.warning("No MCP servers available")

        # Core components
        self.memory = WorkspaceMemory(use_filesystem=enable_filesystem)
        self.queue = TodoQueue()
        self.budget = SimpleBudget()
        self.policy = PolicyEngine()
        self.knowledge_extractor = KnowledgeExtractor(llm_factory, context)
        self.agent_cache = AgentCache()

        # Context management settings
        self.task_context_budget = task_context_budget
        self.context_relevance_threshold = context_relevance_threshold
        self.context_compression_ratio = context_compression_ratio
        self.enable_full_context_propagation = enable_full_context_propagation
        self.context_window_limit = context_window_limit

        # Track context usage
        self.context_usage_stats = {
            "tasks_with_full_context": 0,
            "tasks_with_compressed_context": 0,
            "total_context_tokens": 0,
        }

        # Tracking
        self.objective: str = ""
        self.iteration: int = 0
        self.replan_count: int = 0
        self.start_time: float = 0.0
        self.current_plan: Optional[Plan] = None

        logger.info(
            f"Initialized {name} with {len(self.agents)} agents, "
            f"{len(self.available_servers)} servers, max_iterations={max_iterations}"
        )

    @track_tokens(node_type="workflow")
    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> List[MessageT]:
        """
        Main execution entry point.

        Args:
            message: User objective or message
            request_params: Request parameters

        Returns:
            List of response messages
        """
        tracer = get_tracer(self.context)

        with tracer.start_as_current_span(
            f"{self.__class__.__name__}.generate"
        ) as span:
            # Extract objective
            if isinstance(message, str):
                self.objective = message
            else:
                self.objective = await self._extract_objective(message)

            logger.info(f"Starting execution for objective: {self.objective[:100]}...")
            span.set_attribute("workflow.objective", self.objective[:200])

            # Execute workflow
            try:
                result = await self._execute_workflow(request_params, span)
                span.set_attribute("workflow.success", True)
                span.set_attribute("workflow.iterations", self.iteration)
                span.set_attribute("workflow.tokens_used", self.budget.tokens_used)
                span.set_attribute("workflow.cost", self.budget.cost_incurred)

                logger.info(
                    f"Execution completed successfully: "
                    f"{self.iteration} iterations, "
                    f"{self.budget.tokens_used} tokens, "
                    f"${self.budget.cost_incurred:.2f} cost"
                )

                # Log context usage statistics
                context_stats = self.get_context_usage_stats()
                logger.info(
                    f"Context usage: {context_stats['tasks_with_full_context']} tasks with full context, "
                    f"{context_stats['tasks_with_compressed_context']} compressed, "
                    f"avg {context_stats['average_context_tokens']:.0f} tokens/task"
                )

                return result

            except Exception as e:
                span.set_attribute("workflow.success", False)
                span.record_exception(e)
                logger.error(f"Workflow failed: {e}", exc_info=True)

                # Try to provide some value even on failure
                return await self._emergency_completion(str(e))

    async def _execute_workflow(
        self, request_params: Optional[RequestParams], span: "Span"
    ) -> List[MessageT]:
        """
        Core workflow execution logic with enhanced control.

        Args:
            request_params: Request parameters
            span: Tracing span

        Returns:
            Final response messages
        """
        self.start_time = time.time()
        self.iteration = 0
        self.replan_count = 0

        # Phase 1: Initial Planning
        span.add_event("phase_1_initial_planning")
        logger.info("Phase 1: Creating initial plan")

        initial_plan = await self._create_full_plan()

        if initial_plan.is_complete:
            logger.info("Objective already satisfied according to planner")
            return await self._create_simple_response(
                "The objective appears to be already satisfied."
            )

        self.queue.load_plan(initial_plan)

        # Main execution loop
        while self.iteration < self.max_iterations:
            self.iteration += 1

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Iteration {self.iteration} starting")
            logger.info(f"Queue status: {self.queue.get_progress_summary()}")
            logger.info(
                f"Budget usage: tokens={self.budget.tokens_used}, cost=${self.budget.cost_incurred:.2f}"
            )

            span.add_event(
                f"iteration_{self.iteration}_start",
                {
                    "queue_size": len(self.queue.pending_steps),
                    "completed": len(self.queue.completed_steps),
                    "tokens_used": self.budget.tokens_used,
                },
            )

            # Check if we need to take action based on policy
            verification_result = None
            if self.queue.is_empty():
                verification_result = await self._verify_completion()

            action = self.policy.decide_action(
                queue_empty=self.queue.is_empty(),
                verification_result=verification_result,
                budget=self.budget,
                iteration=self.iteration,
                max_iterations=self.max_iterations,
            )

            logger.info(f"Policy decision: {action}")

            if action == PolicyAction.FORCE_COMPLETE:
                logger.warning("Forcing completion due to resource constraints")
                break

            elif action == PolicyAction.EMERGENCY_STOP:
                logger.error("Emergency stop triggered")
                raise RuntimeError("Emergency stop due to repeated failures")

            elif action == PolicyAction.REPLAN:
                if self.replan_count >= self.max_replans:
                    logger.warning("Max replans reached, forcing completion")
                    break

                span.add_event(f"replanning_{self.replan_count + 1}")
                logger.info(
                    f"Replanning (attempt {self.replan_count + 1}/{self.max_replans})"
                )

                new_plan = await self._create_full_plan()

                if new_plan.is_complete:
                    logger.info("Objective complete according to new plan")
                    break

                added = self.queue.merge_plan(new_plan)
                if added == 0:
                    logger.info("No new steps from replanning, completing")
                    break

                self.replan_count += 1
                continue

            # Execute next step
            next_step = self.queue.get_next_step()
            if not next_step:
                logger.info("No more steps to execute")
                break

            logger.info(
                f"Executing step: {next_step.description} ({len(next_step.tasks)} tasks)"
            )
            span.add_event(
                "executing_step",
                {"step": next_step.description, "tasks": len(next_step.tasks)},
            )

            # Execute all tasks in the step
            step_success = await self._execute_step(next_step, request_params)

            # Complete the step
            self.queue.complete_step(next_step)

            # Update policy based on results
            if step_success:
                self.policy.record_success()
            else:
                self.policy.record_failure()

            # Check context window and trim if needed
            context_size = self.memory.estimate_context_size()
            if context_size > 40000:  # Getting close to typical limits
                logger.warning(f"Context size high: ~{context_size} tokens")
                self.memory.trim_for_context(30000)

        # Phase 3: Final Synthesis
        span.add_event("phase_3_final_synthesis")
        logger.info("\nPhase 3: Creating final synthesis")
        return await self._create_final_synthesis()

    async def _create_full_plan(self) -> Plan:
        """
        Create a comprehensive execution plan with XML-structured prompts.

        Returns:
            Complete execution plan
        """
        # Build planning context
        completed_steps = [step.description for step in self.queue.completed_steps[-5:]]
        relevant_knowledge = self.memory.get_relevant_knowledge(
            self.objective, limit=10
        )

        # Convert knowledge items to dict format for prompt
        knowledge_items = [
            {
                "key": item.key,
                "value": item.value,
                "confidence": item.confidence,
                "category": item.category,
            }
            for item in relevant_knowledge
        ]

        # Create planning agent
        planner = Agent(
            name="StrategicPlanner",
            instruction=PLANNER_INSTRUCTION,
            context=self.context,
        )

        llm = self.llm_factory(planner)

        # Try to create a valid plan with retries
        max_verification_attempts = 3
        previous_plan: Plan = None
        previous_errors: PlanVerificationResult = None

        for attempt in range(max_verification_attempts):
            # Build context (may include previous errors)
            context = get_planning_context(
                objective=self.objective,
                progress_summary=self.queue.get_progress_summary()
                if self.queue.completed_steps
                else "",
                completed_steps=completed_steps,
                knowledge_items=knowledge_items,
                available_servers=self.available_servers,
                available_agents=self.agents,
            )

            # Add previous plan and errors if this is a retry
            if previous_plan and previous_errors:
                context += "\n\n<previous_failed_plan>\n"
                context += previous_plan.model_dump_json(indent=2)
                context += "\n</previous_failed_plan>"

                context += f"\n\n<plan_errors>\n{previous_errors.get_error_summary()}\n</plan_errors>"
                context += "\n<important>The previous plan shown above had errors. Create a new plan that fixes ALL the issues listed. Pay special attention to:"
                context += "\n  - Only use MCP servers from the available_servers list"
                context += "\n  - Ensure all task names are unique"
                context += (
                    "\n  - Dependencies can only reference tasks from previous steps"
                )
                context += "\n</important>"

            # Track tokens for budget
            tokens_before = self._get_current_tokens()

            # Get structured plan
            prompt = get_full_plan_prompt(context)
            plan: Plan = await retry_with_backoff(
                lambda: llm.generate_structured(message=prompt, response_model=Plan),
                max_attempts=2,
            )

            # Update budget
            tokens_after = self._get_current_tokens()
            self.budget.update_tokens(tokens_after - tokens_before)

            # Verify the plan
            verification_result = self._verify_plan(plan)

            if verification_result.is_valid:
                logger.info(
                    f"Created valid plan: {len(plan.steps)} steps, reasoning: {plan.reasoning[:100]}..."
                )
                if verification_result.warnings:
                    logger.warning(
                        f"Plan warnings: {', '.join(verification_result.warnings)}"
                    )

                self.current_plan = plan
                return plan

            else:
                logger.warning(
                    f"Plan verification failed (attempt {attempt + 1}/{max_verification_attempts}): "
                    f"{len(verification_result.errors)} errors found"
                )

                # Store for next iteration
                previous_plan = plan
                previous_errors = verification_result

                if attempt == max_verification_attempts - 1:
                    # Final attempt failed
                    logger.error(
                        f"Failed to create valid plan after {max_verification_attempts} attempts"
                    )
                    logger.error(verification_result.get_error_summary())

                    # Return the plan anyway with a warning
                    self.current_plan = plan
                    return plan

        # Should not reach here
        raise RuntimeError("Failed to create a valid plan")

    def _verify_plan(self, plan: Plan) -> PlanVerificationResult:
        """
        Verify the plan for correctness, collecting all errors.

        Returns a PlanVerificationResult with all errors found.
        This method is modular - add more verification steps as needed.
        """
        result = PlanVerificationResult(is_valid=True)

        # Verification step 1: Check MCP server validity
        self._verify_mcp_servers(plan, result)

        # Verification step 2: Check agent name validity
        self._verify_agent_names(plan, result)

        # Verification step 3: Check task name uniqueness
        self._verify_task_names(plan, result)

        # Verification step 4: Check dependency references
        self._verify_dependencies(plan, result)

        # Verification step 5: Check for basic task validity
        self._verify_task_validity(plan, result)

        # Log successful verification
        if result.is_valid:
            logger.info("Plan verification succeeded")

        return result

    def _verify_mcp_servers(self, plan: Plan, result: PlanVerificationResult) -> None:
        """Verify all MCP servers in the plan are valid."""
        available_set = set(self.available_servers)

        for step_idx, step in enumerate(plan.steps):
            for task in step.tasks:
                if task.servers:
                    for server in task.servers:
                        if server not in available_set:
                            result.add_error(
                                category="invalid_server",
                                message=f"Server '{server}' is not available (available: {', '.join(self.available_servers) if self.available_servers else 'None'})",
                                step_index=step_idx,
                                task_name=task.name,
                                details={
                                    "invalid_server": server,
                                    "available_servers": list(self.available_servers),
                                    "step_description": step.description,
                                },
                            )

    def _verify_agent_names(self, plan: Plan, result: PlanVerificationResult) -> None:
        """Verify all specified agent names are valid."""
        available_agent_names = set(self.agents.keys())

        for step_idx, step in enumerate(plan.steps):
            for task in step.tasks:
                # Only verify if agent is specified (not None)
                if task.agent is not None:
                    if task.agent not in available_agent_names:
                        result.add_error(
                            category="invalid_agent",
                            message=f"Agent '{task.agent}' is not available (available: {', '.join(available_agent_names) if available_agent_names else 'None'})",
                            step_index=step_idx,
                            task_name=task.name,
                            details={
                                "invalid_agent": task.agent,
                                "available_agents": list(available_agent_names),
                                "step_description": step.description,
                                "task_description": task.description,
                            },
                        )

    def _verify_task_names(self, plan: Plan, result: PlanVerificationResult) -> None:
        """Verify all task names are unique."""
        seen_names = {}

        for step_idx, step in enumerate(plan.steps):
            for task in step.tasks:
                if task.name in seen_names:
                    first_step_idx, first_step_desc = seen_names[task.name]
                    result.add_error(
                        category="duplicate_name",
                        message=f"Task name '{task.name}' is duplicated (first seen in step {first_step_idx + 1}: {first_step_desc})",
                        step_index=step_idx,
                        task_name=task.name,
                        details={
                            "first_occurrence_step": first_step_idx + 1,
                            "duplicate_step": step_idx + 1,
                        },
                    )
                else:
                    seen_names[task.name] = (step_idx, step.description)

    def _verify_dependencies(self, plan: Plan, result: PlanVerificationResult) -> None:
        """Verify all task dependencies reference valid previous tasks."""
        # Build a map of task names to their step index
        task_step_map = {}
        for step_idx, step in enumerate(plan.steps):
            for task in step.tasks:
                task_step_map[task.name] = step_idx

        # Check each task's dependencies
        for step_idx, step in enumerate(plan.steps):
            for task in step.tasks:
                if task.requires_context_from:
                    for dep_name in task.requires_context_from:
                        if dep_name not in task_step_map:
                            result.add_error(
                                category="invalid_dependency",
                                message=f"References non-existent task '{dep_name}'",
                                step_index=step_idx,
                                task_name=task.name,
                                details={
                                    "missing_dependency": dep_name,
                                    "available_tasks": list(task_step_map.keys()),
                                },
                            )
                        elif task_step_map[dep_name] >= step_idx:
                            dep_step = task_step_map[dep_name]
                            result.add_error(
                                category="invalid_dependency",
                                message=f"References task '{dep_name}' from step {dep_step + 1} (can only reference previous steps)",
                                step_index=step_idx,
                                task_name=task.name,
                                details={
                                    "dependency_name": dep_name,
                                    "dependency_step": dep_step + 1,
                                    "current_step": step_idx + 1,
                                },
                            )

    def _verify_task_validity(self, plan: Plan, result: PlanVerificationResult) -> None:
        """Verify basic task validity."""
        for step_idx, step in enumerate(plan.steps):
            # Check step has tasks
            if not step.tasks:
                result.add_error(
                    category="empty_step",
                    message=f"Step '{step.description}' has no tasks",
                    step_index=step_idx,
                    details={"step_description": step.description},
                )

            for task in step.tasks:
                # Check task has a name
                if not task.name or not task.name.strip():
                    result.add_error(
                        category="invalid_task",
                        message="Task has no name",
                        step_index=step_idx,
                        details={"task_description": task.description},
                    )

                # Check task has a description
                if not task.description or not task.description.strip():
                    result.add_error(
                        category="invalid_task",
                        message=f"Task '{task.name}' has no description",
                        step_index=step_idx,
                        task_name=task.name,
                    )

                # Warn about extremely high context budgets
                if task.context_window_budget > 80000:
                    result.warnings.append(
                        f"Task '{task.name}' has very high context budget ({task.context_window_budget} tokens)"
                    )

    async def _execute_step(
        self, step: Step, request_params: Optional[RequestParams]
    ) -> bool:
        """
        Execute all tasks in a step with parallel support.

        Args:
            step: Step to execute
            request_params: Request parameters

        Returns:
            True if all tasks succeeded
        """
        logger.info(f"Executing step with {len(step.tasks)} tasks")

        # Prepare tasks for execution
        if self.enable_parallel and self.executor and len(step.tasks) > 1:
            # Parallel execution with streaming results
            logger.info("Executing tasks in parallel")
            task_coroutines = [
                self._execute_task(task, request_params) for task in step.tasks
            ]
            results = await self.executor.execute_many(task_coroutines)
        else:
            # Sequential execution
            logger.info("Executing tasks sequentially")
            results = []
            for task in step.tasks:
                result = await self._execute_task(task, request_params)
                results.append(result)

        # Check overall success
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        logger.info(
            f"Step execution complete: {successful} successful, {failed} failed"
        )

        return failed == 0

    async def _execute_task(
        self, task: Task, request_params: Optional[RequestParams]
    ) -> TaskResult:
        """
        Execute a single task with retry logic.

        Args:
            task: Task to execute
            request_params: Request parameters

        Returns:
            Task execution result
        """
        logger.info(f"Executing task: {task.description[:100]}...")

        # Try with retries
        for attempt in range(self.max_task_retries):
            try:
                result = await self._execute_task_once(task, request_params, attempt)

                if result.success:
                    return result

                # Task failed, maybe retry
                if attempt < self.max_task_retries - 1:
                    logger.warning(
                        f"Task failed, retrying (attempt {attempt + 2}/{self.max_task_retries})"
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff

            except Exception as e:
                logger.error(f"Task execution error: {e}")
                if attempt == self.max_task_retries - 1:
                    # Final attempt, return failure
                    return TaskResult(
                        task_name=task.name,
                        status=TaskStatus.FAILED,
                        error=str(e),
                        retry_count=attempt + 1,
                    )

        # All retries exhausted
        return result

    async def _execute_task_once(
        self, task: Task, request_params: Optional[RequestParams], attempt: int
    ) -> TaskResult:
        """
        Execute a single task attempt.

        Args:
            task: Task to execute
            request_params: Request parameters
            attempt: Current attempt number

        Returns:
            Task execution result
        """
        start_time = time.time()
        result = TaskResult(
            task_name=task.name, status=TaskStatus.IN_PROGRESS, retry_count=attempt
        )

        try:
            # Get or create agent
            if task.agent is None:
                # Check cache first
                cache_key = self.agent_cache.get_key(task.description, task.servers)
                agent = self.agent_cache.get(cache_key)

                if not agent:
                    agent = await self._create_dynamic_agent(task)
                    self.agent_cache.put(cache_key, agent)

            elif task.agent and task.agent in self.agents:
                agent = self.agents[task.agent]
                logger.debug(f"Using predefined agent: {task.agent}")

            else:
                # Default agent
                logger.warning(
                    f'Task "{task.name}" ({task.description}) requested agent "{task.agent}" which is not available. '
                    f"Creating default agent. Available agents: {list(self.agents.keys())}"
                )
                agent = Agent(
                    name=f"TaskExecutor_{task.name}",
                    instruction="You are a capable task executor. Complete the given task thoroughly using available tools.",
                    server_names=task.servers,
                    context=self.context,
                )

            # Build task context
            if task.requires_context_from:
                # Use explicit dependencies if specified
                task_context = self._build_relevant_task_context(task)
            elif self.enable_full_context_propagation:
                task_context = self._build_full_task_context(task)
            else:
                task_context = self._build_task_context(task)

            # Track tokens before
            tokens_before = self._get_current_tokens()

            # Execute with agent
            if isinstance(agent, AugmentedLLM):
                output = await agent.generate_str(
                    message=task_context,
                    request_params=request_params or RequestParams(max_iterations=10),
                )
            else:
                async with agent:
                    llm = await agent.attach_llm(self.llm_factory)
                    output = await llm.generate_str(
                        message=task_context,
                        request_params=request_params
                        or RequestParams(max_iterations=10),
                    )

            # Update tokens and budget
            tokens_after = self._get_current_tokens()
            tokens_used = tokens_after - tokens_before
            result.tokens_used = tokens_used
            self.budget.update_tokens(tokens_used)

            # Success
            result.status = TaskStatus.COMPLETED
            result.output = output
            result.duration_seconds = time.time() - start_time

            # Extract artifacts if mentioned
            if any(
                phrase in output.lower()
                for phrase in ["created file:", "saved to:", "wrote to:"]
            ):
                result.artifacts[f"task_{task.name}_output"] = output

            # Extract knowledge
            knowledge_items = await self.knowledge_extractor.extract_knowledge(
                result, self.objective
            )
            result.knowledge_extracted = knowledge_items

            # Update task status
            task.status = TaskStatus.COMPLETED

            logger.info(
                f"Task completed: {task.name} "
                f"(duration: {result.duration_seconds:.1f}s, tokens: {tokens_used})"
            )

        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.duration_seconds = time.time() - start_time
            task.status = TaskStatus.FAILED
            logger.error(f"Task {task.name} failed: {e}")

        # Record result
        self.memory.add_task_result(result)
        return result

    async def _create_dynamic_agent(self, task: Task) -> Agent:
        """
        Dynamically create an optimized agent for a task.

        Args:
            task: Task to create agent for

        Returns:
            Dynamically created agent
        """
        logger.debug(f"Creating dynamic agent for task: {task.description[:50]}...")

        # Agent designer
        designer = Agent(
            name="AgentDesigner",
            instruction=AGENT_DESIGNER_INSTRUCTION,
            context=self.context,
        )

        llm = self.llm_factory(designer)

        # Design agent
        design_prompt = get_agent_design_prompt(
            task.description, task.servers, self.objective
        )

        design = await llm.generate_structured(
            message=design_prompt, response_model=AgentDesign
        )

        # Build comprehensive instruction
        instruction = build_agent_instruction(design.model_dump())

        agent = Agent(
            name=design.name,
            instruction=instruction,
            server_names=task.servers,
            context=self.context,
        )

        logger.debug(f"Created agent '{design.name}' with role: {design.role}")
        return agent

    def _build_task_context(self, task: Task) -> str:
        """
        Build context for task execution, including relevant knowledge and artifacts.

        Args:
            task: Task to build context for

        Returns:
            Task context string
        """
        # Get relevant knowledge
        relevant_knowledge = self.memory.get_relevant_knowledge(
            task.description, limit=5
        )

        # Convert to dict format
        knowledge_items = [
            {"key": item.key, "value": item.value, "confidence": item.confidence}
            for item in relevant_knowledge
        ]

        # Get available artifacts
        artifact_names = (
            list(self.memory.artifacts.keys())[-5:] if self.memory.artifacts else None
        )

        # Get scratchpad path
        scratchpad_path = (
            str(self.memory.get_scratchpad_path())
            if self.memory.get_scratchpad_path()
            else None
        )

        return get_task_context(
            objective=self.objective,
            task_description=task.description,
            relevant_knowledge=knowledge_items,
            available_artifacts=artifact_names,
            scratchpad_path=scratchpad_path,
            required_servers=task.servers,
        )

    def _build_full_task_context(self, task: Task) -> str:
        """
        Build comprehensive context for task execution with smart token management.
        This includes all prior task results and relevant knowledge.

        Args:
            task: Task to build context for

        Returns:
            Task context string
        """
        # Start with essential context
        essential_parts = [
            f"<objective>{self.objective}</objective>",
            f"<task>{task.description}</task>",
        ]

        # Estimate tokens for essential parts
        essential_tokens = self._estimate_tokens("\n".join(essential_parts))
        remaining_budget = self.task_context_budget - essential_tokens

        # Gather all available context sources with relevance scores
        context_sources = self._gather_context_sources(task)

        # Sort by relevance and recency
        context_sources.sort(
            key=lambda x: (x["relevance"], x["timestamp"]), reverse=True
        )

        # Build context within budget
        context_parts = essential_parts.copy()

        if self.enable_full_context_propagation and remaining_budget > 0:
            context_parts.append("<previous_task_results>")

            added_sources = []
            current_tokens = essential_tokens

            for source in context_sources:
                source_tokens = source["estimated_tokens"]

                # Check if we can fit this source
                if current_tokens + source_tokens <= self.task_context_budget:
                    context_parts.append(source["content"])
                    added_sources.append(source["id"])
                    current_tokens += source_tokens
                else:
                    # Try compression if we're close to the limit
                    if (
                        current_tokens / self.task_context_budget
                        >= self.context_compression_ratio
                    ):
                        compressed = self._compress_context_source(source)
                        compressed_tokens = compressed["estimated_tokens"]

                        if (
                            current_tokens + compressed_tokens
                            <= self.task_context_budget
                        ):
                            context_parts.append(compressed["content"])
                            added_sources.append(f"{source['id']}_compressed")
                            current_tokens += compressed_tokens
                            self.context_usage_stats[
                                "tasks_with_compressed_context"
                            ] += 1

            context_parts.append("</previous_task_results>")

            # Log context usage
            logger.debug(
                f"Task context built: {current_tokens}/{self.task_context_budget} tokens, "
                f"{len(added_sources)} sources included"
            )
            self.context_usage_stats["total_context_tokens"] += current_tokens

            if len(added_sources) == len(context_sources):
                self.context_usage_stats["tasks_with_full_context"] += 1

        # Always add relevant knowledge (compact representation)
        knowledge_budget = min(
            5000, remaining_budget // 4
        )  # Reserve some space for knowledge
        relevant_knowledge = self._get_prioritized_knowledge(task, knowledge_budget)

        if relevant_knowledge:
            context_parts.append("<relevant_knowledge>")
            for item in relevant_knowledge:
                context_parts.append(
                    f'  <knowledge confidence="{item.confidence:.2f}" category="{item.category}">'
                )
                context_parts.append(f"    <insight>{item.key}: {item.value}</insight>")
                context_parts.append("  </knowledge>")
            context_parts.append("</relevant_knowledge>")

        # Add tool requirements
        if task.servers:
            context_parts.append("<required_tools>")
            for server in task.servers:
                context_parts.append(f"  <tool>{server}</tool>")
            context_parts.append("</required_tools>")

        # Add any existing artifacts
        if self.memory.artifacts:
            context_parts.append("<available_artifacts>")
            for name in list(self.memory.artifacts.keys())[-5:]:  # Last 5 artifacts
                context_parts.append(f"  <artifact>{name}</artifact>")
            context_parts.append("</available_artifacts>")

        return "\n".join(context_parts)

    def _build_relevant_task_context(self, task: Task) -> str:
        """
        Build task context with explicitly requested dependencies.

        This method uses the task's requires_context_from field to include
        only the outputs from specifically requested previous tasks.

        Args:
            task: Task to build context for

        Returns:
            Task context string with requested dependencies
        """
        # Start with essential context
        essential_parts = [
            f"<objective>{self.objective}</objective>",
            f"<task>{task.description}</task>",
        ]

        # Track tokens for budget management
        essential_tokens = self._estimate_tokens("\n".join(essential_parts))
        budget = task.context_window_budget
        remaining_budget = budget - essential_tokens

        # Build context parts
        context_parts = essential_parts.copy()
        current_tokens = essential_tokens

        # Add requested task outputs
        if task.requires_context_from and remaining_budget > 0:
            context_parts.append("<required_context>")

            # Gather requested task results as context sources
            requested_sources = []
            for task_name in task.requires_context_from:
                # Find the task by name
                referenced_task = self.queue.get_task_by_name(task_name)
                if not referenced_task:
                    logger.warning(
                        f"Task '{task.name}' requested context from unknown task '{task_name}'"
                    )
                    continue

                # Find the result for this task
                result = self._find_task_result_by_name(referenced_task.name)
                if not result:
                    logger.warning(f"No result found for task '{task_name}'")
                    continue

                if not result.success or not result.output:
                    logger.warning(f"Task '{task_name}' failed or has no output")
                    continue

                # Get the step description for this task
                step_description = self._find_step_for_task(referenced_task.name)

                # Format using existing method
                content = self._format_task_result_for_context(
                    step_description=step_description or "Unknown Step",
                    task=referenced_task,
                    result=result,
                )

                requested_sources.append(
                    {
                        "id": f"task_{referenced_task.name}",
                        "name": task_name,
                        "type": "requested_dependency",
                        "relevance": 1.0,  # Explicitly requested, so max relevance
                        "content": content,
                        "estimated_tokens": self._estimate_tokens(content),
                        "original_result": result,
                    }
                )

            # Sort by order in requires_context_from to maintain priority
            ordered_sources = []
            for task_name in task.requires_context_from:
                for source in requested_sources:
                    if source["name"] == task_name:
                        ordered_sources.append(source)
                        break

            # Add sources within budget
            for source in ordered_sources:
                source_tokens = source["estimated_tokens"]

                if current_tokens + source_tokens <= budget:
                    context_parts.append(source["content"])
                    current_tokens += source_tokens
                else:
                    # Try compression
                    compressed = self._compress_context_source(source)
                    compressed_tokens = compressed["estimated_tokens"]

                    if current_tokens + compressed_tokens <= budget:
                        context_parts.append(compressed["content"])
                        current_tokens += compressed_tokens
                        logger.info(
                            f"Compressed output for task '{source['name']}' to fit budget"
                        )
                    else:
                        logger.warning(
                            f"Cannot fit task '{source['name']}' in context even with compression "
                            f"(needs {compressed_tokens} tokens, only {budget - current_tokens} available)"
                        )

            context_parts.append("</required_context>")

        # Add relevant knowledge using existing method
        knowledge_budget = min(5000, remaining_budget // 4)
        relevant_knowledge = self._get_prioritized_knowledge(task, knowledge_budget)

        if relevant_knowledge:
            context_parts.append("<relevant_knowledge>")
            for item in relevant_knowledge:
                context_parts.append(
                    f'  <knowledge confidence="{item.confidence:.2f}" category="{item.category}" source="{item.source}">'
                )
                context_parts.append(f"    <insight>{item.key}: {item.value}</insight>")
                context_parts.append("  </knowledge>")
            context_parts.append("</relevant_knowledge>")

        # Add tool requirements
        if task.servers:
            context_parts.append("<required_tools>")
            for server in task.servers:
                context_parts.append(f"  <tool>{server}</tool>")
            context_parts.append("</required_tools>")

        # Add available artifacts (let the method decide how many based on space)
        if self.memory.artifacts and current_tokens < budget - 1000:
            context_parts.append("<available_artifacts>")
            artifacts_added = 0
            for name in reversed(list(self.memory.artifacts.keys())):
                artifact_line = f"  <artifact>{name}</artifact>"
                artifact_tokens = self._estimate_tokens(artifact_line)
                if current_tokens + artifact_tokens < budget - 500:  # Leave some buffer
                    context_parts.append(artifact_line)
                    current_tokens += artifact_tokens
                    artifacts_added += 1
                    if artifacts_added >= 5:  # Reasonable limit
                        break
            context_parts.append("</available_artifacts>")

        # Add scratchpad path if available
        scratchpad_path = self.memory.get_scratchpad_path()
        if scratchpad_path:
            context_parts.append(
                f"<scratchpad_path>{scratchpad_path}</scratchpad_path>"
            )

        final_context = "\n".join(context_parts)
        final_tokens = self._estimate_tokens(final_context)

        logger.debug(
            f"Built relevant context for task '{task.name}': "
            f"{len(task.requires_context_from)} dependencies requested, "
            f"{final_tokens} tokens used (budget: {budget})"
        )

        return final_context

    def _compress_task_output(self, output: str, max_tokens: int) -> str:
        """
        Compress task output to fit within token budget.

        Args:
            output: Original output to compress
            max_tokens: Maximum tokens allowed

        Returns:
            Compressed output
        """
        # Estimate characters from tokens (rough: 1 token ≈ 4 chars)
        max_chars = max_tokens * 4

        if len(output) <= max_chars:
            return output

        # Take beginning and end to preserve context
        chunk_size = max_chars // 2 - 50  # Leave room for ellipsis
        compressed = (
            output[:chunk_size]
            + "\n... [content truncated for space] ...\n"
            + output[-chunk_size:]
        )

        return compressed

    def _gather_context_sources(self, task: Task) -> List[Dict[str, Any]]:
        """Gather all potential context sources with relevance scoring."""
        sources = []

        # Get all completed task results
        for step in self.queue.completed_steps:
            for step_task in step.tasks:
                result = self._find_task_result_by_name(step_task.name)
                if result and result.success and result.output:
                    # Calculate relevance score
                    relevance = self._calculate_relevance(
                        task_description=task.description,
                        source_task_description=step_task.description,
                        source_output=result.output,
                        source_step=step.description,
                    )

                    # Format the source content
                    content = self._format_task_result_for_context(
                        step_description=step.description, task=step_task, result=result
                    )

                    sources.append(
                        {
                            "id": f"task_{step_task.name}",
                            "type": "task_result",
                            "relevance": relevance,
                            "timestamp": result.duration_seconds,  # Use as proxy for recency
                            "content": content,
                            "estimated_tokens": self._estimate_tokens(content),
                            "original_result": result,
                        }
                    )

        return sources

    def _find_task_result_by_name(self, task_name: str) -> Optional[TaskResult]:
        """Find a task result by task name."""
        for result in self.memory.task_results:
            if result.task_name == task_name:
                return result
        return None

    def _find_step_for_task(self, task_name: str) -> Optional[str]:
        """Find the step description that contains a task."""
        for step in self.queue.completed_steps:
            for task in step.tasks:
                if task.name == task_name:
                    return step.description
        return None

    def _calculate_relevance(
        self,
        task_description: str,
        source_task_description: str,
        source_output: str,
        source_step: str,
    ) -> float:
        """Calculate relevance score between current task and a source."""

        # Simple keyword-based relevance (can be enhanced with embeddings)
        task_words = set(task_description.lower().split())
        source_words = set(source_task_description.lower().split())
        output_words = set(source_output.lower().split()[:100])  # First 100 words
        step_words = set(source_step.lower().split())

        # Check for explicit references
        if any(
            ref in task_description.lower()
            for ref in ["previous", "all", "comprehensive", "synthesize", "compile"]
        ):
            base_relevance = 0.8
        else:
            base_relevance = 0.5

        # Calculate word overlap
        task_overlap = (
            len(task_words & source_words) / len(task_words) if task_words else 0
        )
        output_overlap = (
            len(task_words & output_words) / len(task_words) if task_words else 0
        )
        step_overlap = (
            len(task_words & step_words) / len(task_words) if task_words else 0
        )

        # Weighted relevance
        relevance = (
            base_relevance * 0.4
            + task_overlap * 0.3
            + output_overlap * 0.2
            + step_overlap * 0.1
        )

        # Boost relevance for certain patterns
        if (
            "report" in task_description.lower()
            and "analysis" in source_task_description.lower()
        ):
            relevance = min(1.0, relevance + 0.2)

        return min(1.0, relevance)

    def _format_task_result_for_context(
        self, step_description: str, task: Task, result: TaskResult
    ) -> str:
        """Format a task result for inclusion in context."""
        parts = [
            f'  <step_result step="{step_description}">',
            f'    <task name="{task.name}">{task.description}</task>',
            f"    <output>{result.output}</output>",
        ]

        # Include key knowledge if available
        if result.knowledge_extracted:
            parts.append("    <key_findings>")
            for item in result.knowledge_extracted[:5]:  # Top 5 findings
                parts.append(f"      - {item.key}: {item.value}")
            parts.append("    </key_findings>")

        parts.append("  </step_result>")
        return "\n".join(parts)

    def _compress_context_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Compress a context source to fit within budget."""
        result = source["original_result"]

        # Simple compression: truncate output and keep only key findings
        compressed_output = (
            result.output[:500] + "..." if len(result.output) > 500 else result.output
        )

        parts = [
            f'  <step_result_compressed step="{source["id"]}">',
            f"    <summary>{compressed_output}</summary>",
        ]

        if result.knowledge_extracted:
            parts.append("    <key_findings>")
            for item in result.knowledge_extracted[:3]:  # Even fewer findings
                parts.append(f"      - {item.key}")
            parts.append("    </key_findings>")

        parts.append("  </step_result_compressed>")

        content = "\n".join(parts)

        return {
            "id": source["id"],
            "content": content,
            "estimated_tokens": self._estimate_tokens(content),
        }

    def _get_prioritized_knowledge(
        self, task: Task, token_budget: int
    ) -> List[KnowledgeItem]:
        """Get knowledge items prioritized by relevance within token budget."""
        if not self.memory.knowledge:
            return []

        # Score all knowledge items
        scored_items = []
        for item in self.memory.knowledge:
            relevance = self._calculate_knowledge_relevance(task.description, item)
            if relevance >= self.context_relevance_threshold:
                scored_items.append((relevance, item))

        # Sort by relevance and recency
        scored_items.sort(
            key=lambda x: (x[0], x[1].timestamp.timestamp()), reverse=True
        )

        # Select items within budget
        selected = []
        current_tokens = 0

        for relevance, item in scored_items:
            item_tokens = self._estimate_tokens(f"{item.key}: {item.value}")
            if current_tokens + item_tokens <= token_budget:
                selected.append(item)
                current_tokens += item_tokens
            else:
                break

        return selected

    def _calculate_knowledge_relevance(
        self, task_description: str, item: KnowledgeItem
    ) -> float:
        """Calculate relevance of a knowledge item to a task."""
        # Simple implementation - can be enhanced
        task_words = set(task_description.lower().split())
        item_words = set(item.key.lower().split()) | set(
            str(item.value).lower().split()[:20]
        )

        overlap = len(task_words & item_words) / len(task_words) if task_words else 0

        # Boost by confidence and category relevance
        category_boost = (
            0.2 if item.category in ["findings", "analysis", "errors"] else 0
        )

        return min(1.0, overlap + category_boost) * item.confidence

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple heuristic: 1 token ≈ 4 characters
        # Can be replaced with actual tokenizer
        return len(text) // 4

    def get_context_usage_stats(self) -> Dict[str, Any]:
        """Get statistics about context usage."""
        total_tasks = (
            self.context_usage_stats["tasks_with_full_context"]
            + self.context_usage_stats["tasks_with_compressed_context"]
        )

        stats = {
            "tasks_with_full_context": self.context_usage_stats[
                "tasks_with_full_context"
            ],
            "tasks_with_compressed_context": self.context_usage_stats[
                "tasks_with_compressed_context"
            ],
            "total_tasks_with_context": total_tasks,
            "average_context_tokens": self.context_usage_stats["total_context_tokens"]
            / total_tasks
            if total_tasks > 0
            else 0,
            "total_context_tokens": self.context_usage_stats["total_context_tokens"],
            "context_propagation_enabled": self.enable_full_context_propagation,
            "context_budget": self.task_context_budget,
        }

        return stats

    async def _verify_completion(self) -> tuple[bool, float]:
        """
        Verify if the objective has been completed.

        Returns:
            Tuple of (is_complete, confidence)
        """
        logger.info("Verifying objective completion...")

        verifier = Agent(
            name="ObjectiveVerifier",
            instruction=VERIFIER_INSTRUCTION,
            context=self.context,
        )

        llm = self.llm_factory(verifier)

        # Build verification context
        context = get_verification_context(
            objective=self.objective,
            progress_summary=self.queue.get_progress_summary(),
            knowledge_summary=self.memory.get_knowledge_summary(limit=15),
            artifacts=self.memory.artifacts,
        )

        prompt = get_verification_prompt(context)

        result = await llm.generate_structured(
            message=prompt, response_model=VerificationResult
        )

        logger.info(
            f"Verification result: complete={result.is_complete}, "
            f"confidence={result.confidence}, "
            f"missing={len(result.missing_elements)}, "
            f"reasoning: {result.reasoning[:100]}..."
        )

        return result.is_complete, result.confidence

    async def _create_final_synthesis(self) -> List[MessageT]:
        """
        Create the final deliverable from all work.

        Returns:
            Final synthesis messages
        """
        logger.info("Creating final synthesis of all work...")

        synthesizer = Agent(
            name="FinalSynthesizer",
            instruction=SYNTHESIZER_INSTRUCTION,
            server_names=self.available_servers,
            context=self.context,
        )

        # Build synthesis context
        execution_summary = {
            "iterations": self.iteration,
            "steps_completed": len(self.queue.completed_steps),
            "tasks_completed": len(self.queue.completed_task_names),
            "tokens_used": self.budget.tokens_used,
            "cost": self.budget.cost_incurred,
        }

        # Prepare completed steps with results
        completed_steps = []
        for step in self.queue.completed_steps:
            step_data = {"description": step.description, "task_results": []}

            # Get results for tasks in this step
            step_task_names = {t.name for t in step.tasks}
            step_results = [
                r for r in self.memory.task_results if r.task_name in step_task_names
            ]

            for result in step_results:
                if result.success and result.output:
                    task = self.queue.all_tasks.get(result.task_name)
                    task_desc = task.description if task else "Unknown task"

                    step_data["task_results"].append(
                        {
                            "description": task_desc,
                            "output": result.output,
                            "success": True,
                        }
                    )

            completed_steps.append(step_data)

        # Group knowledge by category
        knowledge_by_category = defaultdict(list)
        for item in self.memory.knowledge:
            knowledge_by_category[item.category].append(item)

        context = get_synthesis_context(
            objective=self.objective,
            execution_summary=execution_summary,
            completed_steps=completed_steps,
            knowledge_by_category=dict(knowledge_by_category),
            artifacts=self.memory.artifacts,
        )

        prompt = get_synthesis_prompt(context)

        # Generate synthesis
        async with synthesizer:
            llm = await synthesizer.attach_llm(self.llm_factory)

            result = await llm.generate(
                message=prompt, request_params=RequestParams(max_iterations=5)
            )

            logger.info("Final synthesis completed")
            return result

    async def _emergency_completion(self, error: str) -> List[MessageT]:
        """
        Provide best-effort response when workflow fails.

        Args:
            error: Error message

        Returns:
            Emergency response messages
        """
        logger.warning(f"Entering emergency completion mode due to: {error}")

        emergency_agent = Agent(
            name="EmergencyResponder",
            instruction=EMERGENCY_RESPONDER_INSTRUCTION,
            context=self.context,
        )

        # Prepare partial knowledge
        partial_knowledge = [
            {"key": item.key, "value": item.value}
            for item in self.memory.knowledge[:10]
        ]

        # Get artifact names
        artifacts_created = (
            list(self.memory.artifacts.keys())[:5] if self.memory.artifacts else None
        )

        context = get_emergency_context(
            objective=self.objective,
            error=error,
            progress_summary=self.queue.get_progress_summary(),
            partial_knowledge=partial_knowledge,
            artifacts_created=artifacts_created,
        )

        prompt = get_emergency_prompt(context)

        async with emergency_agent:
            llm = await emergency_agent.attach_llm(self.llm_factory)
            return await llm.generate(message=prompt)

    async def _extract_objective(
        self, message: MessageParamT | List[MessageParamT]
    ) -> str:
        """
        Extract objective from complex message types.

        Args:
            message: Input message

        Returns:
            Extracted objective string
        """
        extractor = Agent(
            name="ObjectiveExtractor",
            instruction="""
            The message that will be provided to you will be a user message. 
            Your job is to extract the user's objective or request from their message. 
            Be concise and clear. You must be able to answer: 'What is the user asking for in this message?'
            """,
            context=self.context,
        )

        llm = self.llm_factory(extractor)

        return await llm.generate_str(
            message=message,
            request_params=RequestParams(max_iterations=1),
        )

    async def _create_simple_response(self, content: str) -> List[MessageT]:
        """
        Create a simple response message.

        Args:
            content: Response content

        Returns:
            Response messages
        """
        simple_agent = Agent(
            name="SimpleResponder",
            instruction="Provide a clear, direct response.",
            context=self.context,
        )

        async with simple_agent:
            llm = await simple_agent.attach_llm(self.llm_factory)
            return await llm.generate(message=content)

    def _get_current_tokens(self) -> int:
        """
        Get current token count from context.

        Returns:
            Current token count
        """
        if self.context and hasattr(self.context, "token_counter"):
            # Get the summary which contains the total tokens
            try:
                summary = self.context.token_counter.get_summary()
                if summary and hasattr(summary, "usage"):
                    return summary.usage.total_tokens
            except Exception as e:
                logger.debug(f"Failed to get tokens from context: {e}")
        return self.budget.tokens_used  # Fallback to budget tracking

    # ========================================================================
    # Additional methods for AugmentedLLM compatibility
    # ========================================================================

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> str:
        """Generate and return string representation."""
        messages = await self.generate(message, request_params)
        if messages:
            # This is simplified - real implementation would use proper message conversion
            return str(messages[0])
        return ""

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        """Generate structured output."""
        result_str = await self.generate_str(message, request_params)

        parser = Agent(
            name="StructuredParser",
            instruction="Parse the content into the requested structure accurately.",
            context=self.context,
        )

        llm = self.llm_factory(parser)

        return await llm.generate_structured(
            message=f"<parse_request>\n{result_str}\n</parse_request>",
            response_model=response_model,
            request_params=RequestParams(max_iterations=1),
        )
