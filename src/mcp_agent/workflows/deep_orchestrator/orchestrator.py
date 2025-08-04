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

# Import all submodules
from mcp_agent.workflows.deep_orchestrator.budget import SimpleBudget
from mcp_agent.workflows.deep_orchestrator.cache import AgentCache
from mcp_agent.workflows.deep_orchestrator.knowledge import KnowledgeExtractor
from mcp_agent.workflows.deep_orchestrator.memory import WorkspaceMemory
from mcp_agent.workflows.deep_orchestrator.models import (
    AgentDesign,
    Plan,
    PolicyAction,
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
        available_agents: Optional[Dict[str, Agent]] = None,
        available_servers: Optional[List[str]] = None,
        max_iterations: int = 20,
        max_replans: int = 3,
        enable_filesystem: bool = True,
        enable_parallel: bool = True,
        max_task_retries: int = 3,
        context: Optional["Context"] = None,
        **kwargs,
    ):
        """
        Initialize the adaptive orchestrator with production features.

        Args:
            llm_factory: Factory function to create LLMs
            name: Name of the orchestrator
            available_agents: Pre-defined agents available for use
            available_servers: List of available MCP servers
            max_iterations: Maximum workflow iterations
            max_replans: Maximum number of replanning attempts
            enable_filesystem: Enable filesystem workspace
            enable_parallel: Enable parallel task execution
            max_task_retries: Maximum retries per task
            context: Application context
            **kwargs: Additional arguments for AugmentedLLM
        """
        super().__init__(
            name=name,
            instruction=ORCHESTRATOR_SYSTEM_INSTRUCTION,
            context=context,
            **kwargs,
        )

        self.llm_factory = llm_factory
        self.available_agents = available_agents or {}
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

        # Tracking
        self.objective: str = ""
        self.iteration: int = 0
        self.replan_count: int = 0
        self.start_time: float = 0.0
        self.current_plan: Optional[Plan] = None

        logger.info(
            f"Initialized {name} with {len(self.available_agents)} agents, "
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

                return result

            except Exception as e:
                span.set_attribute("workflow.success", False)
                span.record_exception(e)
                logger.error(f"Workflow failed: {e}", exc_info=True)

                # Try to provide some value even on failure
                return await self._emergency_completion(str(e))

    async def _execute_workflow(
        self, request_params: Optional[RequestParams], span: Any
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

        context = get_planning_context(
            objective=self.objective,
            progress_summary=self.queue.get_progress_summary()
            if self.queue.completed_steps
            else "",
            completed_steps=completed_steps,
            knowledge_items=knowledge_items,
            available_servers=self.available_servers,
            available_agents=self.available_agents,
        )

        # Create planning agent
        planner = Agent(
            name="StrategicPlanner",
            instruction=PLANNER_INSTRUCTION,
            context=self.context,
        )

        llm = self.llm_factory(planner)

        # Track tokens for budget
        tokens_before = self._get_current_tokens()

        # Get structured plan
        prompt = get_full_plan_prompt(context)
        plan = await retry_with_backoff(
            lambda: llm.generate_structured(message=prompt, response_model=Plan),
            max_attempts=2,
        )

        # Update budget
        tokens_after = self._get_current_tokens()
        self.budget.update_tokens(tokens_after - tokens_before)

        logger.info(
            f"Created plan: {len(plan.steps)} steps, reasoning: {plan.reasoning[:100]}..."
        )

        self.current_plan = plan

        return plan

    async def _execute_step(
        self, step: Any, request_params: Optional[RequestParams]
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

        # Get executor - with proper fallback
        executor = getattr(self, "executor", None)

        # Prepare tasks for execution
        if self.enable_parallel and executor and len(step.tasks) > 1:
            # Parallel execution
            logger.info("Executing tasks in parallel")
            task_coroutines = [
                self._execute_task(task, request_params) for task in step.tasks
            ]
            results = await executor.execute_many(task_coroutines)
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
                        task_id=task.id,
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
            task_id=task.id, status=TaskStatus.IN_PROGRESS, retry_count=attempt
        )

        try:
            # Get or create agent
            if task.agent == "AUTO":
                # Check cache first
                cache_key = self.agent_cache.get_key(task.description, task.servers)
                agent = self.agent_cache.get(cache_key)

                if not agent:
                    agent = await self._create_dynamic_agent(task)
                    self.agent_cache.put(cache_key, agent)

            elif task.agent and task.agent in self.available_agents:
                agent = self.available_agents[task.agent]
                logger.debug(f"Using predefined agent: {task.agent}")

            else:
                # Default agent
                agent = Agent(
                    name=f"TaskExecutor_{task.id[:8]}",
                    instruction="You are a capable task executor. Complete the given task thoroughly using available tools.",
                    server_names=task.servers,
                    context=self.context,
                )

            # Build task context
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
                result.artifacts[f"task_{task.id[:8]}_output"] = output

            # Extract knowledge
            knowledge_items = await self.knowledge_extractor.extract_knowledge(
                result, self.objective
            )
            result.knowledge_extracted = knowledge_items

            # Update task status
            task.status = TaskStatus.COMPLETED

            logger.info(
                f"Task completed: {task.id[:8]} "
                f"(duration: {result.duration_seconds:.1f}s, tokens: {tokens_used})"
            )

        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.duration_seconds = time.time() - start_time
            task.status = TaskStatus.FAILED
            logger.error(f"Task {task.id[:8]} failed: {e}")

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
        Build comprehensive context for task execution.

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
            "tasks_completed": len(self.queue.completed_task_ids),
            "tokens_used": self.budget.tokens_used,
            "cost": self.budget.cost_incurred,
        }

        # Prepare completed steps with results
        completed_steps = []
        for step in self.queue.completed_steps:
            step_data = {"description": step.description, "task_results": []}

            # Get results for tasks in this step
            step_task_ids = {t.id for t in step.tasks}
            step_results = [
                r for r in self.memory.task_results if r.task_id in step_task_ids
            ]

            for result in step_results:
                if result.success and result.output:
                    task = self.queue.all_tasks.get(result.task_id)
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

    async def _extract_objective(self, message: Any) -> str:
        """
        Extract objective from complex message types.

        Args:
            message: Input message

        Returns:
            Extracted objective string
        """
        extractor = Agent(
            name="ObjectiveExtractor",
            instruction="Extract the user's objective or request from their message. Be concise and clear.",
            context=self.context,
        )

        llm = self.llm_factory(extractor)

        return await llm.generate_str(
            message=f"What is the user asking for in this message?\n\n{message}",
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
