"""
Adaptive Workflow - Multi-agent system based on Claude Deep Research architecture
"""
import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Type, Any, Set, Tuple
from enum import Enum

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParam,
    MessageParamT, 
    MessageT,
    ModelT,
    RequestParams,
)
from mcp_agent.workflows.llm.llm_selector import ModelSelector
from mcp.types import ModelPreferences
from mcp_agent.tracing.telemetry import get_tracer
from mcp_agent.tracing.semconv import (
    GEN_AI_AGENT_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.types import TextContent

from .models import (
    TaskType,
    TaskComplexity,
    SubagentSpec,
    SubagentTask,
    CitationSource,
    WorkflowStrategy,
    WorkflowMemory,
    WorkflowResult,
    TaskBatch,
    ProgressEvaluation,
)
from .memory import MemoryManager, LearningManager
from .prompts import (
    LEAD_RESEARCHER_PROMPT,
    TASK_ANALYZER_PROMPT,
    STRATEGY_PLANNER_PROMPT,
    SUBAGENT_CREATOR_PROMPT,
    PROGRESS_EVALUATOR_PROMPT,
    RESULT_SYNTHESIZER_PROMPT,
    CITATION_FORMATTER_PROMPT,
    RESEARCH_SUBAGENT_TEMPLATE,
    ACTION_SUBAGENT_TEMPLATE,
    HYBRID_SUBAGENT_TEMPLATE,
)

logger = get_logger(__name__)


class AdaptiveWorkflow(AugmentedLLM[MessageParamT, MessageT]):
    """
    Adaptive multi-agent workflow based on Claude Deep Research architecture.
    
    Key features:
    - Dynamic task analysis and strategy selection
    - Non-cascading iteration limits 
    - Comprehensive cost and time tracking
    - Memory persistence and learning
    - Parallel subagent execution
    - Citation tracking for research
    - Adaptive behavior based on progress
    """
    
    def __init__(
        self,
        llm_factory: Callable[[Agent], AugmentedLLM[MessageParamT, MessageT]],
        name: str | None = None,
        available_servers: List[str] | None = None,
        time_budget: timedelta = timedelta(minutes=30),
        cost_budget: float = 10.0,
        max_iterations: int = 20,
        max_subagents: int = 50,
        enable_parallel: bool = True,
        enable_learning: bool = True,
        memory_manager: Optional[MemoryManager] = None,
        learning_manager: Optional[LearningManager] = None,
        model_preferences: Optional[ModelPreferences] = None,
        context: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize Adaptive Workflow
        
        Args:
            llm_factory: Factory to create LLM instances
            name: Workflow name
            available_servers: List of available MCP servers
            time_budget: Maximum time for workflow execution
            cost_budget: Maximum cost in dollars
            max_iterations: Maximum iterations (non-cascading)
            max_subagents: Maximum total subagents to create
            enable_parallel: Enable parallel subagent execution
            enable_learning: Enable learning from executions
            memory_manager: Custom memory manager (defaults to in-memory)
            learning_manager: Custom learning manager
            model_preferences: Preferences for model selection
            context: Workflow context
        """
        super().__init__(
            name=name or "AdaptiveWorkflow",
            instruction=LEAD_RESEARCHER_PROMPT,
            model_preferences=model_preferences,
            context=context,
            **kwargs,
        )
        
        self.llm_factory = llm_factory
        self.available_servers = available_servers or []
        self.time_budget = time_budget
        self.cost_budget = cost_budget
        self.max_iterations = max_iterations
        self.max_subagents = max_subagents
        self.enable_parallel = enable_parallel
        self.enable_learning = enable_learning
        
        # Initialize managers
        self.memory_manager = memory_manager or MemoryManager()
        self.learning_manager = learning_manager or LearningManager()
        
        # Model selector for choosing appropriate models
        self.model_selector = ModelSelector()
        
        # Track current workflow
        self._current_workflow_id: Optional[str] = None
        self._current_memory: Optional[WorkflowMemory] = None
        
    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> List[MessageT]:
        """Execute the adaptive workflow"""
        tracer = get_tracer(self.context)
        
        with tracer.start_as_current_span(
            f"{self.__class__.__name__}.generate"
        ) as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.name)
            span.set_attribute("workflow.time_budget_seconds", self.time_budget.total_seconds())
            span.set_attribute("workflow.cost_budget", self.cost_budget)
            span.set_attribute("workflow.max_iterations", self.max_iterations)
            
            try:
                # Extract objective from message
                objective = self._extract_objective(message)
                span.set_attribute("workflow.objective", objective[:200])
                
                # Execute workflow
                result = await self._execute_workflow(objective, request_params, span)
                
                # Record final metrics
                span.set_attribute("workflow.success", result.success)
                span.set_attribute("workflow.total_time", result.total_time_seconds)
                span.set_attribute("workflow.total_cost", result.total_cost)
                span.set_attribute("workflow.tasks_completed", result.tasks_completed)
                
                # Convert to message format
                return self._format_result_as_messages(result)
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> str:
        """Execute workflow and return string result"""
        messages = await self.generate(message, request_params)
        return self._messages_to_string(messages)
    
    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        """Execute workflow and return structured result"""
        # Execute workflow
        result_str = await self.generate_str(message, request_params)
        
        # Use a fast model to structure the result
        structuring_agent = Agent(
            name="StructuringAgent",
            instruction="Convert the workflow result to the requested format",
        )
        structuring_llm = self.llm_factory(structuring_agent)
        
        return await structuring_llm.generate_structured(
            message=result_str,
            response_model=response_model,
            request_params=request_params,
        )
    
    async def _execute_workflow(
        self,
        objective: str,
        request_params: Optional[RequestParams],
        span: trace.Span,
    ) -> WorkflowResult:
        """Main workflow execution"""
        workflow_id = str(uuid.uuid4())
        self._current_workflow_id = workflow_id
        
        # Phase 1: Task Analysis
        span.add_event("phase_1_task_analysis_start")
        task_type, complexity = await self._analyze_task(objective, span)
        
        # Phase 2: Strategy Planning
        span.add_event("phase_2_strategy_planning_start")
        strategy = await self._plan_strategy(objective, task_type, complexity, span)
        
        # Initialize workflow memory
        self._current_memory = WorkflowMemory(
            workflow_id=workflow_id,
            objective=objective,
            task_type=task_type,
            complexity=complexity,
            strategy=strategy,
        )
        
        # Phase 3: Iterative Execution
        span.add_event("phase_3_execution_start")
        await self._execute_iterations(request_params, span)
        
        # Phase 4: Result Synthesis
        span.add_event("phase_4_synthesis_start")
        result = await self._synthesize_results(span)
        
        # Phase 5: Citation Formatting (if research task)
        if task_type in [TaskType.RESEARCH, TaskType.HYBRID]:
            span.add_event("phase_5_citation_formatting_start")
            result = await self._format_citations(result, span)
        
        # Save memory and record learning
        self.memory_manager.save_memory(self._current_memory)
        if self.enable_learning:
            self.learning_manager.record_execution(self._current_memory, result.success)
        
        return result
    
    async def _analyze_task(
        self,
        objective: str,
        span: trace.Span,
    ) -> Tuple[TaskType, TaskComplexity]:
        """Analyze the task to determine type and complexity"""
        # Use a fast model for analysis
        analyzer_agent = Agent(
            name="TaskAnalyzer",
            instruction=TASK_ANALYZER_PROMPT,
        )
        analyzer_llm = self.llm_factory(analyzer_agent)
        
        # Get analysis
        from pydantic import BaseModel, Field
        
        class TaskAnalysis(BaseModel):
            task_type: TaskType
            complexity: TaskComplexity
            key_aspects: List[str] = Field(max_items=10)
            recommended_tools: List[str] = Field(default_factory=list)
            potential_challenges: List[str] = Field(default_factory=list)
        
        analysis = await analyzer_llm.generate_structured(
            message=f"Analyze this objective and determine task type and complexity:\n\n{objective}",
            response_model=TaskAnalysis,
            request_params=RequestParams(
                model_preferences=ModelPreferences(speed=0.8, intelligence=0.2)
            ),
        )
        
        span.add_event("task_analyzed", {
            "task_type": analysis.task_type,
            "complexity": analysis.complexity,
            "key_aspects": len(analysis.key_aspects),
        })
        
        # Use learning manager to refine estimate if available
        if self.enable_learning:
            learned_complexity = self.learning_manager.estimate_complexity(objective, analysis.task_type)
            # Average the learned and analyzed complexity
            if learned_complexity != analysis.complexity:
                logger.info(f"Adjusting complexity from {analysis.complexity} to {learned_complexity} based on learning")
                analysis.complexity = learned_complexity
        
        return analysis.task_type, analysis.complexity
    
    async def _plan_strategy(
        self,
        objective: str,
        task_type: TaskType,
        complexity: TaskComplexity,
        span: trace.Span,
    ) -> WorkflowStrategy:
        """Plan execution strategy"""
        # Check if learning manager has a suggestion
        suggested_strategy = None
        if self.enable_learning:
            suggested_strategy = self.learning_manager.suggest_strategy(task_type, complexity)
        
        # Use strategy planner to create/refine strategy
        planner_agent = Agent(
            name="StrategyPlanner",
            instruction=STRATEGY_PLANNER_PROMPT,
        )
        planner_llm = self.llm_factory(planner_agent)
        
        context = f"""
        Objective: {objective}
        Task Type: {task_type}
        Complexity: {complexity}
        Available MCP Servers: {', '.join(self.available_servers)}
        Time Budget: {self.time_budget.total_seconds() / 60:.1f} minutes
        Cost Budget: ${self.cost_budget}
        """
        
        if suggested_strategy:
            context += f"\n\nSuggested approach based on past success: {suggested_strategy.approach}"
        
        strategy = await planner_llm.generate_structured(
            message=context,
            response_model=WorkflowStrategy,
            request_params=RequestParams(
                model_preferences=ModelPreferences(speed=0.7, intelligence=0.3)
            ),
        )
        
        # Apply constraints
        strategy.parallelism_level = min(strategy.parallelism_level, 10 if self.enable_parallel else 1)
        strategy.subagent_budget = min(strategy.subagent_budget, self.max_subagents)
        
        span.add_event("strategy_planned", {
            "approach": strategy.approach,
            "parallelism": strategy.parallelism_level,
            "subagent_budget": strategy.subagent_budget,
        })
        
        return strategy
    
    async def _execute_iterations(
        self,
        request_params: Optional[RequestParams],
        span: trace.Span,
    ) -> None:
        """Execute the main iteration loop"""
        start_time = time.time()
        
        while self._should_continue(start_time):
            iteration = self._current_memory.iterations + 1
            self._current_memory.iterations = iteration
            
            span.add_event(f"iteration_{iteration}_start", {
                "phase": self._current_memory.phase,
                "subagents_created": self._current_memory.total_subagents_created,
                "cost_so_far": self._current_memory.total_cost,
            })
            
            # Plan next batch of tasks
            task_batch = await self._plan_next_tasks(span)
            if not task_batch or not task_batch.tasks:
                logger.info("No more tasks to execute")
                break
            
            # Check resource limits before executing
            if not self._check_resource_limits(task_batch, span):
                logger.warning("Resource limits would be exceeded, stopping")
                break
            
            # Execute tasks
            if self.enable_parallel and task_batch.can_parallelize and len(task_batch.tasks) > 1:
                await self._execute_parallel(task_batch.tasks, request_params, span)
            else:
                await self._execute_sequential(task_batch.tasks, request_params, span)
            
            # Evaluate progress
            evaluation = await self._evaluate_progress(span)
            
            if evaluation.is_complete and evaluation.confidence > 0.8:
                logger.info(f"Objective achieved with {evaluation.confidence:.1%} confidence")
                self._current_memory.phase = "complete"
                break
            
            if evaluation.should_pivot:
                logger.info(f"Pivoting strategy: {evaluation.pivot_reason}")
                # Could implement strategy adjustment here
            
            # Save checkpoint
            self.memory_manager.save_memory(self._current_memory)
            
            span.add_event(f"iteration_{iteration}_complete", {
                "tasks_executed": len(task_batch.tasks),
                "progress_confidence": evaluation.confidence,
            })
    
    async def _plan_next_tasks(self, span: trace.Span) -> Optional[TaskBatch]:
        """Plan the next batch of tasks"""
        # Build context from memory
        context = self._build_planning_context()
        
        # Use lead researcher to plan tasks
        lead_agent = Agent(
            name="LeadResearcher",
            instruction=LEAD_RESEARCHER_PROMPT,
        )
        lead_llm = self.llm_factory(lead_agent)
        
        prompt = f"""
        Based on our progress so far, plan the next batch of tasks.
        
        {context}
        
        Create 1-{self._current_memory.strategy.parallelism_level} focused tasks that:
        1. Build on completed work
        2. Fill identified gaps
        3. Make concrete progress
        4. Can potentially run in parallel
        
        For each task, be specific about the objective and what MCP servers are needed.
        """
        
        # Get task specifications
        from pydantic import BaseModel, Field
        
        class TaskSpec(BaseModel):
            objective: str
            description: str
            required_servers: List[str]
            depends_on: List[str] = Field(default_factory=list)
            estimated_iterations: int = Field(default=5, le=10)
        
        class TaskPlan(BaseModel):
            tasks: List[TaskSpec] = Field(max_items=10)
            rationale: str
            can_parallelize: bool = True
        
        plan = await lead_llm.generate_structured(
            message=prompt,
            response_model=TaskPlan,
            request_params=RequestParams(
                model_preferences=ModelPreferences(intelligence=0.8, speed=0.2)
            ),
        )
        
        # Convert to SubagentTasks
        tasks = []
        for i, spec in enumerate(plan.tasks):
            # Create appropriate subagent spec based on task type
            agent_spec = await self._create_subagent_spec(
                spec.objective,
                spec.description,
                spec.required_servers,
                spec.estimated_iterations,
            )
            
            task = SubagentTask(
                task_id=f"{self._current_workflow_id}_task_{self._current_memory.iterations}_{i}",
                description=spec.description,
                objective=spec.objective,
                agent_spec=agent_spec,
                dependencies=spec.depends_on,
                context=self._get_relevant_context(spec.objective),
            )
            tasks.append(task)
        
        # Estimate time for the batch
        estimated_time = len(tasks) * 60.0  # Simple estimate, could be refined
        
        return TaskBatch(
            tasks=tasks,
            rationale=plan.rationale,
            estimated_time=estimated_time,
            can_parallelize=plan.can_parallelize,
        )
    
    async def _create_subagent_spec(
        self,
        objective: str,
        description: str,
        required_servers: List[str],
        estimated_iterations: int,
    ) -> SubagentSpec:
        """Create a subagent specification"""
        # Determine template based on task type
        if self._current_memory.task_type == TaskType.RESEARCH:
            template = RESEARCH_SUBAGENT_TEMPLATE
        elif self._current_memory.task_type == TaskType.ACTION:
            template = ACTION_SUBAGENT_TEMPLATE
        else:
            template = HYBRID_SUBAGENT_TEMPLATE
        
        # Format instruction
        instruction = template.format(
            objective=objective,
            instructions=description,
            tools=", ".join(required_servers) if required_servers else "standard tools",
        )
        
        # Create spec
        return SubagentSpec(
            name=f"Subagent_{objective[:30].replace(' ', '_')}",
            instruction=instruction,
            server_names=required_servers,
            expected_iterations=estimated_iterations,
            parallel_tools=True,
            timeout_seconds=300,  # 5 minutes default
        )
    
    async def _execute_parallel(
        self,
        tasks: List[SubagentTask],
        request_params: Optional[RequestParams],
        span: trace.Span,
    ) -> None:
        """Execute tasks in parallel"""
        span.add_event("parallel_execution_start", {"task_count": len(tasks)})
        
        # Create all agents
        agents = []
        for task in tasks:
            agent = Agent(
                name=task.agent_spec.name,
                instruction=task.agent_spec.instruction,
                server_names=task.agent_spec.server_names,
                context=self.context,
            )
            agents.append((task, agent))
        
        # Execute all tasks concurrently
        async with asyncio.TaskGroup() as tg:
            for task, agent in agents:
                tg.create_task(self._execute_single_task(task, agent, request_params, span))
    
    async def _execute_sequential(
        self,
        tasks: List[SubagentTask],
        request_params: Optional[RequestParams],
        span: trace.Span,
    ) -> None:
        """Execute tasks sequentially"""
        span.add_event("sequential_execution_start", {"task_count": len(tasks)})
        
        for task in tasks:
            # Check dependencies
            if not self._dependencies_met(task):
                logger.warning(f"Skipping task {task.task_id} due to unmet dependencies")
                task.status = "skipped"
                continue
            
            agent = Agent(
                name=task.agent_spec.name,
                instruction=task.agent_spec.instruction,
                server_names=task.agent_spec.server_names,
                context=self.context,
            )
            
            await self._execute_single_task(task, agent, request_params, span)
    
    async def _execute_single_task(
        self,
        task: SubagentTask,
        agent: Agent,
        request_params: Optional[RequestParams],
        span: trace.Span,
    ) -> None:
        """Execute a single task with an agent"""
        task_span = tracer.start_as_current_span(
            f"task_{task.task_id}",
            attributes={
                "task.id": task.task_id,
                "task.objective": task.objective[:200],
                "task.agent": agent.name,
            }
        )
        
        try:
            async with agent:
                task.start_time = datetime.now()
                task.status = "running"
                self._current_memory.total_subagents_created += 1
                
                # Create LLM for the agent
                llm = await agent.attach_llm(self.llm_factory)
                
                # Set task-specific parameters
                task_params = RequestParams(
                    max_iterations=task.agent_spec.expected_iterations,
                    parallel_tool_calls=task.agent_spec.parallel_tools,
                    model_preferences=self._get_model_preferences_for_task(task),
                )
                
                # Merge with provided params
                if request_params:
                    task_params = task_params.model_copy(update=request_params.model_dump(exclude_unset=True))
                
                # Execute with timeout
                try:
                    result = await asyncio.wait_for(
                        llm.generate_str(
                            message=f"{task.objective}\n\nContext:\n{task.context}",
                            request_params=task_params,
                        ),
                        timeout=task.agent_spec.timeout_seconds,
                    )
                    
                    task.result = result
                    task.status = "completed"
                    
                    # Extract any citations if present
                    # This is a simplified extraction - could be more sophisticated
                    if "[" in result and "](" in result:
                        task.citations = self._extract_citations(result)
                    
                    # Update key findings
                    self._current_memory.key_findings.append(
                        f"{task.objective}: {result[:200]}..."
                    )
                    
                except asyncio.TimeoutError:
                    task.status = "failed"
                    task.error = f"Timeout after {task.agent_spec.timeout_seconds}s"
                    logger.error(f"Task {task.task_id} timed out")
                
                task.end_time = datetime.now()
                
                # Get token usage from span if available
                if hasattr(task_span, "get_attribute"):
                    task.input_tokens = task_span.get_attribute(GEN_AI_USAGE_INPUT_TOKENS) or 0
                    task.output_tokens = task_span.get_attribute(GEN_AI_USAGE_OUTPUT_TOKENS) or 0
                    # Estimate cost (simplified - would need actual model pricing)
                    task.cost = (task.input_tokens / 1000) * 0.01 + (task.output_tokens / 1000) * 0.03
                
                # Update memory
                self._current_memory.completed_tasks.append(task)
                self._current_memory.total_input_tokens += task.input_tokens
                self._current_memory.total_output_tokens += task.output_tokens
                self._current_memory.total_cost += task.cost
                
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.end_time = datetime.now()
            logger.error(f"Task {task.task_id} failed: {e}")
            task_span.record_exception(e)
            task_span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            task_span.end()
    
    async def _evaluate_progress(self, span: trace.Span) -> ProgressEvaluation:
        """Evaluate progress toward objective"""
        evaluator_agent = Agent(
            name="ProgressEvaluator",
            instruction=PROGRESS_EVALUATOR_PROMPT,
        )
        evaluator_llm = self.llm_factory(evaluator_agent)
        
        context = f"""
        Original objective: {self._current_memory.objective}
        
        Completed tasks: {len(self._current_memory.completed_tasks)}
        Failed tasks: {sum(1 for t in self._current_memory.completed_tasks if t.status == "failed")}
        
        Key findings:
        {self._format_key_findings()}
        
        Time elapsed: {(datetime.now() - self._current_memory.start_time).total_seconds() / 60:.1f} minutes
        Cost so far: ${self._current_memory.total_cost:.2f}
        """
        
        evaluation = await evaluator_llm.generate_structured(
            message=context,
            response_model=ProgressEvaluation,
            request_params=RequestParams(
                model_preferences=ModelPreferences(speed=0.7, intelligence=0.3)
            ),
        )
        
        span.add_event("progress_evaluated", {
            "is_complete": evaluation.is_complete,
            "confidence": evaluation.confidence,
            "should_pivot": evaluation.should_pivot,
        })
        
        return evaluation
    
    async def _synthesize_results(self, span: trace.Span) -> WorkflowResult:
        """Synthesize all results into final output"""
        synthesizer_agent = Agent(
            name="ResultSynthesizer",
            instruction=RESULT_SYNTHESIZER_PROMPT,
        )
        synthesizer_llm = self.llm_factory(synthesizer_agent)
        
        # Gather all successful task results
        successful_tasks = [t for t in self._current_memory.completed_tasks if t.status == "completed"]
        
        context = f"""
        Original objective: {self._current_memory.objective}
        
        Strategy used: {self._current_memory.strategy.approach}
        
        Task results:
        {self._format_task_results(successful_tasks)}
        
        Create a comprehensive response that directly addresses the objective.
        """
        
        final_result = await synthesizer_llm.generate_str(
            message=context,
            request_params=RequestParams(
                model_preferences=ModelPreferences(intelligence=0.9, cost=0.1)
            ),
        )
        
        # Calculate final metrics
        total_time = (datetime.now() - self._current_memory.start_time).total_seconds()
        
        # Determine success and limitations
        failed_tasks = [t for t in self._current_memory.completed_tasks if t.status == "failed"]
        limitations = []
        if failed_tasks:
            limitations.append(f"{len(failed_tasks)} subtasks failed to complete")
        if self._current_memory.total_cost > self.cost_budget * 0.9:
            limitations.append("Approaching cost budget limit")
        if total_time > self.time_budget.total_seconds() * 0.9:
            limitations.append("Approaching time budget limit")
        
        result = WorkflowResult(
            workflow_id=self._current_workflow_id,
            objective=self._current_memory.objective,
            task_type=self._current_memory.task_type,
            result=final_result,
            citations=self._current_memory.all_citations,
            confidence=0.9 if not failed_tasks else 0.7,
            tasks_completed=len(successful_tasks),
            tasks_failed=len(failed_tasks),
            subagents_used=self._current_memory.total_subagents_created,
            total_time_seconds=total_time,
            iterations=self._current_memory.iterations,
            total_input_tokens=self._current_memory.total_input_tokens,
            total_output_tokens=self._current_memory.total_output_tokens,
            total_cost=self._current_memory.total_cost,
            success=len(failed_tasks) == 0,
            limitations=limitations,
            metadata={
                "strategy": self._current_memory.strategy.approach,
                "complexity": self._current_memory.complexity,
            }
        )
        
        span.add_event("results_synthesized", {
            "result_length": len(final_result),
            "confidence": result.confidence,
        })
        
        return result
    
    async def _format_citations(self, result: WorkflowResult, span: trace.Span) -> WorkflowResult:
        """Format citations in the result"""
        if not result.citations:
            return result
        
        citation_agent = Agent(
            name="CitationFormatter",
            instruction=CITATION_FORMATTER_PROMPT,
        )
        citation_llm = self.llm_factory(citation_agent)
        
        # Prepare citations for formatting
        citation_list = "\n".join([
            f"- {c.title} ({c.url or 'no url'}): {c.content_snippet[:100]}..."
            for c in result.citations
        ])
        
        formatted_result = await citation_llm.generate_str(
            message=f"""
            Format citations in this text:
            
            {result.result}
            
            Available sources:
            {citation_list}
            """,
            request_params=RequestParams(
                model_preferences=ModelPreferences(speed=0.8, intelligence=0.2)
            ),
        )
        
        result.result = formatted_result
        span.add_event("citations_formatted", {"citation_count": len(result.citations)})
        
        return result
    
    # Helper methods
    def _should_continue(self, start_time: float) -> bool:
        """Check if workflow should continue"""
        # Time check
        elapsed = time.time() - start_time
        if elapsed > self.time_budget.total_seconds():
            logger.info("Time budget exceeded")
            return False
        
        # Iteration check
        if self._current_memory.iterations >= self.max_iterations:
            logger.info("Maximum iterations reached")
            return False
        
        # Cost check
        if self._current_memory.total_cost > self.cost_budget:
            logger.info("Cost budget exceeded")
            return False
        
        # Subagent check
        if self._current_memory.total_subagents_created >= self.max_subagents:
            logger.info("Maximum subagents reached")
            return False
        
        return True
    
    def _check_resource_limits(self, task_batch: TaskBatch, span: trace.Span) -> bool:
        """Check if executing task batch would exceed limits"""
        # Estimate resource usage
        estimated_subagents = self._current_memory.total_subagents_created + len(task_batch.tasks)
        estimated_cost = self._current_memory.total_cost + (len(task_batch.tasks) * 0.5)  # Rough estimate
        
        if estimated_subagents > self.max_subagents:
            span.add_event("resource_limit_subagents", {
                "current": self._current_memory.total_subagents_created,
                "would_be": estimated_subagents,
                "limit": self.max_subagents,
            })
            return False
        
        if estimated_cost > self.cost_budget:
            span.add_event("resource_limit_cost", {
                "current": self._current_memory.total_cost,
                "would_be": estimated_cost,
                "limit": self.cost_budget,
            })
            return False
        
        return True
    
    def _dependencies_met(self, task: SubagentTask) -> bool:
        """Check if task dependencies are met"""
        if not task.dependencies:
            return True
        
        completed_ids = {t.task_id for t in self._current_memory.completed_tasks if t.status == "completed"}
        return all(dep_id in completed_ids for dep_id in task.dependencies)
    
    def _build_planning_context(self) -> str:
        """Build context for planning next tasks"""
        # Compress memory if needed
        self.memory_manager.compress_memory(self._current_memory)
        
        context_parts = [
            f"Objective: {self._current_memory.objective}",
            f"Strategy: {self._current_memory.strategy.approach}",
            f"Phase: {self._current_memory.phase}",
            f"Iterations: {self._current_memory.iterations}",
            f"Progress: {len(self._current_memory.completed_tasks)} tasks completed",
            "",
            "Recent findings:",
            self._format_key_findings(max_findings=10),
            "",
            "Available MCP servers:",
            ", ".join(self.available_servers),
        ]
        
        return "\n".join(context_parts)
    
    def _get_relevant_context(self, objective: str) -> str:
        """Get relevant context for a specific task"""
        # Find related findings
        relevant_findings = []
        objective_lower = objective.lower()
        
        for finding in self._current_memory.key_findings[-20:]:  # Last 20 findings
            if any(word in finding.lower() for word in objective_lower.split()):
                relevant_findings.append(finding)
        
        if relevant_findings:
            return "Related findings:\n" + "\n".join(f"- {f}" for f in relevant_findings[:5])
        else:
            return "No directly related findings yet."
    
    def _get_model_preferences_for_task(self, task: SubagentTask) -> ModelPreferences:
        """Get appropriate model preferences for a task"""
        # Research tasks need more intelligence
        if self._current_memory.task_type == TaskType.RESEARCH:
            return ModelPreferences(intelligence=0.7, speed=0.2, cost=0.1)
        # Action tasks need speed
        elif self._current_memory.task_type == TaskType.ACTION:
            return ModelPreferences(speed=0.7, intelligence=0.2, cost=0.1)
        # Default balanced
        else:
            return ModelPreferences(intelligence=0.5, speed=0.3, cost=0.2)
    
    def _extract_citations(self, text: str) -> List[CitationSource]:
        """Extract citations from text (simplified)"""
        citations = []
        # This is a very simple extraction - in practice would be more sophisticated
        import re
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(pattern, text)
        
        for title, url in matches:
            citation = CitationSource(
                title=title,
                url=url,
                content_snippet=text[:100],  # Just a snippet
                relevance_score=0.8,  # Would calculate based on context
            )
            citations.append(citation)
            self._current_memory.all_citations.append(citation)
        
        return citations
    
    def _format_key_findings(self, max_findings: int = 20) -> str:
        """Format key findings for display"""
        findings = self._current_memory.key_findings[-max_findings:]
        if not findings:
            return "No key findings yet."
        return "\n".join(f"- {finding}" for finding in findings)
    
    def _format_task_results(self, tasks: List[SubagentTask]) -> str:
        """Format task results for synthesis"""
        if not tasks:
            return "No completed tasks."
        
        results = []
        for task in tasks:
            result_preview = task.result[:500] if task.result else "No result"
            results.append(f"""
Task: {task.objective}
Status: {task.status}
Result: {result_preview}...
Citations: {len(task.citations)} sources
---""")
        
        return "\n".join(results)
    
    def _extract_objective(self, message: Any) -> str:
        """Extract objective string from message"""
        if isinstance(message, str):
            return message
        elif isinstance(message, list):
            # Extract text from message params
            for msg in message:
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, str):
                        return msg.content
                    elif isinstance(msg.content, list):
                        for content in msg.content:
                            if hasattr(content, 'text'):
                                return content.text
        return str(message)
    
    def _messages_to_string(self, messages: List[MessageT]) -> str:
        """Convert messages to string"""
        parts = []
        for msg in messages:
            if hasattr(msg, 'content'):
                if isinstance(msg.content, str):
                    parts.append(msg.content)
                elif isinstance(msg.content, list):
                    for content in msg.content:
                        if isinstance(content, TextContent):
                            parts.append(content.text)
        return "\n".join(parts)
    
    def _format_result_as_messages(self, result: WorkflowResult) -> List[MessageT]:
        """Format workflow result as messages"""
        # Return the synthesized result as a message
        # The actual MessageT type would depend on the LLM provider
        return [result.result]  # Simplified - would create proper message objects