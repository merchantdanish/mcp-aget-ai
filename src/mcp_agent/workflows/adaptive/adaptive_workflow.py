"""
Adaptive Workflow - Following Deep Research Architecture
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Type, Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    MessageTypes,
    ModelT,
    RequestParams,
)
from mcp.types import ModelPreferences
from mcp_agent.tracing.telemetry import get_tracer
from mcp_agent.logging.logger import get_logger

from mcp_agent.workflows.adaptive.models import (
    TaskType,
    ResearchAspect,
    SubagentResult,
    SynthesisDecision,
    ExecutionMemory,
    ExecutionResult,
)
from mcp_agent.workflows.adaptive.memory import MemoryManager
from mcp_agent.workflows.adaptive.prompts import (
    LEAD_RESEARCHER_ANALYZE_PROMPT,
    LEAD_RESEARCHER_PLAN_PROMPT,
    LEAD_RESEARCHER_SYNTHESIZE_PROMPT,
    LEAD_RESEARCHER_DECIDE_PROMPT,
    RESEARCH_SUBAGENT_TEMPLATE,
    ACTION_SUBAGENT_TEMPLATE,
    FINAL_REPORT_PROMPT,
)

logger = get_logger(__name__)


class AdaptiveWorkflow(AugmentedLLM[MessageParamT, MessageT]):
    """
    Adaptive workflow following the Deep Research architecture.

    Key principles:
    - Lead Researcher maintains control throughout
    - Subagents created dynamically for specific aspects
    - Iterative refinement until objective is met
    - Clear synthesis and decision phases
    """

    def __init__(
        self,
        llm_factory: Callable[[Agent], AugmentedLLM[MessageParamT, MessageT]],
        name: str | None = None,
        available_agents: List[Agent | AugmentedLLM] | None = None,
        available_servers: List[str] | None = None,
        time_budget: timedelta = timedelta(minutes=30),
        cost_budget: float = 10.0,
        max_iterations: int = 10,
        enable_parallel: bool = True,
        memory_manager: Optional[MemoryManager] = None,
        model_preferences: Optional[ModelPreferences] = None,
        context: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize Adaptive Workflow V2"""
        super().__init__(
            name=name or "AdaptiveWorkflowV2",
            instruction="I am a lead researcher coordinating investigation into your objective.",
            context=context,
            **kwargs,
        )

        self.llm_factory = llm_factory
        self.available_agents = {}
        # Process available agents - handle both Agent and AugmentedLLM instances
        for agent in available_agents or []:
            if isinstance(agent, Agent):
                self.available_agents[agent.name] = agent
            elif hasattr(agent, "name"):
                # It's an AugmentedLLM instance
                self.available_agents[agent.name] = agent

        self.available_servers = available_servers or []
        self.time_budget = time_budget
        self.cost_budget = cost_budget
        self.max_iterations = max_iterations
        self.enable_parallel = enable_parallel
        self.memory_manager = memory_manager or MemoryManager()
        self.model_preferences = model_preferences

        # Track current workflow
        self._current_execution_id: Optional[str] = None
        self._current_memory: Optional[ExecutionMemory] = None

    async def generate(
        self,
        message: MessageTypes,
        request_params: RequestParams | None = None,
    ) -> List[MessageT]:
        """Execute the adaptive workflow"""
        tracer = get_tracer(self.context)

        with tracer.start_as_current_span(
            f"{self.__class__.__name__}.generate"
        ) as span:
            try:
                # Extract objective
                objective = await self._extract_objective(message)
                span.set_attribute("workflow.objective", objective[:200])

                # Execute workflow
                result = await self._execute_workflow(objective, request_params, span)

                # Record metrics
                span.set_attribute("workflow.success", result.success)
                span.set_attribute("workflow.iterations", result.iterations)
                span.set_attribute("workflow.total_cost", result.total_cost)

                return self._format_result_as_messages(result)

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    async def _execute_workflow(
        self,
        objective: str,
        request_params: Optional[RequestParams],
        span: trace.Span,
    ) -> ExecutionResult:
        """Main workflow execution following Deep Research pattern"""
        execution_id = str(uuid.uuid4())
        self._current_execution_id = execution_id

        # Initialize memory
        self._current_memory = ExecutionMemory(
            execution_id=execution_id,
            objective=objective,
            start_time=datetime.now(),
        )

        # Phase 1: Initial Analysis
        span.add_event("phase_1_initial_analysis")
        task_type = await self._analyze_objective(objective, span)
        self._current_memory.task_type = task_type

        # Main iterative loop
        start_time = time.time()

        while self._should_continue(start_time):
            iteration = self._current_memory.iterations + 1
            self._current_memory.iterations = iteration

            span.add_event(
                f"iteration_{iteration}_start",
                {
                    "research_conducted": len(self._current_memory.research_history),
                    "cost_so_far": self._current_memory.total_cost,
                },
            )

            # Phase 2: Plan Research
            aspects = await self._plan_research(span)
            if not aspects:
                logger.info("No more aspects to research")
                break

            # Phase 3: Execute Research
            results = await self._execute_research(aspects, request_params, span)

            # Phase 4: Synthesize Results
            synthesis_messages = await self._synthesize_results(results, span)
            self._current_memory.research_history.append(synthesis_messages)

            # Phase 5: Decide Next Steps
            decision = await self._decide_next_steps(synthesis_messages, span)

            if decision.is_complete:
                logger.info("Research objective achieved")
                break

            if decision.new_aspects:
                logger.info(
                    f"Identified {len(decision.new_aspects)} new aspects to research"
                )

            # Save checkpoint
            await self.memory_manager.save_memory(self._current_memory)

        # Phase 6: Generate Final Report
        span.add_event("phase_6_final_report")
        final_report = await self._generate_final_report(span)

        # Create result
        total_time = time.time() - start_time
        result = ExecutionResult(
            execution_id=execution_id,
            objective=objective,
            task_type=task_type,
            result_messages=final_report,
            confidence=0.9,
            iterations=self._current_memory.iterations,
            subagents_used=len(self._current_memory.subagent_results),
            total_time_seconds=total_time,
            total_cost=self._current_memory.total_cost,
            success=True,
        )

        return result

    async def _analyze_objective(self, objective: str, span: trace.Span) -> TaskType:
        """Analyze the objective to determine task type"""
        # Create analysis agent
        analyzer = Agent(
            name="ObjectiveAnalyzer",
            instruction=LEAD_RESEARCHER_ANALYZE_PROMPT,
        )
        llm = self.llm_factory(analyzer)

        from pydantic import BaseModel

        class Analysis(BaseModel):
            task_type: TaskType
            key_aspects: List[str]
            estimated_scope: str

        analysis = await llm.generate_structured(
            message=f"Analyze this objective: {objective}",
            response_model=Analysis,
        )

        span.add_event("objective_analyzed", {"task_type": analysis.task_type})
        return analysis.task_type

    async def _plan_research(self, span: trace.Span) -> List[ResearchAspect]:
        """Plan what aspects to research next"""
        # Build context from memory
        context = self._build_context()

        # Use lead researcher to identify aspects
        lead_agent = Agent(
            name="LeadResearcher",
            instruction=LEAD_RESEARCHER_PLAN_PROMPT,
        )
        llm = self.llm_factory(lead_agent)

        from pydantic import BaseModel, Field

        class ResearchPlan(BaseModel):
            aspects: List[ResearchAspect] = Field(max_length=5)
            rationale: str

        plan = await llm.generate_structured(
            message=f"{context}\n\nWhat aspects should we investigate next?",
            response_model=ResearchPlan,
        )

        span.add_event("research_planned", {"aspects": len(plan.aspects)})
        return plan.aspects

    async def _execute_research(
        self,
        aspects: List[ResearchAspect],
        request_params: Optional[RequestParams],
        span: trace.Span,
    ) -> List[SubagentResult]:
        """Execute research using subagents"""
        span.add_event("research_execution_start", {"aspects": len(aspects)})

        if self.enable_parallel and len(aspects) > 1:
            # Execute in parallel
            tasks = []
            for aspect in aspects:
                task = self._execute_single_aspect(aspect, request_params, span)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
        else:
            # Execute sequentially
            results = []
            for aspect in aspects:
                result = await self._execute_single_aspect(aspect, request_params, span)
                results.append(result)

        return results

    async def _execute_single_aspect(
        self,
        aspect: ResearchAspect,
        request_params: Optional[RequestParams],
        span: trace.Span,
    ) -> SubagentResult:
        """Execute research for a single aspect"""

        # Check if we should use a predefined agent
        if (
            aspect.use_predefined_agent
            and aspect.use_predefined_agent in self.available_agents
        ):
            agent = self.available_agents[aspect.use_predefined_agent]
            logger.info(f"Using predefined agent: {aspect.use_predefined_agent}")

            # For predefined agents, we'll provide the specific objective
            # but let them use their existing instruction and servers
            use_predefined = True
        else:
            # Create new agent as before
            # Select template based on task type
            if self._current_memory.task_type == TaskType.RESEARCH:
                template = RESEARCH_SUBAGENT_TEMPLATE
            else:
                template = ACTION_SUBAGENT_TEMPLATE

            # Create instruction
            instruction = template.format(
                aspect=aspect.name,
                objective=aspect.objective,
                tools=", ".join(aspect.required_servers)
                if aspect.required_servers
                else "standard tools",
            )

            # Create subagent
            agent = Agent(
                name=f"Subagent_{aspect.name.replace(' ', '_')}",
                instruction=instruction,
                server_names=aspect.required_servers,
                context=self.context,
            )
            use_predefined = False

        result = SubagentResult(
            aspect_name=aspect.name,
            start_time=datetime.now(),
        )

        try:
            # Handle predefined agents differently
            if use_predefined and isinstance(agent, AugmentedLLM):
                # Predefined agent is already an AugmentedLLM
                llm = agent

                # Execute with limited iterations
                params = RequestParams(
                    max_iterations=5,  # Non-cascading limit
                    parallel_tool_calls=True,
                )
                if request_params:
                    params = params.model_copy(
                        update=request_params.model_dump(exclude_unset=True)
                    )

                # Execute research with the specific objective
                response = await llm.generate_str(
                    message=f"Please help with the following task: {aspect.objective}",
                    request_params=params,
                )

                result.findings = response
                result.success = True
                result.end_time = datetime.now()
                result.cost = 0.5  # Would calculate from actual usage

            else:
                # Handle both new agents and predefined Agent instances
                async with agent:
                    if isinstance(agent, Agent):
                        llm = await agent.attach_llm(self.llm_factory)
                    else:
                        # This shouldn't happen, but handle gracefully
                        llm = agent

                    # Execute with limited iterations
                    params = RequestParams(
                        max_iterations=5,  # Non-cascading limit
                        parallel_tool_calls=True,
                    )
                    if request_params:
                        params = params.model_copy(
                            update=request_params.model_dump(exclude_unset=True)
                        )

                    # Execute research
                    response = await llm.generate_str(
                        message=f"Research: {aspect.objective}",
                        request_params=params,
                    )

                    result.findings = response
                    result.success = True
                    result.end_time = datetime.now()

                    # Simple cost estimate
                    result.cost = 0.5  # Would calculate from actual usage

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.end_time = datetime.now()
            logger.error(f"Subagent {aspect.name} failed: {e}")

        # Update memory
        self._current_memory.subagent_results.append(result)
        self._current_memory.total_cost += result.cost

        return result

    async def _synthesize_results(
        self,
        results: List[SubagentResult],
        span: trace.Span,
    ) -> List[MessageT]:
        """Synthesize results from subagents"""
        # Create synthesis agent
        synthesizer = Agent(
            name="Synthesizer",
            instruction=LEAD_RESEARCHER_SYNTHESIZE_PROMPT,
        )
        llm = self.llm_factory(synthesizer)

        # Format results
        results_text = "\n\n".join(
            [
                f"**{r.aspect_name}**:\n{r.findings}"
                for r in results
                if r.success and r.findings
            ]
        )

        synthesis_messages = await llm.generate(
            message=f"Synthesize these research findings:\n\n{results_text}",
        )

        span.add_event("results_synthesized")
        return synthesis_messages

    async def _decide_next_steps(
        self,
        synthesis_messages: List[MessageT],
        span: trace.Span,
    ) -> SynthesisDecision:
        """Decide whether to continue research or conclude"""
        # Create decision agent
        decider = Agent(
            name="DecisionMaker",
            instruction=LEAD_RESEARCHER_DECIDE_PROMPT,
        )
        llm = self.llm_factory(decider)

        # Get synthesis as string - just the last message (assistant's synthesis)
        synthesis_str = ""
        if synthesis_messages:
            synthesis_str = llm.message_str(synthesis_messages[-1], content_only=True)

        context = f"""
        Objective: {self._current_memory.objective}
        Iterations completed: {self._current_memory.iterations}
        
        Latest synthesis:
        {synthesis_str}
        
        Previous research:
        {self._format_research_history()}
        """

        decision = await llm.generate_structured(
            message=context,
            response_model=SynthesisDecision,
        )

        span.add_event(
            "decision_made",
            {
                "is_complete": decision.is_complete,
                "confidence": decision.confidence,
                "new_aspects": len(decision.new_aspects) if decision.new_aspects else 0,
            },
        )

        return decision

    async def _generate_final_report(self, span: trace.Span) -> List[MessageT]:
        """Generate the final report"""
        # Create report writer
        writer = Agent(
            name="ReportWriter",
            instruction=FINAL_REPORT_PROMPT,
        )
        llm = self.llm_factory(writer)

        # Build comprehensive context
        context = f"""
        Objective: {self._current_memory.objective}
        Task Type: {self._current_memory.task_type}
        
        Research conducted ({self._current_memory.iterations} iterations):
        {self._format_research_history()}
        
        All findings:
        {self._format_all_findings()}
        """

        report_messages = await llm.generate(
            message=f"Generate a comprehensive report based on this research:\n\n{context}",
        )

        return report_messages

    # Helper methods
    def _should_continue(self, start_time: float) -> bool:
        """Check if workflow should continue"""
        # Time check
        if time.time() - start_time > self.time_budget.total_seconds():
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

        return True

    def _build_context(self) -> str:
        """Build context for planning"""
        # Format available agents if any
        agents_info = ""
        if self.available_agents:
            agents_info = "\n\nAvailable Agents:\n" + "\n".join(
                [
                    f"- {self._format_agent_info(agent_name)}"
                    for agent_name in self.available_agents.keys()
                ]
            )

        return f"""
        Objective: {self._current_memory.objective}
        Task Type: {self._current_memory.task_type}
        Iterations: {self._current_memory.iterations}
        
        Research conducted:
        {self._format_research_history()}
        
        Available MCP servers: {", ".join(self.available_servers)}
        {agents_info}
        """

    def _format_research_history(self) -> str:
        """Format research history"""
        if not self._current_memory.research_history:
            return "No research conducted yet."

        # We need an LLM instance to convert messages to strings
        # Use a simple agent for this
        formatter = Agent(
            name="HistoryFormatter",
            instruction="Format research history",
            context=self.context,
        )
        llm = self.llm_factory(formatter)

        history_parts = []
        for i, synthesis_messages in enumerate(self._current_memory.research_history):
            if synthesis_messages:
                # Convert the last message (synthesis) to string
                synthesis_str = llm.message_str(
                    synthesis_messages[-1], content_only=True
                )
                history_parts.append(f"Iteration {i + 1}:\n{synthesis_str}")

        return "\n\n".join(history_parts)

    def _format_all_findings(self) -> str:
        """Format all subagent findings"""
        findings = []
        for result in self._current_memory.subagent_results:
            if result.success and result.findings:
                findings.append(f"**{result.aspect_name}**:\n{result.findings}")

        return "\n\n".join(findings)

    async def _extract_objective(self, message: Any) -> str:
        """Extract objective using the LLM for robust parsing"""
        # For simple string messages, return directly
        if isinstance(message, str):
            return message

        # For complex messages, use LLM to extract the objective
        # This handles provider-specific message formats gracefully
        try:
            # Create a simple extraction agent
            extractor = Agent(
                name="ObjectiveExtractor",
                instruction="Extract and return the user's request as a clear objective statement.",
                context=self.context,
            )

            llm = self.llm_factory(extractor)

            # Just get the objective as a string - let the Lead Researcher handle any complexity
            objective = await llm.generate_str(
                message=f"What is the user asking for in this message?\n\n{message}",
                request_params=RequestParams(max_iterations=1),
            )

            return objective

        except Exception as e:
            logger.warning(f"Failed to extract objective using LLM: {e}")
            # Fallback to string conversion
            return str(message)

    def _format_result_as_messages(self, result: ExecutionResult) -> List[MessageT]:
        """Format workflow result as messages"""
        # Return the stored messages directly
        return result.result_messages

    def _format_agent_info(self, agent_name: str) -> str:
        """Format agent information for display to Lead Researcher"""
        agent = self.available_agents.get(agent_name)
        if not agent:
            return f"{agent_name} (not found)"

        # Get agent details
        if isinstance(agent, AugmentedLLM):
            name = agent.agent.name
            instruction = agent.agent.instruction
            server_names = agent.agent.server_names
        elif isinstance(agent, Agent):
            name = agent.name
            instruction = agent.instruction
            server_names = agent.server_names
        else:
            return f"{agent_name} (unknown type)"

        # Format server list
        servers_str = ", ".join(server_names) if server_names else "no specific servers"

        return f"{name}: {instruction} (servers: {servers_str})"

    async def generate_str(
        self,
        message: MessageTypes,
        request_params: RequestParams | None = None,
    ) -> str:
        """Request an LLM generation and return the string representation of the result"""
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(
            f"{self.__class__.__name__}.{self.name}.generate_str"
        ) as span:
            span.set_attribute("agent.name", self.agent.name)

            params = self.get_request_params(request_params)

            if self.context.tracing_enabled:
                AugmentedLLM.annotate_span_with_request_params(span, params)

            result = await self.generate(
                message=message,
                request_params=params,
            )

            res = str(result[0]) if result else ""
            span.set_attribute("result", res)

            return res

    async def generate_structured(
        self,
        message: MessageTypes,
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        """Request a structured LLM generation and return the result as a Pydantic model."""
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(
            f"{self.__class__.__name__}.{self.name}.generate_structured"
        ) as span:
            span.set_attribute("agent.name", self.agent.name)

            params = self.get_request_params(request_params)

            if self.context.tracing_enabled:
                AugmentedLLM.annotate_span_with_request_params(span, params)

            result_str = await self.generate_str(message=message, request_params=params)

            llm: AugmentedLLM = self.llm_factory(
                agent=Agent(
                    name="Structured Output",
                    instruction="Produce a structured output given a message",
                )
            )

            structured_result = await llm.generate_structured(
                message=result_str,
                response_model=response_model,
                request_params=params,
            )

            if self.context.tracing_enabled:
                try:
                    span.set_attribute(
                        "structured_response_json", structured_result.model_dump_json()
                    )
                except Exception:
                    # Ignore serialization errors
                    pass

            return structured_result
