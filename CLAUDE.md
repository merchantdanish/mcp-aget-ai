# Reliable Conversation Manager (RCM) - Opinionated Design & Implementation Plan

## Executive Summary

The Reliable Conversation Manager (RCM) is an mcp-agent server application that implements research findings from "LLMs Get Lost in Multi-Turn Conversation" to create more reliable multi-turn conversational AI systems. This document provides an opinionated, detail-oriented implementation plan focusing on canonical mcp-agent patterns, debuggability, and testability.

### Core Design Principles

1. **Conversation-as-Workflow**: The entire conversation is a single workflow instance, NOT individual turns
2. **Quality-First**: Every response undergoes mandatory quality evaluation and potential refinement
3. **Fail-Fast**: Detect quality issues early and fix them before they compound
4. **Observable**: Every decision point is logged and traceable
5. **Testable**: Components are isolated with clear interfaces

## Architecture Decisions

### Why mcp-agent?

The mcp-agent framework provides critical abstractions that align perfectly with RCM requirements:

```python
# From examples/basic/mcp_basic_agent/main.py - canonical agent pattern
async with finder_agent:
    logger.info("finder: Connected to server, calling list_tools...")
    result = await finder_agent.list_tools()
    llm = await finder_agent.attach_llm(OpenAIAugmentedLLM)
```

**Decision**: Use mcp-agent's Agent abstraction for ALL LLM interactions, including quality evaluation. This ensures consistent tool access, logging, and error handling.

### Workflow Architecture Pattern

Based on analysis of mcp-agent examples, there are two patterns:

1. **Turn-as-Workflow** (REJECTED):

```python
# From original design doc - this neutralizes Temporal benefits
@app.workflow
class TurnProcessorWorkflow(Workflow[Dict[str, Any]]):
    async def run(self, args: Dict[str, Any]) -> WorkflowResult[Dict[str, Any]]:
        # Process one turn... loses conversation state
```

2. **Conversation-as-Workflow** (ADOPTED):

```python
# From examples/mcp_agent_server/temporal/basic_agent_server.py - pattern we'll extend
@app.workflow
class BasicAgentWorkflow(Workflow[str]):
    @app.workflow_run
    async def run(self, input: str = "What is the Model Context Protocol?") -> WorkflowResult[str]:
        # Maintains state across entire conversation
```

**Decision**: Implement conversation-as-workflow with internal state management and user input waiting.

### Quality Control Architecture

The paper identifies four key failure modes:

1. **Premature Answer Attempts** (39% of failures)
2. **Answer Bloat** (20-300% length increase)
3. **Lost-in-Middle-Turns** (forget middle context)
4. **Unreliability** (112% increase in multi-turn)

**Decision**: Implement mandatory quality pipeline with LLM-as-judge pattern:

```python
# Based on paper's quality dimensions
quality_dimensions = {
    "clarity": "Clear, well-structured response",
    "completeness": "Addresses all user requirements",
    "assumptions": "Minimizes unsupported assumptions (LOWER IS BETTER)",
    "verbosity": "Concise without bloat (LOWER IS BETTER)",
    "premature_attempt": "Boolean - attempted answer without info",
    "middle_turn_reference": "References information from middle turns",
    "requirement_tracking": "Tracks user requirements across turns"
}
```

## Detailed Component Design

### 1. Core Workflow Implementation

```python
# examples/reliable_conversation/src/workflows/conversation_workflow.py
"""
Conversation-as-workflow implementation following mcp-agent patterns.
Based on examples/workflows/workflow_swarm/main.py signal handling patterns.
"""

from mcp_agent.executor.workflow import Workflow, WorkflowResult
from mcp_agent.app import MCPApp
from typing import Dict, Any, Optional, List
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class ConversationMessage:
    """Single message in conversation - matches paper's Message model"""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    turn_number: int = 0

@dataclass
class Requirement:
    """Tracked requirement from paper Section 5.1"""
    id: str
    description: str
    source_turn: int
    status: Literal["pending", "addressed", "confirmed"] = "pending"
    confidence: float = 1.0

@dataclass
class QualityMetrics:
    """From paper Table 1 - all metrics 0-1 scale"""
    clarity: float
    completeness: float
    assumptions: float  # Lower is better
    verbosity: float    # Lower is better
    premature_attempt: bool = False
    middle_turn_reference: float = 0.0
    requirement_tracking: float = 0.0

    @property
    def overall_score(self) -> float:
        """Paper's composite scoring formula"""
        base = (self.clarity + self.completeness + self.middle_turn_reference +
                self.requirement_tracking + (1 - self.assumptions) + (1 - self.verbosity)) / 6
        if self.premature_attempt:
            base *= 0.5  # Heavy penalty from paper
        return base

@dataclass
class ConversationState:
    """Complete conversation state - maintained in workflow"""
    conversation_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    requirements: List[Requirement] = field(default_factory=list)
    consolidated_context: str = ""
    quality_history: List[QualityMetrics] = field(default_factory=list)
    current_turn: int = 0

    # Paper metrics
    first_answer_attempt_turn: Optional[int] = None
    answer_lengths: List[int] = field(default_factory=list)
    consolidation_turns: List[int] = field(default_factory=list)

    # Execution state
    is_temporal_mode: bool = False
    is_active: bool = True

@app.workflow
class ConversationWorkflow(Workflow[Dict[str, Any]]):
    """
    Core conversation workflow implementing paper findings.
    Supports both AsyncIO and Temporal execution modes.
    """

    def __init__(self):
        super().__init__()
        self.state: Optional[ConversationState] = None
        self.config: Optional[ConversationConfig] = None

    @app.workflow_run
    async def run(self, args: Dict[str, Any]) -> WorkflowResult[Dict[str, Any]]:
        """Main conversation loop - handles both execution modes"""

        # Initialize from args
        self.config = ConversationConfig(**args.get("config", {}))

        # Determine execution mode from context
        execution_engine = self.context.config.execution_engine

        if execution_engine == "temporal":
            return await self._run_temporal_conversation(args)
        else:
            return await self._run_asyncio_conversation(args)

    async def _run_asyncio_conversation(self, args: Dict[str, Any]) -> WorkflowResult[Dict[str, Any]]:
        """AsyncIO mode - single turn processing for REPL"""

        # Initialize or restore state
        if "state" in args:
            self.state = ConversationState(**args["state"])
        else:
            self.state = ConversationState(
                conversation_id=args.get("conversation_id", f"rcm_{int(time.time())}"),
                is_temporal_mode=False
            )
            # Add system message on first turn
            await self._add_system_message()

        # Process single turn
        user_input = args["user_input"]
        await self._process_turn(user_input)

        # Return updated state
        return WorkflowResult(
            value={
                "response": self.state.messages[-1].content,
                "state": self.state.__dict__,
                "metrics": self.state.quality_history[-1].__dict__ if self.state.quality_history else {},
                "turn_number": self.state.current_turn
            }
        )

    async def _process_turn(self, user_input: str):
        """
        Process single conversation turn with quality control.
        This is the heart of the RCM implementation.
        """

        # Increment turn counter
        self.state.current_turn += 1

        # Add user message
        self.state.messages.append(
            ConversationMessage(
                role="user",
                content=user_input,
                turn_number=self.state.current_turn
            )
        )

        # Execute quality-controlled response generation
        result = await self.context.executor.execute(
            "process_turn_with_quality",
            {
                "state": self.state.__dict__,
                "config": self.config.__dict__
            }
        )

        # Update state with results
        self.state.messages.append(
            ConversationMessage(
                role="assistant",
                content=result["response"],
                turn_number=self.state.current_turn
            )
        )

        # Update tracked state
        self.state.requirements = [Requirement(**r) for r in result["requirements"]]
        self.state.consolidated_context = result["consolidated_context"]
        self.state.quality_history.append(QualityMetrics(**result["metrics"]))
        self.state.answer_lengths.append(len(result["response"]))

        # Track paper metrics
        if result.get("context_consolidated"):
            self.state.consolidation_turns.append(self.state.current_turn)

        if result["metrics"]["premature_attempt"] and self.state.first_answer_attempt_turn is None:
            self.state.first_answer_attempt_turn = self.state.current_turn
```

### 2. Quality Control Task Implementation

```python
# examples/reliable_conversation/src/tasks/quality_control.py
"""
Core quality control implementation from paper Section 5.4.
Uses mcp-agent task decorators for executor compatibility.
"""

from mcp_agent.app import app
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
import json

# Quality evaluation prompt from paper Appendix
QUALITY_EVALUATOR_PROMPT = """You are an expert evaluator assessing conversation quality based on research findings.

Evaluate responses across these research-backed dimensions:

1. CLARITY (0-1, higher better): Is the response clear, well-structured, and easy to understand?
2. COMPLETENESS (0-1, higher better): Does it appropriately address pending user requirements?
3. ASSUMPTIONS (0-1, LOWER better): Does it make unsupported assumptions about unstated details?
4. VERBOSITY (0-1, LOWER better): Is it unnecessarily long or repetitive? (Research shows 20-300% bloat)
5. PREMATURE_ATTEMPT (boolean): Is this attempting a complete answer without sufficient information?
6. MIDDLE_TURN_REFERENCE (0-1, higher better): Does it reference information from middle conversation turns?
7. REQUIREMENT_TRACKING (0-1, higher better): Does it track and reference user requirements across turns?

Research context: Multi-turn conversations show 39% performance degradation due to instruction forgetting,
answer bloat, premature attempts, and lost-in-middle-turns phenomena."""

@app.workflow_task(name="process_turn_with_quality")
async def process_turn_with_quality(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main turn processing implementing paper's quality refinement methodology.
    From paper Section 5.4.1 - uses real LLMs for requirement extraction, quality evaluation, and response refinement.
    """
    state = ConversationState(**params["state"])
    config = params["config"]

    # Step 1: Extract requirements using LLM (prevents "instruction forgetting")
    requirements = await app.context.executor.execute(
        "extract_requirements_with_llm",
        {
            "messages": [m.__dict__ for m in state.messages],
            "existing_requirements": [r.__dict__ for r in state.requirements]
        }
    )

    # Step 2: Consolidate context if needed (prevents "lost-in-middle-turns")
    consolidated_context = state.consolidated_context
    context_consolidated = False

    if _should_consolidate_context(state, config):
        consolidated_context = await app.context.executor.execute(
            "consolidate_context_with_llm",
            {
                "messages": [m.__dict__ for m in state.messages],
                "requirements": requirements,
                "previous_context": state.consolidated_context
            }
        )
        context_consolidated = True

    # Step 3: Generate response with quality refinement loop
    best_response = ""
    best_metrics = None
    max_attempts = config.get("max_refinement_attempts", 3)

    for attempt in range(max_attempts):
        # Generate response
        response = await app.context.executor.execute(
            "generate_response_with_constraints",
            {
                "messages": [m.__dict__ for m in state.messages],
                "consolidated_context": consolidated_context,
                "requirements": requirements,
                "attempt": attempt,
                "previous_issues": [] if attempt == 0 else best_metrics.get("issues", []),
                "config": config
            }
        )

        # Evaluate quality using LLM
        evaluation = await app.context.executor.execute(
            "evaluate_quality_with_llm",
            {
                "response": response,
                "consolidated_context": consolidated_context,
                "requirements": requirements,
                "turn_number": state.current_turn,
                "conversation_history": [m.__dict__ for m in state.messages]
            }
        )

        metrics = QualityMetrics(**evaluation["metrics"])

        # Track best response
        if best_metrics is None or metrics.overall_score > best_metrics["overall_score"]:
            best_response = response
            best_metrics = {
                "metrics": metrics.__dict__,
                "issues": evaluation.get("issues", []),
                "overall_score": metrics.overall_score
            }

        # Check quality threshold
        quality_threshold = config.get("quality_threshold", 0.8)
        if metrics.overall_score >= quality_threshold:
            break

    return {
        "response": best_response,
        "requirements": requirements,
        "consolidated_context": consolidated_context,
        "context_consolidated": context_consolidated,
        "metrics": best_metrics["metrics"],
        "refinement_attempts": attempt + 1
    }

def _should_consolidate_context(state: ConversationState, config: Dict[str, Any]) -> bool:
    """Determine if context consolidation is needed based on paper findings"""
    consolidation_interval = config.get("consolidation_interval", 3)

    return (
        state.current_turn % consolidation_interval == 0 or  # Every N turns
        len(state.consolidated_context) > 2000 or           # Long context threshold
        state.current_turn == 1                             # Always consolidate first turn
    )
```

### 3. LLM Evaluation Tasks

```python
# examples/reliable_conversation/src/tasks/llm_evaluators.py
"""
LLM-based evaluation tasks implementing paper methodologies.
Each task uses mcp-agent patterns for consistency.
"""

@app.workflow_task(name="evaluate_quality_with_llm")
async def evaluate_quality_with_llm(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM-based quality evaluation implementing paper's quality dimensions.
    From paper Section 5.4.2.
    """
    response = params["response"]
    consolidated_context = params["consolidated_context"]
    requirements = params["requirements"]
    turn_number = params["turn_number"]
    conversation_history = params["conversation_history"]

    # Detect premature attempts based on pending requirements
    pending_reqs = [r for r in requirements if r["status"] == "pending"]
    has_complete_solution_markers = _detect_complete_solution_attempt(response)

    # Create evaluator agent with specialized prompt
    evaluator_agent = Agent(
        name="quality_evaluator",
        instruction=QUALITY_EVALUATOR_PROMPT,
        server_names=[]  # No MCP servers needed for evaluation
    )

    async with evaluator_agent:
        # Get LLM based on config
        llm_class = _get_llm_class(app.context.config.rcm.evaluator_model_provider)
        llm = await evaluator_agent.attach_llm(llm_class)

        evaluation_prompt = f"""Evaluate this conversation response for quality issues identified in research.

RESPONSE TO EVALUATE:
{response}

CONVERSATION CONTEXT:
{consolidated_context}

PENDING REQUIREMENTS:
{json.dumps([r["description"] for r in pending_reqs], indent=2)}

CONVERSATION HISTORY LENGTH: {len(conversation_history)} messages
TURN NUMBER: {turn_number}

ADDITIONAL CONTEXT:
- Has complete solution markers: {has_complete_solution_markers}
- Pending requirements count: {len(pending_reqs)}

Evaluate each dimension carefully and return JSON with exact format:
{{
    "clarity": 0.0-1.0,
    "completeness": 0.0-1.0,
    "assumptions": 0.0-1.0,
    "verbosity": 0.0-1.0,
    "premature_attempt": true/false,
    "middle_turn_reference": 0.0-1.0,
    "requirement_tracking": 0.0-1.0,
    "issues": ["specific issue 1", "specific issue 2"],
    "strengths": ["strength 1", "strength 2"],
    "improvement_suggestions": ["suggestion 1", "suggestion 2"]
}}

Focus on research findings: instruction forgetting, answer bloat, premature attempts, lost-in-middle-turns."""

        try:
            result = await llm.generate_str(evaluation_prompt)

            # Parse JSON response with validation
            data = json.loads(result)

            # Apply paper-based heuristics
            if has_complete_solution_markers and len(pending_reqs) > 2:
                data["premature_attempt"] = True
                data["issues"].append("Complete solution attempt with multiple pending requirements")

            # Apply verbosity penalty for answer bloat
            response_length = len(response)
            if turn_number > 1 and response_length > 500:
                verbosity_penalty = min(0.3, (response_length - 500) / 1000)
                data["verbosity"] = max(0, data["verbosity"] - verbosity_penalty)
                data["issues"].append(f"Response length ({response_length} chars) shows potential answer bloat")

            return {
                "metrics": data,
                "issues": data.get("issues", []),
                "evaluator_raw_response": result
            }

        except Exception as e:
            # Fallback scores if evaluation fails
            app.logger.error(f"Quality evaluation failed: {str(e)}")
            return {
                "metrics": {
                    "clarity": 0.5,
                    "completeness": 0.5,
                    "assumptions": 0.7,
                    "verbosity": 0.6,
                    "premature_attempt": has_complete_solution_markers and len(pending_reqs) > 1,
                    "middle_turn_reference": 0.3,
                    "requirement_tracking": 0.4,
                },
                "issues": [f"Quality evaluation error: {str(e)}"],
                "evaluator_raw_response": str(e)
            }

def _detect_complete_solution_attempt(response: str) -> bool:
    """Detect if response contains markers of complete solution attempts"""
    solution_markers = [
        "here's the complete",
        "here is the full",
        "final solution",
        "complete implementation",
        "this should handle everything",
        "final answer",
        "complete response",
        "here's everything you need"
    ]

    response_lower = response.lower()
    return any(marker in response_lower for marker in solution_markers)

@app.workflow_task(name="extract_requirements_with_llm")
async def extract_requirements_with_llm(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    LLM-based requirement extraction to prevent instruction forgetting.
    From paper Section 5.4.3.
    """
    messages = params["messages"]
    existing_requirements = params["existing_requirements"]

    # Create requirement extraction agent
    extractor_agent = Agent(
        name="requirement_extractor",
        instruction="""You extract and track user requirements across conversation turns to prevent instruction forgetting.

Your task:
1. Identify explicit and implicit user requirements from the conversation
2. Track requirements that span multiple turns
3. Update status of existing requirements based on conversation progress
4. Distinguish between different types of requirements (functional, constraints, preferences)

Focus on preventing the "instruction forgetting" phenomenon identified in research.""",
        server_names=[]
    )

    async with extractor_agent:
        llm = await extractor_agent.attach_llm(_get_llm_class())

        # Build conversation context
        conversation_text = "\n".join([
            f"Turn {msg['turn_number']} ({msg['role']}): {msg['content']}"
            for msg in messages if msg['role'] != 'system'
        ])

        existing_req_text = "\n".join([
            f"- {req['id']}: {req['description']} (Status: {req['status']})"
            for req in existing_requirements
        ])

        extraction_prompt = f"""Analyze this conversation to extract and update user requirements.

CONVERSATION:
{conversation_text}

EXISTING REQUIREMENTS:
{existing_req_text}

Extract requirements and return JSON array with this exact format:
[
    {{
        "id": "existing_id_or_new_uuid",
        "description": "clear requirement description",
        "source_turn": turn_number,
        "status": "pending|addressed|confirmed",
        "confidence": 0.0-1.0
    }}
]

Rules:
1. Update existing requirements if mentioned in latest turns
2. Add new requirements from user messages
3. Mark requirements as "addressed" if assistant has handled them
4. Mark as "confirmed" if user explicitly confirms satisfaction
5. Include both explicit and reasonable implicit requirements
6. Maintain requirement IDs for tracking across turns"""

        try:
            result = await llm.generate_str(extraction_prompt)
            requirements_data = json.loads(result)

            # Validate and add IDs if missing
            import uuid
            for req in requirements_data:
                if "id" not in req or not req["id"]:
                    req["id"] = str(uuid.uuid4())[:8]
                if "confidence" not in req:
                    req["confidence"] = 0.8

            return requirements_data

        except Exception as e:
            app.logger.error(f"Requirement extraction failed: {str(e)}")
            # Preserve existing requirements on failure
            return existing_requirements
```

### 4. Task Handlers

```python
# examples/reliable_conversation/src/tasks/code_handler.py
"""
Code task handler with Claude Code SDK integration option.
Follows mcp-agent patterns from examples.
"""

@app.workflow_task(name="handle_code_task")
async def handle_code_task(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle code-related queries with quality control.
    Can optionally use Claude Code SDK for execution.
    """
    state = params["state"]
    user_input = params["user_input"]
    config = params["config"]

    # Create code specialist agent
    code_agent = Agent(
        name="code_specialist",
        instruction="""You are a code specialist that generates high-quality, production-ready code.
        Follow these principles:
        1. Write clean, documented code
        2. Include error handling
        3. Follow language best practices
        4. Consider edge cases
        5. Make code testable""",
        server_names=["filesystem"]  # Access to codebase
    )

    async with code_agent:
        llm = await code_agent.attach_llm(_get_llm_class())

        # Check if we should use Claude Code SDK
        if config.get("use_claude_code", False) and _is_code_generation_request(user_input):
            response = await _generate_with_claude_code(user_input, state, code_agent)
        else:
            response = await _generate_code_response(user_input, state, llm)

        return {"response": response}

async def _generate_with_claude_code(user_input: str, state: Dict, agent: Agent) -> str:
    """
    Use Claude Code SDK for code generation.
    Based on SDK documentation patterns.
    """
    import subprocess
    import json

    # Prepare prompt with context
    full_prompt = f"""
    {user_input}

    Previous code context:
    {_extract_code_context(state)}

    Requirements to address:
    {_format_requirements(state.get('requirements', []))}
    """

    # Call Claude Code via subprocess
    cmd = [
        "claude", "-p", full_prompt,
        "--output-format", "json",
        "--max-turns", "3",
        "--system-prompt", "You are a senior software engineer. Write clean, production-ready code."
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        response_data = json.loads(result.stdout)

        if response_data.get("is_error"):
            return f"Error generating code: {response_data.get('error_message', 'Unknown error')}"

        return response_data.get("result", "")

    except subprocess.CalledProcessError as e:
        app.logger.error(f"Claude Code SDK error: {e}")
        # Fallback to regular generation
        return await _generate_code_response(user_input, state, agent.llm)

# examples/reliable_conversation/src/tasks/chat_handler.py
"""
Chat task handler with MCP tool integration.
Based on examples/basic/mcp_basic_agent patterns.
"""

@app.workflow_task(name="handle_chat_task")
async def handle_chat_task(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle general chat queries with MCP tool support.
    """
    state = params["state"]
    user_input = params["user_input"]
    config = params["config"]

    # Create chat agent with MCP servers
    chat_agent = Agent(
        name="chat_assistant",
        instruction="""You are a helpful assistant with access to various tools.
        Use tools when they would help answer the user's query more accurately.
        Be concise but complete in your responses.""",
        server_names=config.get("mcp_servers", ["fetch", "filesystem"])
    )

    async with chat_agent:
        llm = await chat_agent.attach_llm(_get_llm_class())

        # List available tools
        tools = await chat_agent.list_tools()
        app.logger.debug(f"Available tools: {[t.name for t in tools.tools]}")

        # Generate response with potential tool use
        response = await llm.generate_str(
            user_input,
            context={
                "available_tools": [t.name for t in tools.tools],
                "conversation_history": _format_conversation_history(state),
                "requirements": state.get("requirements", [])
            }
        )

        # Check for answer bloat
        if _is_bloated_response(response, user_input):
            response = await _reduce_bloat(response, user_input, llm)

        return {"response": response}

def _is_bloated_response(response: str, query: str) -> bool:
    """Detect answer bloat based on paper findings"""
    query_words = len(query.split())
    response_words = len(response.split())

    # Paper shows 20-300% bloat in multi-turn
    return response_words > query_words * 20
```

### 5. Main Application Entry Point

```python
# examples/reliable_conversation/main.py
"""
Main entry point for Reliable Conversation Manager.
Implements REPL with conversation-as-workflow pattern.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_agent.app import MCPApp
from workflows.conversation_workflow import ConversationWorkflow, ConversationConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Create app instance
app = MCPApp(name="reliable_conversation_manager")

# Import all tasks to register them
from tasks import quality_control
from tasks import llm_evaluators
from tasks import code_handler
from tasks import chat_handler

async def run_repl():
    """Run the RCM REPL interface"""

    async with app.run() as rcm_app:
        # Display welcome message
        console.print(Panel.fit(
            "[bold blue]Reliable Conversation Manager[/bold blue]\n\n"
            "Multi-turn chat with quality control based on 'LLMs Get Lost' research\n"
            f"Execution Engine: {rcm_app.context.config.execution_engine}\n\n"
            "Commands: /stats, /requirements, /exit",
            border_style="blue"
        ))

        # Load configuration
        config = ConversationConfig(
            quality_threshold=rcm_app.context.config.get("rcm.quality_threshold", 0.8),
            max_refinement_attempts=rcm_app.context.config.get("rcm.max_refinement_attempts", 3),
            consolidation_interval=rcm_app.context.config.get("rcm.consolidation_interval", 3),
            use_claude_code=rcm_app.context.config.get("rcm.use_claude_code", False)
        )

        # Create workflow instance
        workflow = ConversationWorkflow()
        conversation_state = None

        while True:
            # Get user input
            try:
                user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
            except (EOFError, KeyboardInterrupt):
                break

            # Handle commands
            if user_input.lower() == "/exit":
                break
            elif user_input.lower() == "/stats":
                _display_stats(conversation_state)
                continue
            elif user_input.lower() == "/requirements":
                _display_requirements(conversation_state)
                continue

            # Process turn through workflow
            with console.status("[bold green]Processing...[/bold green]"):
                result = await workflow.run({
                    "user_input": user_input,
                    "state": conversation_state.__dict__ if conversation_state else None,
                    "config": config.__dict__
                })

            # Extract response and state
            response_data = result.value
            conversation_state = ConversationState(**response_data["state"])

            # Display response
            console.print(f"\n[bold green]Assistant:[/bold green] {response_data['response']}")

            # Display quality metrics if verbose
            if rcm_app.context.config.get("rcm.verbose_metrics", False):
                _display_quality_metrics(response_data.get("metrics", {}))

        # Display final summary
        if conversation_state and conversation_state.turns:
            _display_final_summary(conversation_state)

def _display_quality_metrics(metrics: Dict[str, Any]):
    """Display quality metrics in a table"""
    if not metrics:
        return

    table = Table(title="Response Quality Metrics", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")

    for key, value in metrics.items():
        if key != "issues":
            display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
            table.add_row(key.replace("_", " ").title(), display_value)

    console.print(table)

    if metrics.get("issues"):
        console.print("\n[yellow]Issues detected:[/yellow]")
        for issue in metrics["issues"]:
            console.print(f"  • {issue}")

def _display_stats(state: ConversationState):
    """Display conversation statistics"""
    if not state:
        console.print("[yellow]No conversation started yet[/yellow]")
        return

    table = Table(title="Conversation Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Turns", str(state.current_turn))
    table.add_row("Requirements Tracked", str(len(state.requirements)))

    pending = len([r for r in state.requirements if r.status == "pending"])
    table.add_row("Pending Requirements", str(pending))

    if state.quality_history:
        avg_quality = sum(q.overall_score for q in state.quality_history) / len(state.quality_history)
        table.add_row("Average Quality Score", f"{avg_quality:.2f}")

    if state.answer_lengths:
        avg_length = sum(state.answer_lengths) / len(state.answer_lengths)
        table.add_row("Avg Response Length", f"{avg_length:.0f} chars")

        # Check for bloat
        if len(state.answer_lengths) > 2:
            bloat = state.answer_lengths[-1] / state.answer_lengths[0]
            table.add_row("Response Bloat Ratio", f"{bloat:.1f}x")

    console.print(table)

if __name__ == "__main__":
    asyncio.run(run_repl())
```

## Implementation Phases

### Phase 1: Foundation (Week 1)

#### Task 1.1: Project Setup (4 hours)

**Goal**: Create project structure with proper mcp-agent patterns

**Subtasks**:

1. Create directory structure:

   ```
   examples/reliable_conversation/
   ├── src/
   │   ├── workflows/
   │   │   └── conversation_workflow.py
   │   ├── tasks/
   │   │   ├── __init__.py
   │   │   ├── quality_control.py
   │   │   ├── llm_evaluators.py
   │   │   ├── code_handler.py
   │   │   └── chat_handler.py
   │   ├── models/
   │   │   └── conversation_models.py
   │   └── utils/
   │       ├── logging.py
   │       └── config.py
   ├── tests/
   │   ├── unit/
   │   └── integration/
   ├── main.py
   ├── mcp_agent.config.yaml
   ├── requirements.txt
   └── README.md
   ```

2. Set up configuration:

   ```yaml
   # mcp_agent.config.yaml
   execution_engine: asyncio # Change to temporal later

   logger:
     type: file
     level: debug
     path: logs/rcm.jsonl

   mcp:
     servers:
       fetch:
         command: uvx
         args: [mcp-server-fetch]
       filesystem:
         command: npx
         args: [-y, "@modelcontextprotocol/server-filesystem", "./workspace"]

   openai:
     default_model: gpt-4

   anthropic:
     default_model: claude-3-sonnet-20240229

   rcm:
     quality_threshold: 0.8
     max_refinement_attempts: 3
     consolidation_interval: 3
     verbose_metrics: false
     use_claude_code: false
     evaluator_model_provider: openai # or anthropic
   ```

3. Create base models from paper:

   - Copy dataclasses from design (ConversationMessage, Requirement, QualityMetrics, ConversationState)
   - Add validation methods
   - Add serialization/deserialization

4. Set up logging utilities:

   ```python
   # src/utils/logging.py
   from mcp_agent.logging.logger import get_logger

   def get_rcm_logger(name: str):
       """Get logger with RCM-specific formatting"""
       logger = get_logger(f"rcm.{name}")
       # Add conversation ID to all logs
       return logger
   ```

**Success Criteria**:

- Project structure created
- Can import all modules without errors
- Configuration loads correctly
- `python main.py` shows welcome message

#### Task 1.2: Basic Workflow Implementation (8 hours)

**Goal**: Implement ConversationWorkflow with AsyncIO support

**Subtasks**:

1. Implement ConversationWorkflow class:

   - Copy from design doc
   - Add proper error handling
   - Add debug logging at each decision point

2. Implement state management:

   - State serialization/deserialization
   - State validation
   - State history tracking for debugging

3. Create workflow factory:

   ```python
   @app.workflow
   class ConversationWorkflow(Workflow[Dict[str, Any]]):
       # Implementation from design
   ```

4. Implement basic REPL loop:
   - User input handling
   - Basic response generation (no quality control yet)
   - State persistence between turns

**Success Criteria**:

- Can have basic conversation
- State persists between turns
- All interactions logged
- Graceful error handling

### Phase 2: Quality Control (Week 2)

#### Task 2.1: LLM Evaluator Implementation (8 hours)

**Goal**: Implement quality evaluation from paper

**Subtasks**:

1. Implement quality evaluation task:

   - Copy `evaluate_quality_with_llm` from design
   - Add all 7 quality dimensions from paper
   - Implement JSON parsing with fallbacks

2. Create evaluator agents:

   ```python
   evaluator_agent = Agent(
       name="quality_evaluator",
       instruction=QUALITY_EVALUATOR_PROMPT,
       server_names=[]
   )
   ```

3. Implement premature answer detection:

   - Pattern matching for solution markers
   - Requirement counting logic
   - Confidence scoring

4. Add quality metrics tracking:
   - Store all evaluations
   - Calculate running averages
   - Identify quality trends

**Success Criteria**:

- Quality evaluation returns valid scores
- Premature answers detected correctly
- Metrics logged and trackable
- Can identify quality issues

#### Task 2.2: Requirement Tracking (6 hours)

**Goal**: Implement requirement extraction and tracking

**Subtasks**:

1. Implement requirement extraction task:

   - Copy `extract_requirements_with_llm` from design
   - Add requirement deduplication
   - Implement confidence scoring

2. Create requirement status updates:

   - Pending → Addressed logic
   - Addressed → Confirmed logic
   - Requirement expiration handling

3. Add requirement visualization:
   - `/requirements` command implementation
   - Color coding by status
   - Show source turn numbers

**Success Criteria**:

- Requirements extracted from conversation
- Status updates work correctly
- Can view all requirements
- No duplicate requirements

#### Task 2.3: Context Consolidation (6 hours)

**Goal**: Implement context consolidation to prevent lost-in-middle

**Subtasks**:

1. Implement consolidation task:

   - Focus on middle turns
   - Preserve all requirements
   - Keep under token limits

2. Add consolidation triggers:

   - Every N turns
   - Context size threshold
   - Quality degradation trigger

3. Implement context pruning:
   - Remove redundant information
   - Preserve critical details
   - Maintain chronological order

**Success Criteria**:

- Context stays manageable
- Middle turn info preserved
- Consolidation improves quality
- No information loss

### Phase 3: Task Handlers (Week 3)

#### Task 3.1: Code Task Handler (8 hours)

**Goal**: Implement code-specific handling

**Subtasks**:

1. Implement code task classification:

   - Detect code-related queries
   - Identify language/framework
   - Determine code operation type

2. Add code-specific quality checks:

   - Syntax validation
   - Import checking
   - Error handling verification

3. Integrate Claude Code SDK (optional):

   - Subprocess wrapper
   - Response parsing
   - Fallback handling

4. Add code context tracking:
   - Previous code snippets
   - Import statements
   - Variable/function definitions

**Success Criteria**:

- Code queries handled correctly
- Generated code is valid
- Context preserved across turns
- Quality checks pass

#### Task 3.2: Chat Task Handler with MCP (8 hours)

**Goal**: Implement chat handling with tool use

**Subtasks**:

1. Implement tool requirement detection:

   - Analyze query for tool needs
   - Map to available tools
   - Generate tool arguments

2. Add MCP tool execution:

   - Tool permission handling
   - Result formatting
   - Error handling

3. Implement answer bloat prevention:

   - Response length analysis
   - Conciseness scoring
   - Automatic summarization

4. Add tool result integration:
   - Format tool results
   - Integrate into response
   - Maintain conversation flow

**Success Criteria**:

- Tools used when appropriate
- Results integrated smoothly
- No unnecessary tool calls
- Responses stay concise

### Phase 4: Integration and Refinement (Week 4)

#### Task 4.1: Quality Refinement Loop (6 hours)

**Goal**: Implement response refinement

**Subtasks**:

1. Implement refinement logic:

   - Quality threshold checking
   - Issue-specific refinement
   - Max attempt limiting

2. Add refinement strategies:

   - Address specific issues
   - Preserve good content
   - Incremental improvement

3. Implement refinement tracking:
   - Log all attempts
   - Track improvement rates
   - Identify patterns

**Success Criteria**:

- Low-quality responses refined
- Quality improves with refinement
- No infinite loops
- Refinement metrics available

#### Task 4.2: Advanced REPL Features (6 hours)

**Goal**: Add production REPL features

**Subtasks**:

1. Add command system:

   - `/stats` - conversation statistics
   - `/requirements` - requirement tracking
   - `/quality` - quality metrics
   - `/export` - export conversation

2. Implement rich formatting:

   - Syntax highlighting for code
   - Quality score indicators
   - Requirement status badges

3. Add conversation management:
   - Save/load conversations
   - Resume previous sessions
   - Clear conversation

**Success Criteria**:

- All commands work
- Rich formatting displays correctly
- Can save/resume conversations
- Good user experience

### Phase 5: Testing and Debugging (Week 5)

#### Task 5.1: Unit Testing (8 hours)

**Goal**: Comprehensive unit test coverage

**Subtasks**:

1. Test quality evaluation:

   ```python
   async def test_quality_evaluation_detects_premature():
       response = "Here's the complete solution..."
       requirements = [{"status": "pending"}, {"status": "pending"}]
       result = await evaluate_quality_with_llm({
           "response": response,
           "requirements": requirements
       })
       assert result["metrics"]["premature_attempt"] == True
   ```

2. Test requirement extraction:

   - Multi-turn extraction
   - Status updates
   - Edge cases

3. Test context consolidation:

   - Middle turn preservation
   - Size constraints
   - Information retention

4. Test task handlers:
   - Code vs chat classification
   - Tool selection
   - Response generation

**Success Criteria**:

- 80%+ test coverage
- All edge cases handled
- Tests run in CI
- Clear test documentation

#### Task 5.2: Integration Testing (8 hours)

**Goal**: Test full conversation flows

**Subtasks**:

1. Create test scenarios from paper:

   - Premature answer scenario
   - Answer bloat scenario
   - Lost-in-middle scenario
   - Multi-requirement scenario

2. Implement conversation simulators:

   ```python
   async def simulate_conversation(turns: List[str]) -> ConversationState:
       workflow = ConversationWorkflow()
       state = None
       for turn in turns:
           result = await workflow.run({
               "user_input": turn,
               "state": state
           })
           state = result.value["state"]
       return state
   ```

3. Add quality assertions:
   - Quality improves over baseline
   - Requirements tracked correctly
   - No premature answers
   - Context preserved

**Success Criteria**:

- All scenarios pass
- Quality metrics improve
- No regressions
- Performance acceptable

### Phase 6: Temporal Migration (Week 6)

#### Task 6.1: Temporal Workflow Implementation (8 hours)

**Goal**: Add Temporal support to workflow

**Subtasks**:

1. Update ConversationWorkflow:

   - Add `_run_temporal_conversation`
   - Implement signal handling
   - Add pause/resume support

2. Create Temporal worker:

   ```python
   # run_worker.py
   from mcp_agent.executor.temporal import create_temporal_worker_for_app
   from main import app

   async def main():
       async with create_temporal_worker_for_app(app) as worker:
           await worker.run()
   ```

3. Update configuration:

   - Add Temporal settings
   - Queue configuration
   - Worker options

4. Implement state persistence:
   - Workflow state in Temporal
   - Recovery handling
   - Migration from AsyncIO state

**Success Criteria**:

- Can run with Temporal executor
- State persists across restarts
- Signals work correctly
- No functionality lost

#### Task 6.2: Production Configuration (4 hours)

**Goal**: Prepare for production deployment

**Subtasks**:

1. Add environment-based config:

   ```python
   # Detect environment
   if os.getenv("RCM_EXECUTOR") == "temporal":
       config.execution_engine = "temporal"
   ```

2. Implement monitoring:

   - Conversation metrics
   - Quality tracking
   - Performance monitoring
   - Error rates

3. Add operational tooling:
   - Health checks
   - Graceful shutdown
   - State export/import
   - Debug mode

**Success Criteria**:

- Runs in production environment
- Monitoring working
- Operational tools available
- Performance acceptable

## Testing Strategy

### Unit Test Structure

```python
# tests/unit/test_quality_evaluation.py
import pytest
from src.tasks.llm_evaluators import evaluate_quality_with_llm

@pytest.mark.asyncio
async def test_detects_premature_attempt():
    """Test from paper - premature attempts with pending requirements"""
    # Arrange
    params = {
        "response": "Here's the complete solution to your problem",
        "requirements": [
            {"description": "Need Python code", "status": "pending"},
            {"description": "Must handle errors", "status": "pending"}
        ],
        "turn_number": 1
    }

    # Act
    result = await evaluate_quality_with_llm(params)

    # Assert
    assert result["metrics"]["premature_attempt"] == True
    assert "pending requirements" in str(result["issues"]).lower()
```

### Integration Test Patterns

```python
# tests/integration/test_conversation_flows.py
async def test_prevents_answer_bloat():
    """Test from paper - responses should not grow 20-300%"""
    workflow = ConversationWorkflow()

    # Simulate multi-turn conversation
    responses = []
    state = None

    for i in range(5):
        result = await workflow.run({
            "user_input": f"Add requirement {i}",
            "state": state
        })
        state = result.value["state"]
        responses.append(result.value["response"])

    # Check bloat
    first_length = len(responses[0])
    last_length = len(responses[-1])
    bloat_ratio = last_length / first_length

    assert bloat_ratio < 2.0, f"Excessive bloat: {bloat_ratio:.1f}x"
```

## Debugging Tools

### Conversation Inspector

```python
# src/utils/inspector.py
class ConversationInspector:
    """Debug tool for analyzing conversation state"""

    def print_quality_timeline(self, state: ConversationState):
        """Show quality scores over time"""
        for i, (turn, metrics) in enumerate(zip(state.messages, state.quality_history)):
            if turn.role == "assistant":
                print(f"Turn {i}: {metrics.overall_score:.2f} "
                      f"(premature: {metrics.premature_attempt})")

    def analyze_requirements(self, state: ConversationState):
        """Show requirement lifecycle"""
        for req in state.requirements:
            print(f"{req.id}: {req.description}")
            print(f"  Source: Turn {req.source_turn}")
            print(f"  Status: {req.status}")
            print(f"  Confidence: {req.confidence}")
```

### Quality Profiler

```python
# src/utils/profiler.py
class QualityProfiler:
    """Profile quality issues across conversations"""

    def __init__(self):
        self.issues = defaultdict(int)

    def profile_conversation(self, state: ConversationState):
        """Identify patterns in quality issues"""
        for metrics in state.quality_history:
            if metrics.premature_attempt:
                self.issues["premature_attempts"] += 1
            if metrics.verbosity > 0.7:
                self.issues["high_verbosity"] += 1
            if metrics.middle_turn_reference < 0.3:
                self.issues["lost_middle_turns"] += 1
```

## Performance Considerations

### Token Usage Optimization

1. **Context Consolidation**: Consolidate every 3 turns to stay under limits
2. **Selective History**: Only include relevant previous turns
3. **Efficient Prompts**: Use concise evaluation prompts

### Response Time Optimization

1. **Parallel Evaluation**: Evaluate quality dimensions in parallel
2. **Early Termination**: Stop refinement once quality threshold met
3. **Caching**: Cache requirement extractions when possible

### Memory Management

1. **State Pruning**: Remove old quality metrics after 20 turns
2. **Context Windows**: Limit conversation history to recent N turns
3. **Lazy Loading**: Load conversation history on demand

## Configuration Reference

### Environment Variables

```bash
# Execution engine
export RCM_EXECUTOR=asyncio  # or temporal

# Model selection
export RCM_MODEL_PROVIDER=openai  # or anthropic
export RCM_MODEL=gpt-4  # or claude-3-sonnet-20240229

# Quality control
export RCM_QUALITY_THRESHOLD=0.8
export RCM_MAX_REFINEMENTS=3

# Claude Code SDK
export RCM_USE_CLAUDE_CODE=false
```

### Configuration Schema

```yaml
rcm:
  # Quality control settings
  quality_threshold: 0.8 # Minimum acceptable quality score
  max_refinement_attempts: 3 # Max refinement iterations
  consolidation_interval: 3 # Consolidate context every N turns

  # Feature flags
  use_claude_code: false # Use Claude Code SDK for code tasks
  verbose_metrics: false # Show quality metrics in REPL

  # Model configuration
  evaluator_model_provider: openai # LLM for quality evaluation

  # Conversation limits
  max_turns: 50 # Maximum conversation turns
  max_context_tokens: 8000 # Maximum context size

  # MCP servers
  mcp_servers:
    - fetch
    - filesystem
```

## Conclusion

This implementation plan provides a complete, opinionated approach to building the Reliable Conversation Manager using mcp-agent patterns. The phased approach ensures quick initial results while building toward a production-ready system with sophisticated quality control.

Key aspects:

1. **Canonical mcp-agent usage** throughout
2. **Comprehensive quality control** from the paper
3. **Testable and debuggable** architecture
4. **Clear migration path** from AsyncIO to Temporal
5. **Production-ready** monitoring and configuration

The implementation follows research findings to prevent LLMs from getting lost in conversation while maintaining the flexibility to handle both code and general chat tasks with MCP tool integration.
