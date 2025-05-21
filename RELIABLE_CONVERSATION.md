Okay, this is the definitive Design Document and Implementation Plan, incorporating all elements, including Temporal, and addressing the "Lost in Conversation" paper's findings. It's structured for clarity and aimed at being digestible for a coding assistant to help scaffold.

## Design Document: Reliable Conversation Manager (RCM) with Temporal Orchestration

**Version:** 3.0
**Date:** 2025-11-01
**Author:** AI Assistant

**1. Introduction & Motivation**

This document outlines the design for a **Reliable Conversation Manager (RCM)**, a core component for AI agent frameworks, specifically built upon the `mcp-agent` toolkit. The RCM aims to significantly enhance the reliability and coherence of multi-turn conversations with Large Language Models (LLMs). It directly addresses common LLM pitfalls such as context loss (the "Lost in the Middle" problem described in arXiv:2505.06120v1), instruction forgetting, premature solution generation, and excessive verbosity, which degrade user experience and task success rates.

The RCM achieves this through a multi-faceted strategy:

- **Durable State Management via Temporal:** Each RCM-managed conversation is orchestrated as a Temporal workflow, ensuring that the rich conversational state (`ReliableConversationState`) is persisted, auditable, and recoverable.
- **Proactive Contextual Synthesis:** The RCM actively maintains and utilizes summaries of user intent, accumulated task knowledge, and a detailed interaction history.
- **Explicit Requirement Lifecycle Management:** User requirements are identified, tracked, and their fulfillment is verified throughout the conversation.
- **Iterative Self-Correction Loop:** Internal evaluation stages assess assistant responses against quality criteria and tracked requirements, enabling the RCM to detect and mitigate LLM failure modes by initiating corrective actions or refinement loops. These evaluations and refinements are themselves durable steps within the Temporal workflow.
- **Strategic Conversational Staging:** The RCM guides interactions through distinct logical stages (e.g., input evaluation, context consolidation, response planning, generation, evaluation), promoting focused and comprehensive processing.
- **Defined Recovery Protocols:** Low-confidence states or detected conversational drift trigger specific, durable recovery strategies within the workflow.
- **Controlled Human Intervention:** The system provides points for human feedback via Temporal signals, allowing workflows to pause and await guidance.

A comprehensive simulation framework, leveraging Temporal for executing individual simulation runs, is included for systematic testing and benchmarking. The design also details the `ProgrammaticToolManager` for reliably interacting with external tools like CLI-based coding assistants, treating these as managed, durable sub-conversations.

**2. Goals**

- **Improve Conversational Reliability:** Substantially reduce context loss, instruction forgetting, premature/verbose responses, and unaddressed requirements in multi-turn dialogues.
- **Durable Memory Management:** Implement `ReliableConversationState` as the central, Temporal-persisted memory for each conversation.
- **Enable Self-Correction:** Autonomously detect and recover from common LLM failure modes identified in research (e.g., arXiv:2505.06120v1).
- **Modularity & Reusability:**
  - Core RCM logic encapsulated in `RCMInternalLogic`, orchestrated by `RCMConversationWorkflow` (a Temporal workflow).
  - Strategy modules (evaluators, consolidators) designed as Temporal activities.
  - `ProgrammaticToolManager` for external tools, integrable as activities/child workflows.
- **Testability & Observability:** Use a Temporal-based simulation framework for benchmarking. Leverage Temporal UI for workflow visibility.
- **Configurability:** Provide `RCMConfig` for fine-grained control over RCM strategies and thresholds.
- **Specialized Tool Integration:** Enable reliable, multi-turn, durable interactions with programmatic tools.
- **Human Feedback Loops:** Integrate human intervention via Temporal signals.

**3. Non-Goals**

- Replacing high-level agent orchestration patterns (e.g., `mcp-agent`'s `Orchestrator`).
- Achieving infallible conversational abilities.
- Developing full SDKs for external tools if existing interfaces are sufficient.

**4. System Architecture Overview**

The RCM system comprises several key components:

1.  **`RCMConversationWorkflow` (Temporal Workflow):**

    - The top-level orchestrator for a single, end-to-end conversation.
    - Holds the `ReliableConversationState` as its primary, durable state.
    - Manages the overall turn-taking loop, receiving user inputs via Temporal signals.
    - Invokes the `RCMInternalLogic` to process each turn.
    - Handles signals for human feedback and termination.

2.  **`RCMInternalLogic` (Python Class):**

    - Encapsulates the stage-based processing logic for a single conversational turn.
    - Operates on the `ReliableConversationState` passed to it by the workflow.
    - Invokes various "Strategy Modules" as Temporal Activities.
    - Invokes the "Base LLM" for response generation as a Temporal Activity.

3.  **Strategy Modules (Temporal Activities):**

    - Specialized modules responsible for specific meta-tasks within the RCM loop (e.g., `LLMInputEvaluatorActivity`, `LLMRequirementsExtractorActivity`, `LLMContextConsolidatorActivity`, `LLMResponseEvaluatorActivity`, `VerbosityDetectorActivity`, `PrematureAssumptionDetectorActivity`).
    - Each module that requires LLM capabilities will make calls to an LLM (typically the `base_llm` or a dedicated meta-LLM) which itself will be an activity.

4.  **Base LLM (as Temporal Activities):**

    - The underlying LLM used for generating assistant responses and for powering LLM-backed strategy modules.
    - Its core generation methods (e.g., `OpenAIAugmentedLLM.generate_activity`) are implemented as Temporal Activities, ensuring retries, timeouts, and isolation for these external calls.
    - This will be an instance of a class derived from `mcp_agent.workflows.llm.augmented_llm.AugmentedLLM`.

5.  **Data Structures (Pydantic Models):**

    - `ReliableConversationState`, `Message`, `Requirement`, `ContentEvaluation`, `InputEvaluation`, `RCMConfig`, Enums (`ConversationStage`, `MessageRole`). These are designed for Temporal serialization.

6.  **`ProgrammaticToolManager` (e.g., `ClaudeCodeReliableManager`):**
    - Can be invoked as a long-running Temporal Activity or a Child Workflow from `RCMConversationWorkflow`.
    - Manages multi-step interactions with external CLI tools, applying RCM-like principles internally (evaluation, refinement of CLI calls).

**Diagrammatic Flow (Conceptual for one RCM turn within the Workflow):**

```
RCMConversationWorkflow (Temporal Workflow)
  |
  | (User Input via Signal)
  v
  RCMInternalLogic.process_turn(input, current_ReliableConversationState)
  |
  |--> LLMInputEvaluatorActivity -> (uses BaseLLMActivity)
  |
  |--> LLMRequirementsExtractorActivity -> (uses BaseLLMActivity)
  |
  |--> (Heuristic Stage Determination)
  |
  |--> [IF Recovery Stage] -> Recovery Logic (may call ConsolidationActivity)
  |
  |--> [IF Consolidate Stage] -> LLMContextConsolidatorActivity -> (uses BaseLLMActivity)
  |
  |--> [IF Recap Stage] -> (uses BaseLLMActivity for recap generation)
  |
  |--> PromptConstructor.build_prompt_for_base_llm(...)
  |
  |--> BaseLLMActivity (for generating assistant response)
  |
  |--> VerbosityDetectorActivity -> (uses BaseLLMActivity) (optional)
  |
  |--> PrematureAssumptionDetectorActivity -> (uses BaseLLMActivity) (optional)
  |
  |--> LLMResponseEvaluatorActivity -> (uses BaseLLMActivity)
  |
  | (Updates ReliableConversationState held by Workflow)
  v
RCMConversationWorkflow (awaits next signal or terminates)
```

**5. Detailed Component Design**

**5.1. Core Data Structures (Pydantic Models)**

Location: `mcp_agent_rcm/reliable_conversation/state.py`

- **`ReliableConversationState`**:

  - `session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))`
  - `messages: List[Message] = Field(default_factory=list)`
  - `current_turn_id: int = 0`
  - `key_requirements: List[Requirement] = Field(default_factory=list)`
  - `consolidated_user_intent: str = ""`
  - `consolidated_task_knowledge: str = ""`
  - `active_conversation_stage: ConversationStage = ConversationStage.INITIALIZING`
  - `last_assistant_response_evaluation: Optional[ContentEvaluation] = None`
  - `last_user_input_evaluation: Optional[InputEvaluation] = None`
  - `is_final_answer_attempted: bool = False`
  - `reliability_metrics_snapshot: Dict[str, Any] = Field(default_factory=dict, description="Trend data: context_preservation_score (list), verbosity_score_avg, premature_assumption_incidents_count")`
  - `recovery_actions_taken: List[Dict[str, Any]] = Field(default_factory=list)`
  - `human_intervention_points: List[Dict[str, Any]] = Field(default_factory=list)`
  - `metadata: Dict[str, Any] = Field(default_factory=dict)`

- **`Message`**:

  - `message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))`
  - `role: MessageRole`
  - `text_content: Optional[str] = None`
  - `structured_content: Optional[Dict[str, Any]] = None` (Must be JSON-serializable for Temporal)
  - `turn_id: int`
  - `timestamp: datetime = Field(default_factory=datetime.utcnow)`
  - `metadata: Optional[Dict[str, Any]] = None`

- **`Requirement`**:

  - `requirement_id: str = Field(default_factory=lambda: str(uuid.uuid4()))`
  - `description: str`
  - `source_message_id: str`
  - `status: Literal["pending", "clarification_needed", "in_progress", "addressed", "partially_addressed", "failed_to_address", "validation_pending", "user_validated"] = "pending"`
  - `satisfaction_assessment: Optional[str] = None`
  - `validation_history: List[Dict[str, Any]] = Field(default_factory=list)` (Stores serialized `ContentEvaluation`)
  - `priority: Optional[Literal["low", "medium", "high"]] = "medium"`
  - `is_implicit: bool = False`

- **`ContentEvaluation`**:

  - `evaluation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))`
  - `evaluated_message_id: str`
  - `verbosity_score: float = Field(ge=0, le=1)`
  - `premature_assumption_score: float = Field(ge=0, le=1)`
  - `context_preservation_score: float = Field(ge=0, le=1)`
  - `requirements_coverage: Dict[str, Literal["fully_addressed", "partially_addressed", "not_addressed", "newly_implied_by_response", "irrelevant_to_this_response"]] = Field(default_factory=dict)`
  - `hallucination_score: float = Field(ge=0, le=1)`
  - `overall_quality_score: float = Field(ge=0, le=1)`
  - `reasoning: str`
  - `suggested_next_stage: Optional[ConversationStage] = None`
  - `suggested_refinements_for_llm: Optional[str] = None`
  - `identified_new_sub_requirements: List[str] = Field(default_factory=list)`

- **`InputEvaluation`**:

  - `evaluation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))`
  - `evaluated_message_id: str`
  - `clarity_score: float = Field(ge=0, le=1)`
  - `actionability_score: float = Field(ge=0, le=1)`
  - `new_requirements_pre_identified: List[str] = Field(default_factory=list)`
  - `potential_conflicts_with_prior_state: List[str] = Field(default_factory=list)`
  - `reasoning: str`

- **`ConversationStage` (Enum)**: `INITIALIZING`, `EVALUATE_USER_INPUT`, `EXTRACT_UPDATE_REQUIREMENTS`, `CHECK_RELIABILITY_AND_RECOVER`, `CONSOLIDATE_CONTEXT`, `RECAPITULATE_CONTEXT`, `PLAN_RESPONSE_STRATEGY`, `GENERATE_ASSISTANT_RESPONSE`, `EVALUATE_ASSISTANT_RESPONSE`, `REQUEST_USER_CLARIFICATION`, `AWAIT_HUMAN_FEEDBACK`, `FINALIZE_RESPONSE`, `TERMINATED`.

- **`MessageRole` (Enum)**: `USER`, `ASSISTANT`, `SYSTEM`, `INTERNAL_META_PROMPT`, `INTERNAL_EVALUATION`, `TOOL_CALL`, `TOOL_RESULT`, `HUMAN_FEEDBACK`.

**5.2. RCM Configuration (`RCMConfig`)**

Location: `mcp_agent_rcm/reliable_conversation/config.py`

```python
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

class RCMConfig(BaseModel):
    max_internal_refinement_loops: int = Field(default=2, ge=0)
    recovery_strategy_on_low_reliability: Literal["none", "force_consolidate", "force_recapitulate", "request_human_feedback"] = "force_consolidate"
    low_reliability_threshold: float = Field(default=0.4, ge=0, le=1)
    max_wait_for_user_input_hours: int = Field(default=24, ge=1, description="Max hours workflow waits for user input signal.")
    # ... (all other fields from previous version) ...
    strategy_module_configs: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict, description="Per-strategy-module specific configs, e.g., model overrides for evaluation LLM calls.")
    strategy_module_prompts: Dict[str, str] = Field(default_factory=dict, description="Custom prompts for strategy modules. Keys match module names.")
    default_llm_request_params: Optional[Dict[str, Any]] = Field(default=None, description="Default mcp_agent.RequestParams for the main/base LLM generation, passed as dict.")
    default_llm_request_params_for_meta_tasks: Optional[Dict[str, Any]] = Field(default=None, description="Default mcp_agent.RequestParams for internal RCM LLM calls (eval, consolidate etc.), passed as dict.")
    enable_explicit_verbose_detection_module: bool = True
    enable_explicit_premature_assumption_detection_module: bool = True
    # ... other fields like context_window_config_main_llm, verbosity_alert_threshold etc.
```

**5.3. `RCMConversationWorkflow` (Temporal Workflow)**

Location: `mcp_agent_rcm/workflows/rcm_conversation_workflow.py`
_(Core structure as defined in the prior "Temporal Integration" section. This workflow holds `_conversation_state` and calls `_rcm_internal_logic.process_turn`.)_

Key aspects:

- Uses `@app.workflow`, `@app.workflow_run`, `@app.workflow_signal`, `@workflow.query` from `mcp-agent` and `temporalio`.
- `run` method initializes `_rcm_config` and `_conversation_state`.
- Main loop awaits `submit_user_input` signal, then calls `await self._rcm_internal_logic.process_turn(...)`.
- Handles `AWAIT_HUMAN_FEEDBACK` stage by `await workflow.wait_for_signal("submit_human_feedback_signal", ...)`.
- The `mcp_agent_context` is available as `self.context` (from `Workflow` base class via `MCPApp`).

**5.4. `RCMInternalLogic` (Core RCM Processing)**

Location: `mcp_agent_rcm/reliable_conversation/rcm_internal_logic.py`
_(Structure as defined in the prior "Temporal Integration" section. This class contains the main `process_turn` method and helper methods.)_

Key aspects:

- Constructor takes `rcm_config`, `initial_agent_instruction`, `mcp_agent_context`.
- Initializes strategy modules. These modules will use the `mcp_agent_context.executor` to run their LLM-dependent parts as activities.
- `process_turn` method:
  - Takes `current_user_input: str` and `conv_state: ReliableConversationState` (which it modifies).
  - Iterates through RCM stages (`EVALUATE_USER_INPUT` -> ... -> `EVALUATE_ASSISTANT_RESPONSE`).
  - Calls strategy module methods as **Temporal Activities** using `await self.mcp_context.executor.execute(activity_function, *activity_args)`.
  - Calls the base LLM for assistant response generation as a **Temporal Activity**.
  - Updates `conv_state` durably.
- Helper methods for building prompts, checking recovery, executing recovery, etc., are mostly pure Python logic operating on `conv_state`, but recovery/recap might trigger further activities.

**5.5. Strategy Modules (as Temporal Activities)**

Location: `mcp_agent_rcm/reliable_conversation/strategies/*.py`

- **`BaseStrategyModule(ABC)`**:
  - `__init__(self, llm_activity_proxy: Any, mcp_context: MCPContext, config: Optional[Dict], default_request_params_dict: Optional[Dict], prompt_override: Optional[str])`
    - `llm_activity_proxy`: This would be a reference or name that allows calling the base LLM's generation methods _as activities_. For example, it could be the `AugmentedLLM` instance itself if its methods are already activities, or just the activity name string.
    - `mcp_context`: Used to call `executor.execute`.
- **`LLMInputEvaluatorActivity`**:
  - `@app.workflow_task(name="RCM_InputEvaluationActivity")`
  - `async def evaluate(self, user_rcm_message_dict: Dict) -> Dict:` (Takes and returns dicts for Temporal)
    - Reconstructs `RCMMessage` from dict.
    - Builds prompt.
    - Calls `self.llm_activity_proxy.generate_structured_activity(prompt, response_model=InputEvaluation, ...)`
    - Returns `InputEvaluation.model_dump()`.
- **`LLMRequirementsExtractorActivity`**:
  - `@app.workflow_task(name="RCM_RequirementsExtractionActivity")`
  - `async def extract_and_update(self, conv_state_dict: Dict, new_requirements_texts: List[str]) -> Dict:`
    - Reconstructs `ReliableConversationState`.
    - Builds prompt.
    - Calls LLM for extraction (activity call).
    - Updates `conv_state.key_requirements`.
    - Returns `conv_state.model_dump()`.
- **`LLMContextConsolidatorActivity`**:
  - `@app.workflow_task(name="RCM_ContextConsolidationActivity")`
  - `async def consolidate(self, conv_state_dict: Dict) -> Dict:`
    - Reconstructs state. Calls LLM for intent, then knowledge (two activity calls). Updates state. Returns state dict.
- **`LLMResponseEvaluatorActivity`**:
  - `@app.workflow_task(name="RCM_ResponseEvaluationActivity")`
  - `async def evaluate(self, conv_state_dict: Dict, assistant_response_text: str, assistant_response_structured_content: Optional[Dict], explicit_verbosity_score: Optional[float], explicit_premature_assumption_score: Optional[float]) -> Dict:`
    - Reconstructs state. Builds prompt incorporating explicit scores. Calls LLM. Returns `ContentEvaluation.model_dump()`.
- **`VerbosityDetectorActivity`**:
  - `@app.workflow_task(name="RCM_VerbosityDetectionActivity")`
  - `async def detect(self, response_text: str) -> float:` (Calls LLM. Returns score).
- **`PrematureAssumptionDetectorActivity`**:
  - `@app.workflow_task(name="RCM_PrematureAssumptionDetectionActivity")`
  - `async def detect(self, response_text: str, conv_state_dict: Dict) -> float:` (Calls LLM. Returns score).
- **`HeuristicStageDeterminer`** and **`PromptConstructor`**: These remain pure Python classes/methods called directly by `RCMInternalLogic` within the workflow's execution thread. They do not make external calls themselves.

**5.6. Base LLM Activities**

Location: `mcp_agent_rcm/activities/base_llm_activities.py` (or within the specific `AugmentedLLM` implementation in `mcp-agent` if preferred)

The chosen base `AugmentedLLM` (e.g., `OpenAIAugmentedLLM`) needs its core methods exposed as activities.

```python
# Example for OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM, OpenAICompletionTasks, RequestCompletionRequest, RequestStructuredCompletionRequest
from mcp_agent.executor.workflow_task import workflow_task # static decorator
from mcp_agent.config import OpenAISettings
from mcp_agent.app import MCPApp # Assuming an app instance `app`
# from my_project.app import app # If app is defined project-wide

app = MCPApp(name="base_llm_activities_app_placeholder") # Placeholder if no global app

# Make the core OpenAI call an activity
@app.workflow_task(name="OpenAI_RequestCompletionActivity")
async def request_completion_activity(request_dict: Dict) -> Dict:
    request = RequestCompletionRequest(**request_dict)
    # Original OpenAICompletionTasks.request_completion_task logic here
    # Ensure result is serializable (e.g., model_dump)
    # This is already handled by mcp-agent's Temporal integration for Pydantic models.
    raw_response = await OpenAICompletionTasks.request_completion_task(request)
    return raw_response.model_dump(mode='json')


@app.workflow_task(name="OpenAI_RequestStructuredCompletionActivity")
async def request_structured_completion_activity(request_dict: Dict) -> Dict:
    # Need to handle response_model deserialization carefully if passed as string
    # For Temporal, custom serializers or passing schema might be better.
    # PydanticTypeSerializer from mcp-agent.utils can be used here.
    from mcp_agent.utils.pydantic_type_serializer import deserialize_model

    config = OpenAISettings(**request_dict['config'])
    serialized_response_model_str = request_dict['serialized_response_model']
    response_model_class = deserialize_model(serialized_response_model_str)

    request = RequestStructuredCompletionRequest(
        config=config,
        response_model=response_model_class, # This is now a type
        serialized_response_model=None, # It's deserialized
        response_str=request_dict['response_str'],
        model=request_dict['model']
    )
    raw_response = await OpenAICompletionTasks.request_structured_completion_task(request)
    return raw_response.model_dump(mode='json')


# The OpenAIAugmentedLLM methods would then call these activities:
# class OpenAIAugmentedLLM:
#     async def generate_activity_call(self, messages_dict_list: List[Dict], request_params_dict: Dict) -> List[Dict]:
#         # ... prepare payload for request_completion_activity ...
#         response_dict = await self.context.executor.execute(
#             "OpenAI_RequestCompletionActivity", # Activity name
#             {"config": self.context.config.openai.model_dump(), "payload": payload_dict}
#         )
#         # ... process response_dict into List[ChatCompletionMessage] and then List[Dict] ...
#         return processed_responses_dict_list
```

_Note: The `AugmentedLLM` classes in `mcp-agent` like `OpenAIAugmentedLLM` already use `@workflow_task` for their core `request_completion_task`. This makes them directly usable as activities. The key is to ensure the arguments and return types are Temporal-serializable (Pydantic models are fine with the default `pydantic_data_converter`)._

**5.7. `ProgrammaticToolManager` (e.g., `ClaudeCodeReliableManager`)**

Location: `mcp_agent_rcm/reliable_conversation/programmatic_tool_manager.py`

- Can be a Temporal Workflow (`ClaudeCodeWorkflow`) or a long-running Activity.
- **If Workflow:**
  - Takes initial coding prompt, files, `RCMConfig` for its internal RCM-like loop.
  - `_invoke_claude_code_cli` method becomes an Activity.
  - Uses a meta-LLM (via activities) for evaluating code output.
  - Signals back progress or final results.
- **If Activity:**
  - The entire `execute_coding_task` method is one activity.
  - Less internal state durability than a full workflow, but simpler.
  - Heartbeating is essential for long CLI calls.

**5.8. Simulation Framework (`RCMBenchmarkSuite`)**

Location: `mcp_agent_rcm/simulations/`
_(Largely as described previously, with the `RCMBenchmarkSuite` acting as a Temporal client.)_

- `ShardedInstruction` and `SimulationRunResult` models.
- `run_single_simulation` method starts `RCMConversationWorkflow` instances.
- Interacts via signals (`submit_user_input`).
- Uses queries (`get_conversation_state_snapshot`) and final workflow result for metrics.
- Baseline runs (single-turn, naive multi-turn) would also ideally be Temporal workflows/activities for consistent measurement and execution.

**6. Implementation Plan (Phased)**

**Phase 0: Core Models, Configs, Basic Temporal Setup**

1.  **Pydantic Models & Enums (`state.py`):** Implement all data structures.
    - _Testable:_ Models validate and are serializable by Temporal's Pydantic data converter.
2.  **`RCMConfig` (`config.py`):** Implement the configuration model.
    - _Testable:_ Loads defaults, allows overrides.
3.  **`MCPApp` Instance (`mcp_agent_rcm/app.py`):** Define a global `MCPApp` instance for the RCM project. This app will register all RCM-specific workflows and activities.
4.  **Basic `RCMConversationWorkflow` Skeleton (`workflows/rcm_conversation_workflow.py`):**
    - Define with `@app.workflow`, `@app.workflow_run`.
    - Implement `submit_user_input` and `terminate_conversation` signals.
    - Implement `get_conversation_state_snapshot` query.
    - Initial `run` method:
      - Instantiates `RCMConfig` from input dict.
      - Initializes `_conversation_state` (e.g., `ReliableConversationState(session_id=workflow.info().workflow_id, ...)`).
      - Logs initial prompt.
      - `await workflow.wait_for_signal("submit_user_input")` in a loop.
      - Logs received input and loops or terminates.
    - _Testable:_ Deploy worker. Start workflow via Temporal client. Send signals. Query state. Check logs.
5.  **Temporal Worker Script (`workers/rcm_worker.py`):**
    - Imports `app` and `RCMConversationWorkflow`.
    - Uses `create_temporal_worker_for_app(app)` to start.
    - _Testable:_ Worker starts, connects to Temporal, and can poll for `RCMConversationWorkflow` tasks.

**Phase 1: `RCMInternalLogic` and Base LLM as Activity**

1.  **Base LLM Activity Registration:**
    - Ensure the chosen base `AugmentedLLM` from `mcp-agent` (e.g., `OpenAIAugmentedLLM`) has its core `request_completion_task` (and `request_structured_completion_task`) methods decorated with `@app.workflow_task` (they are already).
    - These will be registered with the `app` instance in `mcp_agent_rcm/app.py`.
2.  **`RCMInternalLogic` Skeleton (`reliable_conversation/rcm_internal_logic.py`):**
    - Constructor: `rcm_config`, `initial_agent_instruction`, `mcp_agent_context`.
    - `add_message_to_durable_state` helper.
    - `process_turn` method:
      - Takes `current_user_input: str`, `conv_state: ReliableConversationState`.
      - Adds user message to `conv_state`.
      - Makes a single call to the base LLM (as an activity) using `await self.mcp_context.executor.execute("ActivityNameForBaseLLMGenerate", activity_args)`.
      - Adds assistant response to `conv_state`.
3.  **Integrate `RCMInternalLogic` into `RCMConversationWorkflow`:**
    - Workflow's `run` method instantiates `RCMInternalLogic`.
    - After receiving user input via signal, calls `await self._rcm_internal_logic.process_turn(self._current_user_input, self._conversation_state)`.
    - _Testable:_ Workflow runs a turn. A base LLM activity is executed (visible in Temporal UI). `ReliableConversationState` (queried) shows user and assistant messages.

**Phase 2: Implement Strategy Modules as Activities**

1.  **`BaseStrategyModule` (`strategies/base_strategy.py`):**
    - Define common constructor as described in section 5.5.
2.  **Implement each LLM-backed strategy module activity:**
    - For `LLMInputEvaluator`, `LLMRequirementsExtractor`, `LLMContextConsolidator`, `LLMResponseEvaluator`, `VerbosityDetector`, `PrematureAssumptionDetector`.
    - Each will have a main method (e.g., `async def evaluate(self, args_dict: Dict) -> Dict:`) decorated with `@app.workflow_task(name="UniqueActivityName")`.
    - These activities will take serialized inputs (dicts) and return serialized outputs (dicts).
    - Inside, they reconstruct Pydantic models, build prompts, and call the base LLM's structured generation activity (e.g., `"OpenAI_RequestStructuredCompletionActivity"`), passing the appropriate `response_model` (serialized if necessary, then deserialized by the activity).
    - Register these activities with the `MCPApp` in `mcp_agent_rcm/app.py`.
3.  **Update `RCMInternalLogic.process_turn()`:**
    - Replace stubs with `await self.mcp_context.executor.execute("StrategyActivityName", strategy_args_dict)`.
    - Implement `HeuristicStageDeterminer` and `PromptConstructor` (pure Python logic).
    - _Testable:_ Each strategy module activity can be unit-tested by invoking it via `TemporalTestEnvironment` or a simple client. Full workflow runs show a sequence of strategy activities and base LLM activities per turn.

**Phase 3: Full RCM Loop in `RCMInternalLogic` & Workflow Signal Integration**

1.  **Complete `RCMInternalLogic.process_turn()`:** Implement the full stage-based loop, internal refinement logic, recovery checks, and execution of recovery strategies (which may call activities like consolidation).
2.  **Human Feedback in `RCMConversationWorkflow`:**
    - When `RCMInternalLogic.process_turn` causes `_conversation_state.active_conversation_stage` to become `AWAIT_HUMAN_FEEDBACK`:
      - The `RCMConversationWorkflow.run()` method's main loop will detect this.
      - It then `await workflow.wait_for_signal("submit_human_feedback_signal", timeout=...)`.
    - The `submit_human_feedback` signal handler (already defined) updates `_conversation_state` and sets `_next_user_input_event`.
    - _Testable:_ Workflow demonstrates full internal RCM logic flow. It correctly pauses for human feedback signals and resumes. Recovery strategies trigger expected activities.

**Phase 4: `ProgrammaticToolManager` (e.g., `ClaudeCodeReliableManager`)**

1.  **Define as a Temporal Workflow (`ClaudeCodeWorkflow`):**
    - Location: `mcp_agent_rcm/workflows/claude_code_workflow.py`.
    - Takes parameters like initial prompt, `RCMConfig` (for its internal RCM-like loop).
    - The method that invokes the Claude Code CLI (e.g., `_invoke_claude_code_cli`) **must** be an activity `@app.workflow_task(name="InvokeClaudeCodeCLIActivity")`. This activity needs to handle subprocess execution, capture stdout/stderr, and manage timeouts/heartbeating if the CLI call is very long.
    - Uses a meta-LLM (via activities) to evaluate Claude Code's output and generate refinement prompts for subsequent CLI calls.
2.  **Integration (Optional):** `RCMConversationWorkflow` could potentially start `ClaudeCodeWorkflow` as a child workflow if a coding task is delegated.
    - _Testable:_ `ClaudeCodeWorkflow` can be started independently. The CLI invocation activity runs, retries on failure. The internal loop refines CLI calls.

**Phase 5: Simulation Framework & Initial Benchmarking**

1.  **Models (`simulations/types.py`):** `ShardedInstruction`, `SimulationRunResult`.
2.  **`RCMBenchmarkSuite` (`simulations/benchmark_suite.py`):**
    - Acts as a Temporal client.
    - `run_single_benchmark`: Starts `RCMConversationWorkflow` via `executor.start_workflow`. Interacts using `handle.signal("submit_user_input", ...)`. Retrieves final state/result via `handle.result()` and queries.
    - Implement baseline runs (single-turn, naive multi-turn). These might also be simple Temporal workflows or direct activity calls to the base LLM.
    - _Testable:_ Simulation suite can execute a set of sharded instructions against the RCM workflow, collect metrics, and generate a comparative report.

**Phase 6: Tuning, Advanced Features, and Optimizations**

1.  **Tune `RCMConfig` Defaults:** Based on simulation results.
2.  **User Guidance Messages:** Refine prompts for `REQUEST_USER_CLARIFICATION` and `AWAIT_HUMAN_FEEDBACK`.
3.  **Memory Optimization in `PromptConstructor`:** Implement token counting and summarization for long histories, potentially as an activity if LLM-based summarization is used.
    - _Testable:_ Performance improvements on benchmarks. Better handling of long conversations.

**Phase 7: Documentation, Examples, and Release Preparation**

1.  **Documentation:** Detail RCM architecture, Temporal integration, configuration, strategy module customization, and simulation usage.
2.  **Examples (`examples/`):**
    - `temporal_client_chat.py`: Client to interact with `RCMConversationWorkflow`.
    - `temporal_client_claude_code.py`: Client for `ClaudeCodeWorkflow`.
    - `run_simulation_temporal.py`: Script to run benchmark suite.
3.  **Code Polishing & API Review.**

This comprehensive plan ensures that each part of the RCM is built with Temporal's capabilities in mind from the outset, leading to a truly robust and reliable conversational system. The separation of concerns between the `RCMConversationWorkflow` (managing overall flow and durable state) and `RCMInternalLogic` (handling per-turn processing by invoking activities) is key to this design.
