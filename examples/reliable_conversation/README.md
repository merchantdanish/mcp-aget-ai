# Reliable Conversation Manager (RCM)

Implementation of research findings from "LLMs Get Lost in Multi-Turn Conversation" using mcp-agent framework.

## Phase 2 Implementation Status ✅

### Completed Features

- **Project Structure**: Complete directory structure following mcp-agent patterns
- **Base Models**: All core data models (ConversationMessage, Requirement, QualityMetrics, ConversationState)
- **Configuration**: YAML config with RCM-specific settings + canonical mcp-agent config loading
- **Logging**: RCM-specific logging utilities following mcp-agent patterns
- **ConversationWorkflow**: Quality-controlled workflow with AsyncIO support
- **REPL Interface**: Interactive command-line interface with Rich formatting
- **Quality Control Pipeline**: Complete LLM-based quality evaluation system
- **Requirement Tracking**: LLM-based requirement extraction and status tracking
- **Context Consolidation**: Prevents lost-in-middle-turns phenomenon
- **Robust Fallbacks**: System works even when LLM providers are unavailable
- **Real LLM Integration**: Works with OpenAI and Anthropic APIs
- **Research Metrics**: Tracks all metrics from the paper (bloat, premature attempts, etc.)

### Architecture

```
examples/reliable_conversation/
├── src/
│   ├── workflows/
│   │   └── conversation_workflow.py    # Main workflow implementation
│   ├── models/
│   │   └── conversation_models.py      # Data models from paper
│   ├── utils/
│   │   ├── logging.py                  # RCM logging utilities
│   │   └── config.py                   # Configuration helpers
│   └── tasks/                          # (Phase 2: Quality control tasks)
├── main.py                             # REPL entry point
├── mcp_agent.config.yaml              # Configuration
└── requirements.txt                    # Dependencies
```

### Current Capabilities

1. **Multi-turn Conversation**: Maintains state across conversation turns
2. **Rich Interface**: Color-coded REPL with status indicators
3. **Statistics**: `/stats` command shows conversation metrics
4. **Logging**: Comprehensive logging of all conversation events
5. **Model Support**: Configurable OpenAI/Anthropic model support
6. **MCP Integration**: Access to filesystem and fetch tools

### Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the REPL
python main.py
```

### REPL Commands

- `/stats` - Show conversation statistics
- `/requirements` - Show tracked requirements (Phase 2)
- `/exit` - Exit the conversation

### Configuration

Edit `mcp_agent.config.yaml` to configure:

```yaml
rcm:
  quality_threshold: 0.8              # Quality threshold for responses
  max_refinement_attempts: 3          # Max refinement iterations (Phase 2)
  consolidation_interval: 3           # Context consolidation frequency (Phase 2)
  evaluator_model_provider: "openai"  # LLM for quality evaluation (Phase 2)
  verbose_metrics: false              # Show quality metrics in REPL
```

### Research Implementation

Based on "LLMs Get Lost in Multi-Turn Conversation" paper findings:

- **Conversation-as-Workflow**: Entire conversation is single workflow instance
- **State Persistence**: Complete conversation state maintained across turns
- **Quality Metrics**: Framework for 7-dimension quality evaluation (Phase 2)
- **Requirement Tracking**: Infrastructure for cross-turn requirement tracking (Phase 2)

### Implementation Highlights

**Quality Control Pipeline:**
- 7-dimension quality evaluation based on research paper
- LLM-based requirement extraction to prevent instruction forgetting  
- Context consolidation every N turns to prevent lost-in-middle-turns
- Response refinement loop with quality thresholds
- Premature answer detection with completion markers

**Robust Architecture:**
- Canonical mcp-agent patterns (no custom task registration)
- Comprehensive fallbacks at every level
- Works with real API keys or gracefully degrades
- All functions are regular async functions, not framework tasks

### Development

The implementation follows canonical mcp-agent patterns:

```python
# Workflow pattern
@app.workflow
class ConversationWorkflow(Workflow[Dict[str, Any]]):
    @app.workflow_run
    async def run(self, args: Dict[str, Any]) -> WorkflowResult[Dict[str, Any]]:
        # Implementation

# Agent pattern
agent = Agent(
    name="responder",
    instruction="...",
    server_names=["fetch", "filesystem"]
)

async with agent:
    llm = await agent.attach_llm(OpenAIAugmentedLLM)
    result = await llm.generate_str(message)
```

### Testing Phase 1

#### Automated Test
```bash
# Run basic functionality test
python test_basic.py
```

#### Manual Test (REPL)
```bash
# Test interactive conversation
python main.py
> Hello, how are you?
> Can you help me with Python?
> /stats
> /exit
```

Expected behavior:
- Maintains conversation context across turns
- Shows statistics after multiple turns
- Logs all events to `logs/rcm-*.jsonl`
- Graceful error handling

#### Test Results
- ✅ Workflow creation and registration
- ✅ Multi-turn conversation state persistence
- ✅ Quality metrics framework
- ✅ REPL commands and Rich formatting
- ✅ Logging and error handling