# Adaptive Workflow

An adaptive multi-agent workflow implementation based on the Claude Deep Research architecture.

## Design Philosophy

The Adaptive Workflow follows these key principles from the Deep Research architecture:

1. **Dynamic Task Decomposition**: Instead of fixed patterns, the workflow analyzes each objective and creates appropriate subtasks
2. **Parallel Exploration**: Multiple subagents work simultaneously on different aspects
3. **Non-cascading Limits**: Subagents have their own iteration limits that don't cascade (preventing explosion)
4. **Learning from Experience**: The workflow tracks patterns of success and failure to improve over time
5. **Resource Awareness**: Time and cost budgets prevent runaway execution

## Key Components

### Task Analysis
The workflow starts by analyzing the objective to determine:
- **Task Type**: Research (information gathering), Action (making changes), or Hybrid
- **Initial Strategy**: How to approach the task (breadth-first, depth-first, etc.)

### Dynamic Subagent Creation
Instead of using pre-defined agents, the workflow creates specialized subagents on-demand with:
- Focused instructions for their specific subtask
- Only the MCP servers they need
- Appropriate resource limits

### Memory and Learning
The workflow maintains memory at two levels:
- **Workflow Memory**: Tracks progress within a single execution
- **Learning Manager**: Tracks patterns across executions to improve future runs

## Areas for Improvement

Based on the implementation, here are key areas that could be enhanced:

### 1. Remove Over-Engineering
- **TaskComplexity enum**: This arbitrary categorization doesn't add real value. The workflow should dynamically adapt based on what it discovers, not pre-determined buckets.
- **Complex Strategy objects**: Could be simplified to just track the approach and let the workflow adapt dynamically.

### 2. Better Strategy Selection
Instead of complex enums and categories, use simple heuristics:
```python
# Current: Complex categorization
if complexity == TaskComplexity.SIMPLE:
    parallelism = 3
elif complexity == TaskComplexity.MODERATE:
    parallelism = 5
    
# Better: Dynamic adaptation
parallelism = min(
    len(aspects_to_explore),  # Based on discovered aspects
    available_budget / estimated_cost_per_agent,  # Based on resources
    max_parallelism  # Hard limit
)
```

### 3. Improved Progress Evaluation
Current implementation uses confidence scores, but could be more sophisticated:
- Track coverage of the original objective
- Identify diminishing returns
- Detect when subagents are finding redundant information

### 4. Better Error Recovery
Current implementation marks tasks as failed but doesn't retry intelligently:
- Retry with different approaches
- Adjust strategy based on failure patterns
- Fallback to simpler methods

### 5. Smarter Resource Allocation
Instead of fixed budgets per agent:
- Allocate more resources to promising paths
- Reduce resources for areas with diminishing returns
- Dynamic reallocation based on discoveries

### 6. Real Citation Management
Current implementation has basic citation extraction. Could improve with:
- Proper source verification
- Deduplication of sources
- Quality scoring of sources
- Citation format standardization

## Usage Patterns

### Simple Research
```python
workflow = AdaptiveWorkflow(llm_factory=factory, available_servers=["filesystem", "github"])
result = await workflow.generate_str("What are the main features of our authentication system?")
```

### Complex Investigation
```python
result = await workflow.generate_str("""
    Analyze all customer feedback from the last quarter across all channels 
    (support tickets, reviews, social media) and identify the top 5 issues 
    with proposed solutions for each.
""")
```

### Hybrid Task
```python
result = await workflow.generate_str("""
    Research best practices for API rate limiting, then implement a rate 
    limiting system for our user API endpoints with configurable limits.
""")
```

## Implementation Notes

1. **Non-cascading Iterations**: Each subagent has its own `expected_iterations` limit (typically 3-7) that doesn't cascade. This prevents the explosion problem seen in the original Orchestrator.

2. **Parallel Execution**: When `enable_parallel=True`, independent subtasks run simultaneously using Python's async capabilities.

3. **Memory Compression**: The workflow automatically compresses memory to avoid context window issues in long-running tasks.

4. **Learning Persistence**: Currently uses in-memory learning. Could be extended to persist learning across runs.

## Future Enhancements

1. **Streaming Results**: Support streaming responses as subagents complete their work
2. **Interactive Mode**: Allow users to guide the workflow mid-execution
3. **Visualization**: Create visual representations of the task decomposition and execution
4. **Plugin System**: Allow custom strategies and evaluators to be plugged in
5. **Distributed Execution**: Support running subagents across multiple machines for scale