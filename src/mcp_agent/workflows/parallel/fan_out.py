import functools
from typing import Any, Callable, Coroutine, Dict, List, Type

from mcp_agent.agents.agent import Agent
from mcp_agent.executor.executor import Executor, AsyncioExecutor
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
)
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


class FanOut:
    """
    Distribute work to multiple parallel tasks.

    This is a building block of the Parallel workflow, which can be used to fan out
    work to multiple agents or other parallel tasks, and then aggregate the results.
    """

    def __init__(
        self,
        agents: List[Agent | AugmentedLLM[MessageParamT, MessageT]] | None = None,
        functions: List[Callable[[MessageParamT], List[MessageT]]] | None = None,
        llm_factory: Callable[[Agent], AugmentedLLM[MessageParamT, MessageT]] = None,
        executor: Executor | None = None,
    ):
        """
        Initialize the FanOut with a list of agents, functions, or LLMs.
        If agents are provided, they will be wrapped in an AugmentedLLM using llm_factory if not already done s.
        If functions are provided, they will be invoked in parallel directly.
        """

        self.executor = executor or AsyncioExecutor()
        self.agents: List[AugmentedLLM[MessageParamT, MessageT]] = []
        self.llm_factory = llm_factory or (
            lambda agent: AugmentedLLM[MessageParamT, MessageT](agent=agent)
        )
        for agent in agents or []:
            if isinstance(agent, AugmentedLLM):
                self.agents.append(agent)
            else:
                self.agents.append(self.llm_factory(agent=agent))

        self.functions: List[Callable[[MessageParamT], MessageT]] = functions or []

        if not self.agents and not self.functions:
            raise ValueError(
                "At least one agent or function must be provided for fan-out to work"
            )

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> Dict[str, List[MessageT]]:
        """
        Request fan-out agent/function generations, and return the results as a dictionary.
        The keys are the names of the agents or functions that generated the results.
        """
        tasks: List[
            Callable[..., List[MessageT]] | Coroutine[Any, Any, List[MessageT]]
        ] = []
        task_names: List[str] = []

        # Create bound methods for each agent's generate function
        for agent in self.agents:
            tasks.append(
                agent.generate(
                    message=message,
                    use_history=use_history,
                    max_iterations=max_iterations,
                    model=model,
                    stop_sequences=stop_sequences,
                    max_tokens=max_tokens,
                    parallel_tool_calls=parallel_tool_calls,
                )
            )
            task_names.append(agent.name)

        # Create bound methods for regular functions
        for function in self.functions:
            tasks.append(function(message))
            task_names.append(function.__name__ or id(function))

        logger.debug("Running fan-out tasks:", data=task_names)
        task_results = await self.executor.execute(*tasks)
        logger.debug(
            "Fan-out tasks completed:", data=dict(zip(task_names, task_results))
        )
        return dict(zip(task_names, task_results))

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        use_history: bool = True,
        max_iterations: int = 10,
        model: str = None,
        stop_sequences: List[str] = None,
        max_tokens: int = 2048,
        parallel_tool_calls: bool = True,
    ) -> Dict[str, str]:
        """
        Request fan-out agent/function generations and return the string results as a dictionary.
        The keys are the names of the agents or functions that generated the results.
        """

        def fn_result_to_string(fn, message):
            return str(fn(message))

        tasks: List[Callable[..., str] | Coroutine[Any, Any, str]] = []
        task_names: List[str] = []

        # Create bound methods for each agent's generate function
        for agent in self.agents:
            bound_generate = functools.partial(
                agent.generate_str,
                message=message,
                use_history=use_history,
                max_iterations=max_iterations,
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )
            tasks.append(bound_generate)
            task_names.append(agent.name)

        # Create bound methods for regular functions
        for function in self.functions:
            bound_function = functools.partial(fn_result_to_string, function, message)
            tasks.append(bound_function)
            task_names.append(function.__name__ or id(function))

        task_results = await self.executor.execute(*tasks)
        return dict(zip(task_names, task_results))

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
    ) -> Dict[str, ModelT]:
        """
        Request a structured fan-out agent/function generation and return the result as a Pydantic model.
        The keys are the names of the agents or functions that generated the results.
        """
        tasks = []
        task_names = []

        # Create bound methods for each agent's generate function
        for agent in self.agents:
            bound_generate = functools.partial(
                agent.generate_structured,
                message=message,
                response_model=response_model,
                use_history=use_history,
                max_iterations=max_iterations,
                model=model,
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
                parallel_tool_calls=parallel_tool_calls,
            )
            tasks.append(bound_generate)
            task_names.append(agent.name)

        # Create bound methods for regular functions
        for function in self.functions:
            bound_function = functools.partial(function, message)
            tasks.append(bound_function)
            task_names.append(function.__name__ or id(function))

        task_results = await self.executor.execute(*tasks)
        return dict(zip(task_names, task_results))
