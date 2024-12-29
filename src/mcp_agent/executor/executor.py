import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Coroutine, List, TypeVar

# Type variable for the return type of tasks
R = TypeVar("R")


class Executor(ABC):
    """Abstract base class for different execution backends"""

    @abstractmethod
    async def execute(
        self,
        *tasks: Callable[..., R] | Coroutine[Any, Any, R],
        **kwargs: Any,
    ) -> List[R | BaseException]:
        """Execute a list of tasks and return their results"""

    @abstractmethod
    async def execute_streaming(
        self,
        *tasks: List[Callable[..., R] | Coroutine[Any, Any, R]],
        **kwargs: Any,
    ) -> AsyncIterator[R | BaseException]:
        """Execute tasks and yield results as they complete"""


class AsyncioExecutor(Executor):
    """Default executor using asyncio"""

    async def execute(
        self,
        *tasks: Callable[..., R] | Coroutine[Any, Any, R],
        **kwargs: Any,
    ) -> List[R | BaseException]:
        async def run_task(task: Callable[..., R] | Coroutine[Any, Any, R]) -> R:
            if asyncio.iscoroutine(task):
                return await task
            elif asyncio.iscoroutinefunction(task):
                return await task(**kwargs)
            return task(**kwargs)

        return await asyncio.gather(
            *(run_task(task) for task in tasks), return_exceptions=True
        )

    async def execute_streaming(
        self,
        *tasks: List[Callable[..., R] | Coroutine[Any, Any, R]],
        **kwargs: Any,
    ) -> AsyncIterator[R | BaseException]:
        async def run_task(task: Callable[..., R] | Coroutine[Any, Any, R]) -> R:
            if asyncio.iscoroutine(task):
                return await task
            elif asyncio.iscoroutinefunction(task):
                return await task(**kwargs)
            return task(**kwargs)

        # Create futures for all tasks
        futures = [asyncio.create_task(run_task(task)) for task in tasks]
        pending = set(futures)

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for future in done:
                yield await future
