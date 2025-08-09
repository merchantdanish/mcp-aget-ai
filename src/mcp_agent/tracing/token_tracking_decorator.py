"""
Token tracking decorator for AugmentedLLM methods
"""

import functools
from typing import TypeVar, Callable

T = TypeVar("T")


def track_tokens(
    node_type: str = "llm",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to track token usage for AugmentedLLM methods.
    Automatically pushes/pops token context around method execution.

    Args:
        node_type: The type of node for token tracking. Default is "llm" for base AugmentedLLM classes.
                  Higher-order AugmentedLLM classes should use "agent".
    """

    def decorator(method: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(method)
        async def wrapper(self, *args, **kwargs) -> T:
            # Only track if we have a token counter in context
            if hasattr(self, "context") and self.context and self.context.token_counter:
                # Build metadata
                metadata = {
                    "method": method.__name__,
                    "class": self.__class__.__name__,
                }

                # Add any model info if available
                if hasattr(self, "provider"):
                    metadata["provider"] = getattr(self, "provider")

                async with self.context.token_counter.scope(
                    name=getattr(self, "name", self.__class__.__name__),
                    node_type=node_type,
                    metadata=metadata,
                ):
                    return await method(self, *args, **kwargs)
            else:
                # No token counter, just execute normally
                return await method(self, *args, **kwargs)

        return wrapper

    return decorator
