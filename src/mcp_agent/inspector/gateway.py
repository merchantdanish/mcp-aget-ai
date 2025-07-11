"""Gateway module for mcp-agent-inspector."""

from typing import Optional, Any


def mount(app: Optional[Any] = None, *, expose: bool = False, auth: Optional[Any] = None) -> None:
    """
    Mount the inspector on an existing FastAPI application.
    
    Args:
        app: Optional FastAPI application instance. If None, will spawn internal server.
        expose: If True, allow external connections (default: False, localhost only)
        auth: Authentication provider (for future use in M5)
    
    This is a stub implementation for M0-A milestone.
    Full implementation comes in M0-B.
    """
    # M0-A: Stub function for re-export
    # Full implementation in M0-B
    pass