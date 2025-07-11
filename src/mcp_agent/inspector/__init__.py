"""
mcp-agent-inspector: Zero-dependency debugging and observability tool for mcp-agent.
"""

from .gateway import mount
from .version import __version__

__all__ = ["mount", "__version__"]