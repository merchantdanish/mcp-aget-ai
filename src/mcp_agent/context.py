"""
A central context object to store global state that is shared across the application.
"""

from pydantic import BaseModel
from mcp import ServerSession

from .config import Settings
from .mcp_server_registry import ServerRegistry


class Context(BaseModel):
    """
    Context that is passed around through the application.
    This is a global context that is shared across the application.
    """

    def __init__(self, **data):
        super().__init__(**data)
        self.upstream_session: ServerSession = None
        self.server_registry: ServerRegistry = None
        self.plugins = []
        # Possibly store workflow intermediate data references
        # For caching and memory:
        self.memory_cache = {}
        # store intermediate data by workflow_id
        self.workflow_data_store = {}


global_context = Context()


async def initialize_context(config: Settings):
    """
    Initialize the global application context.
    """

    context = Context()
    context.server_registry = ServerRegistry(config.config_yaml)


def get_current_context():
    """
    Get the current application context.
    """
    return global_context
