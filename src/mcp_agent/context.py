from pydantic import BaseModel


class Context(BaseModel):
    def __init__(self, **data):
        super().__init__(**data)
        self.upstream_session = None
        self.registry = {}
        self.plugins = []
        # Possibly store workflow intermediate data references
        # For caching and memory:
        self.memory_cache = {}
        # store intermediate data by workflow_id
        self.workflow_data_store = {}


global_context = Context()


def get_current_context():
    return global_context
