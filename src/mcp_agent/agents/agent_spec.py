from __future__ import annotations

from typing import Callable, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class AgentSpec(BaseModel):
    """
    Canonical, strongly-typed Agent specification used across the system.

    This represents a declarative way to define an Agent without constructing it yet.
    AgentSpec is used to create an Agent instance.
    It can be defined as a config (loaded from a md, yaml, json, etc.), or
    it can be created programmatically.
    """

    name: str
    instruction: Optional[Union[str, Callable[[dict], str]]] = None
    server_names: List[str] = Field(default_factory=list)
    functions: List[Callable] = Field(default_factory=list)
    connection_persistence: bool = True
    human_input_callback: Optional[Callable] = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
