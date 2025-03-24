from typing import List, Union
from pydantic import BaseModel

from mcp.types import (
    PromptMessage,
    TextContent,
    ImageContent,
    EmbeddedResource,
    Role,
    GetPromptResult,
)


class PromptMessageMultipart(BaseModel):
    """
    Extension of PromptMessage that handles multiple content parts.
    Internally converts to/from a sequence of standard PromptMessages.
    """

    role: Role
    content: List[Union[TextContent, ImageContent, EmbeddedResource]]

    @classmethod
    def from_prompt_messages(
        cls, messages: List[PromptMessage]
    ) -> List["PromptMessageMultipart"]:
        """Convert a sequence of PromptMessages into PromptMessageMultipart objects."""
        if not messages:
            return []

        result = []
        current_group = None
        current_role = None

        for msg in messages:
            if msg.role != current_role:
                # Role changed, start new message
                if current_group is not None:
                    result.append(current_group)
                current_role = msg.role
                current_group = cls(role=msg.role, content=[msg.content])
            else:
                # Same role, add to current message
                current_group.content.append(msg.content)

        # Add the last group
        if current_group is not None:
            result.append(current_group)

        return result

    def to_prompt_messages(self) -> List[PromptMessage]:
        """Convert this PromptMessageMultipart to a sequence of standard PromptMessages."""
        return [
            PromptMessage(role=self.role, content=content_part)
            for content_part in self.content
        ]

    @classmethod
    def parse_get_prompt_result(
        cls, result: GetPromptResult
    ) -> List["PromptMessageMultipart"]:
        """Parse a GetPromptResult into PromptMessageMultipart objects."""
        return cls.from_prompt_messages(result.messages)
