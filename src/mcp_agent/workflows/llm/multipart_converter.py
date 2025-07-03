from typing import Generic, Protocol

from mcp_agent.utils.prompt_message_multipart import PromptMessageMultipart
from mcp.types import PromptMessage
from mcp_agent.workflows.llm.augmented_llm import MessageTypes
from mcp_agent.workflows.llm.augmented_llm import MessageParamT, MessageT


class MessageConverter(Protocol, Generic[MessageParamT, MessageT]):
    @staticmethod
    def from_prompt_message_multipart(
        multipart_msg: PromptMessageMultipart, concatenate_text_blocks: bool = False
    ) -> MessageParamT:
        """Convert a PromptMessageMultipart to a Provider-compatible message param type"""
        ...

    @staticmethod
    def from_prompt_message(message: PromptMessage) -> MessageParamT:
        """Convert a PromptMessage to a Provider-compatible message param type"""
        ...

    @staticmethod
    def from_mixed_messages(message: MessageTypes) -> list[MessageParamT]:
        """Convert a mixed message type to a list of Provider-compatible message param types"""
        ...
