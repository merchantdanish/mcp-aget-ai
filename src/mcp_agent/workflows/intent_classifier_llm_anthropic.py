from typing import List

from .augmented_llm_anthropic import AnthropicAugmentedLLM
from .intent_classifier import Intent
from .intent_classifier_llm import LLMIntentClassifier

CLASSIFIER_SYSTEM_INSTRUCTION = """
You are a precise intent classifier that analyzes input requests to determine their intended action or purpose.
You are provided with a request and a list of intents to choose from.
You can choose one or more intents, or choose none if no intent is appropriate.
"""


class AnthropicLLMIntentClassifier(LLMIntentClassifier):
    """
    An LLM router that uses an Anthropic model to make routing decisions.
    """

    def __init__(
        self,
        intents: List[Intent],
        classification_instruction: str | None = None,
    ):
        anthropic_llm = AnthropicAugmentedLLM(instruction=CLASSIFIER_SYSTEM_INSTRUCTION)

        super().__init__(
            llm=anthropic_llm,
            intents=intents,
            classification_instruction=classification_instruction,
        )

    @classmethod
    async def create(
        cls,
        intents: List[Intent],
        classification_instruction: str | None = None,
    ) -> "AnthropicLLMIntentClassifier":
        """
        Factory method to create and initialize a classifier.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            intents=intents,
            classification_instruction=classification_instruction,
        )
        await instance.initialize()
        return instance
