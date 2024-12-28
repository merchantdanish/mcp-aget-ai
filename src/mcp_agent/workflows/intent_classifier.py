from abc import ABC, abstractmethod
from typing import Dict, List
from pydantic import BaseModel, Field


class Intent(BaseModel):
    """A class that represents a single intent category"""

    name: str
    """The name of the intent"""

    description: str | None = None
    """A description of what this intent represents"""

    examples: List[str] = Field(default_factory=list)
    """Example phrases or requests that match this intent"""

    metadata: Dict[str, str] = Field(default_factory=dict)
    """Additional metadata about the intent that might be useful for classification"""


class IntentClassificationResult(BaseModel):
    """A class that represents the result of intent classification"""

    intent: str
    """The classified intent name"""

    p_score: float | None = None
    """
    The probability score (i.e. 0->1) of the classification. 
    This is optional and may only be provided if the classifier is probabilistic (e.g. a probabilistic binary classifier).
    """

    extracted_entities: Dict[str, str] = Field(default_factory=dict)
    """Any entities or parameters extracted from the input request that are relevant to the intent"""


class IntentClassifier(ABC):
    """
    Base class for intent classification. This can be implemented using different approaches
    like LLMs, embedding models, traditional ML classification models, or rule-based systems.

    When to use this:
        - When you need to understand the user's intention before routing or processing
        - When you want to extract structured information from natural language inputs
        - When you need to handle multiple related but distinct types of requests

    Examples:
        - Classifying customer service requests (complaint, question, feedback)
        - Understanding user commands in a chat interface
        - Determining the type of analysis requested for a dataset
    """

    def __init__(self, intents: List[Intent]):
        self.intents = {intent.name: intent for intent in intents}
        self.initialized: bool = False

        if not self.intents:
            raise ValueError("At least one intent must be provided")

    @abstractmethod
    async def classify(
        self, request: str, top_k: int = 1
    ) -> List[IntentClassificationResult]:
        """
        Classify the input request into one or more intents.

        Args:
            request: The input text to classify
            top_k: Maximum number of top intent matches to return. May return fewer.

        Returns:
            List of classification results, ordered by confidence
        """

    async def initialize(self):
        """Initialize the classifier. Override this method if needed."""
        self.initialized = True
