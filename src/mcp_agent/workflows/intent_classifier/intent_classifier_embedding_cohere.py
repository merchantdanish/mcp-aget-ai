from typing import List

from mcp_agent.workflows.embedding.embedding_cohere import CohereEmbeddingModel
from mcp_agent.workflows.intent_classifier.intent_classifier_base import Intent
from mcp_agent.workflows.intent_classifier.intent_classifier_embedding import (
    EmbeddingIntentClassifier,
)


class CohereEmbeddingIntentClassifier(EmbeddingIntentClassifier):
    """
    An intent classifier that uses Cohere's embedding models for computing semantic simiarity based classifications.
    """

    def __init__(
        self,
        intents: List[Intent],
        embedding_model: CohereEmbeddingModel | None = None,
    ):
        embedding_model = embedding_model or CohereEmbeddingModel()
        super().__init__(embedding_model=embedding_model, intents=intents)

    @classmethod
    async def create(
        cls,
        intents: List[Intent],
        embedding_model: CohereEmbeddingModel | None = None,
    ) -> "CohereEmbeddingIntentClassifier":
        """
        Factory method to create and initialize a classifier.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            intents=intents,
            embedding_model=embedding_model,
        )
        await instance.initialize()
        return instance
