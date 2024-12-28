from typing import List

from .embedding_openai import OpenAIEmbeddingModel
from .intent_classifier import Intent
from .intent_classifier_embedding import EmbeddingIntentClassifier


class OpenAIEmbeddingIntentClassifier(EmbeddingIntentClassifier):
    """
    An intent classifier that uses OpenAI's embedding models for computing semantic simiarity based classifications.
    """

    def __init__(
        self,
        intents: List[Intent],
        embedding_model: OpenAIEmbeddingModel | None = None,
    ):
        embedding_model = embedding_model or OpenAIEmbeddingModel()
        super().__init__(embedding_model=embedding_model, intents=intents)

    @classmethod
    async def create(
        cls,
        intents: List[Intent],
        embedding_model: OpenAIEmbeddingModel | None = None,
    ) -> "OpenAIEmbeddingIntentClassifier":
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
