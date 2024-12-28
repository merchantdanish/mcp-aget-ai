from typing import List

from numpy import array, float32, stack
from openai import OpenAI

from ..context import get_current_config
from .intent_classifier import Intent
from .intent_classifier_embedding import (
    EmbeddingModel,
    EmbeddingIntentClassifier,
    FloatArray,
)


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model implementation"""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=get_current_config().openai.api_key)
        self.model = model
        # Cache the dimension since it's fixed per model
        self._embedding_dim = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }[model]

    async def embed(self, data: List[str]) -> FloatArray:
        response = self.client.embeddings.create(
            model=self.model, input=data, encoding_format="float"
        )

        # Sort the embeddings by their index to ensure correct order
        sorted_embeddings = sorted(response.data, key=lambda x: x["index"])

        # Stack all embeddings into a single array
        embeddings = stack(
            [
                array(embedding["embedding"], dtype=float32)
                for embedding in sorted_embeddings
            ]
        )
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class OpenAIEmbeddingIntentClassifier(EmbeddingIntentClassifier):
    """
    An intent classifier that uses OpenAI's embedding models for computing semantic simiarity based classifications.
    """

    def __init__(
        self,
        intents: List[Intent],
        model: str = "text-embedding-3-small",
    ):
        embedding_model = OpenAIEmbeddingModel(model=model)
        super().__init__(embedding_model=embedding_model, intents=intents)
