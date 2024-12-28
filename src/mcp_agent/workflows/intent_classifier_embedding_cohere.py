from typing import List

from cohere import Client
from numpy import array, float32
from numpy.typing import NDArray

from ..context import get_current_config
from .intent_classifier import Intent
from .intent_classifier_embedding import (
    EmbeddingModel,
    EmbeddingIntentClassifier,
)

FloatArray = NDArray[float32]


class CohereEmbeddingModel(EmbeddingModel):
    """Cohere embedding model implementation"""

    def __init__(self, model: str = "embed-multilingual-v3.0"):
        self.client = Client(api_key=get_current_config().cohere.api_key)
        self.model = model
        # Cache the dimension since it's fixed per model
        # https://docs.cohere.com/v2/docs/cohere-embed
        self._embedding_dim = {
            "embed-english-v2.0": 4096,
            "embed-english-light-v2.0": 1024,
            "embed-english-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-v2.0": 768,
            "embed-multilingual-v3.0": 1024,
            "embed-multilingual-light-v3.0": 384,
        }[model]

    async def embed(self, data: List[str]) -> FloatArray:
        response = self.client.embed(
            texts=data,
            model=self.model,
            input_type="classification",
            embedding_types=["float"],
        )

        embeddings = array(response.embeddings, dtype=float32)
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class CohereEmbeddingIntentClassifier(EmbeddingIntentClassifier):
    """
    An intent classifier that uses Cohere's embedding models for computing semantic simiarity based classifications.
    """

    def __init__(
        self,
        intents: List[Intent],
        model: str = "embed-multilingual-v3.0",
    ):
        embedding_model = CohereEmbeddingModel(model=model)
        super().__init__(embedding_model=embedding_model, intents=intents)
