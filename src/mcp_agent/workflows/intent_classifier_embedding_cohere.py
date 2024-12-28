from typing import List

from .embedding_cohere import CohereEmbeddingModel
from .intent_classifier import Intent
from .intent_classifier_embedding import EmbeddingIntentClassifier


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
