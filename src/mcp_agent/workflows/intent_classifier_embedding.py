from typing import List

from numpy import mean

from .embedding import (
    FloatArray,
    EmbeddingModel,
    compute_confidence,
    compute_similarity_scores,
)
from .intent_classifier import Intent, IntentClassifier, IntentClassificationResult


class EmbeddingIntent(Intent):
    """An intent with embedding information"""

    embedding: FloatArray | None = None
    """Pre-computed embedding for this intent"""


class EmbeddingIntentClassifier(IntentClassifier):
    """
    An intent classifier that uses embedding similarity for classification.
    Supports different embedding models through the EmbeddingModel interface.

    Features:
    - Semantic similarity based classification
    - Support for example-based learning
    - Flexible embedding model support
    - Multiple similarity computation strategies
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        intents: List[Intent],
    ):
        super().__init__(intents=intents)
        self.embedding_model = embedding_model
        self.initialized = False

    async def initialize(self):
        """
        Precompute embeddings for all intents by combining their
        descriptions and examples
        """
        for intent in self.intents.values():
            # Combine all text for a rich intent representation
            intent_texts = [intent.name, intent.description] + intent.examples

            # Get embeddings for all texts
            embeddings = await self.embedding_model.embed(intent_texts)

            # Use mean pooling to combine embeddings
            embedding = mean(embeddings, axis=0)

            # Create intents with embeddings
            self.intents[intent.name] = EmbeddingIntent(
                **intent,
                embedding=embedding,
            )

        self.initialized = True

    async def classify(
        self, request: str, top_k: int = 1
    ) -> List[IntentClassificationResult]:
        """
        Classify the input text into one or more intents

        Args:
            text: Input text to classify
            top_k: Maximum number of top matches to return

        Returns:
            List of classification results, ordered by confidence
        """
        if not self.initialized:
            await self.initialize()

        # Get embedding for input
        embeddings = await self.embedding_model.embed([request])
        request_embedding = embeddings[0]  # Take first since we only embedded one text

        results: List[IntentClassificationResult] = []
        for intent_name, intent in self.intents.items():
            if intent.embedding is None:
                continue

            similarity_scores = compute_similarity_scores(
                request_embedding, intent.embedding
            )

            # Compute overall confidence score
            confidence = compute_confidence(similarity_scores)

            results.append(
                IntentClassificationResult(
                    intent=intent_name,
                    p_score=confidence,
                )
            )

        results.sort(key=lambda x: x.p_score, reverse=True)
        return results[:top_k]
