from abc import ABC, abstractmethod
from typing import Dict, List

from numpy import float32, mean
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity

from .intent_classifier import Intent, IntentClassifier, IntentClassificationResult

FloatArray = NDArray[float32]


class EmbeddingIntent(Intent):
    """An intent with embedding information"""

    embedding: FloatArray | None = None
    """Pre-computed embedding for this intent"""


class EmbeddingModel(ABC):
    """Abstract interface for embedding models"""

    @abstractmethod
    async def embed(self, data: List[str]) -> FloatArray:
        """
        Generate embeddings for a list of messages

        Args:
            data: List of text strings to embed

        Returns:
            Array of embeddings, shape (len(texts), embedding_dim)
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings"""


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

            similarity_scores = self._compute_similarity_scores(
                request_embedding, intent.embedding
            )

            # Compute overall confidence score
            confidence = self._compute_confidence(similarity_scores)

            results.append(
                IntentClassificationResult(
                    intent=intent_name,
                    p_score=confidence,
                )
            )

        results.sort(key=lambda x: x.p_score, reverse=True)
        return results[:top_k]

    def _compute_similarity_scores(
        self, text_embedding: FloatArray, intent_embedding: FloatArray
    ) -> Dict[str, float]:
        """
        Compute different similarity metrics between embeddings
        """
        # Reshape for sklearn's cosine_similarity
        text_emb = text_embedding.reshape(1, -1)
        intent_emb = intent_embedding.reshape(1, -1)

        cosine_sim = float(cosine_similarity(text_emb, intent_emb)[0, 0])

        # Could add other similarity metrics here
        return {
            "cosine": cosine_sim,
            # "euclidean": float(euclidean_similarity),
            # "dot_product": float(dot_product)
        }

    def _compute_confidence(self, similarity_scores: Dict[str, float]) -> float:
        """
        Compute overall confidence score from individual similarity metrics
        """
        # For now, just use cosine similarity as confidence
        # Could implement more sophisticated combination of scores
        return similarity_scores["cosine"]
