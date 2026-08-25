"""
Text Retriever: semantic vector retrieval for text queries.

Responsibilities:
- Accepts a natural language text query.
- Generates a 1536-dimensional embedding using EmbeddingService.
- Queries pgvector via ProductRepository.
- Returns candidate products with cosine similarity scores.

Note:
- Does NOT contain business rules or deterministic filters (price, stock, size, category).
- Delegates pure database access to ProductRepository.
"""

from typing import Optional, List, Tuple, TYPE_CHECKING

from app.ai.embeddings import EmbeddingService, get_embedding_service
from app.models.product import Product
from app.retrieval.filters import ProductFilters

if TYPE_CHECKING:
    from app.repositories.product_repository import ProductRepository


class TextRetriever:
    """
    Handles text-based semantic retrieval against product vector embeddings.
    """

    def __init__(
        self,
        repository: "ProductRepository",
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        self.repository = repository
        self._embedding_service = embedding_service

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def retrieve(
        self,
        query: str,
        filters: Optional[ProductFilters] = None,
        limit: int = 10,
    ) -> List[Tuple[Product, float]]:
        """
        Executes text semantic retrieval:
        1. Generates 1536-dim vector for query using Gemini Embedding 2.
        2. Retrieves closest product matches from repository with similarity scores.
        """
        query_vector = self.embedding_service.embed_text(query)
        return self.repository.vector_search(
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )
