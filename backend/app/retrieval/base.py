"""
Base Retriever Protocol for MerchantAI.

Defines the conceptual contract for all product retrievers (Text, Image, Hybrid):
    retrieve(query, filters, limit) -> List[Tuple[Product, float]]
"""

from typing import Protocol, Optional, List, Tuple, Any, runtime_checkable

from app.models.product import Product
from app.retrieval.filters import ProductFilters


@runtime_checkable
class Retriever(Protocol):
    """
    Standard interface/protocol for candidate product retrieval.
    
    Any concrete retriever (TextRetriever, ImageRetriever, HybridRetriever)
    implements a retrieve() method accepting a query, optional deterministic filters,
    and a limit, returning a list of (Product, similarity_score) tuples.
    """

    def retrieve(
        self,
        query: Any,
        filters: Optional[ProductFilters] = None,
        limit: int = 10,
    ) -> List[Tuple[Product, float]]:
        """
        Retrieves candidate products matching the query and filters.
        """
        ...
