"""
Base Retriever Protocol for MerchantAI with multi-merchant scoping.

Defines the conceptual contract for all product retrievers (Text, Image, Hybrid):
    retrieve(merchant_id, query, filters, limit) -> List[Tuple[Product, float]]
"""

import uuid
from typing import Protocol, Optional, List, Tuple, Any, runtime_checkable

from app.models.product import Product
from app.retrieval.filters import ProductFilters


@runtime_checkable
class Retriever(Protocol):
    """
    Standard interface/protocol for candidate product retrieval with mandatory tenant isolation.
    
    Any concrete retriever (TextRetriever, ImageRetriever, HybridRetriever)
    implements a retrieve() method accepting merchant_id, query, optional deterministic filters,
    and a limit, returning a list of (Product, similarity_score) tuples strictly scoped to merchant_id.
    """

    def retrieve(
        self,
        *,
        merchant_id: uuid.UUID,
        query: Any,
        filters: Optional[ProductFilters] = None,
        limit: int = 10,
    ) -> List[Tuple[Product, float]]:
        """
        Retrieves candidate products matching the query and filters within merchant_id boundary.
        """
        ...
