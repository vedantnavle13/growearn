"""
Hybrid Retriever (Interface / Stub).

Defines the contract for future multi-modal hybrid retrieval (Text + Image + Filters).

Planned Future Workflow:
1. Concurrently query TextRetriever and ImageRetriever.
2. Combine and normalize similarity scores using Reciprocal Rank Fusion (RRF)
   or weighted score blending:
   score = (w_text * text_score) + (w_image * image_score).
3. Apply deterministic business filters.
4. Return fused candidate product rankings.
"""

from typing import Optional, List, Tuple, Any

from app.models.product import Product
from app.retrieval.filters import ProductFilters
from app.retrieval.text_retriever import TextRetriever
from app.retrieval.image_retriever import ImageRetriever


class HybridRetriever:
    """
    Interface and stub for future hybrid (text + image) multi-modal retrieval.
    
    Not implemented yet. Will be activated when image retrieval is implemented.
    """

    def __init__(
        self,
        text_retriever: TextRetriever,
        image_retriever: Optional[ImageRetriever] = None,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
    ) -> None:
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever or ImageRetriever()
        self.text_weight = text_weight
        self.image_weight = image_weight

    def retrieve(
        self,
        query: Any = None,
        filters: Optional[ProductFilters] = None,
        limit: int = 10,
        image_data: Optional[bytes] = None,
        image_url: Optional[str] = None,
    ) -> List[Tuple[Product, float]]:
        """
        Execute multi-modal hybrid retrieval.
        """
        raise NotImplementedError(
            "HybridRetriever.retrieve() is a staged architecture stub and will be implemented when multi-modal search is activated."
        )
