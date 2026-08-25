"""
Image Retriever (Interface / Stub).

Defines the contract for future image-based visual product retrieval.

Planned Future Workflow:
1. Accept image bytes or public image URL.
2. Generate 1536-dimensional image embedding via Gemini Multimodal Embedding model.
3. Query pgvector cosine distance on `Product.image_embedding`.
4. Return candidate products with visual similarity scores.
"""

from typing import Optional, List, Tuple, Any, TYPE_CHECKING

from app.models.product import Product
from app.retrieval.filters import ProductFilters

if TYPE_CHECKING:
    from app.repositories.product_repository import ProductRepository


class ImageRetriever:
    """
    Interface and stub for future visual/image semantic retrieval.
    
    Not implemented yet. Will be activated in the multimodal search step.
    """

    def __init__(self, repository: Optional["ProductRepository"] = None) -> None:
        self.repository = repository

    def retrieve(
        self,
        query: Any,
        filters: Optional[ProductFilters] = None,
        limit: int = 10,
    ) -> List[Tuple[Product, float]]:
        """
        Retrieve products by visual similarity from image data or image URL.
        """
        raise NotImplementedError(
            "ImageRetriever.retrieve() is a staged architecture stub and will be implemented in the multimodal step."
        )

    def retrieve_by_url(
        self,
        image_url: str,
        filters: Optional[ProductFilters] = None,
        limit: int = 10,
    ) -> List[Tuple[Product, float]]:
        """
        Retrieve products by visual similarity from an image URL.
        """
        raise NotImplementedError(
            "ImageRetriever.retrieve_by_url() is a staged architecture stub and will be implemented in the multimodal step."
        )
