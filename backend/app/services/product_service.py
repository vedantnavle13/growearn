"""
Product Service: business logic layer orchestrating retrieval and response formatting.

Responsibilities:
- Receives validated search parameters from the API router.
- Constructs deterministic business filters (ProductFilters).
- Delegates semantic retrieval to TextRetriever.
- Delegates database access to ProductRepository.
- Shapes ORM objects into Pydantic response schemas (calculating in_stock and stripping cost_price).
"""

from decimal import Decimal
from typing import Optional, List

from sqlalchemy.orm import Session

from app.ai.embeddings import get_embedding_service, EmbeddingService
from app.repositories.product_repository import ProductRepository
from app.retrieval.filters import ProductFilters
from app.retrieval.text_retriever import TextRetriever
from app.schemas.product import (
    ProductSearchResponse,
    ProductSearchResult,
    SemanticSearchResponse,
    SemanticSearchResult,
    VariantSummary,
)


class ProductService:
    """Orchestrates product retrieval and shapes results for the API."""

    def __init__(
        self,
        db: Session,
        embedding_service: Optional[EmbeddingService] = None,
        text_retriever: Optional[TextRetriever] = None,
    ) -> None:
        self.repository = ProductRepository(db)
        self._embedding_service = embedding_service
        self.text_retriever = text_retriever or TextRetriever(
            repository=self.repository,
            embedding_service=self._embedding_service,
        )

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def search_products(
        self,
        *,
        q: Optional[str],
        category: Optional[str],
        min_price: Optional[Decimal],
        max_price: Optional[Decimal],
        color: Optional[str],
        size: Optional[str],
        in_stock: Optional[bool],
        limit: int,
    ) -> ProductSearchResponse:
        """
        Execute keyword and filter search and return a validated response schema.
        """
        filters = ProductFilters(
            category=category,
            min_price=min_price,
            max_price=max_price,
            color=color,
            size=size,
            in_stock=in_stock,
        )

        products, total = self.repository.search(
            q=q,
            filters=filters,
            limit=limit,
        )

        results = [self._to_schema(p) for p in products]

        return ProductSearchResponse(
            total=total,
            limit=limit,
            results=results,
        )

    def semantic_search_products(
        self,
        *,
        q: str,
        category: Optional[str] = None,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None,
        color: Optional[str] = None,
        size: Optional[str] = None,
        in_stock: Optional[bool] = None,
        limit: int = 10,
    ) -> SemanticSearchResponse:
        """
        Execute semantic vector search with deterministic filters:
        API -> ProductService -> TextRetriever -> ProductRepository
        """
        # 1. Encapsulate deterministic business filters
        filters = ProductFilters(
            category=category,
            min_price=min_price,
            max_price=max_price,
            color=color,
            size=size,
            in_stock=in_stock,
        )

        # 2. Delegate semantic retrieval to TextRetriever
        candidates = self.text_retriever.retrieve(
            query=q,
            limit=limit,
            filters=filters,
        )

        # 3. Shape ORM products into response schema
        results = [
            self._to_semantic_schema(product, similarity)
            for product, similarity in candidates
        ]

        return SemanticSearchResponse(
            query=q,
            total=len(results),
            limit=limit,
            results=results,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_schema(product) -> ProductSearchResult:
        """Map an ORM Product (with pre-loaded variants) to a response schema."""
        variant_summaries = []
        for variant in product.variants:
            available = 0
            if variant.inventory is not None:
                available = variant.inventory.quantity - variant.inventory.reserved_quantity

            variant_summaries.append(
                VariantSummary(
                    id=variant.id,
                    sku=variant.sku,
                    size=variant.size,
                    color=variant.color,
                    price=variant.price,
                    in_stock=available > 0,
                )
            )

        return ProductSearchResult(
            id=product.id,
            title=product.title,
            description=product.description,
            price=product.price,
            attributes=product.attributes,
            variants=variant_summaries,
        )

    @staticmethod
    def _to_semantic_schema(product, similarity_score: float) -> SemanticSearchResult:
        """Map an ORM Product and similarity score to SemanticSearchResult."""
        variant_summaries = []
        for variant in product.variants:
            available = 0
            if variant.inventory is not None:
                available = variant.inventory.quantity - variant.inventory.reserved_quantity

            variant_summaries.append(
                VariantSummary(
                    id=variant.id,
                    sku=variant.sku,
                    size=variant.size,
                    color=variant.color,
                    price=variant.price,
                    in_stock=available > 0,
                )
            )

        return SemanticSearchResult(
            id=product.id,
            title=product.title,
            description=product.description,
            price=product.price,
            attributes=product.attributes,
            variants=variant_summaries,
            similarity_score=similarity_score,
        )
