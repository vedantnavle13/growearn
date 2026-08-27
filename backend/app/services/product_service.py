"""
Product Service: business logic layer orchestrating multi-tenant product retrieval and response formatting.

Responsibilities:
- Receives validated search parameters and explicit merchant context from the API router.
- Constructs deterministic business filters (ProductFilters).
- Delegates intent extraction to IntentService.
- Delegates semantic retrieval to TextRetriever with merchant_id boundary.
- Delegates database access to ProductRepository with merchant_id boundary.
- Shapes ORM objects into Pydantic response schemas (calculating in_stock and stripping cost_price).
"""

import uuid
from decimal import Decimal
from typing import Optional, List

from sqlalchemy.orm import Session

from app.ai.embeddings import get_embedding_service, EmbeddingService
from app.repositories.product_repository import ProductRepository
from app.retrieval.filters import ProductFilters
from app.retrieval.ranker import ProductRanker
from app.retrieval.text_retriever import TextRetriever
from app.core.config import settings
from app.schemas.intent import CommerceIntent
from app.schemas.preference import CustomerPreferences
from app.schemas.product import (
    ProductSearchResponse,
    ProductSearchResult,
    SemanticSearchResponse,
    SemanticSearchResult,
    IntentSearchResponse,
    IntentSearchResult,
    VariantSummary,
)
from app.services.category_concept_service import CategoryConceptService
from app.services.customer_preference_service import CustomerPreferenceService
from app.services.intent_service import IntentService


class ProductService:
    """Orchestrates product retrieval and shapes results for the API with tenant isolation."""

    def __init__(
        self,
        db: Session,
        embedding_service: Optional[EmbeddingService] = None,
        text_retriever: Optional[TextRetriever] = None,
        intent_service: Optional[IntentService] = None,
        ranker: Optional[ProductRanker] = None,
        preference_service: Optional[CustomerPreferenceService] = None,
    ) -> None:
        self.db = db
        self.repository = ProductRepository(db)
        self._embedding_service = embedding_service
        self.text_retriever = text_retriever or TextRetriever(
            repository=self.repository,
            embedding_service=self._embedding_service,
        )
        self._intent_service = intent_service
        self._ranker = ranker
        self._preference_service = preference_service
        self._category_concept_service: Optional[CategoryConceptService] = None

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    @property
    def intent_service(self) -> IntentService:
        if self._intent_service is None:
            self._intent_service = IntentService()
        return self._intent_service

    @property
    def ranker(self) -> ProductRanker:
        if self._ranker is None:
            self._ranker = ProductRanker()
        return self._ranker

    @property
    def preference_service(self) -> CustomerPreferenceService:
        if self._preference_service is None:
            self._preference_service = CustomerPreferenceService(self.db)
        return self._preference_service

    @property
    def category_concept_service(self) -> CategoryConceptService:
        if self._category_concept_service is None:
            self._category_concept_service = CategoryConceptService(self.db)
        return self._category_concept_service

    def search_products(
        self,
        *,
        merchant_id: uuid.UUID,
        q: Optional[str] = None,
        category: Optional[str] = None,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None,
        color: Optional[str] = None,
        size: Optional[str] = None,
        in_stock: Optional[bool] = None,
        limit: int = 20,
    ) -> ProductSearchResponse:
        """
        Execute keyword and filter search scoped to merchant_id and return a validated response schema.
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
            merchant_id=merchant_id,
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
        merchant_id: uuid.UUID,
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
        Execute semantic vector search with deterministic filters scoped to merchant_id:
        API -> ProductService -> TextRetriever -> ProductRepository (WHERE merchant_id = :id)
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

        # 2. Delegate semantic retrieval to TextRetriever with merchant_id boundary
        candidates = self.text_retriever.retrieve(
            merchant_id=merchant_id,
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

    def intent_search_products(
        self,
        *,
        merchant_id: uuid.UUID,
        raw_query: str,
        customer_id: Optional[uuid.UUID] = None,
        external_customer_id: Optional[str] = None,
        limit: int = settings.DEFAULT_SEARCH_LIMIT,
    ) -> IntentSearchResponse:
        """
        End-to-end intent-driven product search scoped to merchant_id:

        raw_query -> IntentService -> CommerceIntent -> CategoryConceptService -> ProductFilters -> TextRetriever(merchant_id) -> ProductRanker(with CustomerPreferences(merchant_id)) -> top N results
        """
        # 1. Extract structured intent from natural language
        intent = self.intent_service.extract_intent(raw_query)

        # 2. Determine effective limit: use requested_limit from intent if specified, otherwise use passed limit
        # Clamp to MAX_SEARCH_LIMIT
        effective_limit = intent.requested_limit if intent.requested_limit is not None else limit
        effective_limit = min(effective_limit, settings.MAX_SEARCH_LIMIT)

        # 3. Resolve category vs category_concept using CategoryConceptService
        concept_service = CategoryConceptService(self.db, str(merchant_id))
        hard_filter_category, concept_categories = concept_service.resolve_category_or_concept(
            exact_category=intent.category,
            category_concept=intent.category_concept
        )

        # 4. Build deterministic hard filters from intent (hard constraints never overridden)
        # Use resolved hard_filter_category (exact match only), NOT category_concept
        filters = ProductFilters(
            category=hard_filter_category,
            brand=intent.brand,
            min_price=intent.min_price,
            max_price=intent.max_price,
            color=intent.color,
            size=intent.size,
            in_stock=True,  # Default: only show available products
        )

        # 5. Retrieve candidate products strictly within merchant boundary passing all hard constraints
        # Retrieve more candidates than needed to account for ranking/filtering
        candidate_limit = min(settings.MAX_SEARCH_LIMIT, max(effective_limit * 3, 30))
        candidates = self.text_retriever.retrieve(
            merchant_id=merchant_id,
            query=intent.query,
            limit=candidate_limit,
            filters=filters,
        )

        # 6. Load customer preferences strictly scoped to merchant_id
        preferences = (
            self.preference_service.get_preferences(
                merchant_id=merchant_id,
                customer_id=customer_id,
                external_customer_id=external_customer_id,
            )
            if (customer_id or external_customer_id)
            else None
        )

        # 7. Multi-signal deterministic ranking (semantic + keyword + dynamic attributes + personalization + category_concept)
        scored_products = self.ranker.rank(
            candidates=candidates,
            intent=intent,
            preferences=preferences,
            concept_categories=concept_categories,
            limit=effective_limit,
        )

        # 8. Shape results using final composite relevance score
        results = [
            self._to_intent_search_schema(sp.product, sp.final_score)
            for sp in scored_products
        ]

        # Include category_concept in response for debugging/transparency
        intent_dict = intent.model_dump(mode="json")
        intent_dict["category_concept_categories"] = concept_categories

        return IntentSearchResponse(
            query=raw_query,
            intent=intent_dict,
            total=len(results),
            limit=effective_limit,
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

    @staticmethod
    def _to_intent_search_schema(product, similarity_score: float) -> IntentSearchResult:
        """Map an ORM Product and similarity score to IntentSearchResult."""
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

        return IntentSearchResult(
            id=product.id,
            title=product.title,
            description=product.description,
            price=product.price,
            attributes=product.attributes,
            variants=variant_summaries,
            similarity_score=similarity_score,
        )
