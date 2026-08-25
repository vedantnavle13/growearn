"""
Product API router: HTTP handling only.

Responsibilities:
- Parse and validate query parameters via Pydantic.
- Inject a database session via FastAPI's Depends.
- Delegate all logic to ProductService.
- Return the validated Pydantic response.
"""

from decimal import Decimal
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.product import (
    ProductSearchParams,
    ProductSearchResponse,
    SemanticSearchParams,
    SemanticSearchResponse,
)
from app.services.product_service import ProductService

router = APIRouter(prefix="/api/products", tags=["products"])


@router.get(
    "/search",
    response_model=ProductSearchResponse,
    summary="Search products",
    description=(
        "Keyword and filter-based product search. "
        "Supports filtering by category, price range, color, size, and stock availability."
    ),
)
def search_products(
    q: Annotated[Optional[str], Query(description="Keyword to search in title and description")] = None,
    category: Annotated[Optional[str], Query(description="Filter by category (e.g. Shirts, Jeans)")] = None,
    min_price: Annotated[Optional[Decimal], Query(ge=0, description="Minimum price")] = None,
    max_price: Annotated[Optional[Decimal], Query(ge=0, description="Maximum price")] = None,
    color: Annotated[Optional[str], Query(description="Filter by color (e.g. Navy Blue)")] = None,
    size: Annotated[Optional[str], Query(description="Only show products with a variant in this size")] = None,
    in_stock: Annotated[Optional[bool], Query(description="If true, only return in-stock products")] = None,
    limit: Annotated[int, Query(ge=1, le=50, description="Max results to return (default 20, max 50)")] = 20,
    db: Session = Depends(get_db),
) -> ProductSearchResponse:
    """
    Search products by keyword, category, price, color, size, and stock status.
    cost_price is never exposed in the response.
    """
    # Validate combined parameters using the schema
    try:
        params = ProductSearchParams(
            q=q,
            category=category,
            min_price=min_price,
            max_price=max_price,
            color=color,
            size=size,
            in_stock=in_stock,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    service = ProductService(db)
    return service.search_products(
        q=params.q,
        category=params.category,
        min_price=params.min_price,
        max_price=params.max_price,
        color=params.color,
        size=params.size,
        in_stock=params.in_stock,
        limit=params.limit,
    )


@router.get(
    "/semantic-search",
    response_model=SemanticSearchResponse,
    summary="Semantic vector search",
    description=(
        "Cross-modal/natural-language semantic search powered by Gemini Embedding 2 and pgvector. "
        "Calculates cosine similarity against product embeddings and applies hard deterministic filters."
    ),
)
def semantic_search_products(
    q: Annotated[str, Query(min_length=1, description="Natural language search query")],
    category: Annotated[Optional[str], Query(description="Deterministic filter: product category")] = None,
    min_price: Annotated[Optional[Decimal], Query(ge=0, description="Deterministic filter: minimum price")] = None,
    max_price: Annotated[Optional[Decimal], Query(ge=0, description="Deterministic filter: maximum price")] = None,
    color: Annotated[Optional[str], Query(description="Deterministic filter: color")] = None,
    size: Annotated[Optional[str], Query(description="Deterministic filter: variant size availability")] = None,
    in_stock: Annotated[Optional[bool], Query(description="Deterministic filter: available inventory check")] = None,
    limit: Annotated[int, Query(ge=1, le=50, description="Max results to return (default 10, max 50)")] = 10,
    db: Session = Depends(get_db),
) -> SemanticSearchResponse:
    """
    Search products semantically using Gemini Embedding 2 vectors and pgvector cosine distance.
    Applies deterministic filters (category, price range, color, size, in-stock) alongside vector ranking.
    """
    try:
        params = SemanticSearchParams(
            q=q,
            category=category,
            min_price=min_price,
            max_price=max_price,
            color=color,
            size=size,
            in_stock=in_stock,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    service = ProductService(db)
    return service.semantic_search_products(
        q=params.q,
        category=params.category,
        min_price=params.min_price,
        max_price=params.max_price,
        color=params.color,
        size=params.size,
        in_stock=params.in_stock,
        limit=params.limit,
    )

