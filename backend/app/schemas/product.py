"""
Pydantic schemas for Product API.

Defines request query parameters and response shapes.
cost_price is intentionally excluded from all response schemas.
"""

import uuid
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class VariantSummary(BaseModel):
    """Lightweight variant info returned with a product search result."""

    id: uuid.UUID
    sku: str
    size: Optional[str] = None
    color: Optional[str] = None
    price: Decimal
    in_stock: bool  # quantity - reserved_quantity > 0

    model_config = {"from_attributes": True}


class ProductSearchResult(BaseModel):
    """Single product returned by the search endpoint.

    cost_price is intentionally absent to avoid leaking margin data.
    """

    id: uuid.UUID
    title: str
    description: Optional[str] = None
    price: Decimal
    attributes: dict
    variants: list[VariantSummary] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class ProductSearchResponse(BaseModel):
    """Paginated wrapper around search results."""

    total: int
    limit: int
    results: list[ProductSearchResult]


# ---------------------------------------------------------------------------
# Query parameter schema
# ---------------------------------------------------------------------------

class ProductSearchParams(BaseModel):
    """Validated query parameters for GET /api/products/search."""

    q: Optional[str] = Field(None, description="Keyword search across title and description")
    category: Optional[str] = Field(None, description="Filter by product category (stored in JSONB attributes)")
    min_price: Optional[Decimal] = Field(None, ge=Decimal("0"), description="Minimum price (inclusive)")
    max_price: Optional[Decimal] = Field(None, ge=Decimal("0"), description="Maximum price (inclusive)")
    color: Optional[str] = Field(None, description="Filter by color stored in JSONB attributes")
    size: Optional[str] = Field(None, description="Only return products that have a variant in this size")
    in_stock: Optional[bool] = Field(None, description="If true, only return products with available inventory")
    limit: int = Field(20, ge=1, le=50, description="Number of results to return (max 50)")

    @field_validator("max_price")
    @classmethod
    def max_price_gte_min_price(cls, v: Optional[Decimal], info) -> Optional[Decimal]:
        """Ensure max_price >= min_price when both are provided."""
        min_price = info.data.get("min_price")
        if v is not None and min_price is not None and v < min_price:
            raise ValueError("max_price must be greater than or equal to min_price")
        return v


# ---------------------------------------------------------------------------
# Semantic search schemas
# ---------------------------------------------------------------------------

class SemanticSearchResult(BaseModel):
    """Product returned by semantic vector search with similarity score.

    cost_price and raw embedding vectors are never exposed.
    """

    id: uuid.UUID
    title: str
    description: Optional[str] = None
    price: Decimal
    attributes: dict
    variants: list[VariantSummary] = Field(default_factory=list)
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")

    model_config = {"from_attributes": True}


class SemanticSearchResponse(BaseModel):
    """Wrapper around semantic search results."""

    query: str
    total: int
    limit: int
    results: list[SemanticSearchResult]


class SemanticSearchParams(BaseModel):
    """Validated query parameters for GET /api/products/semantic-search."""

    q: str = Field(..., min_length=1, max_length=500, description="Natural language search query")
    category: Optional[str] = Field(None, description="Deterministic filter: product category")
    min_price: Optional[Decimal] = Field(None, ge=Decimal("0"), description="Deterministic filter: minimum price")
    max_price: Optional[Decimal] = Field(None, ge=Decimal("0"), description="Deterministic filter: maximum price")
    color: Optional[str] = Field(None, description="Deterministic filter: color")
    size: Optional[str] = Field(None, description="Deterministic filter: variant size availability")
    in_stock: Optional[bool] = Field(None, description="Deterministic filter: available inventory check")
    limit: int = Field(10, ge=1, le=50, description="Number of results to return (default 10, max 50)")

    @field_validator("q")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string 'q' must not be empty or whitespace only")
        return v.strip()

    @field_validator("max_price")
    @classmethod
    def max_price_gte_min_price(cls, v: Optional[Decimal], info) -> Optional[Decimal]:
        min_price = info.data.get("min_price")
        if v is not None and min_price is not None and v < min_price:
            raise ValueError("max_price must be greater than or equal to min_price")
        return v

