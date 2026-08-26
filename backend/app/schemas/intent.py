"""
Pydantic schemas for Generic Commerce Intent Extraction.

Supports ANY merchant product category (clothing, electronics, stationery, watches, furniture, etc.).
"""

from decimal import Decimal
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator, model_validator


class CommerceIntent(BaseModel):
    """
    Generic, cross-category structured commerce intent extracted from natural language user queries.
    
    Fields:
    - query: Clean semantic search query expressing the core user product search intent (without conversational filler).
    - category: Inferred product category (e.g., Shirts, Laptops, Stationery), or null if not specified/confident.
    - brand: Explicitly requested brand name, or null.
    - min_price: Optional minimum price filter (inclusive, >= 0).
    - max_price: Optional maximum price filter (inclusive, >= 0).
    - color: Optional color or finish.
    - size: Optional product/variant size.
    - attributes: Key-value dictionary of category-specific constraints/preferences strongly implied by user request.
    """

    query: str = Field(
        ...,
        description="Meaningful semantic search query expressing product intent without conversational filler",
    )
    category: Optional[str] = Field(
        default=None,
        description="Broad or specific product category (e.g. Shirts, Laptops, Stationery), or null",
    )
    brand: Optional[str] = Field(
        default=None,
        description="Explicitly requested brand, or null",
    )
    min_price: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Minimum price boundary (inclusive), or null",
    )
    max_price: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Maximum price boundary (inclusive), or null",
    )
    color: Optional[str] = Field(
        default=None,
        description="Requested color or finish, or null",
    )
    size: Optional[str] = Field(
        default=None,
        description="Requested size, or null",
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Category-specific key-value constraints actually present in or strongly implied by the request",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string cannot be empty or whitespace only")
        return v.strip()

    @field_validator("attributes")
    @classmethod
    def validate_attributes_dict(cls, v: Any) -> Dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("Attributes must be a dictionary")
        return v

    @model_validator(mode="after")
    def validate_price_range(self) -> "CommerceIntent":
        if self.min_price is not None and self.max_price is not None:
            if self.min_price > self.max_price:
                raise ValueError(
                    f"min_price ({self.min_price}) must be less than or equal to max_price ({self.max_price})"
                )
        return self


# ---------------------------------------------------------------------------
# Request / Response schemas for intent endpoints
# ---------------------------------------------------------------------------

class IntentParseRequest(BaseModel):
    """Request body for POST /api/intent/parse."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language user query to extract intent from",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query must not be empty or whitespace only")
        return v.strip()


import uuid


class IntentSearchRequest(BaseModel):
    """Request body for POST /api/products/search/intent."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language user query for intent-driven product search",
    )
    customer_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional customer ID to apply lightweight customer personalization signals",
    )
    external_customer_id: Optional[str] = Field(
        default=None,
        description="Optional external customer ID from merchant system for customer personalization",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results (default 10, max 50)",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query must not be empty or whitespace only")
        return v.strip()

