"""
Pydantic schemas for Generic Commerce Intent Extraction.

Supports ANY merchant product category (clothing, electronics, stationery, watches, furniture, etc.).
"""

from decimal import Decimal
from enum import Enum
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator, model_validator


class CommerceIntent(BaseModel):
    """
    Generic, cross-category structured commerce intent extracted from natural language user queries.
    
    Fields:
    - query: Clean semantic search query expressing the core user product search intent (without conversational filler).
    - category: Inferred EXACT product category (e.g., Shirts, Laptops, Stationery), or null if not specified/confident.
                Use ONLY when user specifies a specific category that likely exists in the database.
    - category_concept: Broad semantic category concept (e.g., clothing, electronics, footwear), or null.
                        Use for broad user terms like "clothes", "electronics", "shoes" that map to multiple DB categories.
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
        description="EXACT product category (e.g. Shirts, Laptops, Stationery), or null. Only use when user specifies a specific category.",
    )
    category_concept: Optional[str] = Field(
        default=None,
        description="Broad semantic category concept (e.g. clothing, electronics, footwear), or null. Use for broad terms like 'clothes', 'electronics', 'shoes'.",
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
    requested_limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="User-specified result count limit (e.g., 'top 5', 'show me 3'). Null if not specified.",
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

    @field_validator("category_concept")
    @classmethod
    def validate_category_concept(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return v.strip().lower()
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
from app.core.config import settings


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
        default=settings.DEFAULT_SEARCH_LIMIT,
        ge=1,
        le=settings.MAX_SEARCH_LIMIT,
        description=f"Maximum number of results (default {settings.DEFAULT_SEARCH_LIMIT}, max {settings.MAX_SEARCH_LIMIT})",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query must not be empty or whitespace only")
        return v.strip()


# ---------------------------------------------------------------------------
# Purchase Intent — distinct from CommerceIntent (search).
# Used to drive the three checkout modes:
#   SINGLE_PRODUCT  — buy a specific item from last search results or by ID
#   CART            — checkout the entire active cart
# The LLM fills this via tool parameters; backend resolves the actual IDs.
# ---------------------------------------------------------------------------

class PurchaseScope(str, Enum):
    """Whether the user wants to buy a single product or their whole cart."""
    SINGLE_PRODUCT = "SINGLE_PRODUCT"
    CART = "CART"


class ProductReference(str, Enum):
    """How the specific product was referenced by the user."""
    SEARCH_RESULT = "SEARCH_RESULT"   # "the 1st one", "second product"
    PRODUCT_ID = "PRODUCT_ID"         # explicit UUID from user
    NONE = "NONE"                     # no product reference (cart checkout)


class PurchaseIntent(BaseModel):
    """
    Structured purchase intent extracted from LLM tool calls.

    Examples:
      "buy the first one"
        → purchase_scope=SINGLE_PRODUCT, product_reference=SEARCH_RESULT, reference_position=1

      "checkout my cart"
        → purchase_scope=CART, product_reference=NONE

      "buy product <UUID>"
        → purchase_scope=SINGLE_PRODUCT, product_reference=PRODUCT_ID, product_id="<UUID>"

    The backend (PurchaseIntentResolver) converts reference_position into
    the actual database UUIDs. The LLM MUST NOT invent IDs.
    """

    purchase_scope: PurchaseScope
    product_reference: ProductReference = ProductReference.NONE
    reference_position: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="1-based position in last_search_results (for SEARCH_RESULT reference)",
    )
    product_id: Optional[str] = Field(
        default=None,
        description="Explicit product UUID string (for PRODUCT_ID reference only)",
    )
    quantity: int = Field(
        default=1,
        ge=1,
        le=99,
        description="Quantity to purchase",
    )
    size: Optional[str] = Field(
        default=None,
        description="User-specified size/variant (e.g. 'L', 'M', 'XL')",
    )
    address_hint: Optional[str] = Field(
        default=None,
        description="Address reference: 'default', 'home', 'office', '2', or null",
    )
