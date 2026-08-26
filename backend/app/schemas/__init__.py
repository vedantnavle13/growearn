"""Pydantic schemas and DTOs package."""

from app.schemas.integration import (
    MerchantProductInput,
    MerchantVariantInput,
    MerchantCustomerInput,
)
from app.schemas.intent import CommerceIntent, IntentParseRequest, IntentSearchRequest
from app.schemas.preference import CustomerPreferences
from app.schemas.product import (
    ProductSearchParams,
    ProductSearchResponse,
    ProductSearchResult,
    SemanticSearchParams,
    SemanticSearchResponse,
    SemanticSearchResult,
    IntentSearchResponse,
    IntentSearchResult,
    VariantSummary,
)

__all__ = [
    "MerchantProductInput",
    "MerchantVariantInput",
    "MerchantCustomerInput",
    "CommerceIntent",
    "IntentParseRequest",
    "IntentSearchRequest",
    "CustomerPreferences",
    "ProductSearchParams",
    "ProductSearchResponse",
    "ProductSearchResult",
    "SemanticSearchParams",
    "SemanticSearchResponse",
    "SemanticSearchResult",
    "IntentSearchResponse",
    "IntentSearchResult",
    "VariantSummary",
]
