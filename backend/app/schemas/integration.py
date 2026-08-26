"""
Pydantic schemas and integration contracts for external merchant platforms.

These schemas define the input contracts for future merchant platform connectors
(Shopify, WooCommerce, Custom Storefronts, etc.).

All external identifiers are scoped per merchant. Internal UUIDs remain internal.
"""

from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class MerchantVariantInput(BaseModel):
    """
    Contract schema for ingesting a product variant from a merchant system.
    """
    external_variant_id: Optional[str] = Field(
        default=None,
        description="Variant identifier in the merchant's external platform (e.g. 'variant-123-M')",
    )
    sku: Optional[str] = Field(
        default=None,
        description="Stock Keeping Unit identifier",
    )
    size: Optional[str] = Field(
        default=None,
        description="Variant size (e.g. 'S', 'M', 'L', 'XL', '42')",
    )
    color: Optional[str] = Field(
        default=None,
        description="Variant color name (e.g. 'black', 'navy')",
    )
    price: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Variant price",
    )
    stock: int = Field(
        default=0,
        ge=0,
        description="Available inventory stock quantity",
    )


class MerchantProductInput(BaseModel):
    """
    Contract schema for ingesting a product from a merchant system.
    """
    external_product_id: str = Field(
        ...,
        min_length=1,
        description="Product identifier in the merchant's external platform (e.g. 'product-123')",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Product title / name",
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed product description",
    )
    category: Optional[str] = Field(
        default=None,
        description="Product category (e.g. 'Shirts', 'Laptops', 'Footwear')",
    )
    brand: Optional[str] = Field(
        default=None,
        description="Brand name (e.g. 'UrbanThreads')",
    )
    price: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Base / display price of the product",
    )
    currency: str = Field(
        default="INR",
        description="Currency code (e.g. 'INR', 'USD')",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Public image URL for multimodal search and display",
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary product attributes (e.g. {'color': 'black', 'fit': 'slim', 'material': 'cotton'})",
    )
    variants: List[MerchantVariantInput] = Field(
        default_factory=list,
        description="List of product variants (sizes, colors, stock)",
    )

    @field_validator("external_product_id", "name")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return v.strip()


class MerchantCustomerInput(BaseModel):
    """
    Contract schema for ingesting customer data from a merchant system.
    """
    external_customer_id: str = Field(
        ...,
        min_length=1,
        description="Customer identifier in the merchant's external platform (e.g. 'customer-123')",
    )
    name: Optional[str] = Field(
        default=None,
        description="Customer full name",
    )
    email: Optional[str] = Field(
        default=None,
        description="Customer email address",
    )
    phone: Optional[str] = Field(
        default=None,
        description="Customer phone number",
    )

    @field_validator("external_customer_id")
    @classmethod
    def validate_external_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("external_customer_id cannot be empty or whitespace only")
        return v.strip()
