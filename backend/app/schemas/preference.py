"""
Pydantic schemas for lightweight customer personalization and preference signals.
"""

import uuid
from decimal import Decimal
from typing import Optional, List
from pydantic import BaseModel, Field


class CustomerPreferences(BaseModel):
    """
    Lightweight, deterministic customer preference profile derived from historical events and orders.

    Fields:
    - customer_id: UUID of the customer if identified, or None.
    - preferred_categories: Ordered list of categories the customer interacts with or purchases most frequently.
    - preferred_brands: Ordered list of preferred brands.
    - preferred_colors: Ordered list of preferred colors (normalized lowercase).
    - preferred_price_min: Lower bound of typical purchasing/viewing price range.
    - preferred_price_max: Upper bound of typical purchasing/viewing price range.
    - preferred_product_ids: List of product IDs previously purchased or added to cart.
    - total_events_analyzed: Count of historical interactions analyzed.
    """

    customer_id: Optional[uuid.UUID] = None
    preferred_categories: List[str] = Field(default_factory=list)
    preferred_brands: List[str] = Field(default_factory=list)
    preferred_colors: List[str] = Field(default_factory=list)
    preferred_price_min: Optional[Decimal] = None
    preferred_price_max: Optional[Decimal] = None
    preferred_product_ids: List[str] = Field(default_factory=list)
    total_events_analyzed: int = 0
