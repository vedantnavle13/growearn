"""
Deterministic business and database filters for product catalog queries.

Contains pure, deterministic filtering logic (price range, stock, size, category, color)
independent of vector similarity or ranking algorithms.

IMPORTANT BUSINESS RULES:
- Price, color, size, and stock constraints are VARIANT-level constraints.
  A product qualifies if there EXISTS at least one variant satisfying ALL
  requested variant-level constraints simultaneously.
- Category is a product-level attribute (stored in JSONB attributes->>'category').
- All constraints are HARD: they are enforced at the PostgreSQL WHERE level,
  not by post-retrieval Python filtering.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, List

from sqlalchemy import select, and_
from sqlalchemy.sql.elements import ColumnElement

from app.models.product import Product, ProductVariant, Inventory


@dataclass(frozen=True)
class ProductFilters:
    """
    Encapsulates hard deterministic filter criteria for product search queries.

    Fields:
    - category: Match against product JSONB attributes->>'category' (ILIKE)
    - min_price: Inclusive minimum VARIANT price
    - max_price: Inclusive maximum VARIANT price
    - color: Match against VARIANT color column (ILIKE)
    - size: Match against VARIANT size column (ILIKE)
    - in_stock: Variant must have available inventory (quantity - reserved_quantity) > 0
    """
    category: Optional[str] = None
    brand: Optional[str] = None
    min_price: Optional[Decimal] = None
    max_price: Optional[Decimal] = None
    color: Optional[str] = None
    size: Optional[str] = None
    in_stock: Optional[bool] = None

    def to_sqlalchemy_clauses(self) -> List[ColumnElement[bool]]:
        """
        Translates filter parameters into SQLAlchemy WHERE clauses.

        Category and brand are product-level filters.
        Price, color, size, and stock are variant-level filters combined into a
        single EXISTS subquery so that all constraints must be satisfied by the
        SAME variant (a product qualifies if at least one such variant exists).
        """
        clauses: List[ColumnElement[bool]] = [Product.is_active.is_(True)]

        # -----------------------------------------------------------------
        # Product-level filter: category (stored in JSONB attributes)
        # -----------------------------------------------------------------
        if self.category:
            clauses.append(
                Product.attributes["category"].astext.ilike(self.category.strip())
            )

        # -----------------------------------------------------------------
        # Product-level filter: brand (stored in JSONB attributes)
        # -----------------------------------------------------------------
        if self.brand:
            clauses.append(
                Product.attributes["brand"].astext.ilike(f"%{self.brand.strip()}%")
            )

        # -----------------------------------------------------------------
        # Variant-level filters: price, color, size, stock
        # All must be satisfied by the SAME variant via a single EXISTS subquery.
        # -----------------------------------------------------------------
        variant_conditions: List[ColumnElement[bool]] = [
            ProductVariant.product_id == Product.id,
        ]

        if self.min_price is not None:
            variant_conditions.append(ProductVariant.price >= self.min_price)

        if self.max_price is not None:
            variant_conditions.append(ProductVariant.price <= self.max_price)

        if self.color:
            variant_conditions.append(
                ProductVariant.color.ilike(f"%{self.color.strip()}%")
            )

        if self.size:
            variant_conditions.append(
                ProductVariant.size.ilike(f"%{self.size.strip()}%")
            )

        # Build the subquery: if stock constraint is needed, join Inventory
        has_variant_filters = (
            self.min_price is not None
            or self.max_price is not None
            or self.color is not None
            or self.size is not None
            or self.in_stock is True
        )

        if has_variant_filters:
            if self.in_stock is True:
                # Join inventory and require available stock
                variant_subq = (
                    select(ProductVariant.id)
                    .join(Inventory, Inventory.variant_id == ProductVariant.id)
                    .where(
                        and_(
                            *variant_conditions,
                            (Inventory.quantity - Inventory.reserved_quantity) > 0,
                        )
                    )
                    .correlate(Product)
                    .exists()
                )
            else:
                variant_subq = (
                    select(ProductVariant.id)
                    .where(and_(*variant_conditions))
                    .correlate(Product)
                    .exists()
                )

            clauses.append(variant_subq)

        return clauses
