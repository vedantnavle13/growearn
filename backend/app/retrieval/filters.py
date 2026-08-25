"""
Deterministic business and database filters for product catalog queries.

Contains pure, deterministic filtering logic (price range, stock, size, category, color)
independent of vector similarity or ranking algorithms.
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
    - category: Exact or substring match in product JSONB attributes
    - min_price: Inclusive minimum product price
    - max_price: Inclusive maximum product price
    - color: Exact or substring match in product JSONB attributes
    - size: Subquery checking if product has at least one variant in this size
    - in_stock: Subquery checking if product has at least one variant with (quantity - reserved_quantity) > 0
    """
    category: Optional[str] = None
    min_price: Optional[Decimal] = None
    max_price: Optional[Decimal] = None
    color: Optional[str] = None
    size: Optional[str] = None
    in_stock: Optional[bool] = None

    def to_sqlalchemy_clauses(self) -> List[ColumnElement[bool]]:
        """
        Translates filter parameters into SQLAlchemy WHERE clauses.
        """
        clauses: List[ColumnElement[bool]] = [Product.is_active.is_(True)]

        # Category: stored in JSONB attributes->>'category'
        if self.category:
            clauses.append(
                Product.attributes["category"].astext.ilike(f"%{self.category.strip()}%")
            )

        # Price range: base product price
        if self.min_price is not None:
            clauses.append(Product.price >= self.min_price)

        if self.max_price is not None:
            clauses.append(Product.price <= self.max_price)

        # Color: stored in JSONB attributes->>'color'
        if self.color:
            clauses.append(
                Product.attributes["color"].astext.ilike(f"%{self.color.strip()}%")
            )

        # Size: product has a variant in this size
        if self.size:
            size_subq = (
                select(ProductVariant.id)
                .where(
                    and_(
                        ProductVariant.product_id == Product.id,
                        ProductVariant.size.ilike(f"%{self.size.strip()}%"),
                    )
                )
                .correlate(Product)
                .exists()
            )
            clauses.append(size_subq)

        # In-stock: product has at least one variant with available inventory > 0
        if self.in_stock is True:
            stock_subq = (
                select(Inventory.id)
                .join(ProductVariant, ProductVariant.id == Inventory.variant_id)
                .where(
                    and_(
                        ProductVariant.product_id == Product.id,
                        (Inventory.quantity - Inventory.reserved_quantity) > 0,
                    )
                )
                .correlate(Product)
                .exists()
            )
            clauses.append(stock_subq)

        return clauses
