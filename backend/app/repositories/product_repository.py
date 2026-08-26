"""
Product Repository: all database access for product search with strict tenant isolation.

Rules:
- Only this module may issue SQLAlchemy queries for products.
- Every query MUST enforce `Product.merchant_id == merchant_id` at the SQL level.
- Never accept raw strings to inject into SQL; always use ORM operators
  or SQLAlchemy's parameterized bind values.
- Never expose cost_price in returned data (callers receive ORM objects,
  but the service layer is responsible for shaping the response schema).
"""

import uuid
from decimal import Decimal
from typing import Optional, List, Tuple

from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session, joinedload

from app.models.product import Product, ProductVariant
from app.retrieval.filters import ProductFilters


class ProductRepository:
    """Data-access layer for products, variants, and inventory with tenant isolation."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def search(
        self,
        *,
        merchant_id: uuid.UUID,
        q: Optional[str] = None,
        filters: Optional[ProductFilters] = None,
        category: Optional[str] = None,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None,
        color: Optional[str] = None,
        size: Optional[str] = None,
        in_stock: Optional[bool] = None,
        limit: int = 20,
    ) -> Tuple[List[Product], int]:
        """
        Query products scoped to merchant_id that match keyword search and deterministic filters.
        Returns a tuple of (list[Product], total_count).
        """
        if filters is None:
            filters = ProductFilters(
                category=category,
                min_price=min_price,
                max_price=max_price,
                color=color,
                size=size,
                in_stock=in_stock,
            )

        # Base tenant isolation constraint + filters
        clauses = [Product.merchant_id == merchant_id]
        clauses.extend(filters.to_sqlalchemy_clauses())

        # Keyword search: ILIKE over title and description
        if q and q.strip():
            pattern = f"%{q.strip()}%"
            clauses.append(
                Product.title.ilike(pattern) | Product.description.ilike(pattern)
            )

        # Count query
        count_stmt = select(func.count(Product.id)).where(and_(*clauses))
        total: int = self.db.execute(count_stmt).scalar_one()

        if total == 0:
            return [], 0

        # Main query with eager-loaded variants + inventory
        stmt = (
            select(Product)
            .where(and_(*clauses))
            .options(
                joinedload(Product.variants).joinedload(ProductVariant.inventory)
            )
            .order_by(Product.created_at.desc())
            .limit(limit)
        )

        products = list(self.db.execute(stmt).unique().scalars())
        return products, total

    def vector_search(
        self,
        *,
        merchant_id: uuid.UUID,
        query_vector: List[float],
        filters: Optional[ProductFilters] = None,
        limit: int = 10,
    ) -> List[Tuple[Product, float]]:
        """
        Execute pgvector cosine distance search against Product.embedding scoped to merchant_id.
        Hard tenant boundary is enforced in SQL WHERE BEFORE pgvector distance calculation and ranking.
        Returns a list of (Product, similarity_score) tuples ordered by highest similarity.
        """
        clauses = [
            Product.merchant_id == merchant_id,
            Product.is_active.is_(True),
            Product.embedding.is_not(None),
        ]

        if filters is not None:
            # Append deterministic filter clauses (ignoring is_active since already added)
            for clause in filters.to_sqlalchemy_clauses():
                clauses.append(clause)

        # Distance expression using pgvector cosine_distance (<=>)
        distance_expr = Product.embedding.cosine_distance(query_vector).label("distance")

        # Main query: select Product and distance, ordered by distance ascending
        stmt = (
            select(Product, distance_expr)
            .where(and_(*clauses))
            .options(
                joinedload(Product.variants).joinedload(ProductVariant.inventory)
            )
            .order_by(distance_expr.asc())
            .limit(limit)
        )

        rows = self.db.execute(stmt).unique().all()

        results: List[Tuple[Product, float]] = []
        for product, distance in rows:
            dist_val = float(distance) if distance is not None else 1.0
            similarity = round(max(0.0, 1.0 - dist_val), 4)
            results.append((product, similarity))

        return results

    def semantic_search(
        self,
        *,
        merchant_id: uuid.UUID,
        query_vector: List[float],
        category: Optional[str] = None,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None,
        color: Optional[str] = None,
        size: Optional[str] = None,
        in_stock: Optional[bool] = None,
        limit: int = 10,
    ) -> List[Tuple[Product, float]]:
        """
        Backward-compatible wrapper around vector_search scoped to merchant_id.
        """
        filters = ProductFilters(
            category=category,
            min_price=min_price,
            max_price=max_price,
            color=color,
            size=size,
            in_stock=in_stock,
        )
        return self.vector_search(
            merchant_id=merchant_id,
            query_vector=query_vector,
            filters=filters,
            limit=limit,
        )
