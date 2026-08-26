"""
Integration tests for hard-constraint enforcement in the semantic search retrieval pipeline with tenant isolation.

These tests run against the LIVE PostgreSQL database (with seeded data) and verify
that price, category, color, size, and stock constraints are enforced at the SQL
level — not by post-retrieval Python filtering.
"""

import os
import sys
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.db.database import SessionLocal
from app.models.merchant import Merchant
from app.models.product import Product, ProductVariant, Inventory
from app.repositories.product_repository import ProductRepository
from app.retrieval.filters import ProductFilters

from sqlalchemy import and_, select


def _has_qualifying_variant(db, product_id, max_price=None, color=None, size=None, in_stock=None):
    """Check if a product has at least one variant satisfying all the given constraints."""
    conditions = [ProductVariant.product_id == product_id]
    if max_price is not None:
        conditions.append(ProductVariant.price <= max_price)
    if color is not None:
        conditions.append(ProductVariant.color.ilike(f"%{color}%"))
    if size is not None:
        conditions.append(ProductVariant.size.ilike(f"%{size}%"))

    if in_stock:
        stmt = (
            select(ProductVariant.id)
            .join(Inventory, Inventory.variant_id == ProductVariant.id)
            .where(
                and_(
                    *conditions,
                    (Inventory.quantity - Inventory.reserved_quantity) > 0,
                )
            )
        )
    else:
        stmt = select(ProductVariant.id).where(and_(*conditions))

    result = db.execute(stmt).first()
    return result is not None


@pytest.fixture
def db():
    """Fresh session per test to avoid PendingRollbackError cascades."""
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def merchant_id(db):
    m = db.query(Merchant).first()
    assert m is not None, "Seeded merchant required for live database integration tests"
    return m.id


@pytest.fixture
def repo(db):
    return ProductRepository(db)


# ---------------------------------------------------------------------------
# Test 1: max_price constraint
# ---------------------------------------------------------------------------
class TestMaxPriceConstraint:
    def test_max_price_2500_no_variant_above(self, db, repo, merchant_id):
        """Every returned product must have at least one variant with price <= 2500."""
        filters = ProductFilters(max_price=Decimal("2500"))
        results = repo.vector_search(
            merchant_id=merchant_id,
            query_vector=[0.0] * 1536,
            filters=filters,
            limit=10,
        )
        for product, _sim in results:
            assert _has_qualifying_variant(db, product.id, max_price=Decimal("2500")), (
                f"Product '{product.title}' (product price={product.price}) was returned "
                f"but has NO variant with price <= 2500"
            )


# ---------------------------------------------------------------------------
# Test 2: category constraint
# ---------------------------------------------------------------------------
class TestCategoryConstraint:
    def test_category_shirts_excludes_others(self, db, repo, merchant_id):
        """All returned products must have category=Shirts (exact, not substring)."""
        filters = ProductFilters(category="Shirts")
        results = repo.vector_search(
            merchant_id=merchant_id,
            query_vector=[0.0] * 1536,
            filters=filters,
            limit=10,
        )
        assert len(results) > 0, "Expected at least some Shirts"
        for product, _sim in results:
            cat = product.attributes.get("category", "")
            assert cat == "Shirts", (
                f"Product '{product.title}' has category='{cat}', expected 'Shirts'"
            )

    def test_category_t_shirts_does_not_return_shirts(self, db, repo, merchant_id):
        """Searching for T-Shirts must not return Shirts category products."""
        filters = ProductFilters(category="T-Shirts")
        results = repo.vector_search(
            merchant_id=merchant_id,
            query_vector=[0.0] * 1536,
            filters=filters,
            limit=10,
        )
        for product, _sim in results:
            cat = product.attributes.get("category", "")
            assert cat == "T-Shirts", (
                f"Product '{product.title}' has category='{cat}', expected 'T-Shirts'"
            )


# ---------------------------------------------------------------------------
# Test 3: color constraint (variant-level)
# ---------------------------------------------------------------------------
class TestColorConstraint:
    def test_color_black_variant_exists(self, db, repo, merchant_id):
        """Every returned product must have at least one variant with color containing 'black'."""
        filters = ProductFilters(color="black")
        results = repo.vector_search(
            merchant_id=merchant_id,
            query_vector=[0.0] * 1536,
            filters=filters,
            limit=10,
        )
        assert len(results) > 0, "Expected some products with a black variant"
        for product, _sim in results:
            assert _has_qualifying_variant(db, product.id, color="black"), (
                f"Product '{product.title}' was returned but has NO variant with color containing 'black'"
            )


# ---------------------------------------------------------------------------
# Test 4: size constraint (variant-level)
# ---------------------------------------------------------------------------
class TestSizeConstraint:
    def test_size_m_variant_exists(self, db, repo, merchant_id):
        """Every returned product must have at least one variant with size='M'."""
        filters = ProductFilters(size="M")
        results = repo.vector_search(
            merchant_id=merchant_id,
            query_vector=[0.0] * 1536,
            filters=filters,
            limit=10,
        )
        assert len(results) > 0, "Expected some products with an M variant"
        for product, _sim in results:
            assert _has_qualifying_variant(db, product.id, size="M"), (
                f"Product '{product.title}' was returned but has NO variant with size 'M'"
            )


# ---------------------------------------------------------------------------
# Test 5: in_stock constraint
# ---------------------------------------------------------------------------
class TestStockConstraint:
    def test_in_stock_true_has_available_inventory(self, db, repo, merchant_id):
        """Every returned product must have at least one variant with available stock > 0."""
        filters = ProductFilters(in_stock=True)
        results = repo.vector_search(
            merchant_id=merchant_id,
            query_vector=[0.0] * 1536,
            filters=filters,
            limit=10,
        )
        assert len(results) > 0, "Expected some in-stock products"
        for product, _sim in results:
            assert _has_qualifying_variant(db, product.id, in_stock=True), (
                f"Product '{product.title}' was returned but has NO in-stock variant"
            )


# ---------------------------------------------------------------------------
# Test 6: Combined constraints (the original failing query)
# ---------------------------------------------------------------------------
class TestCombinedConstraints:
    def test_shirts_black_under_2500_in_stock(self, db, repo, merchant_id):
        """
        Combined: category=Shirts, color=black, max_price=2500, in_stock=true.
        Every result must satisfy ALL constraints via a single qualifying variant.
        """
        filters = ProductFilters(
            category="Shirts",
            color="black",
            max_price=Decimal("2500"),
            in_stock=True,
        )
        results = repo.vector_search(
            merchant_id=merchant_id,
            query_vector=[0.0] * 1536,
            filters=filters,
            limit=10,
        )
        for product, _sim in results:
            cat = product.attributes.get("category", "")
            assert cat == "Shirts", (
                f"Product '{product.title}' has category='{cat}', expected 'Shirts'"
            )
            assert _has_qualifying_variant(
                db, product.id,
                max_price=Decimal("2500"),
                color="black",
                in_stock=True,
            ), (
                f"Product '{product.title}' has no variant that is black, <= 2500, and in stock"
            )


# ---------------------------------------------------------------------------
# Test 7: Semantic similarity orders results AFTER hard constraints
# ---------------------------------------------------------------------------
class TestSimilarityOrdering:
    def test_results_ordered_by_similarity_descending(self, db, repo, merchant_id):
        """Results from vector_search must be ordered by similarity (descending)."""
        filters = ProductFilters(category="Shirts")
        results = repo.vector_search(
            merchant_id=merchant_id,
            query_vector=[0.0] * 1536,
            filters=filters,
            limit=10,
        )
        if len(results) > 1:
            scores = [sim for _, sim in results]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"Results not ordered by similarity: {scores[i]} < {scores[i+1]}"
                )
