"""
Tests for GET /api/products/search

Tests are organized by scenario:
- keyword search (q)
- category filtering
- price range filtering
- out-of-stock filtering
- invalid query parameter validation

Uses FastAPI's TestClient and monkeypatches ProductService to avoid
hitting the real database, keeping tests fast and deterministic.
"""

import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.product import (
    ProductSearchResponse,
    ProductSearchResult,
    VariantSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_variant(
    *,
    size: str = "M",
    color: str = "Navy Blue",
    price: Decimal = Decimal("1999.00"),
    in_stock: bool = True,
) -> VariantSummary:
    return VariantSummary(
        id=uuid.uuid4(),
        sku=f"UT-TST-{uuid.uuid4().hex[:6].upper()}",
        size=size,
        color=color,
        price=price,
        in_stock=in_stock,
    )


def _make_product(
    *,
    title: str = "Test Product",
    description: str = "A test product",
    price: Decimal = Decimal("1999.00"),
    category: str = "Shirts",
    color: str = "Navy Blue",
    variants: list | None = None,
) -> ProductSearchResult:
    return ProductSearchResult(
        id=uuid.uuid4(),
        title=title,
        description=description,
        price=price,
        attributes={"category": category, "color": color, "brand": "UrbanThreads Studio"},
        variants=variants or [_make_variant(color=color)],
    )


def _mock_response(results: list[ProductSearchResult], limit: int = 20) -> ProductSearchResponse:
    return ProductSearchResponse(total=len(results), limit=limit, results=results)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests: keyword search
# ---------------------------------------------------------------------------

class TestKeywordSearch:
    def test_keyword_returns_matching_products(self, client):
        shirt = _make_product(title="Oxford Button-Down Shirt", category="Shirts")
        response_data = _mock_response([shirt])

        with patch(
            "app.api.products.ProductService.search_products",
            return_value=response_data,
        ):
            res = client.get("/api/products/search", params={"q": "oxford"})

        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 1
        assert "Oxford" in body["results"][0]["title"]

    def test_empty_keyword_returns_all(self, client):
        products = [_make_product(title=f"Product {i}") for i in range(5)]
        response_data = _mock_response(products)

        with patch(
            "app.api.products.ProductService.search_products",
            return_value=response_data,
        ):
            res = client.get("/api/products/search")

        assert res.status_code == 200
        assert res.json()["total"] == 5

    def test_no_results_returns_empty_list(self, client):
        response_data = _mock_response([])

        with patch(
            "app.api.products.ProductService.search_products",
            return_value=response_data,
        ):
            res = client.get("/api/products/search", params={"q": "xyznonexistentproduct"})

        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 0
        assert body["results"] == []


# ---------------------------------------------------------------------------
# Tests: category filtering
# ---------------------------------------------------------------------------

class TestCategoryFilter:
    def test_category_filter_returns_only_that_category(self, client):
        jeans = _make_product(title="Slim Selvedge Jeans", category="Jeans")
        response_data = _mock_response([jeans])

        with patch(
            "app.api.products.ProductService.search_products",
            return_value=response_data,
        ):
            res = client.get("/api/products/search", params={"category": "Jeans"})

        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 1
        assert body["results"][0]["attributes"]["category"] == "Jeans"

    def test_unknown_category_returns_empty(self, client):
        response_data = _mock_response([])

        with patch(
            "app.api.products.ProductService.search_products",
            return_value=response_data,
        ):
            res = client.get("/api/products/search", params={"category": "Spacesuits"})

        assert res.status_code == 200
        assert res.json()["total"] == 0


# ---------------------------------------------------------------------------
# Tests: price filtering
# ---------------------------------------------------------------------------

class TestPriceFilter:
    def test_min_price_filter(self, client):
        expensive = _make_product(title="Leather Jacket", price=Decimal("7999.00"))
        response_data = _mock_response([expensive])

        with patch(
            "app.api.products.ProductService.search_products",
            return_value=response_data,
        ):
            res = client.get("/api/products/search", params={"min_price": "5000"})

        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 1
        assert Decimal(body["results"][0]["price"]) >= Decimal("5000")

    def test_max_price_filter(self, client):
        cheap = _make_product(title="Basic T-Shirt", price=Decimal("799.00"))
        response_data = _mock_response([cheap])

        with patch(
            "app.api.products.ProductService.search_products",
            return_value=response_data,
        ):
            res = client.get("/api/products/search", params={"max_price": "1000"})

        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 1
        assert Decimal(body["results"][0]["price"]) <= Decimal("1000")

    def test_price_range_filter(self, client):
        mid = _make_product(title="Stretch Chinos", price=Decimal("2499.00"))
        response_data = _mock_response([mid])

        with patch(
            "app.api.products.ProductService.search_products",
            return_value=response_data,
        ):
            res = client.get(
                "/api/products/search",
                params={"min_price": "2000", "max_price": "3000"},
            )

        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 1

    def test_invalid_max_price_less_than_min_price_returns_422(self, client):
        res = client.get(
            "/api/products/search",
            params={"min_price": "5000", "max_price": "2000"},
        )
        assert res.status_code == 422

    def test_negative_min_price_returns_422(self, client):
        res = client.get("/api/products/search", params={"min_price": "-100"})
        assert res.status_code == 422

    def test_negative_max_price_returns_422(self, client):
        res = client.get("/api/products/search", params={"max_price": "-50"})
        assert res.status_code == 422

    def test_non_numeric_price_returns_422(self, client):
        res = client.get("/api/products/search", params={"min_price": "cheap"})
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# Tests: out-of-stock / in-stock filtering
# ---------------------------------------------------------------------------

class TestStockFilter:
    def test_in_stock_true_returns_only_available_products(self, client):
        in_stock_variant = _make_variant(in_stock=True)
        product = _make_product(title="In-Stock Shirt", variants=[in_stock_variant])
        response_data = _mock_response([product])

        with patch(
            "app.api.products.ProductService.search_products",
            return_value=response_data,
        ):
            res = client.get("/api/products/search", params={"in_stock": "true"})

        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 1
        # All returned variants must be in stock
        for result in body["results"]:
            for variant in result["variants"]:
                assert variant["in_stock"] is True

    def test_out_of_stock_product_excluded_when_in_stock_true(self, client):
        # Service returns 0 results when in_stock=True and nothing is available
        response_data = _mock_response([])

        with patch(
            "app.api.products.ProductService.search_products",
            return_value=response_data,
        ):
            res = client.get("/api/products/search", params={"in_stock": "true"})

        assert res.status_code == 200
        assert res.json()["total"] == 0

    def test_in_stock_false_does_not_filter_by_stock(self, client):
        out_of_stock_variant = _make_variant(in_stock=False)
        product = _make_product(title="Out of Stock Shoe", variants=[out_of_stock_variant])
        response_data = _mock_response([product])

        with patch(
            "app.api.products.ProductService.search_products",
            return_value=response_data,
        ):
            res = client.get("/api/products/search", params={"in_stock": "false"})

        assert res.status_code == 200
        assert res.json()["total"] == 1


# ---------------------------------------------------------------------------
# Tests: limit parameter
# ---------------------------------------------------------------------------

class TestLimitParam:
    def test_limit_is_respected(self, client):
        products = [_make_product(title=f"Product {i}") for i in range(3)]
        response_data = _mock_response(products, limit=3)

        with patch(
            "app.api.products.ProductService.search_products",
            return_value=response_data,
        ):
            res = client.get("/api/products/search", params={"limit": "3"})

        assert res.status_code == 200
        assert len(res.json()["results"]) == 3

    def test_limit_exceeding_50_returns_422(self, client):
        res = client.get("/api/products/search", params={"limit": "100"})
        assert res.status_code == 422

    def test_limit_zero_returns_422(self, client):
        res = client.get("/api/products/search", params={"limit": "0"})
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# Tests: cost_price not in response
# ---------------------------------------------------------------------------

class TestCostPriceNotExposed:
    def test_cost_price_absent_from_response(self, client):
        product = _make_product(title="Expensive Jacket")
        response_data = _mock_response([product])

        with patch(
            "app.api.products.ProductService.search_products",
            return_value=response_data,
        ):
            res = client.get("/api/products/search")

        assert res.status_code == 200
        for result in res.json()["results"]:
            assert "cost_price" not in result
