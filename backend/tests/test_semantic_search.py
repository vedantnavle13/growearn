"""
Tests for GET /api/products/semantic-search

Scenarios tested:
- semantic query returns ranked products with similarity scores
- price filtering (min_price, max_price, price range)
- category filtering
- in-stock filtering
- empty query validation (HTTP 422)
- invalid limit validation (HTTP 422)
- invalid price range validation (HTTP 422)
- ensuring cost_price and raw embedding vectors are never exposed
"""

import uuid
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.product import (
    SemanticSearchResponse,
    SemanticSearchResult,
    VariantSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_variant(
    *,
    size: str = "M",
    color: str = "Black",
    price: Decimal = Decimal("2499.00"),
    in_stock: bool = True,
) -> VariantSummary:
    return VariantSummary(
        id=uuid.uuid4(),
        sku=f"UT-SEM-{uuid.uuid4().hex[:6].upper()}",
        size=size,
        color=color,
        price=price,
        in_stock=in_stock,
    )


def _make_semantic_result(
    *,
    title: str = "Black Slim Fit Shirt",
    description: str = "A formal wedding shirt",
    price: Decimal = Decimal("2499.00"),
    category: str = "Shirts",
    color: str = "Black",
    similarity_score: float = 0.8950,
    variants: list | None = None,
) -> SemanticSearchResult:
    return SemanticSearchResult(
        id=uuid.uuid4(),
        title=title,
        description=description,
        price=price,
        attributes={"category": category, "color": color, "brand": "UrbanThreads Studio"},
        variants=variants or [_make_variant(color=color, price=price)],
        similarity_score=similarity_score,
    )


def _mock_semantic_response(
    query: str,
    results: list[SemanticSearchResult],
    limit: int = 10,
) -> SemanticSearchResponse:
    return SemanticSearchResponse(
        query=query,
        total=len(results),
        limit=limit,
        results=results,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests: Semantic Query
# ---------------------------------------------------------------------------

class TestSemanticQuery:
    def test_semantic_query_returns_ranked_results(self, client):
        shirt1 = _make_semantic_result(
            title="Black Slim Fit Formal Shirt",
            similarity_score=0.9250,
        )
        shirt2 = _make_semantic_result(
            title="Oxford Cotton Mandarin Shirt",
            similarity_score=0.8120,
        )
        mock_resp = _mock_semantic_response(
            query="wedding formal shirt",
            results=[shirt1, shirt2],
        )

        with patch(
            "app.api.products.ProductService.semantic_search_products",
            return_value=mock_resp,
        ):
            res = client.get(
                "/api/products/semantic-search",
                params={"q": "wedding formal shirt"},
            )

        assert res.status_code == 200
        body = res.json()
        assert body["query"] == "wedding formal shirt"
        assert body["total"] == 2
        assert len(body["results"]) == 2
        assert body["results"][0]["similarity_score"] == 0.9250
        assert body["results"][1]["similarity_score"] == 0.8120

    def test_empty_query_returns_422(self, client):
        res = client.get("/api/products/semantic-search", params={"q": ""})
        assert res.status_code == 422

    def test_whitespace_query_returns_422(self, client):
        res = client.get("/api/products/semantic-search", params={"q": "   "})
        assert res.status_code == 422

    def test_missing_query_parameter_returns_422(self, client):
        res = client.get("/api/products/semantic-search")
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# Tests: Price Filtering
# ---------------------------------------------------------------------------

class TestSemanticPriceFilter:
    def test_min_price_filtering(self, client):
        expensive = _make_semantic_result(
            title="Italian Leather Jacket",
            price=Decimal("7999.00"),
            similarity_score=0.8800,
        )
        mock_resp = _mock_semantic_response(
            query="winter outerwear",
            results=[expensive],
        )

        with patch(
            "app.api.products.ProductService.semantic_search_products",
            return_value=mock_resp,
        ):
            res = client.get(
                "/api/products/semantic-search",
                params={"q": "winter outerwear", "min_price": "5000"},
            )

        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 1
        assert Decimal(body["results"][0]["price"]) >= Decimal("5000")

    def test_max_price_filtering(self, client):
        budget_tee = _make_semantic_result(
            title="Supima Cotton Tee",
            price=Decimal("999.00"),
            similarity_score=0.8500,
        )
        mock_resp = _mock_semantic_response(
            query="casual cotton tee",
            results=[budget_tee],
        )

        with patch(
            "app.api.products.ProductService.semantic_search_products",
            return_value=mock_resp,
        ):
            res = client.get(
                "/api/products/semantic-search",
                params={"q": "casual cotton tee", "max_price": "1500"},
            )

        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 1
        assert Decimal(body["results"][0]["price"]) <= Decimal("1500")

    def test_invalid_price_range_min_greater_than_max_returns_422(self, client):
        res = client.get(
            "/api/products/semantic-search",
            params={"q": "shoes", "min_price": "5000", "max_price": "2000"},
        )
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# Tests: Category Filtering
# ---------------------------------------------------------------------------

class TestSemanticCategoryFilter:
    def test_category_filter(self, client):
        jeans = _make_semantic_result(
            title="Selvedge Denim Jeans",
            category="Jeans",
            similarity_score=0.9100,
        )
        mock_resp = _mock_semantic_response(
            query="rugged blue pants",
            results=[jeans],
        )

        with patch(
            "app.api.products.ProductService.semantic_search_products",
            return_value=mock_resp,
        ):
            res = client.get(
                "/api/products/semantic-search",
                params={"q": "rugged blue pants", "category": "Jeans"},
            )

        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 1
        assert body["results"][0]["attributes"]["category"] == "Jeans"


# ---------------------------------------------------------------------------
# Tests: In-Stock Filtering
# ---------------------------------------------------------------------------

class TestSemanticStockFilter:
    def test_in_stock_true_filter(self, client):
        in_stock_prod = _make_semantic_result(
            title="Available Sneakers",
            variants=[_make_variant(in_stock=True)],
            similarity_score=0.8700,
        )
        mock_resp = _mock_semantic_response(
            query="minimalist sneakers",
            results=[in_stock_prod],
        )

        with patch(
            "app.api.products.ProductService.semantic_search_products",
            return_value=mock_resp,
        ):
            res = client.get(
                "/api/products/semantic-search",
                params={"q": "minimalist sneakers", "in_stock": "true"},
            )

        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 1
        assert body["results"][0]["variants"][0]["in_stock"] is True


# ---------------------------------------------------------------------------
# Tests: Limit Parameter
# ---------------------------------------------------------------------------

class TestSemanticLimit:
    def test_valid_limit(self, client):
        items = [_make_semantic_result(title=f"Item {i}") for i in range(5)]
        mock_resp = _mock_semantic_response(query="clothes", results=items, limit=5)

        with patch(
            "app.api.products.ProductService.semantic_search_products",
            return_value=mock_resp,
        ):
            res = client.get(
                "/api/products/semantic-search",
                params={"q": "clothes", "limit": "5"},
            )

        assert res.status_code == 200
        assert len(res.json()["results"]) == 5

    def test_limit_exceeding_max_50_returns_422(self, client):
        res = client.get(
            "/api/products/semantic-search",
            params={"q": "shirt", "limit": "100"},
        )
        assert res.status_code == 422

    def test_limit_zero_returns_422(self, client):
        res = client.get(
            "/api/products/semantic-search",
            params={"q": "shirt", "limit": "0"},
        )
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# Tests: Data Security (cost_price and embedding vectors)
# ---------------------------------------------------------------------------

class TestDataPrivacy:
    def test_cost_price_and_embeddings_not_exposed(self, client):
        item = _make_semantic_result(title="Premium Overcoat")
        mock_resp = _mock_semantic_response(query="overcoat", results=[item])

        with patch(
            "app.api.products.ProductService.semantic_search_products",
            return_value=mock_resp,
        ):
            res = client.get("/api/products/semantic-search", params={"q": "overcoat"})

        assert res.status_code == 200
        body = res.json()
        for prod in body["results"]:
            assert "cost_price" not in prod
            assert "embedding" not in prod
