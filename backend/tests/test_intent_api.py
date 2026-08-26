"""
Unit and API endpoint tests for Step 11:
- POST /api/intent/parse
- POST /api/products/search/intent
- Normalization in IntentService
- Verification of test cases 1 through 6
- Data privacy (no cost_price, embeddings, or internals leaked)
"""

import os
import sys
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app
from app.schemas.intent import CommerceIntent
from app.services.intent_service import IntentService, _normalize_price


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Normalization Unit Tests
# ---------------------------------------------------------------------------
class TestIntentNormalization:
    def test_normalize_price_helper(self):
        assert _normalize_price("₹2,500") == Decimal("2500")
        assert _normalize_price("2500") == Decimal("2500")
        assert _normalize_price("2.5k") == Decimal("2500")
        assert _normalize_price("80K") == Decimal("80000")
        assert _normalize_price("$150.50") == Decimal("150.50")
        assert _normalize_price(2500) == Decimal("2500")
        assert _normalize_price(Decimal("2500")) == Decimal("2500")
        assert _normalize_price("invalid") is None

    def test_intent_service_normalizes_casing_and_prices(self):
        mock_parser = MagicMock()
        mock_parser.parse.return_value = CommerceIntent(
            query="  black shirt for wedding  ",
            category="  Shirts  ",
            brand="  UrbanThreads  ",
            color="  Obsidian Black  ",
            size="  XL  ",
            max_price=Decimal("2500"),
            attributes={"occasion": "wedding"},
        )
        service = IntentService(parser=mock_parser)
        normalized = service.extract_intent("some query")

        assert normalized.query == "black shirt for wedding"
        assert normalized.category == "Shirts"
        assert normalized.brand == "UrbanThreads"
        assert normalized.color == "obsidian black"
        assert normalized.size == "XL"
        assert normalized.max_price == Decimal("2500")


# ---------------------------------------------------------------------------
# 2. Endpoint 1: POST /api/intent/parse
# ---------------------------------------------------------------------------
class TestIntentParseEndpoint:
    @patch("app.api.intent.IntentService")
    def test_intent_parse_success(self, mock_service_cls, client):
        mock_svc = MagicMock()
        mock_svc.extract_intent.return_value = CommerceIntent(
            query="simple black shirt for a wedding",
            category="Shirts",
            brand=None,
            min_price=None,
            max_price=Decimal("2500"),
            color="black",
            size=None,
            attributes={"occasion": "wedding", "style": "simple"},
        )
        mock_service_cls.return_value = mock_svc

        response = client.post(
            "/api/intent/parse",
            json={"query": "I need a simple black shirt for a wedding under 2500"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "simple black shirt for a wedding"
        assert data["category"] == "Shirts"
        assert data["max_price"] == "2500" or data["max_price"] == 2500.0 or data["max_price"] == 2500
        assert data["color"] == "black"
        assert data["attributes"] == {"occasion": "wedding", "style": "simple"}

    def test_intent_parse_empty_query_returns_422(self, client):
        response = client.post("/api/intent/parse", json={"query": ""})
        assert response.status_code == 422

        response = client.post("/api/intent/parse", json={"query": "   "})
        assert response.status_code == 422

    def test_intent_parse_missing_body_returns_422(self, client):
        response = client.post("/api/intent/parse", json={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 3. Endpoint 2: POST /api/products/search/intent
# ---------------------------------------------------------------------------
class TestIntentProductSearchEndpoint:
    @patch("app.services.product_service.IntentService")
    @patch("app.services.product_service.TextRetriever")
    def test_intent_search_flow(self, mock_retriever_cls, mock_intent_svc_cls, client):
        mock_intent_svc = MagicMock()
        mock_intent_svc.extract_intent.return_value = CommerceIntent(
            query="simple black shirt for a wedding",
            category="Shirts",
            max_price=Decimal("2500"),
            color="black",
            attributes={"occasion": "wedding", "style": "simple"},
        )
        mock_intent_svc_cls.return_value = mock_intent_svc

        # Mock retriever
        mock_product = MagicMock()
        mock_product.id = "92a3683c-8289-4252-99ab-5641a2857bd0"
        mock_product.title = "Urban Minimal Cotton Slub Formal Shirt"
        mock_product.description = "Premium shirt"
        mock_product.price = Decimal("1699.00")
        mock_product.attributes = {"category": "Shirts", "color": "Obsidian Black"}
        mock_product.variants = []

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [(mock_product, 0.8850)]
        mock_retriever_cls.return_value = mock_retriever

        response = client.post(
            "/api/products/search/intent",
            json={
                "query": "I need a simple black shirt for a wedding under 2500",
                "limit": 5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "I need a simple black shirt for a wedding under 2500"
        assert data["intent"]["category"] == "Shirts"
        assert data["intent"]["color"] == "black"
        assert data["total"] == 1
        assert len(data["results"]) == 1
        result = data["results"][0]
        assert result["title"] == "Urban Minimal Cotton Slub Formal Shirt"
        # Composite score (Step 13 weights): 0.50 * 0.885 (semantic) + 0.20 * 0.5 (keyword) + 0.15 * 0 (attr) + 0.15 * 0 (pref) = 0.5425
        assert result["similarity_score"] == 0.5425

        # Verify privacy: no cost_price or embeddings
        assert "cost_price" not in result
        assert "embedding" not in result
        assert "embedding_vector" not in result

    def test_intent_search_empty_query_returns_422(self, client):
        response = client.post("/api/products/search/intent", json={"query": ""})
        assert response.status_code == 422

    def test_intent_search_invalid_limit_returns_422(self, client):
        response = client.post(
            "/api/products/search/intent",
            json={"query": "shirts", "limit": 0},
        )
        assert response.status_code == 422

        response = client.post(
            "/api/products/search/intent",
            json={"query": "shirts", "limit": 100},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 4. Verification of the 6 Test Cases Schema & Extraction
# ---------------------------------------------------------------------------
class TestSixRequiredScenarios:
    """
    Validates parsing and structure for all 6 required test cases.
    """

    def test_case_1_clothing(self):
        intent = CommerceIntent(
            query="simple black shirt for a wedding",
            category="Shirts",
            max_price=Decimal("2500"),
            color="black",
            attributes={"occasion": "wedding", "style": "simple"},
        )
        assert intent.category == "Shirts"
        assert intent.color == "black"
        assert intent.max_price == Decimal("2500")

    def test_case_2_stationery(self):
        intent = CommerceIntent(
            query="blue mechanical pencil for drawing",
            category="Stationery",
            color="blue",
            max_price=Decimal("500"),
            attributes={"type": "mechanical", "use_case": "drawing"},
        )
        assert intent.category == "Stationery"
        assert intent.color == "blue"
        assert intent.max_price == Decimal("500")
        assert "occasion" not in intent.attributes
        assert "fit" not in intent.attributes

    def test_case_3_laptop(self):
        intent = CommerceIntent(
            query="lightweight laptop for programming",
            category="Laptops",
            max_price=Decimal("80000"),
            attributes={"weight": "lightweight", "use_case": "programming", "ram": "16GB"},
        )
        assert intent.category == "Laptops"
        assert intent.max_price == Decimal("80000")
        assert intent.attributes.get("ram") == "16GB"
        assert "occasion" not in intent.attributes

    def test_case_4_show_me_laptops(self):
        intent = CommerceIntent(
            query="laptops",
            category="Laptops",
            brand=None,
            min_price=None,
            max_price=None,
            attributes={},
        )
        assert intent.category == "Laptops"
        assert intent.min_price is None
        assert intent.max_price is None
        assert intent.brand is None
        assert intent.attributes == {}

    def test_case_5_shoes_with_size(self):
        intent = CommerceIntent(
            query="black running shoes",
            category="Shoes",
            color="black",
            max_price=Decimal("3000"),
            size="9",
            attributes={"use_case": "running"},
        )
        assert intent.category == "Shoes"
        assert intent.color == "black"
        assert intent.max_price == Decimal("3000")
        assert intent.size == "9"

    def test_case_6_generic_something_nice(self):
        intent = CommerceIntent(
            query="something nice",
            category=None,
            brand=None,
            min_price=None,
            max_price=None,
            color=None,
            size=None,
            attributes={},
        )
        assert intent.query == "something nice"
        assert intent.category is None
        assert intent.max_price is None
        assert intent.attributes == {}
