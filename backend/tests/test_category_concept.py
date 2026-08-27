"""
Tests for Category Concept functionality (Step 15 bug fix).

Tests the broad category concept feature:
- category_concept field in CommerceIntent
- CategoryConceptService mapping
- Concept relevance scoring in ranker
- Intent search with broad concepts
"""

import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from app.models.product import Product
from app.schemas.intent import CommerceIntent
from app.services.category_concept_service import CategoryConceptService
from app.retrieval.ranker import ProductRanker


class TestCategoryConceptService:
    """Tests for CategoryConceptService mapping."""

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.fixture
    def service(self, mock_db):
        return CategoryConceptService(mock_db, str(uuid.uuid4()))

    def test_load_available_categories(self, mock_db, service):
        """Test loading distinct categories from database."""
        mock_rows = [
            MagicMock(__getitem__=lambda self, i: "Shirts" if i == 0 else None),
            MagicMock(__getitem__=lambda self, i: "Jeans" if i == 0 else None),
            MagicMock(__getitem__=lambda self, i: "Jackets" if i == 0 else None),
            MagicMock(__getitem__=lambda self, i: "Laptops" if i == 0 else None),
        ]
        mock_db.execute.return_value.fetchall.return_value = mock_rows

        categories = service._load_available_categories()

        assert categories == {"Shirts", "Jeans", "Jackets", "Laptops"}

    def test_get_categories_for_concept_clothing(self, mock_db, service):
        """Test clothing concept maps to clothing categories."""
        mock_rows = [
            MagicMock(__getitem__=lambda self, i: "Shirts"),
            MagicMock(__getitem__=lambda self, i: "Jeans"),
            MagicMock(__getitem__=lambda self, i: "Jackets"),
        ]
        mock_db.execute.return_value.fetchall.return_value = mock_rows

        categories = service.get_categories_for_concept("clothing")

        assert "Shirts" in categories
        assert "Jeans" in categories
        assert "Jackets" in categories

    def test_get_categories_for_concept_electronics(self, mock_db, service):
        """Test electronics concept maps to electronics categories."""
        mock_rows = [
            MagicMock(__getitem__=lambda self, i: "Laptops"),
            MagicMock(__getitem__=lambda self, i: "Phones"),
            MagicMock(__getitem__=lambda self, i: "Headphones"),
        ]
        mock_db.execute.return_value.fetchall.return_value = mock_rows

        categories = service.get_categories_for_concept("electronics")

        assert "Laptops" in categories
        assert "Phones" in categories

    def test_get_categories_for_unknown_concept(self, mock_db, service):
        """Test unknown concept returns empty list."""
        mock_rows = [MagicMock(__getitem__=lambda self, i: "Shirts")]
        mock_db.execute.return_value.fetchall.return_value = mock_rows

        categories = service.get_categories_for_concept("unknown_concept")

        assert categories == []

    def test_resolve_exact_category_only(self, mock_db, service):
        """Test exact category without concept uses exact as hard filter."""
        mock_rows = [MagicMock(__getitem__=lambda self, i: "Shirts")]
        mock_db.execute.return_value.fetchall.return_value = mock_rows

        hard_filter, concept_cats = service.resolve_category_or_concept(
            exact_category="Shirts",
            category_concept=None
        )

        assert hard_filter == "Shirts"
        assert concept_cats == []

    def test_resolve_concept_only(self, mock_db, service):
        """Test concept only uses no hard filter, concept categories for ranking."""
        mock_rows = [
            MagicMock(__getitem__=lambda self, i: "Shirts"),
            MagicMock(__getitem__=lambda self, i: "Jeans"),
            MagicMock(__getitem__=lambda self, i: "Jackets"),
        ]
        mock_db.execute.return_value.fetchall.return_value = mock_rows

        hard_filter, concept_cats = service.resolve_category_or_concept(
            exact_category=None,
            category_concept="clothing"
        )

        assert hard_filter is None
        assert "Shirts" in concept_cats
        assert "Jeans" in concept_cats
        assert "Jackets" in concept_cats

    def test_resolve_both_exact_wins(self, mock_db, service):
        """Test both provided: exact category wins as hard filter, concept for ranking."""
        mock_rows = [
            MagicMock(__getitem__=lambda self, i: "Shirts"),
            MagicMock(__getitem__=lambda self, i: "Jeans"),
            MagicMock(__getitem__=lambda self, i: "Jackets"),
        ]
        mock_db.execute.return_value.fetchall.return_value = mock_rows

        hard_filter, concept_cats = service.resolve_category_or_concept(
            exact_category="Shirts",
            category_concept="clothing"
        )

        assert hard_filter == "Shirts"
        assert "Shirts" in concept_cats
        assert "Jeans" in concept_cats
        assert "Jackets" in concept_cats


class TestConceptRelevanceScoring:
    """Tests for category concept relevance scoring in ProductRanker."""

    @pytest.fixture
    def ranker(self):
        return ProductRanker()

    def _make_product(self, category: str, title: str = "Test Product") -> Product:
        p = Product(
            id=uuid.uuid4(),
            merchant_id=uuid.uuid4(),
            title=title,
            description="Description",
            price=Decimal("1999.00"),
            attributes={"category": category},
            is_active=True,
        )
        p.variants = []
        return p

    def test_concept_score_exact_match(self, ranker):
        """Test concept score = 1.0 when product category in concept categories."""
        product = self._make_product("Shirts")
        concept_categories = ["Shirts", "Jeans", "Jackets"]

        score = ranker._compute_concept_score(product, concept_categories)

        assert score == 1.0

    def test_concept_score_fuzzy_match(self, ranker):
        """Test concept score = 0.75 for fuzzy match."""
        product = self._make_product("T-Shirts")
        concept_categories = ["Shirts", "Jeans"]

        score = ranker._compute_concept_score(product, concept_categories)

        assert score == 0.75  # "shirts" in "t-shirts"

    def test_concept_score_no_match(self, ranker):
        """Test concept score = 0.0 when no match."""
        product = self._make_product("Laptops")
        concept_categories = ["Shirts", "Jeans", "Jackets"]

        score = ranker._compute_concept_score(product, concept_categories)

        assert score == 0.0

    def test_concept_score_empty_concept(self, ranker):
        """Test concept score = 0.0 when no concept categories provided."""
        product = self._make_product("Shirts")

        score = ranker._compute_concept_score(product, None)
        assert score == 0.0

        score = ranker._compute_concept_score(product, [])
        assert score == 0.0

    def test_concept_score_product_no_category(self, ranker):
        """Test concept score = 0.0 when product has no category attribute."""
        product = self._make_product(None)
        product.attributes = {}
        concept_categories = ["Shirts", "Jeans"]

        score = ranker._compute_concept_score(product, concept_categories)

        assert score == 0.0


class TestCommerceIntentCategoryConcept:
    """Tests for CommerceIntent category_concept field."""

    def test_category_concept_in_schema(self):
        """Test category_concept field exists and validates."""
        intent = CommerceIntent(
            query="black clothes",
            category=None,
            category_concept="clothing",
            color="black"
        )

        assert intent.category is None
        assert intent.category_concept == "clothing"
        assert intent.color == "black"

    def test_category_concept_normalized_to_lowercase(self):
        """Test category_concept is normalized to lowercase."""
        intent = CommerceIntent(
            query="CLOTHES",
            category_concept="CLOTHING"
        )

        assert intent.category_concept == "clothing"

    def test_exact_category_and_concept_both_present(self):
        """Test both exact category and concept can coexist."""
        intent = CommerceIntent(
            query="black clothes for wedding",
            category="Shirts",
            category_concept="clothing",
            color="black"
        )

        assert intent.category == "Shirts"
        assert intent.category_concept == "clothing"


class TestIntentParserCategoryConcept:
    """Tests for IntentParser extracting category_concept."""

    @pytest.fixture
    def parser(self):
        from app.ai.intent_parser import IntentParser
        with patch.object(IntentParser, '__init__', lambda self, *args, **kwargs: None):
            parser = IntentParser()
            parser.client = MagicMock()
            parser.model = "gemini-3.6-flash"
            return parser

    def test_clothes_maps_to_clothing_concept(self, parser):
        """Test 'clothes' maps to category_concept 'clothing', not exact category."""
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = '{"query": "black clothes", "category": null, "category_concept": "clothing", "color": "black", "brand": null, "min_price": null, "max_price": null, "size": null, "attributes": {}}'
        parser.client.models.generate_content.return_value = mock_response

        intent = parser.parse("show me black clothes")

        assert intent.category is None
        assert intent.category_concept == "clothing"
        assert intent.color == "black"

    def test_electronics_maps_to_concept(self, parser):
        """Test 'electronics' maps to category_concept."""
        mock_response = MagicMock()
        mock_response.text = '{"query": "electronics under 50000", "category": null, "category_concept": "electronics", "color": null, "brand": null, "min_price": null, "max_price": 50000, "size": null, "attributes": {}}'
        parser.client.models.generate_content.return_value = mock_response

        intent = parser.parse("electronics under 50000")

        assert intent.category is None
        assert intent.category_concept == "electronics"
        assert intent.max_price == 50000

    def test_footwear_maps_to_concept(self, parser):
        """Test 'footwear' maps to category_concept."""
        mock_response = MagicMock()
        mock_response.text = '{"query": "footwear", "category": null, "category_concept": "footwear", "color": null, "brand": null, "min_price": null, "max_price": null, "size": null, "attributes": {}}'
        parser.client.models.generate_content.return_value = mock_response

        intent = parser.parse("show me footwear")

        assert intent.category is None
        assert intent.category_concept == "footwear"

    def test_shirt_maps_to_exact_category(self, parser):
        """Test 'shirt' maps to exact category, not concept."""
        mock_response = MagicMock()
        mock_response.text = '{"query": "black shirt", "category": "Shirts", "category_concept": null, "color": "black", "brand": null, "min_price": null, "max_price": null, "size": null, "attributes": {}}'
        parser.client.models.generate_content.return_value = mock_response

        intent = parser.parse("black shirt")

        assert intent.category == "Shirts"
        assert intent.category_concept is None


class TestEndToEndBroadConceptSearch:
    """Integration tests for broad concept search through ProductService."""

    @patch("app.services.product_service.TextRetriever")
    @patch("app.services.product_service.IntentService")
    @patch("app.services.product_service.CategoryConceptService")
    def test_black_clothes_search(self, mock_concept_cls, mock_intent_cls, mock_retriever_cls):
        """Test 'black clothes' searches across multiple clothing categories."""
        from app.services.product_service import ProductService

        # Setup mocks
        mock_db = MagicMock()
        mock_intent = MagicMock()
        mock_intent.extract_intent.return_value = CommerceIntent(
            query="black clothes",
            category=None,
            category_concept="clothing",
            color="black"
        )
        mock_intent_cls.return_value = mock_intent

        # Mock CategoryConceptService
        mock_concept_service = MagicMock()
        mock_concept_service.resolve_category_or_concept.return_value = (
            None,  # no hard filter
            ["Shirts", "Jeans", "Jackets", "Dresses"]  # concept categories
        )
        mock_concept_cls.return_value = mock_concept_service

        # Mock retriever returns products from different clothing categories
        mock_products = []
        for cat in ["Shirts", "Jeans", "Jackets"]:
            p = Product(
                id=uuid.uuid4(),
                merchant_id=uuid.uuid4(),
                title=f"Black {cat}",
                description=f"Black {cat.lower()}",
                price=Decimal("1999.00"),
                attributes={"category": cat, "color": "black"},
                is_active=True,
            )
            p.variants = []
            mock_products.append(p)

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [(p, 0.8) for p in mock_products]
        mock_retriever_cls.return_value = mock_retriever

        service = ProductService(mock_db)
        result = service.intent_search_products(
            merchant_id=uuid.uuid4(),
            raw_query="black clothes",
            limit=10
        )

        assert result.total == 3
        categories = {r.attributes.get("category") for r in result.results}
        assert "Shirts" in categories
        assert "Jeans" in categories
        assert "Jackets" in categories
        # All results should be black
        for r in result.results:
            assert r.attributes.get("color") == "black"

    @patch("app.services.product_service.TextRetriever")
    @patch("app.services.product_service.IntentService")
    @patch("app.services.product_service.CategoryConceptService")
    def test_black_shirts_exact_category(self, mock_concept_cls, mock_intent_cls, mock_retriever_cls):
        """Test 'black shirts' uses exact category filter."""
        from app.services.product_service import ProductService

        mock_db = MagicMock()
        mock_intent = MagicMock()
        mock_intent.extract_intent.return_value = CommerceIntent(
            query="black shirt",
            category="Shirts",
            category_concept=None,
            color="black"
        )
        mock_intent_cls.return_value = mock_intent

        mock_concept_service = MagicMock()
        mock_concept_service.resolve_category_or_concept.return_value = (
            "Shirts",  # exact hard filter
            []  # no concept categories
        )
        mock_concept_cls.return_value = mock_concept_service

        mock_product = Product(
            id=uuid.uuid4(),
            merchant_id=uuid.uuid4(),
            title="Black Formal Shirt",
            description="Formal shirt",
            price=Decimal("1999.00"),
            attributes={"category": "Shirts", "color": "black"},
            is_active=True,
        )
        mock_product.variants = []

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [(mock_product, 0.9)]
        mock_retriever_cls.return_value = mock_retriever

        service = ProductService(mock_db)
        result = service.intent_search_products(
            merchant_id=uuid.uuid4(),
            raw_query="black shirt",
            limit=10
        )

        assert result.total == 1
        assert result.results[0].attributes.get("category") == "Shirts"

    @patch("app.services.product_service.TextRetriever")
    @patch("app.services.product_service.IntentService")
    @patch("app.services.product_service.CategoryConceptService")
    def test_electronics_under_50000(self, mock_concept_cls, mock_intent_cls, mock_retriever_cls):
        """Test 'electronics under 50000' applies price filter + concept."""
        from app.services.product_service import ProductService

        mock_db = MagicMock()
        mock_intent = MagicMock()
        mock_intent.extract_intent.return_value = CommerceIntent(
            query="electronics under 50000",
            category=None,
            category_concept="electronics",
            max_price=Decimal("50000")
        )
        mock_intent_cls.return_value = mock_intent

        mock_concept_service = MagicMock()
        mock_concept_service.resolve_category_or_concept.return_value = (
            None,  # no hard filter
            ["Laptops", "Phones", "Headphones"]
        )
        mock_concept_cls.return_value = mock_concept_service

        mock_products = []
        for cat in ["Laptops", "Phones"]:
            p = Product(
                id=uuid.uuid4(),
                merchant_id=uuid.uuid4(),
                title=f"{cat} Model X",
                description=f"Latest {cat.lower()}",
                price=Decimal("45000.00"),
                attributes={"category": cat},
                is_active=True,
            )
            p.variants = []
            mock_products.append(p)

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [(p, 0.85) for p in mock_products]
        mock_retriever_cls.return_value = mock_retriever

        service = ProductService(mock_db)
        result = service.intent_search_products(
            merchant_id=uuid.uuid4(),
            raw_query="electronics under 50000",
            limit=10
        )

        assert result.total == 2
        categories = {r.attributes.get("category") for r in result.results}
        assert "Laptops" in categories
        assert "Phones" in categories
        # All results should be under 50000
        for r in result.results:
            assert r.price <= Decimal("50000")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])