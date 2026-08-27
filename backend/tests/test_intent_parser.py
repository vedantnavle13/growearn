"""
Unit and integration tests for CommerceIntent extraction (IntentParser & IntentService).

Covers:
1. Clothing intent extraction
2. Stationery intent extraction
3. Electronics intent extraction
4. Missing optional information handling
5. No price mentioned handling
6. Invalid price range validation
7. Empty input rejection
8. No invented clothing attributes for non-clothing categories
9. IntentService orchestration
10. Optional live integration test with real Gemini API
"""

import os
import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.schemas.intent import CommerceIntent
from app.ai.intent_parser import IntentParser
from app.services.intent_service import IntentService


# ---------------------------------------------------------------------------
# Helpers & Mocks
# ---------------------------------------------------------------------------

def _mock_generate_content_response(payload: dict) -> MagicMock:
    """Helper to construct a mock response matching google-genai SDK."""
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(payload)
    mock_resp.parsed = CommerceIntent.model_validate(payload)
    return mock_resp


# ---------------------------------------------------------------------------
# Unit Tests: Pydantic Schema Validation
# ---------------------------------------------------------------------------

class TestCommerceIntentSchema:
    def test_valid_intent_creation(self):
        intent = CommerceIntent(
            query="simple black shirt",
            category="Shirts",
            max_price=Decimal("2500"),
            color="black",
            attributes={"occasion": "wedding"},
        )
        assert intent.query == "simple black shirt"
        assert intent.category == "Shirts"
        assert intent.max_price == Decimal("2500")
        assert intent.min_price is None
        assert intent.color == "black"
        assert intent.attributes == {"occasion": "wedding"}
        assert intent.requested_limit is None

    def test_valid_intent_with_requested_limit(self):
        intent = CommerceIntent(
            query="black shirts",
            category="Shirts",
            color="black",
            requested_limit=5,
        )
        assert intent.query == "black shirts"
        assert intent.category == "Shirts"
        assert intent.color == "black"
        assert intent.requested_limit == 5

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            CommerceIntent(query="")

        with pytest.raises(ValidationError):
            CommerceIntent(query="   ")

    def test_invalid_price_range_rejected(self):
        # min_price > max_price must raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            CommerceIntent(
                query="running shoes",
                min_price=Decimal("5000"),
                max_price=Decimal("2000"),
            )
        assert "min_price (5000) must be less than or equal to max_price (2000)" in str(exc_info.value)

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            CommerceIntent(query="shoes", min_price=Decimal("-100"))

        with pytest.raises(ValidationError):
            CommerceIntent(query="shoes", max_price=Decimal("-50"))

    def test_requested_limit_validation(self):
        # Test that requested_limit must be >= 1
        with pytest.raises(ValidationError):
            CommerceIntent(query="shirts", requested_limit=0)

        with pytest.raises(ValidationError):
            CommerceIntent(query="shirts", requested_limit=-1)

        # Test that requested_limit must be <= 50 (MAX_SEARCH_LIMIT)
        with pytest.raises(ValidationError):
            CommerceIntent(query="shirts", requested_limit=51)

        # Valid limits
        intent = CommerceIntent(query="shirts", requested_limit=1)
        assert intent.requested_limit == 1

        intent = CommerceIntent(query="shirts", requested_limit=50)
        assert intent.requested_limit == 50


# ---------------------------------------------------------------------------
# Unit Tests: Mocked IntentParser Across Categories
# ---------------------------------------------------------------------------

class TestIntentParserUnit:
    @pytest.fixture
    def mock_parser(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_test_api_key"}):
            with patch("google.genai.Client"):
                parser = IntentParser(api_key="fake_test_api_key")
                return parser

    def test_clothing_intent_extraction(self, mock_parser):
        raw_text = "I need a simple black shirt for a wedding under 2500"
        mock_payload = {
            "query": "simple black shirt for a wedding",
            "category": "Shirts",
            "brand": None,
            "min_price": None,
            "max_price": 2500,
            "color": "black",
            "size": None,
            "attributes": {
                "occasion": "wedding",
                "style": "simple",
            },
        }

        mock_parser.client.models.generate_content.return_value = _mock_generate_content_response(mock_payload)
        intent = mock_parser.parse(raw_text)

        assert intent.query == "simple black shirt for a wedding"
        assert intent.category == "Shirts"
        assert intent.max_price == Decimal("2500")
        assert intent.min_price is None
        assert intent.color == "black"
        assert intent.attributes.get("occasion") == "wedding"
        assert intent.attributes.get("style") == "simple"

    def test_stationery_intent_extraction(self, mock_parser):
        raw_text = "Find me a blue mechanical pencil for drawing under 500"
        mock_payload = {
            "query": "blue mechanical pencil for drawing",
            "category": "Stationery",
            "brand": None,
            "min_price": None,
            "max_price": 500,
            "color": "blue",
            "size": None,
            "attributes": {
                "type": "mechanical",
                "use_case": "drawing",
            },
        }

        mock_parser.client.models.generate_content.return_value = _mock_generate_content_response(mock_payload)
        intent = mock_parser.parse(raw_text)

        assert intent.query == "blue mechanical pencil for drawing"
        assert intent.category == "Stationery"
        assert intent.max_price == Decimal("500")
        assert intent.color == "blue"
        assert intent.attributes.get("type") == "mechanical"
        assert intent.attributes.get("use_case") == "drawing"
        # Ensure no clothing-specific attributes are invented
        assert "style" not in intent.attributes
        assert "occasion" not in intent.attributes

    def test_electronics_intent_extraction(self, mock_parser):
        raw_text = "I need a lightweight laptop for programming under 80000 with 16GB RAM"
        mock_payload = {
            "query": "lightweight laptop for programming",
            "category": "Laptops",
            "brand": None,
            "min_price": None,
            "max_price": 80000,
            "color": None,
            "size": None,
            "attributes": {
                "weight": "lightweight",
                "use_case": "programming",
                "ram": "16GB",
            },
        }

        mock_parser.client.models.generate_content.return_value = _mock_generate_content_response(mock_payload)
        intent = mock_parser.parse(raw_text)

        assert intent.query == "lightweight laptop for programming"
        assert intent.category == "Laptops"
        assert intent.max_price == Decimal("80000")
        assert intent.attributes.get("ram") == "16GB"
        assert intent.attributes.get("weight") == "lightweight"
        assert intent.attributes.get("use_case") == "programming"
        # Ensure no clothing attributes
        assert "size" not in intent.attributes
        assert "fabric" not in intent.attributes

    def test_missing_optional_information(self, mock_parser):
        raw_text = "Show me laptops"
        mock_payload = {
            "query": "laptops",
            "category": "Laptops",
            "brand": None,
            "min_price": None,
            "max_price": None,
            "color": None,
            "size": None,
            "attributes": {},
        }

        mock_parser.client.models.generate_content.return_value = _mock_generate_content_response(mock_payload)
        intent = mock_parser.parse(raw_text)

        assert intent.query == "laptops"
        assert intent.category == "Laptops"
        assert intent.brand is None
        assert intent.min_price is None
        assert intent.max_price is None
        assert intent.color is None
        assert intent.size is None
        assert intent.attributes == {}

    def test_no_price_mentioned(self, mock_parser):
        raw_text = "I want black running shoes"
        mock_payload = {
            "query": "black running shoes",
            "category": "Shoes",
            "brand": None,
            "min_price": None,
            "max_price": None,
            "color": "black",
            "size": None,
            "attributes": {"use_case": "running"},
            "requested_limit": None,
        }

        mock_parser.client.models.generate_content.return_value = _mock_generate_content_response(mock_payload)
        intent = mock_parser.parse(raw_text)

        assert intent.query == "black running shoes"
        assert intent.category == "Shoes"
        assert intent.max_price is None
        assert intent.min_price is None
        assert intent.color == "black"
        assert intent.requested_limit is None

    def test_requested_limit_extraction_top_n(self, mock_parser):
        raw_text = "show me top 5 black shirts"
        mock_payload = {
            "query": "black shirts",
            "category": "Shirts",
            "brand": None,
            "min_price": None,
            "max_price": None,
            "color": "black",
            "size": None,
            "attributes": {},
            "requested_limit": 5,
        }

        mock_parser.client.models.generate_content.return_value = _mock_generate_content_response(mock_payload)
        intent = mock_parser.parse(raw_text)

        assert intent.query == "black shirts"
        assert intent.category == "Shirts"
        assert intent.color == "black"
        assert intent.requested_limit == 5

    def test_requested_limit_extraction_give_me_n(self, mock_parser):
        raw_text = "give me 3 black shirts"
        mock_payload = {
            "query": "black shirts",
            "category": "Shirts",
            "brand": None,
            "min_price": None,
            "max_price": None,
            "color": "black",
            "size": None,
            "attributes": {},
            "requested_limit": 3,
        }

        mock_parser.client.models.generate_content.return_value = _mock_generate_content_response(mock_payload)
        intent = mock_parser.parse(raw_text)

        assert intent.requested_limit == 3

    def test_requested_limit_extraction_show_n(self, mock_parser):
        raw_text = "show me 7 shoes"
        mock_payload = {
            "query": "shoes",
            "category": "Shoes",
            "brand": None,
            "min_price": None,
            "max_price": None,
            "color": None,
            "size": None,
            "attributes": {},
            "requested_limit": 7,
        }

        mock_parser.client.models.generate_content.return_value = _mock_generate_content_response(mock_payload)
        intent = mock_parser.parse(raw_text)

        assert intent.requested_limit == 7

    def test_requested_limit_extraction_top_1(self, mock_parser):
        raw_text = "show me top 1 black shirt"
        mock_payload = {
            "query": "black shirt",
            "category": "Shirts",
            "brand": None,
            "min_price": None,
            "max_price": None,
            "color": "black",
            "size": None,
            "attributes": {},
            "requested_limit": 1,
        }

        mock_parser.client.models.generate_content.return_value = _mock_generate_content_response(mock_payload)
        intent = mock_parser.parse(raw_text)

        assert intent.requested_limit == 1

    def test_requested_limit_extraction_words(self, mock_parser):
        raw_text = "show me five black shirts"
        mock_payload = {
            "query": "black shirts",
            "category": "Shirts",
            "brand": None,
            "min_price": None,
            "max_price": None,
            "color": "black",
            "size": None,
            "attributes": {},
            "requested_limit": 5,
        }

        mock_parser.client.models.generate_content.return_value = _mock_generate_content_response(mock_payload)
        intent = mock_parser.parse(raw_text)

        assert intent.requested_limit == 5

    def test_requested_limit_extraction_first_n(self, mock_parser):
        raw_text = "first 3 black shirts"
        mock_payload = {
            "query": "black shirts",
            "category": "Shirts",
            "brand": None,
            "min_price": None,
            "max_price": None,
            "color": "black",
            "size": None,
            "attributes": {},
            "requested_limit": 3,
        }

        mock_parser.client.models.generate_content.return_value = _mock_generate_content_response(mock_payload)
        intent = mock_parser.parse(raw_text)

        assert intent.requested_limit == 3

    def test_no_requested_limit_when_not_specified(self, mock_parser):
        raw_text = "show me black shirts"
        mock_payload = {
            "query": "black shirts",
            "category": "Shirts",
            "brand": None,
            "min_price": None,
            "max_price": None,
            "color": "black",
            "size": None,
            "attributes": {},
            "requested_limit": None,
        }

        mock_parser.client.models.generate_content.return_value = _mock_generate_content_response(mock_payload)
        intent = mock_parser.parse(raw_text)

        assert intent.requested_limit is None

    def test_empty_string_input_rejected(self, mock_parser):
        with pytest.raises(ValueError) as exc_info:
            mock_parser.parse("")
        assert "non-empty string" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info2:
            mock_parser.parse("   ")
        assert "non-empty string" in str(exc_info2.value)


# ---------------------------------------------------------------------------
# Unit Tests: IntentService
# ---------------------------------------------------------------------------

class TestIntentService:
    def test_intent_service_delegates_to_parser(self):
        mock_parser = MagicMock()
        mock_intent = CommerceIntent(
            query="mechanical pencil",
            category="Stationery",
            max_price=Decimal("300"),
        )
        mock_parser.parse.return_value = mock_intent

        service = IntentService(parser=mock_parser)
        result = service.extract_intent("Find a pencil under 300")

        mock_parser.parse.assert_called_once_with("Find a pencil under 300")
        assert result == mock_intent
        assert result.category == "Stationery"
        assert result.max_price == Decimal("300")

    def test_intent_service_empty_input_rejected(self):
        mock_parser = MagicMock()
        service = IntentService(parser=mock_parser)

        with pytest.raises(ValueError):
            service.extract_intent("")

        with pytest.raises(ValueError):
            service.extract_intent("   ")


# ---------------------------------------------------------------------------
# Optional Integration Test (Real Gemini API)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("RUN_GEMINI_INTEGRATION_TESTS") or not os.getenv("GEMINI_API_KEY"),
    reason="Requires active GEMINI_API_KEY and RUN_GEMINI_INTEGRATION_TESTS=1",
)
class TestGeminiIntentIntegration:
    def test_real_gemini_extraction(self):
        service = IntentService()
        intent = service.extract_intent("Find a lightweight silver laptop for coding under 75000 with 16GB RAM")

        assert isinstance(intent, CommerceIntent)
        assert len(intent.query) > 0
        assert intent.category is not None
        assert "laptop" in intent.query.lower() or "laptop" in intent.category.lower()
        if intent.max_price is not None:
            assert intent.max_price == Decimal("75000")
