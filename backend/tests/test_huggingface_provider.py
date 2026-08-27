"""
Tests for Hugging Face LLM provider integration and dual-provider toggling.
"""

import json
import os
import uuid
from unittest.mock import MagicMock, patch
import pytest

from app.ai.intent_parser import IntentParser, _clean_json_text
from app.schemas.intent import CommerceIntent
from app.services.agent_service import AgentService
from app.core.merchant_context import MerchantContext
from app.schemas.agent import AgentChatResponse


class TestCleanJsonText:
    """Test the JSON cleaning and markdown-stripping utility."""

    def test_clean_plain_json(self):
        raw = '{"query": "black shirt", "category": "Shirts", "requested_limit": 5}'
        assert _clean_json_text(raw) == raw

    def test_clean_markdown_code_block(self):
        raw = '```json\n{"query": "running shoes", "category": "Shoes"}\n```'
        assert _clean_json_text(raw) == '{"query": "running shoes", "category": "Shoes"}'

    def test_clean_markdown_no_lang(self):
        raw = '```\n{"query": "laptop", "category": "Laptops"}\n```'
        assert _clean_json_text(raw) == '{"query": "laptop", "category": "Laptops"}'

    def test_clean_with_surrounding_commentary(self):
        raw = 'Here is the extracted commerce intent:\n```json\n{"query": "blue jeans", "category": "Jeans"}\n```\nHope that helps!'
        assert _clean_json_text(raw) == '{"query": "blue jeans", "category": "Jeans"}'

    def test_clean_raw_text_with_braces(self):
        raw = 'Sure, intent is {"query": "white sneakers", "color": "white"} thank you.'
        assert _clean_json_text(raw) == '{"query": "white sneakers", "color": "white"}'


class TestHuggingFaceIntentParser:
    """Test IntentParser using Hugging Face provider."""

    @pytest.fixture
    def mock_hf_parser(self):
        with patch.dict(os.environ, {"HF_TOKEN": "fake_hf_token", "LLM_PROVIDER": "huggingface"}):
            with patch("app.ai.intent_parser.InferenceClient"):
                parser = IntentParser(api_key="fake_hf_token", provider="huggingface")
                return parser

    def test_huggingface_parse_success(self, mock_hf_parser):
        """Test successful structured intent parsing via Hugging Face."""
        expected_json = {
            "query": "black formal shirt",
            "category": "Shirts",
            "category_concept": None,
            "brand": None,
            "min_price": None,
            "max_price": 2500.0,
            "color": "black",
            "size": "M",
            "attributes": {"style": "formal"},
            "requested_limit": 3,
        }

        # Mock Hugging Face InferenceClient chat_completion response
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(expected_json)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_hf_parser.hf_client.chat_completion.return_value = mock_response

        intent = mock_hf_parser.parse("top 3 black formal shirt size M under 2500")

        assert isinstance(intent, CommerceIntent)
        assert intent.query == "black formal shirt"
        assert intent.category == "Shirts"
        assert intent.color == "black"
        assert intent.size == "M"
        assert intent.max_price == 2500.0
        assert intent.requested_limit == 3
        assert intent.attributes == {"style": "formal"}

    def test_huggingface_parse_with_markdown_fence(self, mock_hf_parser):
        """Test parsing when model wraps JSON inside markdown code fence."""
        expected_json = {
            "query": "gaming laptop",
            "category": "Laptops",
            "category_concept": None,
            "brand": "Asus",
            "min_price": 50000.0,
            "max_price": 90000.0,
            "color": None,
            "size": None,
            "attributes": {"ram": "16GB"},
            "requested_limit": None,
        }

        mock_choice = MagicMock()
        mock_choice.message.content = f"```json\n{json.dumps(expected_json)}\n```"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_hf_parser.hf_client.chat_completion.return_value = mock_response

        intent = mock_hf_parser.parse("asus gaming laptop between 50000 and 90000 with 16GB ram")
        assert intent.brand == "Asus"
        assert intent.category == "Laptops"
        assert intent.min_price == 50000.0
        assert intent.max_price == 90000.0

    def test_provider_toggle_to_gemini(self):
        """Test initializing IntentParser with provider='gemini'."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_gemini_key"}):
            with patch("app.ai.intent_parser.genai.Client"):
                parser = IntentParser(api_key="fake_gemini_key", provider="gemini")
                assert parser.provider == "gemini"
                assert parser.client is not None
                assert parser.hf_client is None


class TestHuggingFaceAgentService:
    """Test AgentService using Hugging Face provider for tool orchestration."""

    def test_huggingface_agent_direct_response(self):
        """Test agent conversational response without tools via Hugging Face."""
        mock_db = MagicMock()
        merchant_context = MerchantContext(merchant_id=uuid.uuid4())

        agent = AgentService(
            db=mock_db,
            merchant_context=merchant_context,
            provider="huggingface",
        )

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello! I am your shopping assistant. How can I help?"
        mock_choice.message.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_hf_client = MagicMock()
        mock_hf_client.chat_completion.return_value = mock_response
        agent._hf_client = mock_hf_client

        mock_session = MagicMock()
        mock_session.current_intent = None
        mock_session.last_search_results = None
        mock_session.cart_id = None

        res = agent._generate_with_tools_huggingface("hello", "", mock_session)
        assert res["text"] == "Hello! I am your shopping assistant. How can I help?"
        assert res["tool_calls"] == []

    def test_huggingface_agent_tool_calling(self):
        """Test agent tool call execution via Hugging Face."""
        mock_db = MagicMock()
        merchant_context = MerchantContext(merchant_id=uuid.uuid4())

        agent = AgentService(
            db=mock_db,
            merchant_context=merchant_context,
            provider="huggingface",
        )

        # Mock initial tool call
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "search_products"
        mock_tool_call.function.arguments = json.dumps({"query": "black shirt", "limit": 5})

        mock_choice_1 = MagicMock()
        mock_choice_1.message.content = None
        mock_choice_1.message.tool_calls = [mock_tool_call]

        mock_response_1 = MagicMock()
        mock_response_1.choices = [mock_choice_1]

        # Mock followup response after tool execution
        mock_choice_2 = MagicMock()
        mock_choice_2.message.content = "I found 5 black shirts for you!"
        mock_response_2 = MagicMock()
        mock_response_2.choices = [mock_choice_2]

        mock_hf_client = MagicMock()
        mock_hf_client.chat_completion.side_effect = [mock_response_1, mock_response_2]
        agent._hf_client = mock_hf_client

        # Mock execute tool
        with patch.object(agent, "_execute_tool", return_value={"success": True, "count": 5}):
            mock_session = MagicMock()
            mock_session.current_intent = None
            mock_session.last_search_results = None
            mock_session.cart_id = None

            res = agent._generate_with_tools_huggingface("find me 5 black shirts", "", mock_session)

            assert len(res["tool_calls"]) == 1
            assert res["tool_calls"][0]["name"] == "search_products"
            assert res["tool_calls"][0]["args"]["query"] == "black shirt"
            assert res["text"] == "I found 5 black shirts for you!"
