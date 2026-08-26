"""
Intent Service: business logic layer for extracting and validating CommerceIntent from user text.

Responsibilities:
1. Validates raw input query string.
2. Calls IntentParser (Gemini Generative LLM).
3. Validates and enforces CommerceIntent schema.
4. Normalizes extracted fields (safe, deterministic normalization only).
5. Returns the validated CommerceIntent model.

Note:
- Does NOT perform database access or call ProductRepository.
- Does NOT generate embeddings or perform vector retrieval.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Optional
from app.schemas.intent import CommerceIntent
from app.ai.intent_parser import IntentParser


class IntentService:
    """
    Coordinates commerce intent extraction from natural language.
    """

    def __init__(self, parser: Optional[IntentParser] = None) -> None:
        self._parser = parser

    @property
    def parser(self) -> IntentParser:
        if self._parser is None:
            self._parser = IntentParser()
        return self._parser

    def extract_intent(self, text: str) -> CommerceIntent:
        """
        Extracts and validates a structured CommerceIntent from user text.

        Args:
            text (str): Raw user query string.

        Returns:
            CommerceIntent: Validated and normalized commerce intent structure.

        Raises:
            ValueError: If input text is empty or invalid.
            RuntimeError: If intent extraction fails.
        """
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Input query text must be a non-empty string.")

        clean_text = text.strip()
        intent = self.parser.parse(clean_text)

        # Ensure returned object is a valid CommerceIntent
        if not isinstance(intent, CommerceIntent):
            intent = CommerceIntent.model_validate(intent)

        # Apply safe, deterministic normalization
        intent = self._normalize(intent)

        return intent

    @staticmethod
    def _normalize(intent: CommerceIntent) -> CommerceIntent:
        """
        Apply safe, deterministic normalization to extracted intent fields.

        Rules:
        - Normalize color casing: "Black" -> "black"
        - Normalize price strings: "₹2,500" / "2.5k" -> Decimal("2500")
        - Strip whitespace from category/brand/size
        - Never invent values that weren't extracted
        - Never use LLM for normalization
        """
        updates = {}

        # Normalize color: lowercase for consistent variant matching
        if intent.color is not None:
            updates["color"] = intent.color.strip().lower()

        # Normalize category: strip whitespace only (preserve casing for exact DB match)
        if intent.category is not None:
            updates["category"] = intent.category.strip()

        # Normalize brand: strip whitespace
        if intent.brand is not None:
            updates["brand"] = intent.brand.strip()

        # Normalize size: strip whitespace, preserve original value
        if intent.size is not None:
            updates["size"] = intent.size.strip()

        # Normalize prices
        if intent.min_price is not None:
            normalized = _normalize_price(intent.min_price)
            if normalized is not None:
                updates["min_price"] = normalized

        if intent.max_price is not None:
            normalized = _normalize_price(intent.max_price)
            if normalized is not None:
                updates["max_price"] = normalized

        if updates:
            return intent.model_copy(update=updates)
        return intent


def _normalize_price(value) -> Optional[Decimal]:
    """
    Normalize a price value to a clean Decimal.

    Handles:
    - Already-valid Decimals/floats/ints
    - String formats: "₹2,500", "2500", "2.5k", "80K"

    Returns None if the value cannot be safely normalized.
    """
    if isinstance(value, Decimal):
        return value

    if isinstance(value, (int, float)):
        return Decimal(str(value))

    if isinstance(value, str):
        # Strip currency symbols and whitespace
        cleaned = re.sub(r'[₹$€£¥,\s]', '', value.strip())

        if not cleaned:
            return None

        # Handle "k" / "K" suffix (e.g. "2.5k" -> 2500)
        k_match = re.match(r'^(\d+(?:\.\d+)?)[kK]$', cleaned)
        if k_match:
            try:
                return Decimal(str(float(k_match.group(1)) * 1000))
            except (ValueError, InvalidOperation):
                return None

        # Try direct decimal parse
        try:
            return Decimal(cleaned)
        except (ValueError, InvalidOperation):
            return None

    return None
