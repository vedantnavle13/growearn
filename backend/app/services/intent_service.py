"""
Intent Service: business logic layer for extracting and validating CommerceIntent from user text.

Responsibilities:
1. Validates raw input query string.
2. Calls IntentParser (Gemini Generative LLM).
3. Validates and enforces CommerceIntent schema.
4. Returns the validated CommerceIntent model.

Note:
- Does NOT perform database access or call ProductRepository.
- Does NOT generate embeddings or perform vector retrieval.
"""

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
            CommerceIntent: Validated commerce intent structure.

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

        return intent
