"""
Embedding Service module using Google GenAI SDK (google-genai) and Gemini Embedding 2.

Provides reusable embedding generation for cross-modal search:
- text -> embedding (implemented)
- image -> embedding (staged for multimodal step)
"""

import os
from typing import Optional, List
from google import genai
from google.genai import types

from app.core.config import settings


class EmbeddingService:
    """
    Service for generating vector embeddings using Google's Gemini Embedding 2 model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Please set GEMINI_API_KEY in your environment or .env file."
            )

        self.model = model or settings.EMBEDDING_MODEL or "gemini-embedding-2"
        self.dimension = dimension or settings.EMBEDDING_DIMENSION or 1536

        # Initialize the official Google GenAI Client
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Google GenAI Client: {exc}") from exc

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a 1536-dimensional vector embedding for the given input text.

        Args:
            text (str): Input text to embed.

        Returns:
            List[float]: 1536-dimensional vector embedding.

        Raises:
            ValueError: If input text is empty or invalid.
            RuntimeError: If the Google GenAI embedding call fails.
        """
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Input text for embedding must be a non-empty string.")

        cleaned_text = text.strip()

        try:
            config = types.EmbedContentConfig(
                output_dimensionality=self.dimension
            )

            response = self.client.models.embed_content(
                model=self.model,
                contents=cleaned_text,
                config=config,
            )

            if not response or not response.embeddings:
                raise RuntimeError("Empty response received from Gemini Embedding API.")

            embedding_values = response.embeddings[0].values
            if not embedding_values:
                raise RuntimeError("No embedding values found in Gemini response.")

            return embedding_values

        except Exception as exc:
            raise RuntimeError(f"Gemini embedding generation failed: {exc}") from exc

    def embed_image(self, image_data: bytes, mime_type: str = "image/jpeg") -> List[float]:
        """
        Staged for multimodal image embedding in upcoming step.
        """
        raise NotImplementedError(
            "embed_image() is staged and will be implemented in the multimodal search step."
        )


# Global singleton instance provider
_embedding_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Returns a singleton or configured instance of EmbeddingService."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance
