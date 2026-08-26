"""
Intent Parser: uses Gemini Generative LLM with structured output to extract generic CommerceIntent.

Supports ANY ecommerce category (clothing, electronics, stationery, watches, books, cosmetics, etc.).
"""

import os
import re
import json
import time
from typing import Optional
from google import genai
from google.genai import types
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.intent import CommerceIntent


SYSTEM_INSTRUCTION = """You are an expert commerce intent extraction engine for an ecommerce platform.
Your task is to analyze natural language customer search queries across ANY product category (including clothing, electronics, stationery, footwear, watches, cosmetics, furniture, tools, books, etc.) and extract a structured JSON response matching the CommerceIntent schema.

Output JSON Schema:
{
  "query": string (Required. The core semantic search query expressing the product desired, stripped of conversational preambles/pleasantries like 'Hey, can you please find me' or 'Looking for'. Example: 'simple black shirt for a wedding'),
  "category": string or null (The product category, e.g. 'Shirts', 'Laptops', 'Stationery', 'Shoes', 'Dresses', or null if unspecified/ambiguous),
  "brand": string or null (Explicit brand mentioned by customer, or null),
  "min_price": number or null (Minimum price boundary in currency units if specified, or null),
  "max_price": number or null (Maximum price boundary in currency units if specified, or null),
  "color": string or null (Requested color or finish, or null),
  "size": string or null (Requested size, or null),
  "attributes": object (JSON key-value dictionary containing category-specific constraints/preferences present in or strongly implied by the request. E.g. {"occasion": "wedding", "style": "simple"} for clothing, {"weight": "lightweight", "ram": "16GB", "use_case": "programming"} for electronics, {"type": "mechanical", "use_case": "drawing"} for stationery)
}

Strict Rules:
1. Support ANY commerce category (stationery, electronics, clothing, cosmetics, furniture, books, etc.).
2. DO NOT invent attributes or values that the customer did not mention or strongly imply.
3. DO NOT assign clothing-specific attributes (like style, fit, occasion) to non-clothing products like laptops or pencils.
4. DO NOT generate SQL, database queries, or product IDs.
5. Return ONLY a valid JSON object matching the schema above.
"""


class IntentParser:
    """
    Parses natural language commerce search queries into structured CommerceIntent models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Please set GEMINI_API_KEY in your environment or .env file."
            )

        self.model = model or getattr(settings, "GEMINI_INTENT_MODEL", None) or os.getenv("GEMINI_INTENT_MODEL", "gemini-3.6-flash")

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Google GenAI Client: {exc}") from exc

    def parse(self, text: str) -> CommerceIntent:
        """
        Parses raw customer text into a validated CommerceIntent.

        Args:
            text (str): Natural language user input.

        Returns:
            CommerceIntent: Validated structured commerce intent.

        Raises:
            ValueError: If input text is empty or invalid.
            RuntimeError: If the Gemini LLM API call fails.
            ValidationError: If the returned JSON does not conform to CommerceIntent rules.
        """
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Input text for intent parsing must be a non-empty string.")

        import time

        user_prompt = f"Customer request: {text.strip()}"
        max_retries = 4
        last_exc = None

        candidate_models = [self.model, "gemini-3.5-flash", "gemini-3.5-flash-lite"]
        candidate_models = list(dict.fromkeys(candidate_models))

        for model_name in candidate_models:
            for attempt in range(max_retries):
                try:
                    config = types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        temperature=0.0,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    )

                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=user_prompt,
                        config=config,
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc)
                    if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str):
                        # Switch to fallback model immediately
                        if model_name != candidate_models[-1]:
                            break
                        if attempt < max_retries - 1:
                            delay_match = re.search(r'retry in (\d+(?:\.\d+)?)s', err_str)
                            wait_sec = float(delay_match.group(1)) + 1.5 if delay_match else (5.0 * (attempt + 1))
                            time.sleep(wait_sec)
                            continue
                    raise RuntimeError(f"Gemini intent extraction call failed: {exc}") from exc

            if last_exc is None:
                break

        if last_exc:
            raise RuntimeError(f"Gemini intent extraction call failed: {last_exc}") from last_exc

        if not response or not response.text:
            raise RuntimeError("Gemini model returned an empty response.")

        # Validate structured output into CommerceIntent Pydantic model
        try:
            return CommerceIntent.model_validate_json(response.text)
        except ValidationError as val_err:
            raise ValidationError.from_exception_data(
                title="CommerceIntentValidation",
                line_errors=val_err.errors(),
            ) from val_err
        except Exception as parse_err:
            raise ValueError(f"Failed to parse model output into CommerceIntent: {parse_err}") from parse_err
