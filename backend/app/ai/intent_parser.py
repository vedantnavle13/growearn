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
from huggingface_hub import InferenceClient
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.intent import CommerceIntent


SYSTEM_INSTRUCTION = """You are an expert commerce intent extraction engine for an ecommerce platform.
Your task is to analyze natural language customer search queries across ANY product category (including clothing, electronics, stationery, footwear, watches, cosmetics, furniture, tools, books, etc.) and extract a structured JSON response matching the CommerceIntent schema.

Output JSON Schema:
{
  "query": string (Required. The core semantic search query expressing the product desired, stripped of conversational preambles/pleasantries like 'Hey, can you please find me' or 'Looking for'. Example: 'simple black shirt for a wedding'),
  "category": string or null (EXACT product category, e.g. 'Shirts', 'Laptops', 'Stationery', 'Jeans', 'Dresses', or null if unspecified/ambiguous. ONLY use when user specifies a specific category that likely exists in the database.),
  "category_concept": string or null (Broad semantic category CONCEPT, e.g. 'clothing', 'electronics', 'footwear', 'stationery', 'furniture', or null. Use for broad user terms like 'clothes', 'electronics', 'shoes', 'things' that map to MULTIPLE database categories.),
  "brand": string or null (Explicit brand mentioned by customer, or null),
  "min_price": number or null (Minimum price boundary in currency units if specified, or null),
  "max_price": number or null (Maximum price boundary in currency units if specified, or null),
  "color": string or null (Requested color or finish, or null),
  "size": string or null (Requested size, or null),
  "attributes": object (JSON key-value dictionary containing category-specific constraints/preferences present in or strongly implied by the request. E.g. {"occasion": "wedding", "style": "simple"} for clothing, {"weight": "lightweight", "ram": "16GB", "use_case": "programming"} for electronics, {"type": "mechanical", "use_case": "drawing"} for stationery),
  "requested_limit": integer or null (User-specified result count limit. Extract from phrases like "top 5", "5 items", "give me 5", "show me five", "first 3", "show 10", "top 1". Return null if no count specified. Must be >= 1. Maximum allowed is 50 - if user requests more, clamp to 50.)
}

CRITICAL DISTINCTION - category vs category_concept:

EXACT CATEGORY (category field):
- User says "black shirt" → category: "Shirts", category_concept: null
- User says "wedding dress" → category: "Dresses", category_concept: null  
- User says "gaming laptop" → category: "Laptops", category_concept: null
- User says "running shoes" → category: "Shoes", category_concept: null

BROAD CATEGORY CONCEPT (category_concept field):
- User says "black clothes" → category: null, category_concept: "clothing"
- User says "electronics under 50000" → category: null, category_concept: "electronics"
- User says "footwear for running" → category: null, category_concept: "footwear"
- User says "something nice" → category: null, category_concept: null (too vague)

AMBIGUOUS - use category_concept when uncertain:
- "show me clothes" → category_concept: "clothing"
- "I need electronics" → category_concept: "electronics"

REQUESTED LIMIT EXAMPLES:
- "top 5 black shirts" → requested_limit: 5
- "give me 3 black shirts" → requested_limit: 3
- "show me 7 shoes" → requested_limit: 7
- "show me 1 black shirt" → requested_limit: 1
- "show me five black shirts" → requested_limit: 5
- "first 3 black shirts" → requested_limit: 3
- "show me black shirts" → requested_limit: null
- "I need a black shirt" → requested_limit: null

Strict Rules:
1. Support ANY commerce category (stationery, electronics, clothing, cosmetics, furniture, books, etc.).
2. DO NOT invent attributes or values that the customer did not mention or strongly imply.
3. DO NOT assign clothing-specific attributes (like style, fit, occasion) to non-clothing products like laptops or pencils.
4. DO NOT generate SQL, database queries, or product IDs.
5. Return ONLY a valid JSON object matching the schema above.
6. NEVER use category_concept for specific categories like "Shirts" or "Laptops" - those go in category field.
7. Extract requested_limit from natural language count expressions. If no count is specified, set requested_limit to null. Do NOT hallucinate a count.
"""


def _clean_json_text(text: str) -> str:
    """Extract and clean raw JSON string from potentially markdown-wrapped LLM outputs."""
    if not text:
        return ""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text


class IntentParser:
    """
    Parses natural language commerce search queries into structured CommerceIntent models.
    Supports both Hugging Face Inference API and Google Gemini providers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        hf_token = (
            getattr(settings, "HF_TOKEN", None)
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
            or os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        gemini_key = (
            getattr(settings, "GEMINI_API_KEY", None)
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        if provider:
            self.provider = provider.lower()
        elif api_key and (api_key.startswith("hf_")):
            self.provider = "huggingface"
        elif api_key and (api_key.startswith("AIza") or api_key.startswith("AQ.")):
            self.provider = "gemini"
        elif os.getenv("LLM_PROVIDER"):
            self.provider = os.getenv("LLM_PROVIDER").lower()
        else:
            self.provider = (
                getattr(settings, "LLM_PROVIDER", None)
                or "huggingface"
            ).lower()

        if not provider:
            explicit_env_provider = os.getenv("LLM_PROVIDER")
            if not explicit_env_provider:
                if self.provider == "huggingface" and not hf_token and gemini_key:
                    self.provider = "gemini"
                elif self.provider == "gemini" and not gemini_key and hf_token:
                    self.provider = "huggingface"

        if self.provider == "gemini":
            self.api_key = api_key or gemini_key
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not configured. "
                    "Please set GEMINI_API_KEY in your environment or .env file."
                )
            self.model = model or getattr(settings, "GEMINI_INTENT_MODEL", None) or os.getenv("GEMINI_INTENT_MODEL", "gemini-2.5-flash")
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize Google GenAI Client: {exc}") from exc
            self.hf_client = None

        else:
            # Default to Hugging Face provider
            self.provider = "huggingface"
            self.api_key = api_key or hf_token
            if not self.api_key:
                raise ValueError(
                    "HF_TOKEN is not configured. "
                    "Please set HF_TOKEN or HUGGINGFACEHUB_ACCESS_TOKEN in your environment or .env file."
                )

            self.model = model or getattr(settings, "HF_MODEL", None) or os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
            try:
                self.hf_client = InferenceClient(
                    token=self.api_key,
                    base_url=getattr(settings, "HF_API_BASE", None),
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize Hugging Face InferenceClient: {exc}") from exc
            self.client = None

    def parse(self, text: str) -> CommerceIntent:
        """
        Parses raw customer text into a validated CommerceIntent with automatic cross-provider fallback.
        """
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Input text for intent parsing must be a non-empty string.")

        provider = getattr(self, "provider", None)
        if not provider:
            if getattr(self, "client", None) is not None:
                provider = "gemini"
            elif getattr(self, "hf_client", None) is not None:
                provider = "huggingface"
            else:
                provider = getattr(settings, "LLM_PROVIDER", "huggingface").lower()

        if provider == "gemini":
            try:
                return self._parse_gemini(text.strip())
            except Exception as gem_exc:
                # Check if Hugging Face fallback is available
                hf_token = (
                    getattr(settings, "HF_TOKEN", None)
                    or os.getenv("HF_TOKEN")
                    or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
                    or os.getenv("HUGGINGFACE_API_KEY")
                )
                if hf_token:
                    try:
                        if not self.hf_client:
                            self.hf_client = InferenceClient(
                                token=hf_token,
                                base_url=getattr(settings, "HF_API_BASE", None),
                            )
                        return self._parse_huggingface(text.strip())
                    except Exception:
                        pass
                raise gem_exc
        else:
            try:
                return self._parse_huggingface(text.strip())
            except Exception as hf_exc:
                # Check if Gemini fallback is available
                gemini_key = (
                    getattr(settings, "GEMINI_API_KEY", None)
                    or os.getenv("GEMINI_API_KEY")
                    or os.getenv("GOOGLE_API_KEY")
                )
                if gemini_key:
                    try:
                        if not self.client:
                            self.client = genai.Client(api_key=gemini_key)
                        return self._parse_gemini(text.strip())
                    except Exception:
                        pass
                raise hf_exc

    def _parse_huggingface(self, text: str) -> CommerceIntent:
        """Parse commerce intent using Hugging Face Inference API."""
        user_prompt = f"Customer request: {text}"
        candidate_models = [self.model, "Qwen/Qwen2.5-72B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
        candidate_models = [m for m in dict.fromkeys(candidate_models) if m]

        max_retries = 3
        last_exc = None
        raw_response_text = None

        for model_name in candidate_models:
            for attempt in range(max_retries):
                try:
                    response = self.hf_client.chat_completion(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_INSTRUCTION},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.0,
                        max_tokens=600,
                    )
                    if response and response.choices and response.choices[0].message:
                        raw_response_text = response.choices[0].message.content
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc)
                    if "429" in err_str or "Rate limit" in err_str or "temporarily unavailable" in err_str:
                        if model_name != candidate_models[-1]:
                            break
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Hugging Face intent extraction call failed: {exc}") from exc

            if last_exc is None and raw_response_text:
                break

        if last_exc and not raw_response_text:
            raise RuntimeError(f"Hugging Face intent extraction call failed: {last_exc}") from last_exc

        if not raw_response_text:
            raise RuntimeError("Hugging Face model returned an empty response.")

        clean_json = _clean_json_text(raw_response_text)

        # Validate structured output into CommerceIntent Pydantic model
        try:
            return CommerceIntent.model_validate_json(clean_json)
        except ValidationError as val_err:
            raise ValidationError.from_exception_data(
                title="CommerceIntentValidation",
                line_errors=val_err.errors(),
            ) from val_err
        except Exception as parse_err:
            raise ValueError(f"Failed to parse model output into CommerceIntent: {parse_err}") from parse_err

    def _parse_gemini(self, text: str) -> CommerceIntent:
        """Parse commerce intent using Google Gemini API."""
        user_prompt = f"Customer request: {text}"
        max_retries = 3
        last_exc = None
        response = None

        primary_model = self.model or getattr(settings, "GEMINI_INTENT_MODEL", None) or "gemini-3.6-flash"
        candidate_models = [primary_model, "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
        candidate_models = [m for m in dict.fromkeys(candidate_models) if m]

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
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        if model_name != candidate_models[-1]:
                            break
                        if attempt < max_retries - 1:
                            delay_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
                            wait_sec = float(delay_match.group(1)) + 1.5 if delay_match else (3.0 * (attempt + 1))
                            time.sleep(wait_sec)
                            continue
                    raise RuntimeError(f"Gemini intent extraction call failed: {exc}") from exc

            if last_exc is None:
                break

        if last_exc:
            raise RuntimeError(f"Gemini intent extraction call failed: {last_exc}") from last_exc

        if not response or not response.text:
            raise RuntimeError("Gemini model returned an empty response.")

        clean_json = _clean_json_text(response.text)

        # Validate structured output into CommerceIntent Pydantic model
        try:
            return CommerceIntent.model_validate_json(clean_json)
        except ValidationError as val_err:
            raise ValidationError.from_exception_data(
                title="CommerceIntentValidation",
                line_errors=val_err.errors(),
            ) from val_err
        except Exception as parse_err:
            raise ValueError(f"Failed to parse model output into CommerceIntent: {parse_err}") from parse_err

