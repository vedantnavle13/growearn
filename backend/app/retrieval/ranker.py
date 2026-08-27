"""
Product Ranker: deterministic candidate ranking using multiple relevance signals and customer personalization.

Signals:
1. Semantic Similarity: pgvector cosine similarity score from vector retrieval.
2. Keyword Relevance: Token overlap between clean query terms and product text fields.
3. Dynamic Attribute Relevance: Intent attributes matched against product JSONB attributes.
4. Category Concept Relevance: Product category alignment with broad user concepts (clothing, electronics, etc.).
5. Personalization Score: Historical customer preference alignment (categories, brands, colors, price, prior orders).

Final Score Formula:
    final_score = (SEMANTIC_WEIGHT * semantic_score)
                + (KEYWORD_WEIGHT * keyword_score)
                + (ATTRIBUTE_WEIGHT * attribute_score)
                + (CONCEPT_WEIGHT * concept_score)
                + (PERSONALIZATION_WEIGHT * personalization_score)

Ranking occurs strictly AFTER hard constraints have eliminated non-qualifying products.
No LLMs or black-box neural networks are used in the ranking phase to maintain low latency, low cost, and explainability.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Tuple, Dict, Any, Set, Optional

from app.models.product import Product
from app.schemas.intent import CommerceIntent
from app.schemas.preference import CustomerPreferences

# ---------------------------------------------------------------------------
# Default Configurable Weights
# ---------------------------------------------------------------------------
SEMANTIC_WEIGHT: float = 0.45
KEYWORD_WEIGHT: float = 0.15
ATTRIBUTE_WEIGHT: float = 0.15
CONCEPT_WEIGHT: float = 0.10
PERSONALIZATION_WEIGHT: float = 0.15

# Common English stop words ignored in keyword scoring
STOP_WORDS: Set[str] = {
    "a", "an", "the", "for", "i", "want", "need", "something",
    "in", "to", "of", "and", "or", "with", "under", "me", "show",
    "find", "is", "are", "by", "on", "at", "it", "this", "that",
    "please", "can", "you", "give", "looking"
}


@dataclass(frozen=True)
class ScoredProduct:
    """
    Candidate product paired with individual and composite ranking scores.
    """
    product: Product
    semantic_score: float
    keyword_score: float
    attribute_score: float
    concept_score: float
    final_score: float
    personalization_score: float = 0.0

    def to_explanation_dict(self) -> Dict[str, Any]:
        """Internal ranking breakdown for debugging."""
        return {
            "product_id": str(self.product.id),
            "title": self.product.title,
            "semantic_score": round(self.semantic_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "attribute_score": round(self.attribute_score, 4),
            "concept_score": round(self.concept_score, 4),
            "personalization_score": round(self.personalization_score, 4),
            "final_score": round(self.final_score, 4),
        }


class ProductRanker:
    """
    Deterministic multi-signal ranker with customer personalization support.
    """

    def __init__(
        self,
        semantic_weight: float = SEMANTIC_WEIGHT,
        keyword_weight: float = KEYWORD_WEIGHT,
        attribute_weight: float = ATTRIBUTE_WEIGHT,
        concept_weight: float = CONCEPT_WEIGHT,
        personalization_weight: float = PERSONALIZATION_WEIGHT,
    ) -> None:
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.attribute_weight = attribute_weight
        self.concept_weight = concept_weight
        self.personalization_weight = personalization_weight

    def rank(
        self,
        candidates: List[Tuple[Product, float]],
        intent: CommerceIntent,
        preferences: Optional[CustomerPreferences] = None,
        concept_categories: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[ScoredProduct]:
        """
        Rank candidate products based on semantic similarity, keyword overlap, attribute alignment,
        category concept alignment, and customer preference signals.

        Args:
            candidates: List of (Product, semantic_similarity) tuples from hard-filtered retrieval.
            intent: Structured CommerceIntent containing clean query and dynamic attributes.
            preferences: Optional CustomerPreferences profile derived from customer activity.
            concept_categories: List of actual database categories that match the user's broad category concept.
                              Used for concept relevance scoring (e.g., "clothing" -> ["Shirts", "Jeans", "Jackets"]).
            limit: Number of top ranked products to return.

        Returns:
            List[ScoredProduct] sorted descending by final_score.
        """
        if not candidates:
            return []

        scored: List[ScoredProduct] = []

        query_tokens = self._tokenize(intent.query)

        for product, sem_score in candidates:
            # 1. Semantic score (clamped between 0.0 and 1.0)
            semantic_score = max(0.0, min(1.0, float(sem_score)))

            # 2. Keyword relevance score
            keyword_score = self._compute_keyword_score(product, query_tokens)

            # 3. Dynamic attribute relevance score
            attribute_score = self._compute_attribute_score(product, intent.attributes)

            # 4. Category concept relevance score
            concept_score = self._compute_concept_score(product, concept_categories)

            # 5. Personalization relevance score
            personalization_score = self._compute_personalization_score(product, preferences)

            # 6. Composite final score
            final_score = (
                (self.semantic_weight * semantic_score)
                + (self.keyword_weight * keyword_score)
                + (self.attribute_weight * attribute_score)
                + (self.concept_weight * concept_score)
                + (self.personalization_weight * personalization_score)
            )

            scored.append(
                ScoredProduct(
                    product=product,
                    semantic_score=round(semantic_score, 4),
                    keyword_score=round(keyword_score, 4),
                    attribute_score=round(attribute_score, 4),
                    concept_score=round(concept_score, 4),
                    personalization_score=round(personalization_score, 4),
                    final_score=round(final_score, 4),
                )
            )

        # Sort descending by final_score, breaking ties by semantic_score
        scored.sort(key=lambda s: (s.final_score, s.semantic_score), reverse=True)

        return scored[:limit]

    # ------------------------------------------------------------------
    # Scoring Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: Optional[str]) -> Set[str]:
        """Extract lowercase alphanumeric tokens excluding stop words."""
        if not text:
            return set()
        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        return {w for w in words if w not in STOP_WORDS and len(w) > 1}

    def _compute_keyword_score(self, product: Product, query_tokens: Set[str]) -> float:
        """
        Compute token overlap between meaningful query terms and product text fields.
        Returns a float in [0.0, 1.0].
        """
        if not query_tokens:
            return 0.0

        # Build document text representation
        doc_parts = [
            product.title or "",
            product.description or "",
        ]

        if product.attributes and isinstance(product.attributes, dict):
            for k in ("category", "brand", "color", "style", "occasion", "material", "tags"):
                val = product.attributes.get(k)
                if isinstance(val, list):
                    doc_parts.extend(str(item) for item in val)
                elif val:
                    doc_parts.append(str(val))

        doc_tokens = self._tokenize(" ".join(doc_parts))

        if not doc_tokens:
            return 0.0

        matching_tokens = query_tokens.intersection(doc_tokens)
        return len(matching_tokens) / len(query_tokens)

    @staticmethod
    def _compute_attribute_score(product: Product, intent_attributes: Dict[str, Any]) -> float:
        """
        Compare CommerceIntent.attributes against Product.attributes.
        Only evaluates attributes requested in intent.
        Returns a float in [0.0, 1.0].
        """
        if not intent_attributes:
            return 0.0

        prod_attrs = product.attributes if isinstance(product.attributes, dict) else {}
        if not prod_attrs:
            return 0.0

        # Case-insensitive map of product attributes
        prod_attrs_lower = {str(k).lower().strip(): v for k, v in prod_attrs.items()}

        total_match_points = 0.0
        total_eval_keys = len(intent_attributes)

        for raw_key, intent_val in intent_attributes.items():
            key_lower = str(raw_key).lower().strip()
            if key_lower not in prod_attrs_lower:
                # Key not present in product
                continue

            prod_val = prod_attrs_lower[key_lower]
            match_score = ProductRanker._compare_attribute_values(intent_val, prod_val)
            total_match_points += match_score

        return total_match_points / total_eval_keys

    @staticmethod
    def _compare_attribute_values(intent_val: Any, prod_val: Any) -> float:
        """
        Deterministic comparison of intent attribute value vs product attribute value.
        Supports strings, numbers, booleans, and simple lists.
        """
        if intent_val is None or prod_val is None:
            return 0.0

        # Boolean comparison
        if isinstance(intent_val, bool) or isinstance(prod_val, bool):
            return 1.0 if bool(intent_val) == bool(prod_val) else 0.0

        # Numeric comparison
        if isinstance(intent_val, (int, float)) and isinstance(prod_val, (int, float)):
            return 1.0 if float(intent_val) == float(prod_val) else 0.0

        # String in List (e.g. tag in tags list)
        if isinstance(prod_val, (list, set, tuple)):
            norm_intent_str = str(intent_val).strip().lower()
            prod_items = {str(item).strip().lower() for item in prod_val}
            return 1.0 if norm_intent_str in prod_items else 0.0

        if isinstance(intent_val, (list, set, tuple)) and isinstance(prod_val, (list, set, tuple)):
            intent_items = {str(item).strip().lower() for item in intent_val}
            prod_items = {str(item).strip().lower() for item in prod_val}
            if not intent_items:
                return 0.0
            return len(intent_items.intersection(prod_items)) / len(intent_items)

        # String to String comparison
        str_intent = str(intent_val).strip().lower()
        str_prod = str(prod_val).strip().lower()

        if str_intent == str_prod:
            return 1.0
        elif str_intent in str_prod or str_prod in str_intent:
            return 0.75

        return 0.0

    def _compute_concept_score(
        self,
        product: Product,
        concept_categories: Optional[List[str]],
    ) -> float:
        """
        Compute category concept relevance score.
        
        Checks if the product's category matches any of the concept categories
        (e.g., "clothing" -> ["Shirts", "Jeans", "Jackets"]).
        
        Returns a float in [0.0, 1.0].
        """
        if not concept_categories:
            return 0.0

        prod_attrs = product.attributes if isinstance(product.attributes, dict) else {}
        if not prod_attrs:
            return 0.0

        prod_cat = prod_attrs.get("category")
        if not prod_cat:
            return 0.0

        prod_cat_clean = str(prod_cat).strip()

        # Direct match: product category is in concept categories
        if prod_cat_clean in concept_categories:
            return 1.0

        # Fuzzy match: check if any concept category is a substring of product category or vice versa
        for concept_cat in concept_categories:
            concept_clean = str(concept_cat).strip().lower()
            prod_lower = prod_cat_clean.lower()
            if concept_clean in prod_lower or prod_lower in concept_clean:
                return 0.75

        return 0.0

    @staticmethod
    def _compute_personalization_score(
        product: Product,
        preferences: Optional[CustomerPreferences],
    ) -> float:
        """
        Calculates customer preference match score for a product.
        Returns a float in [0.0, 1.0]. Returns 0.0 for cold-start (no profile).
        """
        if not preferences:
            return 0.0

        # Check if customer profile has any recorded preferences
        has_signals = (
            bool(preferences.preferred_categories)
            or bool(preferences.preferred_brands)
            or bool(preferences.preferred_colors)
            or (preferences.preferred_price_min is not None)
            or bool(preferences.preferred_product_ids)
        )
        if not has_signals:
            return 0.0

        score = 0.0
        prod_attrs = product.attributes if isinstance(product.attributes, dict) else {}

        # 1. Category alignment (up to 0.35 points)
        prod_cat = prod_attrs.get("category")
        if prod_cat and preferences.preferred_categories:
            prod_cat_clean = str(prod_cat).strip()
            if prod_cat_clean in preferences.preferred_categories:
                idx = preferences.preferred_categories.index(prod_cat_clean)
                score += max(0.15, 0.35 - (idx * 0.10))

        # 2. Brand alignment (up to 0.20 points)
        prod_brand = prod_attrs.get("brand")
        if prod_brand and preferences.preferred_brands:
            prod_brand_clean = str(prod_brand).strip()
            if prod_brand_clean in preferences.preferred_brands:
                score += 0.20

        # 3. Color alignment (up to 0.25 points)
        if preferences.preferred_colors:
            colors_to_check: Set[str] = set()
            if prod_attrs.get("color"):
                colors_to_check.add(str(prod_attrs.get("color")).strip().lower())
            if hasattr(product, "variants") and product.variants:
                for v in product.variants:
                    if v.color:
                        colors_to_check.add(str(v.color).strip().lower())

            pref_colors_set = set(preferences.preferred_colors)
            if colors_to_check.intersection(pref_colors_set):
                score += 0.25

        # 4. Price range proximity (up to 0.20 points)
        if preferences.preferred_price_min is not None and preferences.preferred_price_max is not None:
            p_price = product.price
            p_min = preferences.preferred_price_min
            p_max = preferences.preferred_price_max
            if p_min <= p_price <= p_max:
                score += 0.20
            elif (p_min * Decimal("0.80")) <= p_price <= (p_max * Decimal("1.20")):
                score += 0.10

        # 5. Prior purchase/interaction affinity (up to 0.10 points)
        if preferences.preferred_product_ids and str(product.id) in preferences.preferred_product_ids:
            score += 0.10

        return max(0.0, min(1.0, score))
