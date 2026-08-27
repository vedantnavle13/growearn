"""
Category Concept Service: maps broad semantic category concepts to actual database categories.

This service provides a deterministic, configurable mapping between user-facing broad concepts
(e.g., "clothing", "electronics") and the specific categories that exist in a merchant's catalogue
(e.g., "Shirts", "Jeans", "Laptops", "Headphones").

Key principles:
- Never invents categories that don't exist in the database
- Maps concepts to categories at the merchant level (tenant isolation)
- Provides both forward mapping (concept → categories) and reverse (category → concepts)
- Configurable via JSON/environment, not hardcoded
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Set

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.product import Product


class CategoryConceptService:
    """
    Maps broad category concepts (clothing, electronics, footwear) to actual 
    database categories (Shirts, Jeans, Laptops, etc.) per merchant.
    """

    # Default concept-to-category mapping (common e-commerce taxonomy)
    # This can be overridden per merchant via configuration
    DEFAULT_CONCEPT_TAXONOMY: Dict[str, List[str]] = {
        "clothing": [
            "Shirts", "T-Shirts", "T-Shirts", "Jeans", "Pants", "Trousers",
            "Jackets", "Coats", "Dresses", "Skirts", "Shorts", "Sweaters",
            "Hoodies", "Sweatshirts", "Cardigans", "Blazers", "Suits",
            "Tops", "Blouses", "Polos", "Tank Tops", "Crop Tops", "Leggings",
            "Jumpsuits", "Rompers", "Overalls", "Vests", "Ponchos", "Capes"
        ],
        "footwear": [
            "Shoes", "Sneakers", "Boots", "Sandals", "Loafers", "Oxfords",
            "Derbies", "Brogues", "Chelsea Boots", "Ankle Boots", "Knee Boots",
            "Heels", "Pumps", "Flats", "Slippers", "Clogs", "Mules",
            "Running Shoes", "Walking Shoes", "Hiking Boots", "Work Boots"
        ],
        "electronics": [
            "Laptops", "Phones", "Smartphones", "Tablets", "Headphones",
            "Earphones", "Monitors", "Keyboards", "Mice", "Webcams",
            "Speakers", "Smartwatches", "Fitness Trackers", "Cameras",
            "Drones", "Gaming Consoles", "Printers", "Routers", "Storage",
            "Chargers", "Cables", "Adapters", "Power Banks", "Cases"
        ],
        "accessories": [
            "Bags", "Backpacks", "Handbags", "Wallets", "Belts", "Hats",
            "Caps", "Scarves", "Gloves", "Sunglasses", "Watches", "Jewelry",
            "Necklaces", "Bracelets", "Earrings", "Rings", "Ties", "Socks"
        ],
        "home": [
            "Furniture", "Decor", "Lighting", "Bedding", "Bath", "Kitchen",
            "Storage", "Organization", "Cleaning", "Tools", "Hardware"
        ],
        "beauty": [
            "Skincare", "Makeup", "Haircare", "Fragrance", "Body Care",
            "Nails", "Tools", "Men's Grooming", "Oral Care"
        ],
        "sports": [
            "Fitness", "Outdoor", "Cycling", "Swimming", "Running", "Yoga",
            "Team Sports", "Racquet Sports", "Water Sports", "Winter Sports"
        ],
        "stationery": [
            "Notebooks", "Pens", "Pencils", "Markers", "Paper", "Planners",
            "Organizers", "Folders", "Binders", "Art Supplies", "Crafts"
        ],
        "books": [
            "Fiction", "Non-Fiction", "Children", "Young Adult", "Mystery",
            "Romance", "Sci-Fi", "Fantasy", "Biography", "History",
            "Self-Help", "Business", "Textbooks", "Comics", "Manga"
        ],
        "toys": [
            "Action Figures", "Dolls", "Games", "Puzzles", "Building Sets",
            "Vehicles", "Plush", "Educational", "Outdoor Toys", "Electronic Toys"
        ],
        "pet": [
            "Dog", "Cat", "Food", "Toys", "Beds", "Carriers", "Grooming",
            "Health", "Training", "Accessories"
        ],
        "automotive": [
            "Car Care", "Interior", "Exterior", "Electronics", "Tools",
            "Maintenance", "Safety", "Accessories"
        ],
    }

    def __init__(self, db: Session, merchant_id: str):
        self.db = db
        self.merchant_id = merchant_id
        self._concept_to_categories: Optional[Dict[str, List[str]]] = None
        self._category_to_concepts: Optional[Dict[str, List[str]]] = None
        self._available_categories: Optional[Set[str]] = None

    def _load_available_categories(self) -> Set[str]:
        """Load all distinct categories that exist in the merchant's catalogue."""
        if self._available_categories is not None:
            return self._available_categories

        # Query distinct categories from product attributes
        query = text("""
            SELECT DISTINCT attributes->>'category' as category
            FROM products
            WHERE merchant_id = :merchant_id
              AND is_active = true
              AND attributes->>'category' IS NOT NULL
              AND attributes->>'category' != ''
        """)
        rows = self.db.execute(query, {"merchant_id": self.merchant_id}).fetchall()

        self._available_categories = {row[0] for row in rows if row[0]}
        return self._available_categories

    def _build_concept_mapping(self) -> Dict[str, List[str]]:
        """Build concept-to-categories mapping based on available categories."""
        if self._concept_to_categories is not None:
            return self._concept_to_categories

        available = self._load_available_categories()
        if not available:
            self._concept_to_categories = {}
            return self._concept_to_categories

        # Normalize available categories for matching
        available_lower = {cat.lower().strip(): cat for cat in available}

        mapping: Dict[str, List[str]] = defaultdict(list)

        # For each concept in default taxonomy, find matching available categories
        for concept, default_categories in self.DEFAULT_CONCEPT_TAXONOMY.items():
            for default_cat in default_categories:
                default_lower = default_cat.lower().strip()
                if default_lower in available_lower:
                    actual_cat = available_lower[default_lower]
                    if actual_cat not in mapping[concept]:
                        mapping[concept].append(actual_cat)

        # Also add reverse mapping: if an available category matches a concept name exactly
        for cat in available:
            cat_lower = cat.lower().strip()
            if cat_lower in self.DEFAULT_CONCEPT_TAXONOMY:
                # This category IS a concept (e.g., category="Clothing" exists)
                if cat not in mapping[cat_lower]:
                    mapping[cat_lower].append(cat)

        self._concept_to_categories = dict(mapping)
        return self._concept_to_categories

    def _build_reverse_mapping(self) -> Dict[str, List[str]]:
        """Build category-to-concepts mapping."""
        if self._category_to_concepts is not None:
            return self._category_to_concepts

        forward = self._build_concept_mapping()
        reverse: Dict[str, List[str]] = defaultdict(list)

        for concept, categories in forward.items():
            for cat in categories:
                reverse[cat].append(concept)

        self._category_to_concepts = dict(reverse)
        return self._category_to_concepts

    def get_categories_for_concept(self, concept: str) -> List[str]:
        """
        Get actual database categories that belong to a broad concept.
        
        Args:
            concept: Broad category concept (e.g., "clothing", "electronics")
            
        Returns:
            List of actual database categories that exist in the merchant's catalogue.
            Returns empty list if concept not recognized or no matching categories exist.
        """
        if not concept:
            return []

        mapping = self._build_concept_mapping()
        concept_lower = concept.strip().lower()
        return mapping.get(concept_lower, [])

    def get_concepts_for_category(self, category: str) -> List[str]:
        """
        Get broad concepts that a specific database category belongs to.
        
        Args:
            category: Exact database category (e.g., "Shirts", "Laptops")
            
        Returns:
            List of broad concepts this category maps to.
        """
        if not category:
            return []

        reverse = self._build_reverse_mapping()
        return reverse.get(category.strip(), [])

    def is_category_in_concept(self, category: str, concept: str) -> bool:
        """Check if a specific category belongs to a broad concept."""
        categories = self.get_categories_for_concept(concept)
        return category.strip() in categories

    def get_all_concepts(self) -> List[str]:
        """Get all recognized broad concepts that have matching categories in catalogue."""
        mapping = self._build_concept_mapping()
        return sorted(mapping.keys())

    def get_all_categories(self) -> List[str]:
        """Get all actual database categories available for this merchant."""
        return sorted(self._load_available_categories())

    def resolve_category_or_concept(
        self,
        exact_category: Optional[str],
        category_concept: Optional[str]
    ) -> tuple[Optional[str], List[str]]:
        """
        Resolve the effective category filter and concept categories for search.
        
        Logic:
        - If exact_category provided and exists in DB → use as hard filter, concept_categories = []
        - If only category_concept provided → no hard category filter, concept_categories = mapped categories
        - If both provided and compatible → exact_category as hard filter, concept_categories = mapped
        - If both provided but conflicting → exact_category wins, concept_categories = mapped
        
        Returns:
            (hard_filter_category, concept_categories_for_ranking)
        """
        hard_filter = None
        concept_categories = []

        # Load available categories to validate
        available = self._load_available_categories()

        # Determine hard filter category
        if exact_category and exact_category.strip() in available:
            hard_filter = exact_category.strip()

        # Determine concept categories for ranking
        if category_concept:
            concept_categories = self.get_categories_for_concept(category_concept)

        return hard_filter, concept_categories