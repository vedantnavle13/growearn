"""
Retrieval Layer for MerchantAI.

Provides modular retrievers for text search, future image search, and hybrid ranking,
alongside deterministic business filters.
"""

from app.retrieval.base import Retriever
from app.retrieval.filters import ProductFilters
from app.retrieval.text_retriever import TextRetriever
from app.retrieval.image_retriever import ImageRetriever
from app.retrieval.hybrid_retriever import HybridRetriever

__all__ = [
    "Retriever",
    "ProductFilters",
    "TextRetriever",
    "ImageRetriever",
    "HybridRetriever",
]
