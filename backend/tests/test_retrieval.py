"""
Unit tests for the modular Retrieval Layer (TextRetriever, ImageRetriever, HybridRetriever, and ProductFilters).
"""

import uuid
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from app.models.product import Product
from app.retrieval.base import Retriever
from app.retrieval.filters import ProductFilters
from app.retrieval.text_retriever import TextRetriever
from app.retrieval.image_retriever import ImageRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.product_service import ProductService


def _create_mock_product(title: str = "Test Shirt", price: Decimal = Decimal("1999.00")) -> Product:
    p = Product(
        id=uuid.uuid4(),
        merchant_id=uuid.uuid4(),
        title=title,
        price=price,
        attributes={"category": "Shirts", "color": "Blue"},
        is_active=True,
    )
    p.variants = []
    return p


class TestRetrieverProtocol:
    def test_text_retriever_satisfies_protocol(self):
        mock_repo = MagicMock()
        retriever = TextRetriever(repository=mock_repo)
        assert isinstance(retriever, Retriever)


class TestProductFilters:
    def test_default_filter_clauses(self):
        filters = ProductFilters()
        clauses = filters.to_sqlalchemy_clauses()
        assert len(clauses) == 1  # Product.is_active is True

    def test_filter_clauses_with_all_options(self):
        filters = ProductFilters(
            category="Shirts",
            min_price=Decimal("1000.00"),
            max_price=Decimal("5000.00"),
            color="Navy",
            size="L",
            in_stock=True,
        )
        clauses = filters.to_sqlalchemy_clauses()
        # is_active, category, min_price, max_price, color, size subq, in_stock subq = 7 clauses
        assert len(clauses) == 7

    def test_filters_immutability(self):
        filters = ProductFilters(category="Shoes")
        with pytest.raises(AttributeError):
            filters.category = "Shirts"  # frozen dataclass


class TestTextRetriever:
    def test_text_retriever_delegates_embedding_and_search(self):
        mock_repo = MagicMock()
        mock_embedding_svc = MagicMock()

        dummy_vector = [0.1] * 1536
        mock_embedding_svc.embed_text.return_value = dummy_vector

        sample_product = _create_mock_product("Casual Denim Shirt")
        mock_repo.vector_search.return_value = [(sample_product, 0.9123)]

        retriever = TextRetriever(
            repository=mock_repo,
            embedding_service=mock_embedding_svc,
        )

        filters = ProductFilters(category="Shirts")
        results = retriever.retrieve(query="denim shirt", filters=filters, limit=5)

        mock_embedding_svc.embed_text.assert_called_once_with("denim shirt")
        mock_repo.vector_search.assert_called_once_with(
            query_vector=dummy_vector,
            filters=filters,
            limit=5,
        )

        assert len(results) == 1
        prod, score = results[0]
        assert prod.title == "Casual Denim Shirt"
        assert score == 0.9123


class TestStubs:
    def test_image_retriever_stub_raises_not_implemented(self):
        retriever = ImageRetriever()
        with pytest.raises(NotImplementedError) as exc_info:
            retriever.retrieve(query=b"fake_image_bytes")
        assert "multimodal step" in str(exc_info.value)

        with pytest.raises(NotImplementedError) as exc_info2:
            retriever.retrieve_by_url(image_url="https://example.com/img.jpg")
        assert "multimodal step" in str(exc_info2.value)

    def test_hybrid_retriever_stub_raises_not_implemented(self):
        mock_repo = MagicMock()
        text_retriever = TextRetriever(repository=mock_repo)
        hybrid = HybridRetriever(text_retriever=text_retriever)

        with pytest.raises(NotImplementedError) as exc_info:
            hybrid.retrieve(query="shoes", image_data=b"fake_bytes")
        assert "multi-modal" in str(exc_info.value)


class TestProductServiceOrchestration:
    def test_service_orchestrates_text_retriever(self):
        mock_db = MagicMock()
        mock_text_retriever = MagicMock()

        sample_product = _create_mock_product("Oxford Linen Shirt")
        mock_text_retriever.retrieve.return_value = [(sample_product, 0.8850)]

        service = ProductService(
            db=mock_db,
            text_retriever=mock_text_retriever,
        )

        response = service.semantic_search_products(
            q="linen shirt",
            category="Shirts",
            min_price=Decimal("1500"),
            limit=5,
        )

        assert response.query == "linen shirt"
        assert response.total == 1
        assert response.results[0].title == "Oxford Linen Shirt"
        assert response.results[0].similarity_score == 0.8850

        # Verify TextRetriever was called with proper ProductFilters
        mock_text_retriever.retrieve.assert_called_once()
        call_kwargs = mock_text_retriever.retrieve.call_args[1]
        assert call_kwargs["query"] == "linen shirt"
        assert call_kwargs["limit"] == 5
        assert isinstance(call_kwargs["filters"], ProductFilters)
        assert call_kwargs["filters"].category == "Shirts"
        assert call_kwargs["filters"].min_price == Decimal("1500")
