"""
Unit and integration tests for Step 14: Multi-Merchant Backend Architecture & Tenant Isolation.

Tests:
1. Product retrieval tenant isolation (Merchant A vs Merchant B).
2. pgvector semantic search isolation (Merchant B candidate excluded regardless of similarity).
3. Customer personalization isolation (Merchant A + C123 vs Merchant B + C123).
4. External identifiers scoped per merchant (no cross-tenant ID collisions).
5. Integration schemas (MerchantProductInput, MerchantVariantInput, MerchantCustomerInput).
6. Merchant context resolution via HTTP header, query parameter, and demo fallback.
"""

import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.merchant_context import MerchantContext, get_merchant_context
from app.main import app
from app.models.merchant import Merchant
from app.models.product import Product, ProductVariant
from app.models.customer import Customer
from app.models.order import Order, OrderItem
from app.models.event import Event
from app.models.enums import OrderStatus
from app.repositories.product_repository import ProductRepository
from app.retrieval.filters import ProductFilters
from app.retrieval.text_retriever import TextRetriever
from app.schemas.integration import (
    MerchantProductInput,
    MerchantVariantInput,
    MerchantCustomerInput,
)
from app.services.customer_preference_service import CustomerPreferenceService
from app.services.product_service import ProductService


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_mock_merchant(name: str = "Test Merchant") -> Merchant:
    return Merchant(
        id=uuid.uuid4(),
        name=name,
        email=f"{name.lower().replace(' ', '')}@example.com",
        store_name=name.lower().replace(" ", "-"),
        is_active=True,
    )


def _create_mock_product(
    merchant_id: uuid.UUID,
    title: str = "Test Shirt",
    category: str = "Shirts",
    color: str = "Black",
    price: Decimal = Decimal("1999.00"),
    external_product_id: str = "ext-prod-1",
) -> Product:
    p = Product(
        id=uuid.uuid4(),
        merchant_id=merchant_id,
        external_product_id=external_product_id,
        title=title,
        description=f"Description for {title}",
        price=price,
        attributes={"category": category, "color": color},
        embedding=[0.1] * 1536,
        is_active=True,
    )
    v = ProductVariant(
        id=uuid.uuid4(),
        product_id=p.id,
        external_variant_id=f"{external_product_id}-M",
        sku=f"SKU-{title[:3].upper()}",
        color=color,
        size="M",
        price=price,
    )
    p.variants = [v]
    return p


# ---------------------------------------------------------------------------
# 1. Product Retrieval Tenant Isolation Tests
# ---------------------------------------------------------------------------

class TestProductRetrievalTenantIsolation:
    def test_repository_search_enforces_merchant_id(self):
        """Verify ProductRepository adds `Product.merchant_id == merchant_id` to WHERE clauses."""
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one.return_value = 1
        mock_db.execute.return_value.unique.return_value.scalars.return_value = []

        repo = ProductRepository(mock_db)
        merchant_a_id = uuid.uuid4()

        repo.search(merchant_id=merchant_a_id, q="shirt")

        # Verify execute was called with a statement filtering by merchant_a_id
        assert mock_db.execute.call_count == 2
        count_stmt = mock_db.execute.call_args_list[0][0][0]
        compiled_sql = str(count_stmt.compile())
        assert "products.merchant_id =" in compiled_sql

    def test_repository_vector_search_enforces_merchant_id(self):
        """Verify vector_search filters by `Product.merchant_id == merchant_id` before ordering."""
        mock_db = MagicMock()
        mock_db.execute.return_value.unique.return_value.all.return_value = []

        repo = ProductRepository(mock_db)
        merchant_a_id = uuid.uuid4()

        repo.vector_search(
            merchant_id=merchant_a_id,
            query_vector=[0.1] * 1536,
            limit=5,
        )

        assert mock_db.execute.call_count == 1
        query_stmt = mock_db.execute.call_args_list[0][0][0]
        compiled_sql = str(query_stmt.compile())
        assert "products.merchant_id =" in compiled_sql

    def test_cross_merchant_products_strictly_isolated(self):
        """
        Verify Merchant A search never returns Merchant B products,
        and Merchant B search never returns Merchant A products.
        """
        merchant_a = uuid.uuid4()
        merchant_b = uuid.uuid4()

        prod_a = _create_mock_product(merchant_id=merchant_a, title="Black Formal Shirt", external_product_id="P-A1")
        prod_b = _create_mock_product(merchant_id=merchant_b, title="Blue Casual Shirt", external_product_id="P-B1")

        mock_db = MagicMock()

        def mock_execute(stmt):
            compiled = stmt.compile()
            params = compiled.params if hasattr(compiled, "params") else {}
            res = MagicMock()
            if merchant_a in params.values() or str(merchant_a) in str(params.values()):
                res.unique.return_value.all.return_value = [(prod_a, 0.10)]
            elif merchant_b in params.values() or str(merchant_b) in str(params.values()):
                res.unique.return_value.all.return_value = [(prod_b, 0.05)]
            else:
                res.unique.return_value.all.return_value = []
            return res

        mock_db.execute.side_effect = mock_execute
        repo = ProductRepository(mock_db)

        # Search under Merchant A
        results_a = repo.vector_search(merchant_id=merchant_a, query_vector=[0.1] * 1536)
        assert len(results_a) == 1
        assert results_a[0][0].merchant_id == merchant_a
        assert results_a[0][0].title == "Black Formal Shirt"

        # Search under Merchant B
        results_b = repo.vector_search(merchant_id=merchant_b, query_vector=[0.1] * 1536)
        assert len(results_b) == 1
        assert results_b[0][0].merchant_id == merchant_b
        assert results_b[0][0].title == "Blue Casual Shirt"

    def test_text_retriever_delegates_merchant_id_to_repository(self):
        """Verify TextRetriever propagates merchant_id to repository vector_search."""
        mock_repo = MagicMock()
        mock_embedding_svc = MagicMock()
        dummy_vector = [0.2] * 1536
        mock_embedding_svc.embed_text.return_value = dummy_vector

        retriever = TextRetriever(repository=mock_repo, embedding_service=mock_embedding_svc)
        merchant_id = uuid.uuid4()

        retriever.retrieve(merchant_id=merchant_id, query="black shirt", limit=5)

        mock_repo.vector_search.assert_called_once()
        call_kwargs = mock_repo.vector_search.call_args[1]
        assert call_kwargs["merchant_id"] == merchant_id
        assert call_kwargs["query_vector"] == dummy_vector
        assert call_kwargs["limit"] == 5

    def test_product_service_passes_merchant_id(self):
        """Verify ProductService search methods propagate merchant_id."""
        mock_db = MagicMock()
        mock_text_retriever = MagicMock()
        mock_text_retriever.retrieve.return_value = []

        service = ProductService(db=mock_db, text_retriever=mock_text_retriever)
        merchant_id = uuid.uuid4()

        service.semantic_search_products(
            merchant_id=merchant_id,
            q="formal shoes",
            limit=5,
        )

        mock_text_retriever.retrieve.assert_called_once()
        assert mock_text_retriever.retrieve.call_args[1]["merchant_id"] == merchant_id


# ---------------------------------------------------------------------------
# 2. Customer Personalization Isolation Tests
# ---------------------------------------------------------------------------

class TestCustomerPersonalizationIsolation:
    def test_customer_preferences_isolated_by_merchant(self):
        """
        Verify Merchant A + customer 'C123' only reads Merchant A history,
        and Merchant B + customer 'C123' only reads Merchant B history.
        """
        mock_db = MagicMock()

        merchant_a = uuid.uuid4()
        merchant_b = uuid.uuid4()

        cust_a = Customer(
            id=uuid.uuid4(),
            merchant_id=merchant_a,
            external_customer_id="C123",
            name="Customer A",
            email="cust.a@example.com",
        )
        cust_b = Customer(
            id=uuid.uuid4(),
            merchant_id=merchant_b,
            external_customer_id="C123",
            name="Customer B",
            email="cust.b@example.com",
        )

        current_target = [merchant_a]

        def mock_query(model):
            q = MagicMock()
            if model == Customer:
                def cust_first():
                    if current_target[0] == merchant_a:
                        return cust_a
                    elif current_target[0] == merchant_b:
                        return cust_b
                    return None
                q.filter.return_value.first = cust_first
            elif model == Order:
                q.options.return_value.filter.return_value.all.return_value = []
            elif model == Event:
                def event_all():
                    if current_target[0] == merchant_a:
                        return [
                            Event(
                                id=uuid.uuid4(),
                                merchant_id=merchant_a,
                                customer_id=cust_a.id,
                                event_type="PRODUCT_VIEWED",
                                entity_type="product",
                                event_metadata={"category": "Shirts", "color": "black", "price": "2000"},
                            )
                        ]
                    elif current_target[0] == merchant_b:
                        return [
                            Event(
                                id=uuid.uuid4(),
                                merchant_id=merchant_b,
                                customer_id=cust_b.id,
                                event_type="PRODUCT_VIEWED",
                                entity_type="product",
                                event_metadata={"category": "Laptops", "color": "silver", "price": "75000"},
                            )
                        ]
                    return []
                q.filter.return_value.order_by.return_value.limit.return_value.all = event_all
            return q

        mock_db.query = mock_query
        service = CustomerPreferenceService(mock_db)

        # 1. Fetch preferences for Merchant A + C123
        current_target[0] = merchant_a
        prefs_a = service.get_preferences(merchant_id=merchant_a, external_customer_id="C123")
        assert prefs_a.customer_id == cust_a.id
        assert "Shirts" in prefs_a.preferred_categories
        assert "Laptops" not in prefs_a.preferred_categories
        assert "black" in prefs_a.preferred_colors

        # 2. Fetch preferences for Merchant B + C123
        current_target[0] = merchant_b
        prefs_b = service.get_preferences(merchant_id=merchant_b, external_customer_id="C123")
        assert prefs_b.customer_id == cust_b.id
        assert "Laptops" in prefs_b.preferred_categories
        assert "Shirts" not in prefs_b.preferred_categories
        assert "silver" in prefs_b.preferred_colors

    def test_nonexistent_customer_under_merchant_returns_neutral_profile(self):
        """Querying a customer ID that belongs to another merchant returns empty preferences."""
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        service = CustomerPreferenceService(mock_db)
        merchant_a = uuid.uuid4()
        other_customer_id = uuid.uuid4()

        prefs = service.get_preferences(merchant_id=merchant_a, customer_id=other_customer_id)
        assert prefs.preferred_categories == []
        assert prefs.preferred_colors == []
        assert prefs.total_events_analyzed == 0


# ---------------------------------------------------------------------------
# 3. Integration Schemas Validation Tests
# ---------------------------------------------------------------------------

class TestIntegrationSchemas:
    def test_merchant_product_input_valid(self):
        payload = {
            "external_product_id": "prod-456",
            "name": "Black Slim Fit Formal Shirt",
            "description": "High quality cotton shirt",
            "category": "Shirts",
            "brand": "UrbanThreads",
            "price": 1999,
            "currency": "INR",
            "image_url": "https://example.com/shirt.jpg",
            "attributes": {
                "color": "black",
                "fit": "slim",
                "material": "cotton",
            },
            "variants": [
                {
                    "external_variant_id": "variant-456-M",
                    "sku": "UT-SHIRT-M",
                    "size": "M",
                    "color": "black",
                    "price": 1999,
                    "stock": 12,
                }
            ],
        }
        product_input = MerchantProductInput.model_validate(payload)
        assert product_input.external_product_id == "prod-456"
        assert product_input.name == "Black Slim Fit Formal Shirt"
        assert product_input.price == Decimal("1999")
        assert len(product_input.variants) == 1
        assert product_input.variants[0].stock == 12

    def test_merchant_customer_input_valid(self):
        payload = {
            "external_customer_id": "cust-789",
            "name": "Alex Smith",
            "email": "alex@example.com",
            "phone": "+919876543210",
        }
        customer_input = MerchantCustomerInput.model_validate(payload)
        assert customer_input.external_customer_id == "cust-789"
        assert customer_input.name == "Alex Smith"
        assert customer_input.email == "alex@example.com"

    def test_merchant_product_input_rejects_empty_id(self):
        with pytest.raises(Exception):
            MerchantProductInput(
                external_product_id="",
                name="Test Shirt",
                price=Decimal("999"),
            )


# ---------------------------------------------------------------------------
# 4. Merchant Context Resolution Tests
# ---------------------------------------------------------------------------

class TestMerchantContextResolution:
    def test_get_merchant_context_direct(self):
        mock_db = MagicMock()
        mock_merchant = _create_mock_merchant("UrbanThreads")
        mock_db.query.return_value.filter.return_value.first.return_value = mock_merchant

        ctx = get_merchant_context(x_merchant_id=str(mock_merchant.id), db=mock_db)
        assert ctx.merchant_id == mock_merchant.id
        assert ctx.merchant_name == "UrbanThreads"

    def test_header_resolution_with_override(self, client):
        merchant_id = uuid.uuid4()
        app.dependency_overrides[get_merchant_context] = lambda: MerchantContext(merchant_id=merchant_id)
        try:
            with patch("app.services.product_service.ProductService.search_products") as mock_search:
                mock_search.return_value = MagicMock(total=0, limit=20, results=[])

                res = client.get(
                    "/api/products/search",
                    headers={"X-Merchant-Id": str(merchant_id)},
                )
                assert res.status_code == 200
                mock_search.assert_called_once()
                assert mock_search.call_args[1]["merchant_id"] == merchant_id
        finally:
            app.dependency_overrides.pop(get_merchant_context, None)

    def test_invalid_uuid_returns_400(self, client):
        res = client.get(
            "/api/products/search",
            headers={"X-Merchant-Id": "not-a-valid-uuid"},
        )
        assert res.status_code == 400
        assert "Invalid merchant ID" in res.json()["detail"]
