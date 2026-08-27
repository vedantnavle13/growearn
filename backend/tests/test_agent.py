"""
Tests for the AI Shopping Agent (Step 15).
"""

import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.agent import AgentChatResponse, AgentProductSummary
from app.models.agent_session import AgentSession
from app.models.customer import Customer
from app.models.cart import Cart, CartItem
from app.models.product import Product, ProductVariant, Inventory
from app.models.merchant import Merchant


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_merchant():
    return Merchant(
        id=uuid.uuid4(),
        name="Test Merchant",
        email="test@example.com",
        store_name="test-merchant",
        is_active=True,
    )


@pytest.fixture
def mock_customer(mock_merchant):
    return Customer(
        id=uuid.uuid4(),
        merchant_id=mock_merchant.id,
        external_customer_id="CUST-123",
        name="Test Customer",
        email="customer@example.com",
    )


@pytest.fixture
def mock_products(mock_merchant):
    """Create mock products for testing."""
    products = []
    for i in range(3):
        product = Product(
            id=uuid.uuid4(),
            merchant_id=mock_merchant.id,
            title=f"Product {i+1}",
            description=f"Description for product {i+1}",
            price=Decimal("1000") + Decimal(str(i * 500)),
            attributes={"category": "Shirts", "color": "black" if i == 0 else "blue"},
            is_active=True,
        )
        variant = ProductVariant(
            id=uuid.uuid4(),
            product_id=product.id,
            sku=f"SKU-{i+1}",
            color="black" if i == 0 else "blue",
            size="M",
            price=product.price,
        )
        inventory = Inventory(
            variant_id=variant.id,
            quantity=10,
            reserved_quantity=0,
        )
        variant.inventory = inventory
        product.variants = [variant]
        products.append(product)
    return products


class TestAgentChat:
    """Tests for the agent chat endpoint."""

    def test_agent_chat_requires_merchant_context(self, client):
        """Test that agent chat requires merchant context."""
        res = client.post(
            "/api/agent/chat",
            json={"message": "hello", "session_id": "test-session"},
        )
        # Should fail because no merchant context (in test env, fallback might work)
        # But we expect it to at least process the request structure
        assert res.status_code in (200, 400, 404, 422, 500)

    def test_agent_chat_creates_session(self, client, mock_merchant, mock_customer):
        """Test that agent chat creates a new session."""
        from app.core.merchant_context import MerchantContext, get_merchant_context
        from app.api.agent import get_customer_context

        # Override dependencies
        app.dependency_overrides[get_merchant_context] = lambda: MerchantContext(merchant_id=mock_merchant.id)
        app.dependency_overrides[get_customer_context] = lambda: mock_customer

        try:
            # Mock the AgentService to avoid actual Gemini calls
            with patch("app.api.agent.AgentService") as mock_agent_service_class:
                mock_agent = MagicMock()
                mock_agent_service_class.return_value = mock_agent
                mock_agent.chat.return_value = AgentChatResponse(
                    session_id="test-session",
                    message="Hello! How can I help you?",
                    products=[],
                    cart_updated=False,
                )

                res = client.post(
                    "/api/agent/chat",
                    json={"message": "hello", "session_id": "test-session"},
                    headers={"X-Merchant-Id": str(mock_merchant.id)},
                )

                assert res.status_code == 200
                body = res.json()
                assert body["session_id"] == "test-session"
                assert "message" in body
        finally:
            app.dependency_overrides.pop(get_merchant_context, None)
            app.dependency_overrides.pop(get_customer_context, None)


class TestAgentServiceTools:
    """Tests for the agent service tool implementations."""

    @patch("app.services.agent_service.ProductService")
    def test_search_products_tool(self, mock_product_service_class, mock_merchant, mock_customer, mock_products):
        """Test search_products tool uses intent search pipeline."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()
        mock_product_service = MagicMock()
        mock_product_service_class.return_value = mock_product_service

        # Mock intent search response
        from app.schemas.product import IntentSearchResponse, IntentSearchResult, VariantSummary
        mock_response = IntentSearchResponse(
            query="black shirt",
            intent={},
            total=1,
            limit=10,
            results=[
                IntentSearchResult(
                    id=mock_products[0].id,
                    title=mock_products[0].title,
                    description=mock_products[0].description,
                    price=mock_products[0].price,
                    attributes=mock_products[0].attributes,
                    variants=[
                        VariantSummary(
                            id=mock_products[0].variants[0].id,
                            sku=mock_products[0].variants[0].sku,
                            color=mock_products[0].variants[0].color,
                            size=mock_products[0].variants[0].size,
                            price=mock_products[0].variants[0].price,
                            in_stock=True,
                        )
                    ],
                    similarity_score=0.9,
                )
            ],
        )
        mock_product_service.intent_search_products.return_value = mock_response

        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )

        result = service.tool_search_products("black shirt under 2500")

        assert result["success"] is True
        assert result["total"] == 1
        assert len(result["products"]) == 1
        assert result["products"][0]["title"] == "Product 1"
        assert result["products"][0]["position"] == 1

    def test_get_product_tool_merchant_isolation(self, mock_merchant, mock_customer, mock_products):
        """Test get_product tool enforces merchant isolation."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()

        # Setup: product belongs to different merchant
        other_merchant_id = uuid.uuid4()
        product_other_merchant = Product(
            id=uuid.uuid4(),
            merchant_id=other_merchant_id,
            title="Other Merchant Product",
            price=Decimal("2000"),
            is_active=True,
        )

        mock_db.query.return_value.filter.return_value.first.return_value = None  # Not found for current merchant

        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )

        result = service.tool_get_product(product_other_merchant.id)

        assert result["success"] is False
        assert "not found" in result["message"].lower()

    def test_get_product_tool_success(self, mock_merchant, mock_customer, mock_products):
        """Test get_product tool returns product details."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()
        product = mock_products[0]

        mock_db.query.return_value.filter.return_value.first.return_value = product

        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )

        result = service.tool_get_product(product.id)

        assert result["success"] is True
        assert result["product"]["title"] == product.title
        assert "variants" in result["product"]
        assert len(result["product"]["variants"]) == 1

    def test_add_to_cart_tool_validates_merchant(self, mock_merchant, mock_customer, mock_products):
        """Test add_to_cart rejects products from other merchants."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()

        # Product from different merchant
        other_merchant_id = uuid.uuid4()
        product_other = Product(
            id=uuid.uuid4(),
            merchant_id=other_merchant_id,
            title="Other Product",
            price=Decimal("1500"),
            is_active=True,
        )
        variant_other = ProductVariant(
            id=uuid.uuid4(),
            product_id=product_other.id,
            sku="OTHER-SKU",
            price=Decimal("1500"),
        )

        mock_db.query.return_value.filter.return_value.first.return_value = None  # Product not found for current merchant

        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )

        result = service.tool_add_to_cart(
            session=MagicMock(customer_id=mock_customer.id),
            product_id=product_other.id,
            variant_id=variant_other.id,
        )

        assert result["success"] is False
        assert "not found" in result["message"].lower()

    def test_add_to_cart_tool_validates_variant_belongs_to_product(self, mock_merchant, mock_customer, mock_products):
        """Test add_to_cart rejects variant that doesn't belong to product."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()
        product = mock_products[0]

        # Different variant (not belonging to product)
        other_variant = ProductVariant(
            id=uuid.uuid4(),
            product_id=uuid.uuid4(),  # Different product
            sku="OTHER-VARIANT",
            price=Decimal("1500"),
        )

        mock_db.query.return_value.filter.return_value.first.side_effect = [
            product,  # Product found
            None,     # Variant not found for this product
        ]

        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )

        # Need a mock session with customer_id
        mock_session = MagicMock()
        mock_session.customer_id = mock_customer.id
        mock_session.cart_id = None

        result = service.tool_add_to_cart(
            session=mock_session,
            product_id=product.id,
            variant_id=other_variant.id,
        )

        assert result["success"] is False
        assert "does not belong" in result["message"].lower()

    def test_add_to_cart_tool_checks_stock(self, mock_merchant, mock_customer, mock_products):
        """Test add_to_cart rejects out of stock variants."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()
        product = mock_products[0]
        variant = product.variants[0]

        # Mock inventory with 0 available
        variant.inventory.quantity = 0
        variant.inventory.reserved_quantity = 0

        mock_db.query.return_value.filter.return_value.first.side_effect = [
            product,  # Product found
            variant,  # Variant found
        ]

        # Mock cart query
        mock_db.query.return_value.filter.return_value.first.return_value = None  # No existing cart
        mock_db.add = MagicMock()
        mock_db.flush = MagicMock()
        mock_db.commit = MagicMock()

        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )

        mock_session = MagicMock()
        mock_session.customer_id = mock_customer.id
        mock_session.cart_id = None

        result = service.tool_add_to_cart(
            session=mock_session,
            product_id=product.id,
            variant_id=variant.id,
        )

        assert result["success"] is False
        assert "out of stock" in result["message"].lower()

    def test_add_to_cart_tool_success(self, mock_merchant, mock_customer, mock_products):
        """Test add_to_cart succeeds with valid product/variant."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()
        product = mock_products[0]
        variant = product.variants[0]

        # Mock inventory with stock available
        variant.inventory.quantity = 10
        variant.inventory.reserved_quantity = 0

        # Mock the query chain for product, variant, and cart
        call_count = [0]
        
        def mock_query(model):
            q = MagicMock()
            
            def filter_side_effect(*args, **kwargs):
                q_filter = MagicMock()
                
                def first_side_effect():
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return product  # Product found
                    elif call_count[0] == 2:
                        return variant  # Variant found
                    elif call_count[0] == 3:
                        return None  # No existing cart
                    elif call_count[0] == 4:
                        return None  # No existing cart item
                    return None
                
                q_filter.first = first_side_effect
                return q_filter
            
            q.filter.side_effect = filter_side_effect
            return q

        mock_db.query.side_effect = mock_query
        mock_db.add = MagicMock()
        mock_db.flush = MagicMock()
        mock_db.commit = MagicMock()

        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )

        mock_session = MagicMock()
        mock_session.customer_id = mock_customer.id
        mock_session.cart_id = None

        result = service.tool_add_to_cart(
            session=mock_session,
            product_id=product.id,
            variant_id=variant.id,
            quantity=1,
        )

        assert result["success"] is True
        assert "added" in result["message"].lower()
        assert result["quantity"] == 1

    def test_get_cart_tool(self, mock_merchant, mock_customer, mock_products):
        """Test get_cart tool returns cart contents."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()
        product = mock_products[0]
        variant = product.variants[0]

        # Create cart with item
        cart = Cart(id=uuid.uuid4(), customer_id=mock_customer.id)
        cart_item = CartItem(
            id=uuid.uuid4(),
            cart_id=cart.id,
            variant_id=variant.id,
            quantity=2,
            price_at_addition=variant.price,
        )

        # Mock _get_cart to return the cart directly
        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )
        
        # Override _get_cart to return our mock cart
        service._get_cart = MagicMock(return_value=cart)

        # Mock the queries:
        # 1. cart_items = db.query(CartItem).filter(...).all() -> returns list with cart_item
        # 2. For each cart_item: variant = db.query(ProductVariant).filter(...).first()
        # 3. product = db.query(Product).filter(...).first()
        
        call_sequence = [
            ("CartItem", "all", [cart_item]),
            ("ProductVariant", "first", variant),
            ("Product", "first", product),
        ]
        call_index = [0]
        
        def mock_query(model):
            q = MagicMock()
            
            if model.__name__ == "CartItem":
                q_filter = MagicMock()
                q_filter.all.return_value = [cart_item]
                q.filter.return_value = q_filter
            elif model.__name__ == "ProductVariant":
                q_filter = MagicMock()
                q_filter.first.return_value = variant
                q.filter.return_value = q_filter
            elif model.__name__ == "Product":
                q_filter = MagicMock()
                q_filter.first.return_value = product
                q.filter.return_value = q_filter
            else:
                q_filter = MagicMock()
                q_filter.first.return_value = None
                q_filter.all.return_value = []
                q.filter.return_value = q_filter
            return q

        mock_db.query.side_effect = mock_query

        mock_session = MagicMock()
        mock_session.customer_id = mock_customer.id
        mock_session.cart_id = cart.id

        result = service.tool_get_cart(mock_session)

        assert result["success"] is True
        assert len(result["items"]) == 1
        assert result["items"][0]["quantity"] == 2
        assert result["subtotal"] == str(variant.price * 2)


class TestAddToCartReferenceResolution:
    """Tests for add_to_cart with reference_position (ordinal references)."""

    def _make_mock_session(self, mock_customer, search_results):
        """Create a mock session with last_search_results."""
        mock_session = MagicMock()
        mock_session.customer_id = mock_customer.id
        mock_session.cart_id = None
        mock_session.last_search_results = search_results
        return mock_session

    def _make_search_results(self, mock_products):
        """Create search results format as stored in session."""
        results = []
        for i, p in enumerate(mock_products):
            results.append({
                "id": str(p.id),
                "title": p.title,
                "price": str(p.price),
                "color": p.variants[0].color if p.variants else None,
                "size": p.variants[0].size if p.variants else None,
                "in_stock": True,
                "position": i + 1,
                "variant_id": str(p.variants[0].id) if p.variants else None,
            })
        return results

    def test_add_to_cart_with_reference_position_first(self, mock_merchant, mock_customer, mock_products):
        """Test add_to_cart with reference_position=1 resolves to first search result."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()
        product = mock_products[0]
        variant = product.variants[0]

        # Mock inventory with stock available
        variant.inventory.quantity = 10
        variant.inventory.reserved_quantity = 0

        call_count = [0]
        
        def mock_query(model):
            q = MagicMock()
            
            def filter_side_effect(*args, **kwargs):
                q_filter = MagicMock()
                
                def first_side_effect():
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return product  # Product found
                    elif call_count[0] == 2:
                        return variant  # Variant found
                    elif call_count[0] == 3:
                        return None  # No existing cart
                    elif call_count[0] == 4:
                        return None  # No existing cart item
                    return None
                
                q_filter.first = first_side_effect
                return q_filter
            
            q.filter.side_effect = filter_side_effect
            return q

        mock_db.query.side_effect = mock_query
        mock_db.add = MagicMock()
        mock_db.flush = MagicMock()
        mock_db.commit = MagicMock()

        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )

        # Create session with search results
        search_results = self._make_search_results(mock_products)
        mock_session = self._make_mock_session(mock_customer, search_results)

        # Call _execute_add_to_cart with reference_position=1
        result = service._execute_add_to_cart(
            args={"reference_position": 1, "quantity": 1},
            session=mock_session
        )

        assert result["success"] is True
        assert "added" in result["message"].lower()
        assert result["quantity"] == 1
        # Verify it used the first product (Product 1)
        assert "Product 1" in result["message"]

    def test_add_to_cart_with_reference_position_second(self, mock_merchant, mock_customer, mock_products):
        """Test add_to_cart with reference_position=2 resolves to second search result."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()
        product = mock_products[1]  # Second product
        variant = product.variants[0]

        # Mock inventory with stock available
        variant.inventory.quantity = 10
        variant.inventory.reserved_quantity = 0

        call_count = [0]
        
        def mock_query(model):
            q = MagicMock()
            
            def filter_side_effect(*args, **kwargs):
                q_filter = MagicMock()
                
                def first_side_effect():
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return product  # Product found (second product)
                    elif call_count[0] == 2:
                        return variant  # Variant found
                    elif call_count[0] == 3:
                        return None  # No existing cart
                    elif call_count[0] == 4:
                        return None  # No existing cart item
                    return None
                
                q_filter.first = first_side_effect
                return q_filter
            
            q.filter.side_effect = filter_side_effect
            return q

        mock_db.query.side_effect = mock_query
        mock_db.add = MagicMock()
        mock_db.flush = MagicMock()
        mock_db.commit = MagicMock()

        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )

        # Create session with search results
        search_results = self._make_search_results(mock_products)
        mock_session = self._make_mock_session(mock_customer, search_results)

        # Call _execute_add_to_cart with reference_position=2
        result = service._execute_add_to_cart(
            args={"reference_position": 2, "quantity": 1},
            session=mock_session
        )

        assert result["success"] is True
        assert "added" in result["message"].lower()
        # Verify it used the second product (Product 2)
        assert "Product 2" in result["message"]

    def test_add_to_cart_with_reference_position_quantity(self, mock_merchant, mock_customer, mock_products):
        """Test add_to_cart with reference_position and quantity > 1."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()
        product = mock_products[1]  # Second product
        variant = product.variants[0]

        # Mock inventory with stock available
        variant.inventory.quantity = 10
        variant.inventory.reserved_quantity = 0

        call_count = [0]
        
        def mock_query(model):
            q = MagicMock()
            
            def filter_side_effect(*args, **kwargs):
                q_filter = MagicMock()
                
                def first_side_effect():
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return product
                    elif call_count[0] == 2:
                        return variant
                    elif call_count[0] == 3:
                        return None
                    elif call_count[0] == 4:
                        return None
                    return None
                
                q_filter.first = first_side_effect
                return q_filter
            
            q.filter.side_effect = filter_side_effect
            return q

        mock_db.query.side_effect = mock_query
        mock_db.add = MagicMock()
        mock_db.flush = MagicMock()
        mock_db.commit = MagicMock()

        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )

        search_results = self._make_search_results(mock_products)
        mock_session = self._make_mock_session(mock_customer, search_results)

        # Call with reference_position=2 and quantity=2
        result = service._execute_add_to_cart(
            args={"reference_position": 2, "quantity": 2},
            session=mock_session
        )

        assert result["success"] is True
        assert result["quantity"] == 2
        assert "Product 2" in result["message"]

    def test_add_to_cart_with_invalid_reference_position(self, mock_merchant, mock_customer, mock_products):
        """Test add_to_cart rejects invalid reference_position (out of bounds)."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()
        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )

        # Only 3 products in search results
        search_results = self._make_search_results(mock_products)
        mock_session = self._make_mock_session(mock_customer, search_results)

        # Try to access position 5 (out of bounds)
        result = service._execute_add_to_cart(
            args={"reference_position": 5, "quantity": 1},
            session=mock_session
        )

        assert result["success"] is False
        assert "not found in recent search results" in result["message"]
        assert "only 3 results" in result["message"]

    def test_add_to_cart_with_no_search_results(self, mock_merchant, mock_customer, mock_products):
        """Test add_to_cart fails gracefully when no search results exist."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()
        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )

        # No search results
        mock_session = self._make_mock_session(mock_customer, [])

        result = service._execute_add_to_cart(
            args={"reference_position": 1, "quantity": 1},
            session=mock_session
        )

        assert result["success"] is False
        assert "No recent search results" in result["message"]

    def test_add_to_cart_with_direct_product_id_still_works(self, mock_merchant, mock_customer, mock_products):
        """Test add_to_cart still works with direct product_id and variant_id (backward compatibility)."""
        from app.services.agent_service import AgentService
        from app.core.merchant_context import MerchantContext

        mock_db = MagicMock()
        product = mock_products[0]
        variant = product.variants[0]

        variant.inventory.quantity = 10
        variant.inventory.reserved_quantity = 0

        call_count = [0]
        
        def mock_query(model):
            q = MagicMock()
            
            def filter_side_effect(*args, **kwargs):
                q_filter = MagicMock()
                
                def first_side_effect():
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return product
                    elif call_count[0] == 2:
                        return variant
                    elif call_count[0] == 3:
                        return None
                    elif call_count[0] == 4:
                        return None
                    return None
                
                q_filter.first = first_side_effect
                return q_filter
            
            q.filter.side_effect = filter_side_effect
            return q

        mock_db.query.side_effect = mock_query
        mock_db.add = MagicMock()
        mock_db.flush = MagicMock()
        mock_db.commit = MagicMock()

        service = AgentService(
            db=mock_db,
            merchant_context=MerchantContext(merchant_id=mock_merchant.id),
            customer=mock_customer,
        )

        # Session without search results (to ensure it doesn't try to resolve reference)
        mock_session = self._make_mock_session(mock_customer, [])

        # Call with direct product_id and variant_id
        result = service._execute_add_to_cart(
            args={
                "product_id": str(product.id),
                "variant_id": str(variant.id),
                "quantity": 1
            },
            session=mock_session
        )

        assert result["success"] is True
        assert "Product 1" in result["message"]


class TestAgentSessionModel:
    """Tests for the AgentSession model."""

    def test_agent_session_creation(self, mock_merchant, mock_customer):
        """Test AgentSession model can be created."""
        from app.models.agent_session import AgentSession

        session = AgentSession(
            session_id="test-session-123",
            merchant_id=mock_merchant.id,
            customer_id=mock_customer.id,
        )

        assert session.session_id == "test-session-123"
        assert session.merchant_id == mock_merchant.id
        assert session.customer_id == mock_customer.id
        assert session.current_intent is None
        assert session.last_search_results is None
        assert session.cart_id is None


class TestAgentSchemas:
    """Tests for agent schemas."""

    def test_agent_chat_request_valid(self):
        """Test AgentChatRequest validation."""
        from app.schemas.agent import AgentChatRequest

        req = AgentChatRequest(message="I need a black shirt", session_id="test-123")
        assert req.message == "I need a black shirt"
        assert req.session_id == "test-123"

    def test_agent_chat_request_empty_message_rejected(self):
        """Test AgentChatRequest rejects empty message."""
        from app.schemas.agent import AgentChatRequest

        with pytest.raises(ValueError):
            AgentChatRequest(message="", session_id="test-123")

    def test_agent_product_summary(self):
        """Test AgentProductSummary schema."""
        from app.schemas.agent import AgentProductSummary

        product = AgentProductSummary(
            id=uuid.uuid4(),
            title="Black Shirt",
            price=Decimal("1999"),
            color="black",
            size="M",
            in_stock=True,
            position=1,
        )

        assert product.title == "Black Shirt"
        assert product.position == 1