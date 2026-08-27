"""
Comprehensive test suite for Purchase Intent, Ordinal Resolution, and Customer Address Architecture.

Validates:
1. Purchase Scope Extraction & Schema
2. PurchaseIntentResolver ordinal resolution (e.g. 1st product -> last_search_results[0])
3. Variant handling & size specification
4. Inventory checks
5. AddressService label, default, position lookups and customer isolation
6. Order shipping address snapshots
7. SingleProductCheckoutService & Cart checkout routing
8. Multi-step checkout state machine
"""

import uuid
from decimal import Decimal
import pytest
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.models.merchant import Merchant
from app.models.customer import Customer
from app.models.customer_address import CustomerAddress
from app.models.product import Product, ProductVariant, Inventory
from app.models.cart import Cart, CartItem
from app.models.order import Order, OrderItem, OrderStatus
from app.models.agent_session import AgentSession
from app.core.merchant_context import MerchantContext
from app.services.address_service import AddressService
from app.services.purchase_intent_resolver import (
    PurchaseIntentResolver,
    PurchaseIntentError,
    ProductNotFoundError,
    VariantNotFoundError,
    InsufficientInventoryError,
)
from app.services.checkout_service import (
    SingleProductCheckoutService,
    CheckoutService,
)
from app.services.agent_service import AgentService
from app.schemas.intent import PurchaseIntent, PurchaseScope, ProductReference


@pytest.fixture(scope="module")
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="module")
def setup_data(db: Session):
    # Fetch or create test merchant
    merchant = db.query(Merchant).first()
    if not merchant:
        merchant = Merchant(name="Test Merchant", store_name="test_store")
        db.add(merchant)
        db.flush()

    # Fetch or create test customer
    customer = db.query(Customer).filter(Customer.merchant_id == merchant.id).first()
    if not customer:
        customer = Customer(merchant_id=merchant.id, name="Test Customer", email="test@example.com")
        db.add(customer)
        db.flush()

    # Clean up any test addresses
    db.query(CustomerAddress).filter(CustomerAddress.customer_id == customer.id).delete()
    db.flush()

    # Create 2 addresses: 1 default Home, 1 Office
    addr_home = CustomerAddress(
        customer_id=customer.id,
        label="Home",
        recipient_name="Test Customer",
        address_line_1="123 Main St",
        address_line_2="Apt 4B",
        city="Mumbai",
        state="Maharashtra",
        postal_code="400001",
        country="India",
        is_default=True,
    )
    addr_office = CustomerAddress(
        customer_id=customer.id,
        label="Office",
        recipient_name="Test Work",
        address_line_1="456 Tech Park",
        city="Bengaluru",
        state="Karnataka",
        postal_code="560001",
        country="India",
        is_default=False,
    )
    db.add_all([addr_home, addr_office])
    db.commit()

    return {
        "merchant": merchant,
        "customer": customer,
        "addr_home": addr_home,
        "addr_office": addr_office,
    }


def test_schema_purchase_intent():
    """Test PurchaseIntent schema instantiation and validation."""
    intent1 = PurchaseIntent(
        purchase_scope=PurchaseScope.SINGLE_PRODUCT,
        product_reference=ProductReference.SEARCH_RESULT,
        reference_position=1,
        quantity=2,
        size="L",
        address_hint="default",
    )
    assert intent1.purchase_scope == PurchaseScope.SINGLE_PRODUCT
    assert intent1.reference_position == 1
    assert intent1.quantity == 2

    intent2 = PurchaseIntent(
        purchase_scope=PurchaseScope.CART,
        product_reference=ProductReference.NONE,
    )
    assert intent2.purchase_scope == PurchaseScope.CART


def test_address_service_lookups(db: Session, setup_data):
    """Test AddressService default, label, position lookups and customer isolation."""
    customer = setup_data["customer"]
    addr_service = AddressService(db)

    # 1. Default address
    default_addr = addr_service.get_default_address(customer.id)
    assert default_addr is not None
    assert default_addr.label == "Home"
    assert default_addr.is_default is True

    # 2. Label lookup (case-insensitive)
    office_addr = addr_service.get_address_by_label(customer.id, "office")
    assert office_addr is not None
    assert office_addr.city == "Bengaluru"

    # 3. Position lookup
    first_addr = addr_service.get_address_by_position(customer.id, 1)
    assert first_addr is not None
    assert first_addr.id == default_addr.id

    # 4. Resolve hint
    resolved_home = addr_service.resolve_address_hint(customer.id, "home")
    assert resolved_home.id == default_addr.id

    resolved_office = addr_service.resolve_address_hint(customer.id, "office")
    assert resolved_office.id == office_addr.id

    resolved_work = addr_service.resolve_address_hint(customer.id, "work")
    assert resolved_work.id == office_addr.id

    resolved_none = addr_service.resolve_address_hint(customer.id, None)
    assert resolved_none.id == default_addr.id

    # 5. Customer isolation: random customer should get None
    random_customer_id = uuid.uuid4()
    assert addr_service.get_default_address(random_customer_id) is None
    assert addr_service.list_addresses(random_customer_id) == []


def test_purchase_intent_resolver_ordinal(db: Session, setup_data):
    """Test PurchaseIntentResolver resolving 1st product from last_search_results."""
    merchant = setup_data["merchant"]
    resolver = PurchaseIntentResolver(db=db, merchant_id=merchant.id)

    # Find a product with variants for testing
    product = db.query(Product).filter(
        Product.merchant_id == merchant.id,
        Product.is_active.is_(True)
    ).first()
    assert product is not None

    variant = db.query(ProductVariant).filter(ProductVariant.product_id == product.id).first()
    assert variant is not None

    # Simulate session with last_search_results
    session = AgentSession(
        merchant_id=merchant.id,
        session_id=str(uuid.uuid4()),
        last_search_results=[
            {
                "position": 1,
                "id": str(product.id),
                "variant_id": str(variant.id),
                "title": product.title,
                "price": str(variant.price),
                "size": variant.size,
                "color": variant.color,
                "in_stock": True,
            }
        ]
    )

    # Resolve position 1
    target = resolver.resolve_single_product(
        session=session,
        reference_position=1,
        quantity=1,
    )
    assert target.product.id == product.id
    assert target.variant.id == variant.id
    assert target.quantity == 1
    assert target.unit_price == variant.price
    assert target.needs_variant_selection is False


def test_single_product_checkout_service_order_creation_and_snapshot(db: Session, setup_data):
    """Test SingleProductCheckoutService creates Order with address snapshot."""
    merchant = setup_data["merchant"]
    customer = setup_data["customer"]
    addr_home = setup_data["addr_home"]

    product = db.query(Product).filter(
        Product.merchant_id == merchant.id,
        Product.is_active.is_(True)
    ).first()
    variant = db.query(ProductVariant).filter(ProductVariant.product_id == product.id).first()

    # Ensure inventory exists
    inv = db.query(Inventory).filter(Inventory.variant_id == variant.id).first()
    if not inv:
        inv = Inventory(variant_id=variant.id, quantity=100, reserved_quantity=0)
        db.add(inv)
        db.commit()
    else:
        inv.quantity = max(inv.quantity, 50)
        db.commit()

    ctx = MerchantContext(merchant_id=merchant.id, merchant_name=merchant.name)
    svc = SingleProductCheckoutService(db=db, merchant_context=ctx, customer=customer)

    # Build summary
    summary = svc.build_summary(product=product, variant=variant, quantity=2, address=addr_home)
    assert summary["product_title"] == product.title
    assert summary["quantity"] == 2
    assert Decimal(summary["total"]) == variant.price * 2
    assert "Mumbai" in summary["delivery_address"]

    # Execute checkout
    result = svc.execute_checkout(product=product, variant=variant, quantity=2, address=addr_home)
    assert result.order_id is not None
    assert result.total_amount == variant.price * 2
    assert result.status == "pending_payment"

    # Verify Order in DB and shipping address snapshot
    order = db.query(Order).filter(Order.id == result.order_id).first()
    assert order is not None
    assert order.customer_address_id == addr_home.id
    assert order.shipping_recipient_name == addr_home.recipient_name
    assert order.shipping_address_line_1 == addr_home.address_line_1
    assert order.shipping_city == "Mumbai"
    assert order.shipping_state == "Maharashtra"
    assert order.shipping_postal_code == "400001"
    assert order.shipping_country == "India"
    assert order.cart_id is None  # Single-product checkout has no cart
    assert len(order.items) == 1
    assert order.items[0].variant_id == variant.id
    assert order.items[0].quantity == 2
    assert order.items[0].price == variant.price


def test_agent_service_single_product_flow(db: Session, setup_data):
    """Test AgentService._execute_checkout_single_product and _execute_confirm_checkout."""
    merchant = setup_data["merchant"]
    customer = setup_data["customer"]
    ctx = MerchantContext(merchant_id=merchant.id, merchant_name=merchant.name)

    agent_svc = AgentService(db=db, merchant_context=ctx, customer=customer)

    product = db.query(Product).filter(
        Product.merchant_id == merchant.id,
        Product.is_active.is_(True)
    ).first()
    variant = db.query(ProductVariant).filter(ProductVariant.product_id == product.id).first()

    session_id = f"test_session_{uuid.uuid4()}"
    session = agent_svc.get_or_create_session(session_id)
    session.last_search_results = [
        {
            "position": 1,
            "id": str(product.id),
            "variant_id": str(variant.id),
            "title": product.title,
            "price": str(variant.price),
            "size": variant.size,
            "color": variant.color,
            "in_stock": True,
        }
    ]
    db.flush()

    # Step 1: Call _execute_checkout_single_product for position 1
    res1 = agent_svc._execute_checkout_single_product(
        args={"reference_position": 1, "quantity": 1, "address_hint": "home"},
        session=session,
    )
    assert res1["success"] is True
    assert res1["awaiting_confirmation"] is True
    assert "checkout_summary" in res1
    assert session.checkout_state["step"] == "awaiting_confirmation"
    assert session.checkout_state["mode"] == "SINGLE_PRODUCT"

    # Step 2: Confirm checkout
    res2 = agent_svc._execute_confirm_checkout(
        args={"confirm": True},
        session=session,
    )
    assert res2["success"] is True
    assert "order_id" in res2
    assert session.checkout_state is None  # State cleared after completion


def test_variant_selection_prompting(db: Session, setup_data):
    """Test that when a product has multiple variants and no size is given, resolver flags needs_variant_selection."""
    merchant = setup_data["merchant"]
    resolver = PurchaseIntentResolver(db=db, merchant_id=merchant.id)

    # Create a test product with 2 variants
    prod = Product(
        merchant_id=merchant.id,
        title="Multi-Variant Polo Shirt",
        description="A great polo shirt",
        price=Decimal("1999.00"),
        is_active=True,
    )
    db.add(prod)
    db.flush()

    var_m = ProductVariant(product_id=prod.id, size="M", color="Black", price=Decimal("1999.00"), sku=f"POLO-M-{uuid.uuid4().hex[:6]}")
    var_l = ProductVariant(product_id=prod.id, size="L", color="Black", price=Decimal("1999.00"), sku=f"POLO-L-{uuid.uuid4().hex[:6]}")
    db.add_all([var_m, var_l])
    db.flush()

    inv_m = Inventory(variant_id=var_m.id, quantity=10, reserved_quantity=0)
    inv_l = Inventory(variant_id=var_l.id, quantity=10, reserved_quantity=0)
    db.add_all([inv_m, inv_l])
    db.commit()

    session = AgentSession(
        merchant_id=merchant.id,
        session_id=str(uuid.uuid4()),
        last_search_results=[{"position": 1, "id": str(prod.id), "title": prod.title}]
    )

    # Without size -> needs variant selection
    target = resolver.resolve_single_product(session=session, reference_position=1)
    assert target.needs_variant_selection is True
    assert target.variant is None
    assert len(target.available_variants) == 2

    # With size="L" -> resolves directly
    target_l = resolver.resolve_single_product(session=session, reference_position=1, size="L")
    assert target_l.needs_variant_selection is False
    assert target_l.variant.id == var_l.id


def test_cart_checkout_flow(db: Session, setup_data):
    """Test AgentService._execute_checkout_cart and confirm_checkout."""
    merchant = setup_data["merchant"]
    customer = setup_data["customer"]
    ctx = MerchantContext(merchant_id=merchant.id, merchant_name=merchant.name)
    agent_svc = AgentService(db=db, merchant_context=ctx, customer=customer)

    session_id = f"test_cart_session_{uuid.uuid4()}"
    session = agent_svc.get_or_create_session(session_id)

    # Find a variant and add to cart
    variant = db.query(ProductVariant).join(Product).filter(
        Product.merchant_id == merchant.id,
        Product.is_active.is_(True)
    ).first()
    assert variant is not None

    agent_svc.tool_add_to_cart(session=session, product_id=variant.product_id, variant_id=variant.id, quantity=1)

    # Step 1: Initiate cart checkout
    res1 = agent_svc._execute_checkout_cart(args={"address_hint": "home"}, session=session)
    assert res1["success"] is True
    assert res1["awaiting_confirmation"] is True
    assert "checkout_summary" in res1
    assert session.checkout_state["mode"] == "CART"

    # Step 2: Confirm cart checkout
    res2 = agent_svc._execute_confirm_checkout(args={"confirm": True}, session=session)
    assert res2["success"] is True
    assert "order_id" in res2
    assert session.checkout_state is None

