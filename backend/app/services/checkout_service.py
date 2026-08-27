"""
Checkout Service: Business logic for cart and single-product checkout flows.

Two distinct checkout paths exist:
  CartCheckoutService    — checkout the entire active cart (see execute_cart_checkout)
  SingleProductCheckoutService — checkout a single resolved product/variant

Both paths:
  - Validate product/variant/inventory via authoritative DB values
  - Create Order + OrderItems
  - Copy address snapshot onto Order (immutable historical record)
  - DO NOT create Razorpay orders yet (payment integration deferred)
"""

import uuid
import logging
from decimal import Decimal
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, and_, func

from app.core.merchant_context import MerchantContext
from app.models.cart import Cart, CartItem, CartStatus
from app.models.order import Order, OrderItem, OrderStatus, Payment, PaymentStatus
from app.models.product import Product, ProductVariant, Inventory
from app.models.customer import Customer
from app.models.customer_address import CustomerAddress
from app.services.razorpay_service import RazorpayService, RazorpayOrderCreationError

logger = logging.getLogger(__name__)


class CheckoutError(Exception):
    """Base exception for checkout errors."""

    def __init__(self, message: str, code: str = "CHECKOUT_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class CartNotFoundError(CheckoutError):
    """Cart not found for customer."""

    def __init__(self, message: str = "Cart not found"):
        super().__init__(message, "CART_NOT_FOUND")


class CartEmptyError(CheckoutError):
    """Cart has no items."""

    def __init__(self, message: str = "Cart is empty"):
        super().__init__(message, "CART_EMPTY")


class ProductUnavailableError(CheckoutError):
    """Product or variant is no longer available."""

    def __init__(self, message: str, product_title: str = ""):
        super().__init__(message, "PRODUCT_UNAVAILABLE")
        self.product_title = product_title


class InsufficientStockError(CheckoutError):
    """Not enough stock for requested quantity."""

    def __init__(self, message: str, product_title: str = "", available: int = 0, requested: int = 0):
        super().__init__(message, "INSUFFICIENT_STOCK")
        self.product_title = product_title
        self.available = available
        self.requested = requested


class MerchantMismatchError(CheckoutError):
    """Cart belongs to different merchant."""

    def __init__(self, message: str = "Cart does not belong to this merchant"):
        super().__init__(message, "MERCHANT_MISMATCH")


class CustomerMismatchError(CheckoutError):
    """Cart belongs to different customer."""

    def __init__(self, message: str = "Cart does not belong to this customer"):
        super().__init__(message, "CUSTOMER_MISMATCH")


class DuplicateCheckoutError(CheckoutError):
    """Duplicate checkout attempt detected."""

    def __init__(self, message: str = "Checkout already in progress for this cart"):
        super().__init__(message, "DUPLICATE_CHECKOUT")


@dataclass
class CartValidationResult:
    """Result of cart validation."""

    cart: Cart
    items: List[CartItem]
    subtotal: Decimal
    warnings: List[str]


@dataclass
class CheckoutResult:
    """Result of successful checkout (legacy, used by Razorpay flow)."""

    order_id: uuid.UUID
    razorpay_order_id: str
    amount_paise: int
    currency: str
    key_id: str
    status: str


@dataclass
class OrderResult:
    """
    Result of a successful order creation (pre-payment).
    Razorpay fields are not included — payment integration is deferred.
    """
    order_id: uuid.UUID
    total_amount: Decimal
    currency: str = "INR"
    status: str = "pending_payment"


class CheckoutService:
    """
    Service for processing checkout with authoritative pricing and inventory validation.

    All monetary calculations use database values only.
    Never trusts frontend, LLM, or session prices.
    """

    def __init__(
        self,
        db: Session,
        merchant_context: MerchantContext,
        customer: Optional[Customer] = None,
        razorpay_service: Optional[RazorpayService] = None,
    ) -> None:
        self.db = db
        self.merchant_context = merchant_context
        self.customer = customer
        self._razorpay_service = razorpay_service

    @property
    def razorpay_service(self) -> RazorpayService:
        if self._razorpay_service is None:
            self._razorpay_service = RazorpayService()
        return self._razorpay_service

    def validate_and_calculate(self, cart: Cart) -> CartValidationResult:
        """
        Validate cart contents and calculate authoritative total.

        Re-checks:
        - Product existence and active status
        - Variant existence
        - Inventory availability
        - Current database prices
        - Merchant isolation
        """
        if not cart.items:
            raise CartEmptyError()

        items = []
        subtotal = Decimal("0")
        warnings = []

        for cart_item in cart.items:
            # Load variant with product and inventory
            variant = self.db.query(ProductVariant).options(
                joinedload(ProductVariant.product),
                joinedload(ProductVariant.inventory),
            ).filter(
                ProductVariant.id == cart_item.variant_id,
            ).first()

            if not variant:
                raise ProductUnavailableError(
                    f"Variant {cart_item.variant_id} no longer exists",
                    product_title="Unknown product",
                )

            # Verify product belongs to this merchant and is active
            product = variant.product
            if not product or product.merchant_id != self.merchant_context.merchant_id:
                raise ProductUnavailableError(
                    f"Product {product.id if product else 'unknown'} not available in this store",
                    product_title=product.title if product else "Unknown",
                )

            if not product.is_active:
                raise ProductUnavailableError(
                    f"Product '{product.title}' is no longer available",
                    product_title=product.title,
                )

            # Check inventory
            available = 0
            if variant.inventory:
                available = variant.inventory.quantity - variant.inventory.reserved_quantity

            if available <= 0:
                raise InsufficientStockError(
                    f"'{product.title}' ({variant.color or ''} {variant.size or ''}) is out of stock",
                    product_title=product.title,
                    available=0,
                    requested=cart_item.quantity,
                )

            if cart_item.quantity > available:
                raise InsufficientStockError(
                    f"Only {available} unit(s) available for '{product.title}' "
                    f"({variant.color or ''} {variant.size or ''}), you requested {cart_item.quantity}",
                    product_title=product.title,
                    available=available,
                    requested=cart_item.quantity,
                )

            # Use current database price (authoritative)
            current_price = variant.price
            line_total = current_price * cart_item.quantity
            subtotal += line_total

            # Warn if price changed since added to cart
            if cart_item.price_at_addition != current_price:
                warnings.append(
                    f"Price for '{product.title}' ({variant.color or ''} {variant.size or ''}) "
                    f"changed from {cart_item.price_at_addition} to {current_price}"
                )

            items.append(cart_item)

        return CartValidationResult(
            cart=cart,
            items=items,
            subtotal=subtotal,
            warnings=warnings,
        )

    def get_or_create_active_cart(self, session_id: str) -> Cart:
        """
        Get the active cart for the current customer/session.

        For agent sessions, the cart is linked to the session.
        """
        # First try to find cart via agent session
        from app.models.agent_session import AgentSession
        session = self.db.query(AgentSession).filter(
            AgentSession.session_id == session_id,
            AgentSession.merchant_id == self.merchant_context.merchant_id,
        ).first()

        if session and session.cart_id:
            cart = self.db.query(Cart).filter(
                Cart.id == session.cart_id,
                Cart.status == CartStatus.ACTIVE,
            ).first()
            if cart:
                return cart

        # Resolve customer context
        effective_customer_id = None
        if session and session.customer_id:
            effective_customer_id = session.customer_id
        elif self.customer:
            effective_customer_id = self.customer.id

        if not effective_customer_id:
            raise CartNotFoundError("Customer not identified. Please provide customer context.")

        # Find active cart for customer
        cart = self.db.query(Cart).filter(
            Cart.customer_id == effective_customer_id,
            Cart.status == CartStatus.ACTIVE,
        ).first()

        if not cart:
            raise CartNotFoundError("No active cart found for this customer")

        # Verify cart belongs to this merchant (via customer)
        if cart.customer.merchant_id != self.merchant_context.merchant_id:
            raise MerchantMismatchError()

        # Verify cart belongs to this customer
        if self.customer and cart.customer_id != self.customer.id:
            raise CustomerMismatchError()

        # Link cart to session if not already linked
        if session and not session.cart_id:
            session.cart_id = cart.id
            self.db.flush()

        return cart

    def check_idempotency(self, cart: Cart) -> Optional[Order]:
        """
        Check if checkout was already initiated for this cart.

        Returns existing order if found (idempotent behavior).
        """
        # Check for existing PENDING order for this cart
        existing_order = self.db.query(Order).filter(
            Order.cart_id == cart.id,
            Order.status == OrderStatus.PENDING,
        ).first()

        if existing_order:
            # Check if it has a pending payment
            pending_payment = self.db.query(Payment).filter(
                Payment.order_id == existing_order.id,
                Payment.status == PaymentStatus.PENDING,
            ).first()

            if pending_payment and pending_payment.razorpay_order_id:
                logger.info(f"Idempotent checkout: returning existing order {existing_order.id}")
                return existing_order

        return None

    def create_order(
        self,
        cart: Cart,
        validation_result: CartValidationResult,
        address: Optional[CustomerAddress] = None,
        customer_name: Optional[str] = None,
    ) -> Order:
        """Create internal Order and OrderItems from validated cart, with optional address snapshot."""
        # For now, total = subtotal (no taxes/shipping/discounts)
        total_amount = validation_result.subtotal

        order = Order(
            merchant_id=self.merchant_context.merchant_id,
            customer_id=cart.customer_id,
            cart_id=cart.id,
            total_amount=total_amount,
            status=OrderStatus.PENDING,
        )

        # Snapshot the delivery address onto the order for immutable history
        if address:
            order.customer_address_id = address.id
            order.shipping_recipient_name = address.recipient_name or customer_name or ""
            order.shipping_address_line_1 = address.address_line_1
            order.shipping_address_line_2 = address.address_line_2
            order.shipping_city = address.city
            order.shipping_state = address.state
            order.shipping_postal_code = address.postal_code
            order.shipping_country = address.country

        self.db.add(order)
        self.db.flush()

        # Create OrderItems with current database prices
        for cart_item in validation_result.items:
            variant = self.db.query(ProductVariant).filter(
                ProductVariant.id == cart_item.variant_id,
            ).first()

            if not variant:
                # This shouldn't happen after validation, but defensive
                raise ProductUnavailableError(
                    f"Variant {cart_item.variant_id} disappeared during order creation",
                )

            order_item = OrderItem(
                order_id=order.id,
                variant_id=variant.id,
                quantity=cart_item.quantity,
                price=variant.price,  # Current DB price
            )
            self.db.add(order_item)

        self.db.flush()
        return order

    def create_payment_record(
        self,
        order: Order,
        razorpay_order_id: str,
        amount: Decimal,
        currency: str = "INR",
    ) -> Payment:
        """Create Payment record linked to order and Razorpay order."""
        payment = Payment(
            order_id=order.id,
            razorpay_order_id=razorpay_order_id,
            amount=amount,
            currency=currency,
            status=PaymentStatus.PENDING,
        )
        self.db.add(payment)
        self.db.flush()
        return payment

    def execute_checkout(self, session_id: str) -> CheckoutResult:
        """
        Execute the complete checkout flow.

        1. Get cart from session/customer
        2. Validate cart and calculate authoritative total
        3. Check idempotency
        4. Create internal Order
        5. Create Razorpay order
        6. Create Payment record
        7. Return checkout info for frontend
        """
        # 1. Get cart
        cart = self.get_or_create_active_cart(session_id)

        # 2. Validate and calculate
        validation_result = self.validate_and_calculate(cart)

        # 3. Check idempotency
        existing_order = self.check_idempotency(cart)
        if existing_order:
            # Return existing checkout info
            pending_payment = self.db.query(Payment).filter(
                Payment.order_id == existing_order.id,
                Payment.status == PaymentStatus.PENDING,
            ).first()

            if pending_payment and pending_payment.razorpay_order_id:
                return CheckoutResult(
                    order_id=existing_order.id,
                    razorpay_order_id=pending_payment.razorpay_order_id,
                    amount_paise=int(existing_order.total_amount * 100),
                    currency=pending_payment.currency,
                    key_id=self.razorpay_service.key_id,
                    status="pending",
                )

        try:
            # 4. Create internal Order
            order = self.create_order(cart, validation_result)

            # 5. Create Razorpay order
            amount_paise = int(validation_result.subtotal * 100)  # Convert to paise
            receipt = f"order_{str(order.id)[:8]}"  # Short receipt ID

            razorpay_order = self.razorpay_service.create_order(
                amount_paise=amount_paise,
                currency="INR",
                receipt=receipt,
                notes={
                    "internal_order_id": str(order.id),
                    "merchant_id": str(self.merchant_context.merchant_id),
                    "customer_id": str(cart.customer_id),
                },
            )

            razorpay_order_id = razorpay_order["id"]

            # 6. Create Payment record
            self.create_payment_record(
                order=order,
                razorpay_order_id=razorpay_order_id,
                amount=validation_result.subtotal,
                currency="INR",
            )

            # 7. Commit all changes
            self.db.commit()

            logger.info(f"Checkout completed: order={order.id}, razorpay_order={razorpay_order_id}")

            return CheckoutResult(
                order_id=order.id,
                razorpay_order_id=razorpay_order_id,
                amount_paise=amount_paise,
                currency="INR",
                key_id=self.razorpay_service.key_id,
                status="pending",
            )

        except RazorpayOrderCreationError as e:
            self.db.rollback()
            logger.error(f"Razorpay order creation failed: {e}")
            raise CheckoutError(f"Payment initialization failed: {e.message}", "RAZORPAY_ERROR") from e
        except Exception as e:
            self.db.rollback()
            logger.error(f"Checkout failed: {e}")
            raise CheckoutError(f"Checkout failed: {str(e)}", "CHECKOUT_FAILED") from e


class PaymentVerificationService:
    """
    Service for verifying payments and updating order/payment status.

    Responsibilities:
    - Verify Razorpay payment signature
    - Update Payment status to SUCCESS/FAILED
    - Update Order status to PAID/FAILED
    - Handle idempotency for duplicate verification attempts
    """

    def __init__(
        self,
        db: Session,
        merchant_context: MerchantContext,
        razorpay_service: Optional[RazorpayService] = None,
    ) -> None:
        self.db = db
        self.merchant_context = merchant_context
        self._razorpay_service = razorpay_service

    @property
    def razorpay_service(self) -> RazorpayService:
        if self._razorpay_service is None:
            self._razorpay_service = RazorpayService()
        return self._razorpay_service

    def verify_payment(
        self,
        razorpay_order_id: str,
        razorpay_payment_id: str,
        razorpay_signature: str,
    ) -> Payment:
        """
        Verify payment signature and update records.

        This is called after frontend completes payment.
        """
        # Find payment by razorpay_order_id
        payment = self.db.query(Payment).filter(
            Payment.razorpay_order_id == razorpay_order_id,
        ).first()

        if not payment:
            logger.warning(f"Payment not found for Razorpay order: {razorpay_order_id}")
            raise CheckoutError("Payment record not found", "PAYMENT_NOT_FOUND")

        # Verify this payment belongs to our merchant
        order = self.db.query(Order).filter(
            Order.id == payment.order_id,
            Order.merchant_id == self.merchant_context.merchant_id,
        ).first()

        if not order:
            logger.warning(f"Order not found or merchant mismatch for payment: {payment.id}")
            raise CheckoutError("Order not found", "ORDER_NOT_FOUND")

        # Idempotency: if already SUCCESS, return existing
        if payment.status == PaymentStatus.SUCCESS:
            logger.info(f"Payment already verified: {payment.id}")
            return payment

        # Verify signature
        try:
            self.razorpay_service.verify_payment_signature(
                razorpay_order_id=razorpay_order_id,
                razorpay_payment_id=razorpay_payment_id,
                razorpay_signature=razorpay_signature,
            )
        except Exception as e:
            # Mark payment as FAILED
            payment.status = PaymentStatus.FAILED
            payment.razorpay_payment_id = razorpay_payment_id
            payment.razorpay_signature = razorpay_signature
            self.db.commit()
            logger.warning(f"Payment verification failed: {e}")
            raise CheckoutError("Payment verification failed", "VERIFICATION_FAILED") from e

        # Signature valid - update payment
        payment.status = PaymentStatus.SUCCESS
        payment.razorpay_payment_id = razorpay_payment_id
        payment.razorpay_signature = razorpay_signature

        # Update order status to PAID
        order.status = OrderStatus.PAID

        # Mark cart as CONVERTED
        if order.cart_id:
            cart = self.db.query(Cart).filter(Cart.id == order.cart_id).first()
            if cart:
                cart.status = CartStatus.CONVERTED

        self.db.commit()

        logger.info(f"Payment verified successfully: payment={payment.id}, order={order.id}")
        return payment


# ---------------------------------------------------------------------------
# Single Product Checkout Service
# ---------------------------------------------------------------------------

class SingleProductCheckoutService:
    """
    Handles checkout for a SINGLE product/variant resolved from a purchase intent.

    Unlike CartCheckoutService, this does NOT load the customer's cart.
    It creates a direct Order for one product + variant + quantity.

    Razorpay integration is deferred — this creates the Order in PENDING status.
    """

    def __init__(
        self,
        db: Session,
        merchant_context: MerchantContext,
        customer: Optional[Customer] = None,
    ) -> None:
        self.db = db
        self.merchant_context = merchant_context
        self.customer = customer

    def build_summary(
        self,
        product: "Product",
        variant: "ProductVariant",
        quantity: int,
        address: Optional[CustomerAddress] = None,
    ) -> dict:
        """
        Build a checkout summary dict to present to the user for confirmation.
        All prices come from the DB — never from LLM or session.
        """
        unit_price = variant.price
        subtotal = unit_price * quantity
        address_display = None
        if address:
            parts = [
                address.recipient_name or (self.customer.name if self.customer else ""),
                address.address_line_1,
                address.address_line_2,
                f"{address.city}, {address.state} {address.postal_code}",
                address.country,
            ]
            address_display = ", ".join(p for p in parts if p)

        return {
            "product_title": product.title,
            "variant_id": str(variant.id),
            "variant_size": variant.size,
            "variant_color": variant.color,
            "quantity": quantity,
            "unit_price": str(unit_price),
            "subtotal": str(subtotal),
            "total": str(subtotal),
            "currency": "INR",
            "delivery_address": address_display,
            "address_id": str(address.id) if address else None,
        }

    def execute_checkout(
        self,
        product: "Product",
        variant: "ProductVariant",
        quantity: int,
        address: Optional[CustomerAddress] = None,
    ) -> OrderResult:
        """
        Create an Order for a single product/variant.

        1. Re-validate product belongs to merchant (defensive)
        2. Re-validate inventory (final check before order creation)
        3. Create Order + OrderItem with authoritative DB price
        4. Snapshot delivery address onto Order
        5. Commit and return OrderResult

        Does NOT create a Razorpay payment — payment integration is deferred.
        """
        # --- Re-validate product belongs to this merchant (security) ---
        fresh_product = self.db.query(Product).filter(
            Product.id == product.id,
            Product.merchant_id == self.merchant_context.merchant_id,
            Product.is_active.is_(True),
        ).first()

        if not fresh_product:
            raise ProductUnavailableError(
                f"Product '{product.title}' is not available in this store.",
                product_title=product.title,
            )

        # --- Re-validate variant ---
        fresh_variant = self.db.query(ProductVariant).filter(
            ProductVariant.id == variant.id,
            ProductVariant.product_id == fresh_product.id,
        ).first()

        if not fresh_variant:
            raise ProductUnavailableError(
                f"The selected variant for '{product.title}' is no longer available.",
                product_title=product.title,
            )

        # --- Re-validate inventory ---
        available = 0
        if fresh_variant.inventory:
            available = fresh_variant.inventory.quantity - fresh_variant.inventory.reserved_quantity

        if available <= 0:
            raise InsufficientStockError(
                f"'{fresh_product.title}' ({fresh_variant.color or ''} {fresh_variant.size or ''}).strip() "
                f"is out of stock.",
                product_title=fresh_product.title,
                available=0,
                requested=quantity,
            )

        if quantity > available:
            raise InsufficientStockError(
                f"Only {available} unit(s) available for '{fresh_product.title}' "
                f"({fresh_variant.color or ''} {fresh_variant.size or ''}). You requested {quantity}.",
                product_title=fresh_product.title,
                available=available,
                requested=quantity,
            )

        # --- Resolve customer ---
        customer_id = self.customer.id if self.customer else None
        customer_name = self.customer.name if self.customer else None
        if not customer_id:
            raise CheckoutError("Customer not identified. Please provide customer context.", "NO_CUSTOMER")

        # --- Create Order ---
        authoritative_price = fresh_variant.price
        total_amount = authoritative_price * quantity

        order = Order(
            merchant_id=self.merchant_context.merchant_id,
            customer_id=customer_id,
            cart_id=None,  # Single-product checkout has no associated cart
            total_amount=total_amount,
            status=OrderStatus.PENDING,
        )

        # Snapshot delivery address
        if address:
            order.customer_address_id = address.id
            order.shipping_recipient_name = address.recipient_name or customer_name or ""
            order.shipping_address_line_1 = address.address_line_1
            order.shipping_address_line_2 = address.address_line_2
            order.shipping_city = address.city
            order.shipping_state = address.state
            order.shipping_postal_code = address.postal_code
            order.shipping_country = address.country

        self.db.add(order)
        self.db.flush()

        # Create OrderItem with authoritative DB price
        order_item = OrderItem(
            order_id=order.id,
            variant_id=fresh_variant.id,
            quantity=quantity,
            price=authoritative_price,
        )
        self.db.add(order_item)
        self.db.flush()

        self.db.commit()

        logger.info(
            f"[SingleProductCheckoutService] Order created: order_id={order.id} "
            f"product='{fresh_product.title}' variant={fresh_variant.id} "
            f"qty={quantity} total={total_amount} customer={customer_id}"
        )

        return OrderResult(
            order_id=order.id,
            total_amount=total_amount,
            currency="INR",
            status="pending_payment",
        )