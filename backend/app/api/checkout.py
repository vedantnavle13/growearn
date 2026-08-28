"""
Checkout API Router: POST /api/checkout

Initiates checkout for the current customer's cart.
Creates internal order, Razorpay order, and payment record.
"""

import logging
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.merchant_context import MerchantContext, get_merchant_context
from app.db.database import get_db
from app.models.customer import Customer
from app.schemas.checkout import CheckoutRequest, CheckoutResponse
from app.services.checkout_service import (
    CheckoutService,
    CheckoutError,
    CartNotFoundError,
    CartEmptyError,
    ProductUnavailableError,
    InsufficientStockError,
    MerchantMismatchError,
    CustomerMismatchError,
)

from decimal import Decimal
from app.models.cart import Cart, CartItem, CartStatus
from app.models.product import Product, ProductVariant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Checkout"])


def get_customer_context(
    x_customer_id: Optional[str] = Header(None, alias="X-Customer-Id", description="Customer UUID header"),
    customer_id: Optional[str] = Query(None, description="Customer UUID query parameter"),
    merchant_context: MerchantContext = Depends(get_merchant_context),
    db: Session = Depends(get_db),
) -> Customer | None:
    """
    Resolve customer context for the request.
    Uses the same logic as agent API.
    """
    target_id_str = x_customer_id or customer_id
    if target_id_str:
        try:
            customer_uuid = uuid.UUID(str(target_id_str).strip())
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid customer ID format: '{target_id_str}'. Must be a valid UUID.",
            )

        customer = db.query(Customer).filter(
            Customer.id == customer_uuid,
            Customer.merchant_id == merchant_context.merchant_id,
        ).first()

        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer with ID '{customer_uuid}' not found in this merchant.",
            )
        return customer

    # Development / Demo mode fallback
    if getattr(settings, "DEBUG", False):
        fallback_customer = db.query(Customer).filter(
            Customer.merchant_id == merchant_context.merchant_id
        ).order_by(Customer.created_at.asc()).first()
        if fallback_customer:
            return fallback_customer

    return None


@router.get(
    "/cart",
    summary="Get active cart contents for customer",
)
async def get_active_cart(
    merchant_context: MerchantContext = Depends(get_merchant_context),
    customer: Customer | None = Depends(get_customer_context),
    db: Session = Depends(get_db),
):
    """Retrieve the current active cart items for the authenticated customer."""
    if not customer:
        return {"items": [], "subtotal": 0.0, "item_count": 0}

    cart = db.query(Cart).filter(
        Cart.customer_id == customer.id,
        Cart.status == CartStatus.ACTIVE,
    ).first()

    if not cart or not cart.items:
        return {"cart_id": str(cart.id) if cart else None, "items": [], "subtotal": 0.0, "item_count": 0}

    items = []
    subtotal = Decimal("0")
    for item in cart.items:
        variant = item.variant
        if not variant:
            continue
        product = variant.product
        if not product or product.merchant_id != merchant_context.merchant_id:
            continue

        available = 0
        if variant.inventory:
            available = max(0, variant.inventory.quantity - variant.inventory.reserved_quantity)

        line_total = item.price_at_addition * item.quantity
        subtotal += line_total

        items.append({
            "id": str(product.id),
            "cart_item_id": str(item.id),
            "product_id": str(product.id),
            "variant_id": str(variant.id),
            "title": product.title,
            "price": float(item.price_at_addition),
            "color": variant.color,
            "size": variant.size,
            "quantity": item.quantity,
            "line_total": float(line_total),
            "in_stock": available >= item.quantity,
            "image_url": product.image_url,
        })

    return {
        "cart_id": str(cart.id),
        "items": items,
        "subtotal": float(subtotal),
        "item_count": sum(i["quantity"] for i in items),
    }


@router.delete(
    "/cart/items/{cart_item_id}",
    summary="Remove an item from cart",
)
async def remove_cart_item(
    cart_item_id: uuid.UUID,
    merchant_context: MerchantContext = Depends(get_merchant_context),
    customer: Customer | None = Depends(get_customer_context),
    db: Session = Depends(get_db),
):
    """Remove a specific item from active cart."""
    if not customer:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found")

    item = db.query(CartItem).join(Cart).filter(
        CartItem.id == cart_item_id,
        Cart.customer_id == customer.id,
        Cart.status == CartStatus.ACTIVE,
    ).first()

    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cart item not found")

    db.delete(item)
    db.commit()
    return {"success": True}


@router.post(
    "/checkout",
    response_model=CheckoutResponse,
    summary="Initiate checkout for current cart",
    description=(
        "Creates an internal order, Razorpay order, and payment record. "
        "Returns information needed to initialize Razorpay checkout on frontend. "
        "Validates cart contents, inventory, and calculates authoritative total from database."
    ),
)
async def checkout(
    body: CheckoutRequest,
    merchant_context: MerchantContext = Depends(get_merchant_context),
    customer: Customer | None = Depends(get_customer_context),
    db: Session = Depends(get_db),
) -> CheckoutResponse:
    """
    Initiate checkout process.

    Flow:
    1. Load cart from agent session (linked to customer)
    2. Validate cart items against current inventory and prices
    3. Create internal Order with PENDING status
    4. Create Razorpay Order
    5. Create Payment record with PENDING status
    6. Return checkout data for frontend Razorpay initialization

    All amounts calculated from database - never trusts frontend/LLM.
    """
    try:
        service = CheckoutService(
            db=db,
            merchant_context=merchant_context,
            customer=customer,
        )

        result = service.execute_checkout(body.session_id)

        return CheckoutResponse(
            order_id=result.order_id,
            razorpay_order_id=result.razorpay_order_id,
            amount=result.amount_paise,
            currency=result.currency,
            key_id=result.key_id,
            status=result.status,
        )

    except CartNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        ) from e
    except CartEmptyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        ) from e
    except ProductUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.message,
        ) from e
    except InsufficientStockError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.message,
        ) from e
    except MerchantMismatchError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=e.message,
        ) from e
    except CustomerMismatchError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=e.message,
        ) from e
    except CheckoutError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        ) from e
    except Exception as e:
        logger.error(f"Checkout failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Checkout failed. Please try again.",
        ) from e