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