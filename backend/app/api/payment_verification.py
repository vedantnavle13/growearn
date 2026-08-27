"""
Payment Verification API Router: POST /api/payments/verify

Verifies Razorpay payment signature after frontend completes payment.
Updates payment and order status on successful verification.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.merchant_context import MerchantContext, get_merchant_context
from app.db.database import get_db
from app.schemas.checkout import PaymentVerificationRequest, PaymentVerificationResponse
from app.services.checkout_service import (
    PaymentVerificationService,
    CheckoutError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/payments", tags=["Payments"])


@router.post(
    "/verify",
    response_model=PaymentVerificationResponse,
    summary="Verify Razorpay payment after frontend completion",
    description=(
        "Verifies the Razorpay payment signature on the server side. "
        "Only after successful verification, updates Payment status to SUCCESS "
        "and Order status to PAID. "
        "Frontend 'payment successful' is NOT sufficient - server verification is mandatory."
    ),
)
async def verify_payment(
    body: PaymentVerificationRequest,
    merchant_context: MerchantContext = Depends(get_merchant_context),
    db: Session = Depends(get_db),
) -> PaymentVerificationResponse:
    """
    Verify payment signature and update order/payment status.

    This endpoint MUST be called after the frontend completes the Razorpay checkout.
    The frontend should call this with the payment details from Razorpay's callback.

    Flow:
    1. Find payment by razorpay_order_id
    2. Verify signature using Razorpay secret key
    3. If valid: Payment.status = SUCCESS, Order.status = PAID, Cart.status = CONVERTED
    4. If invalid: Payment.status = FAILED, Order remains PENDING
    """
    try:
        service = PaymentVerificationService(
            db=db,
            merchant_context=merchant_context,
        )

        payment = service.verify_payment(
            razorpay_order_id=body.razorpay_order_id,
            razorpay_payment_id=body.razorpay_payment_id,
            razorpay_signature=body.razorpay_signature,
        )

        return PaymentVerificationResponse(
            order_id=payment.order_id,
            payment_id=payment.id,
            status=payment.status.value,
            message="Payment verified successfully. Order is now PAID.",
        )

    except CheckoutError as e:
        if e.code == "VERIFICATION_FAILED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Payment verification failed. Invalid signature.",
            ) from e
        elif e.code == "PAYMENT_NOT_FOUND":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Payment record not found.",
            ) from e
        elif e.code == "ORDER_NOT_FOUND":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Associated order not found.",
            ) from e
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=e.message,
            ) from e
    except Exception as e:
        logger.error(f"Payment verification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Payment verification failed. Please try again.",
        ) from e