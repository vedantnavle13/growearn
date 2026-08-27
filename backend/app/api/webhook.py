"""
Razorpay Webhook API Router: POST /api/payments/razorpay/webhook

Handles Razorpay webhook events for payment status updates.
Validates webhook signature and processes events idempotently.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status, Header
from sqlalchemy.orm import Session

from app.core.merchant_context import MerchantContext, get_merchant_context
from app.db.database import get_db
from app.models.order import Payment, PaymentStatus, Order, OrderStatus
from app.models.cart import Cart, CartStatus
from app.services.razorpay_service import RazorpayService, RazorpayWebhookError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/payments/razorpay", tags=["Webhooks"])


@router.post(
    "/webhook",
    summary="Razorpay webhook endpoint",
    description=(
        "Receives Razorpay webhook events for payment status changes. "
        "Validates webhook signature and updates payment/order status idempotently. "
        "Handles duplicate webhook deliveries without side effects."
    ),
)
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str | None = Header(None, alias="X-Razorpay-Signature"),
    merchant_context: MerchantContext = Depends(get_merchant_context),
    db: Session = Depends(get_db),
):
    """
    Process Razorpay webhook events.

    Expected events:
    - payment.captured: Payment successful
    - payment.failed: Payment failed
    - order.paid: Order paid

    Idempotency:
    - Uses razorpay_payment_id to detect duplicate events
    - Payment record status prevents duplicate processing
    - Order status transitions are guarded
    """
    # Get raw body for signature verification
    payload = await request.body()

    # Verify webhook signature
    try:
        razorpay_service = RazorpayService()
        razorpay_service.verify_webhook_signature(payload, x_razorpay_signature or "")
    except RazorpayWebhookError as e:
        logger.warning(f"Webhook signature verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook signature",
        ) from e

    # Parse event
    try:
        event = razorpay_service.parse_webhook_event(payload)
    except RazorpayWebhookError as e:
        logger.error(f"Invalid webhook payload: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook payload",
        ) from e

    event_type = event.get("event")
    payload_data = event.get("payload", {})

    logger.info(f"Received Razorpay webhook: {event_type}")

    # Process based on event type
    try:
        if event_type == "payment.captured":
            await handle_payment_captured(payload_data, db, merchant_context)
        elif event_type == "payment.failed":
            await handle_payment_failed(payload_data, db, merchant_context)
        elif event_type == "order.paid":
            await handle_order_paid(payload_data, db, merchant_context)
        else:
            logger.info(f"Unhandled webhook event type: {event_type}")

    except Exception as e:
        logger.error(f"Webhook processing failed for {event_type}: {e}", exc_info=True)
        # Return 200 to acknowledge receipt (Razorpay expects this)
        # but log the error for investigation
        pass

    return {"status": "ok"}


async def handle_payment_captured(payload: dict, db: Session, merchant_context: MerchantContext):
    """Handle payment.captured event."""
    payment_entity = payload.get("payment", {}).get("entity", {})
    razorpay_payment_id = payment_entity.get("id")
    razorpay_order_id = payment_entity.get("order_id")

    if not razorpay_payment_id or not razorpay_order_id:
        logger.warning("Missing payment_id or order_id in payment.captured event")
        return

    # Find payment record
    payment = db.query(Payment).filter(
        Payment.razorpay_order_id == razorpay_order_id,
    ).first()

    if not payment:
        logger.warning(f"No payment record found for Razorpay order: {razorpay_order_id}")
        return

    # Verify merchant isolation
    order = db.query(Order).filter(
        Order.id == payment.order_id,
        Order.merchant_id == merchant_context.merchant_id,
    ).first()

    if not order:
        logger.warning(f"Order not found or merchant mismatch for payment: {payment.id}")
        return

    # Idempotency: already processed
    if payment.status == PaymentStatus.SUCCESS:
        logger.info(f"Payment already processed: {payment.id}")
        return

    # Update payment
    payment.status = PaymentStatus.SUCCESS
    payment.razorpay_payment_id = razorpay_payment_id
    payment.method = payment_entity.get("method")

    # Update order
    order.status = OrderStatus.PAID

    # Mark cart as converted
    if order.cart_id:
        cart = db.query(Cart).filter(Cart.id == order.cart_id).first()
        if cart:
            cart.status = CartStatus.CONVERTED

    db.commit()
    logger.info(f"Webhook: Payment captured - payment={payment.id}, order={order.id}")


async def handle_payment_failed(payload: dict, db: Session, merchant_context: MerchantContext):
    """Handle payment.failed event."""
    payment_entity = payload.get("payment", {}).get("entity", {})
    razorpay_payment_id = payment_entity.get("id")
    razorpay_order_id = payment_entity.get("order_id")

    if not razorpay_payment_id or not razorpay_order_id:
        logger.warning("Missing payment_id or order_id in payment.failed event")
        return

    # Find payment record
    payment = db.query(Payment).filter(
        Payment.razorpay_order_id == razorpay_order_id,
    ).first()

    if not payment:
        logger.warning(f"No payment record found for Razorpay order: {razorpay_order_id}")
        return

    # Verify merchant isolation
    order = db.query(Order).filter(
        Order.id == payment.order_id,
        Order.merchant_id == merchant_context.merchant_id,
    ).first()

    if not order:
        logger.warning(f"Order not found or merchant mismatch for payment: {payment.id}")
        return

    # Idempotency: already processed as failed
    if payment.status == PaymentStatus.FAILED:
        logger.info(f"Payment already marked failed: {payment.id}")
        return

    # Update payment
    payment.status = PaymentStatus.FAILED
    payment.razorpay_payment_id = razorpay_payment_id

    # Update order to FAILED (but don't mark cart as converted)
    order.status = OrderStatus.FAILED

    db.commit()
    logger.info(f"Webhook: Payment failed - payment={payment.id}, order={order.id}")


async def handle_order_paid(payload: dict, db: Session, merchant_context: MerchantContext):
    """Handle order.paid event (backup for payment.captured)."""
    order_entity = payload.get("order", {}).get("entity", {})
    razorpay_order_id = order_entity.get("id")

    if not razorpay_order_id:
        logger.warning("Missing order_id in order.paid event")
        return

    # Find payment record
    payment = db.query(Payment).filter(
        Payment.razorpay_order_id == razorpay_order_id,
    ).first()

    if not payment:
        logger.warning(f"No payment record found for Razorpay order: {razorpay_order_id}")
        return

    # Verify merchant isolation
    order = db.query(Order).filter(
        Order.id == payment.order_id,
        Order.merchant_id == merchant_context.merchant_id,
    ).first()

    if not order:
        logger.warning(f"Order not found or merchant mismatch for payment: {payment.id}")
        return

    # Idempotency
    if payment.status == PaymentStatus.SUCCESS:
        logger.info(f"Order already paid via webhook: {order.id}")
        return

    # Update payment and order
    payment.status = PaymentStatus.SUCCESS
    order.status = OrderStatus.PAID

    # Mark cart as converted
    if order.cart_id:
        cart = db.query(Cart).filter(Cart.id == order.cart_id).first()
        if cart:
            cart.status = CartStatus.CONVERTED

    db.commit()
    logger.info(f"Webhook: Order paid - payment={payment.id}, order={order.id}")