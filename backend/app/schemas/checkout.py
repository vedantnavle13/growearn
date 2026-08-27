"""
Pydantic schemas for Checkout and Payment API.
"""

import uuid
from decimal import Decimal
from typing import Optional
from pydantic import BaseModel, Field


class CheckoutRequest(BaseModel):
    """Request to initiate checkout."""

    session_id: str = Field(..., description="Agent session ID to identify the cart")


class CheckoutResponse(BaseModel):
    """Response with checkout information for frontend payment initialization."""

    order_id: uuid.UUID
    razorpay_order_id: str
    amount: int = Field(..., description="Amount in paise (e.g., 349800 for ₹3498)")
    currency: str = Field(default="INR")
    key_id: str = Field(..., description="Razorpay public key ID")
    status: str = Field(default="pending")


class PaymentVerificationRequest(BaseModel):
    """Request to verify a payment after frontend completion."""

    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class PaymentVerificationResponse(BaseModel):
    """Response after payment verification."""

    order_id: uuid.UUID
    payment_id: uuid.UUID
    status: str
    message: str


class WebhookEvent(BaseModel):
    """Razorpay webhook event payload (subset)."""

    event: str
    payload: dict


# Internal schemas for service layer
class CheckoutResult(BaseModel):
    """Internal result from checkout service."""

    order_id: uuid.UUID
    razorpay_order_id: str
    amount_paise: int
    currency: str
    key_id: str
    status: str


class PaymentVerificationResult(BaseModel):
    """Internal result from payment verification."""

    order_id: uuid.UUID
    payment_id: uuid.UUID
    status: str
    message: str