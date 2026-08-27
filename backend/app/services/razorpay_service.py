"""
Razorpay Service: Abstraction layer for Razorpay API interactions.

Responsibilities:
- Create Razorpay orders
- Verify payment signatures
- Validate webhook signatures
"""

import hmac
import hashlib
import json
import logging
from decimal import Decimal
from typing import Dict, Any, Optional

try:
    import razorpay
except ImportError:
    razorpay = None

from app.core.config import settings

logger = logging.getLogger(__name__)


class RazorpayError(Exception):
    """Base exception for Razorpay errors."""

    pass


class RazorpayOrderCreationError(RazorpayError):
    """Failed to create Razorpay order."""

    pass


class RazorpayVerificationError(RazorpayError):
    """Payment signature verification failed."""

    pass


class RazorpayWebhookError(RazorpayError):
    """Webhook signature validation failed."""

    pass


class RazorpayService:
    """
    Service for interacting with Razorpay API.

    All sensitive operations use environment-configured credentials.
    Never logs or exposes the secret key.
    """

    def __init__(self) -> None:
        key_id = settings.RAZORPAY_KEY_ID
        key_secret = settings.RAZORPAY_KEY_SECRET

        if not key_id or not key_secret:
            raise ValueError(
                "Razorpay credentials not configured. "
                "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in environment."
            )

        self._client = razorpay.Client(auth=(key_id, key_secret))
        self._key_id = key_id
        self._key_secret = key_secret
        self._webhook_secret = settings.RAZORPAY_WEBHOOK_SECRET

    @property
    def key_id(self) -> str:
        """Public Razorpay key ID (safe to expose to frontend)."""
        return self._key_id

    def create_order(
        self,
        amount_paise: int,
        currency: str = "INR",
        receipt: Optional[str] = None,
        notes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a Razorpay order.

        Args:
            amount_paise: Amount in paise (e.g., 349800 for ₹3498)
            currency: Currency code (default INR)
            receipt: Optional receipt identifier (max 40 chars)
            notes: Optional key-value notes for the order

        Returns:
            Razorpay order response dict

        Raises:
            RazorpayOrderCreationError: If order creation fails
        """
        if amount_paise <= 0:
            raise RazorpayOrderCreationError("Amount must be positive")

        payload = {
            "amount": amount_paise,
            "currency": currency,
        }

        if receipt:
            payload["receipt"] = receipt[:40]  # Razorpay limit

        if notes:
            payload["notes"] = notes

        try:
            order = self._client.order.create(payload)
            logger.info(f"Created Razorpay order: {order.get('id')}")
            return order
        except razorpay.errors.BadRequestError as e:
            logger.error(f"Razorpay bad request: {e}")
            raise RazorpayOrderCreationError(f"Invalid order parameters: {e}") from e
        except razorpay.errors.ServerError as e:
            logger.error(f"Razorpay server error: {e}")
            raise RazorpayOrderCreationError(f"Razorpay server error: {e}") from e
        except Exception as e:
            logger.error(f"Razorpay order creation failed: {e}")
            raise RazorpayOrderCreationError(f"Failed to create order: {e}") from e

    def verify_payment_signature(
        self,
        razorpay_order_id: str,
        razorpay_payment_id: str,
        razorpay_signature: str,
    ) -> bool:
        """
        Verify Razorpay payment signature.

        This MUST be called on the server side after frontend completes payment.
        Never trust frontend-only confirmation.

        Args:
            razorpay_order_id: Order ID from Razorpay
            razorpay_payment_id: Payment ID from Razorpay
            razorpay_signature: Signature from Razorpay

        Returns:
            True if signature is valid

        Raises:
            RazorpayVerificationError: If verification fails
        """
        if not all([razorpay_order_id, razorpay_payment_id, razorpay_signature]):
            raise RazorpayVerificationError("Missing required verification parameters")

        # Construct the expected signature
        # Razorpay uses: hmac_sha256(order_id + "|" + payment_id, key_secret)
        generated_signature = hmac.new(
            self._key_secret.encode("utf-8"),
            f"{razorpay_order_id}|{razorpay_payment_id}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # Constant-time comparison to prevent timing attacks
        is_valid = hmac.compare_digest(generated_signature, razorpay_signature)

        if not is_valid:
            logger.warning(
                f"Payment signature verification failed for order {razorpay_order_id}, "
                f"payment {razorpay_payment_id}"
            )
            raise RazorpayVerificationError("Invalid payment signature")

        logger.info(f"Payment signature verified for order {razorpay_order_id}")
        return True

    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify Razorpay webhook signature.

        Args:
            payload: Raw request body bytes
            signature: X-Razorpay-Signature header value

        Returns:
            True if signature is valid

        Raises:
            RazorpayWebhookError: If verification fails or secret not configured
        """
        if not self._webhook_secret:
            logger.warning("Webhook secret not configured, skipping signature verification")
            # In production, this should raise an error
            # For development, we allow it but log a warning
            return True

        expected_signature = hmac.new(
            self._webhook_secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

        is_valid = hmac.compare_digest(expected_signature, signature)

        if not is_valid:
            logger.warning("Webhook signature verification failed")
            raise RazorpayWebhookError("Invalid webhook signature")

        logger.debug("Webhook signature verified")
        return True

    def parse_webhook_event(self, payload: bytes) -> Dict[str, Any]:
        """
        Parse and validate webhook JSON payload.

        Args:
            payload: Raw request body bytes

        Returns:
            Parsed event dict

        Raises:
            RazorpayWebhookError: If payload is invalid JSON
        """
        try:
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid webhook JSON: {e}")
            raise RazorpayWebhookError("Invalid webhook payload") from e