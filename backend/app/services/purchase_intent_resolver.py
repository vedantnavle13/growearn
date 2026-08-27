"""
PurchaseIntentResolver: Resolves ordinal/position references to actual product/variant UUIDs.

CRITICAL PRINCIPLE:
  The LLM must NEVER invent product IDs or variant IDs.
  It passes reference_position (an integer like 1, 2, 3) and the backend
  resolves it to the actual database UUID from session.last_search_results.

Architecture:
  AgentService
      → PurchaseIntentResolver.resolve_single_product()
          → validates product belongs to merchant
          → checks variant / size
          → validates inventory
          → returns ResolvedPurchaseTarget
"""

import logging
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from app.models.product import Product, ProductVariant, Inventory
from app.models.agent_session import AgentSession

logger = logging.getLogger(__name__)


class PurchaseIntentError(Exception):
    """Raised when purchase intent cannot be resolved."""


class ProductNotFoundError(PurchaseIntentError):
    """Product not found or not available for this merchant."""


class VariantNotFoundError(PurchaseIntentError):
    """Requested variant not found."""


class InsufficientInventoryError(PurchaseIntentError):
    """Not enough inventory for requested quantity."""


@dataclass
class ResolvedPurchaseTarget:
    """
    The result of resolving a purchase intent to concrete database objects.

    If needs_variant_selection=True, the caller must ask the user to pick
    a variant before proceeding. available_variants contains the choices.
    """
    product: Product
    variant: Optional[ProductVariant]          # None = still needs selection
    quantity: int
    unit_price: Decimal
    needs_variant_selection: bool
    available_variants: List[Dict[str, Any]] = field(default_factory=list)


class PurchaseIntentResolver:
    """
    Resolves user purchase references (ordinal positions or explicit IDs)
    to actual Product/ProductVariant database objects.

    Enforces merchant isolation: all resolved products must belong to merchant_id.
    """

    def __init__(self, db: Session, merchant_id: uuid.UUID) -> None:
        self.db = db
        self.merchant_id = merchant_id

    def resolve_single_product(
        self,
        session: AgentSession,
        *,
        reference_position: Optional[int] = None,
        product_id_str: Optional[str] = None,
        quantity: int = 1,
        size: Optional[str] = None,
    ) -> ResolvedPurchaseTarget:
        """
        Resolve a single-product purchase reference to an actual Product + Variant.

        Priority:
          1. reference_position → session.last_search_results[position-1] → product UUID
          2. product_id_str → direct UUID

        The LLM is responsible for passing reference_position for positional references.
        The backend resolves to UUID — the LLM NEVER invents UUIDs.

        Args:
            session: The agent session containing last_search_results.
            reference_position: 1-based position in last search results.
            product_id_str: Explicit product UUID string (alternative to position).
            quantity: Quantity to purchase.
            size: User-specified size/variant (e.g. "L", "M").

        Returns:
            ResolvedPurchaseTarget with product, variant (or None), quantity, and flags.

        Raises:
            PurchaseIntentError subclasses on failure.
        """
        product_uuid: Optional[uuid.UUID] = None
        variant_uuid: Optional[uuid.UUID] = None
        reference_type = "UNKNOWN"

        # ----------------------------------------------------------------
        # Step 1: Resolve product UUID from reference_position or explicit ID
        # ----------------------------------------------------------------
        if reference_position is not None:
            reference_type = "SEARCH_RESULT"
            if not session.last_search_results:
                raise PurchaseIntentError(
                    "No recent search results found. Please search for products first, "
                    "then specify which one to purchase."
                )

            idx = reference_position - 1
            if idx < 0 or idx >= len(session.last_search_results):
                count = len(session.last_search_results)
                raise PurchaseIntentError(
                    f"Position {reference_position} is out of range. "
                    f"Only {count} product(s) were found in the last search."
                )

            search_result = session.last_search_results[idx]
            raw_id = search_result.get("id")
            raw_variant_id = search_result.get("variant_id")

            if not raw_id:
                raise PurchaseIntentError(
                    f"Search result at position {reference_position} is missing a product ID."
                )

            try:
                product_uuid = uuid.UUID(str(raw_id))
            except (ValueError, AttributeError):
                raise PurchaseIntentError(
                    f"Search result at position {reference_position} has an invalid product ID: {raw_id}"
                )

            if raw_variant_id:
                try:
                    variant_uuid = uuid.UUID(str(raw_variant_id))
                except (ValueError, AttributeError):
                    variant_uuid = None  # Will be resolved below

            logger.info(
                f"[PURCHASE_DEBUG] REFERENCE_TYPE=SEARCH_RESULT REFERENCE_POSITION={reference_position} "
                f"RAW_PRODUCT_ID={product_uuid} RAW_VARIANT_ID={variant_uuid}"
            )

        elif product_id_str:
            reference_type = "PRODUCT_ID"
            try:
                product_uuid = uuid.UUID(product_id_str.strip())
            except (ValueError, AttributeError):
                raise PurchaseIntentError(
                    f"Invalid product ID format: '{product_id_str}'. Must be a valid UUID."
                )
            logger.info(f"[PURCHASE_DEBUG] REFERENCE_TYPE=PRODUCT_ID PRODUCT_ID={product_uuid}")

        else:
            raise PurchaseIntentError(
                "No product reference provided. "
                "Specify either reference_position or product_id."
            )

        # ----------------------------------------------------------------
        # Step 2: Load and validate the product
        # ----------------------------------------------------------------
        product = self.db.query(Product).filter(
            Product.id == product_uuid,
            Product.merchant_id == self.merchant_id,
            Product.is_active.is_(True),
        ).first()

        if not product:
            raise ProductNotFoundError(
                f"Product not found or not available in this store. "
                f"(reference_type={reference_type}, product_id={product_uuid})"
            )

        logger.info(
            f"[PURCHASE_DEBUG] RESOLVED_PRODUCT_ID={product.id} PRODUCT_TITLE='{product.title}'"
        )

        # ----------------------------------------------------------------
        # Step 3: Load all variants for this product
        # ----------------------------------------------------------------
        all_variants = self.db.query(ProductVariant).filter(
            ProductVariant.product_id == product.id,
        ).all()

        if not all_variants:
            raise ProductNotFoundError(
                f"Product '{product.title}' has no variants configured."
            )

        # ----------------------------------------------------------------
        # Step 4: Resolve variant
        # ----------------------------------------------------------------
        resolved_variant: Optional[ProductVariant] = None

        if size:
            # User explicitly specified a size — find matching variant
            size_normalized = size.strip().upper()
            size_match = next(
                (v for v in all_variants if v.size and v.size.strip().upper() == size_normalized),
                None,
            )
            if not size_match:
                available_sizes = [v.size for v in all_variants if v.size]
                raise VariantNotFoundError(
                    f"Size '{size}' not available for '{product.title}'. "
                    f"Available sizes: {', '.join(available_sizes) if available_sizes else 'none'}."
                )
            resolved_variant = size_match
            logger.info(f"[PURCHASE_DEBUG] VARIANT_RESOLVED_BY=SIZE SIZE={size} VARIANT_ID={resolved_variant.id}")

        elif variant_uuid:
            # Variant was identified from search result — verify it's valid
            variant_from_search = next(
                (v for v in all_variants if v.id == variant_uuid), None
            )
            if variant_from_search:
                resolved_variant = variant_from_search
                logger.info(f"[PURCHASE_DEBUG] VARIANT_RESOLVED_BY=SEARCH_RESULT VARIANT_ID={resolved_variant.id}")
            # If not found (stale ID), fall through to single-variant check or ask

        if resolved_variant is None:
            if len(all_variants) == 1:
                # Only one variant — no need to ask
                resolved_variant = all_variants[0]
                logger.info(f"[PURCHASE_DEBUG] VARIANT_RESOLVED_BY=SINGLE_VARIANT VARIANT_ID={resolved_variant.id}")
            else:
                # Multiple variants, no size specified → ask user
                logger.info(
                    f"[PURCHASE_DEBUG] NEEDS_VARIANT_SELECTION=True PRODUCT='{product.title}' "
                    f"VARIANT_COUNT={len(all_variants)}"
                )
                available_variant_info = [
                    {
                        "id": str(v.id),
                        "size": v.size,
                        "color": v.color,
                        "price": str(v.price),
                        "in_stock": self._is_in_stock(v, quantity),
                        "available_quantity": self._available_qty(v),
                    }
                    for v in all_variants
                ]
                return ResolvedPurchaseTarget(
                    product=product,
                    variant=None,
                    quantity=quantity,
                    unit_price=product.price,
                    needs_variant_selection=True,
                    available_variants=available_variant_info,
                )

        # ----------------------------------------------------------------
        # Step 5: Validate inventory for resolved variant
        # ----------------------------------------------------------------
        available_qty = self._available_qty(resolved_variant)

        if available_qty <= 0:
            raise InsufficientInventoryError(
                f"'{product.title}' "
                f"({resolved_variant.color or ''} {resolved_variant.size or ''}).strip() "
                f"is currently out of stock."
            )

        if quantity > available_qty:
            raise InsufficientInventoryError(
                f"Only {available_qty} unit(s) available for "
                f"'{product.title}' ({resolved_variant.color or ''} {resolved_variant.size or ''}). "
                f"You requested {quantity}."
            )

        logger.info(
            f"[PURCHASE_DEBUG] RESOLVED_VARIANT_ID={resolved_variant.id} "
            f"QUANTITY={quantity} AVAILABLE={available_qty} UNIT_PRICE={resolved_variant.price}"
        )

        return ResolvedPurchaseTarget(
            product=product,
            variant=resolved_variant,
            quantity=quantity,
            unit_price=resolved_variant.price,
            needs_variant_selection=False,
            available_variants=[],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _available_qty(self, variant: ProductVariant) -> int:
        """Return the available (unreserved) quantity for a variant."""
        if variant.inventory:
            return variant.inventory.quantity - variant.inventory.reserved_quantity
        return 0

    def _is_in_stock(self, variant: ProductVariant, quantity: int = 1) -> bool:
        return self._available_qty(variant) >= quantity
