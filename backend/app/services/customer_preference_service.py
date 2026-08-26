"""
Customer Preference Service: aggregates historical customer behavior into lightweight preference signals
with strict multi-merchant boundary enforcement.

Responsibilities:
- Reads customer events and order history scoped to a specific merchant_id.
- Supports lookup via internal customer UUID or merchant-scoped external_customer_id.
- Calculates deterministic preference signals (categories, brands, colors, price envelope).
- Returns a structured CustomerPreferences object for ranker integration.

Tenant Isolation Guarantees:
- Customer C123 under Merchant A NEVER accesses or affects Customer C123 under Merchant B.
- All database queries enforce `merchant_id == merchant_id`.
"""

import uuid
from collections import Counter
from decimal import Decimal
from typing import Optional, List, Set

from sqlalchemy.orm import Session, joinedload

from app.models.customer import Customer
from app.models.event import Event
from app.models.order import Order, OrderItem
from app.models.enums import OrderStatus
from app.models.product import Product, ProductVariant
from app.schemas.preference import CustomerPreferences


class CustomerPreferenceService:
    """
    Derives deterministic preference profiles from customer activity within a merchant tenant.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_preferences(
        self,
        *,
        merchant_id: uuid.UUID,
        customer_id: Optional[uuid.UUID] = None,
        external_customer_id: Optional[str] = None,
    ) -> CustomerPreferences:
        """
        Builds a CustomerPreferences profile from historical orders and events scoped to merchant_id.

        Args:
            merchant_id: UUID of the merchant tenant.
            customer_id: Internal UUID of the customer, or None.
            external_customer_id: Merchant's external customer identifier, or None.

        Returns:
            CustomerPreferences: Aggregated preference signals strictly isolated to the merchant.
        """
        if not customer_id and not external_customer_id:
            return CustomerPreferences()

        # -----------------------------------------------------------------
        # Resolve customer within the merchant tenant boundary
        # -----------------------------------------------------------------
        resolved_customer_id: Optional[uuid.UUID] = None

        if customer_id:
            cust = (
                self.db.query(Customer)
                .filter(
                    Customer.id == customer_id,
                    Customer.merchant_id == merchant_id,
                )
                .first()
            )
            if cust:
                resolved_customer_id = cust.id
        elif external_customer_id:
            cust = (
                self.db.query(Customer)
                .filter(
                    Customer.external_customer_id == external_customer_id.strip(),
                    Customer.merchant_id == merchant_id,
                )
                .first()
            )
            if cust:
                resolved_customer_id = cust.id

        if not resolved_customer_id:
            # Customer does not exist under this merchant -> return empty cold-start profile
            return CustomerPreferences(customer_id=customer_id)

        category_weights: Counter[str] = Counter()
        brand_weights: Counter[str] = Counter()
        color_weights: Counter[str] = Counter()
        collected_prices: List[Decimal] = []
        purchased_product_ids: Set[str] = set()
        total_interactions = 0

        # -----------------------------------------------------------------
        # 1. Analyze Completed Orders Scoped to Merchant (Weight: 3.0)
        # -----------------------------------------------------------------
        orders = (
            self.db.query(Order)
            .options(
                joinedload(Order.items)
                .joinedload(OrderItem.variant)
                .joinedload(ProductVariant.product)
            )
            .filter(
                Order.merchant_id == merchant_id,
                Order.customer_id == resolved_customer_id,
                Order.status == OrderStatus.PAID,
            )
            .all()
        )

        for order in orders:
            for item in order.items:
                total_interactions += 1
                variant = item.variant
                if not variant:
                    continue

                product = variant.product
                if product:
                    purchased_product_ids.add(str(product.id))
                    if item.price:
                        collected_prices.append(Decimal(str(item.price)))

                    attrs = product.attributes or {}
                    cat = attrs.get("category")
                    if cat:
                        category_weights[str(cat).strip()] += 3.0

                    brand = attrs.get("brand")
                    if brand:
                        brand_weights[str(brand).strip()] += 3.0

                if variant.color:
                    color_weights[str(variant.color).strip().lower()] += 3.0

        # -----------------------------------------------------------------
        # 2. Analyze Customer Events Scoped to Merchant (Views, Clicks, Cart additions)
        # -----------------------------------------------------------------
        events = (
            self.db.query(Event)
            .filter(
                Event.merchant_id == merchant_id,
                Event.customer_id == resolved_customer_id,
            )
            .order_by(Event.created_at.desc())
            .limit(100)
            .all()
        )

        for ev in events:
            total_interactions += 1
            meta = ev.event_metadata or {}
            event_type = ev.event_type

            # Assign interaction weight
            if event_type == "ADD_TO_CART":
                weight = 2.0
            elif event_type in ("PRODUCT_VIEWED", "PRODUCT_CLICKED"):
                weight = 1.0
            else:
                weight = 0.5

            cat = meta.get("category")
            if cat:
                category_weights[str(cat).strip()] += weight

            brand = meta.get("brand")
            if brand:
                brand_weights[str(brand).strip()] += weight

            color = meta.get("color")
            if color:
                color_weights[str(color).strip().lower()] += weight

            raw_price = meta.get("price") or meta.get("cart_total")
            if raw_price:
                try:
                    collected_prices.append(Decimal(str(raw_price)))
                except Exception:
                    pass

            if ev.entity_type == "product" and ev.entity_id:
                purchased_product_ids.add(str(ev.entity_id))

        # -----------------------------------------------------------------
        # 3. Aggregate Preferences
        # -----------------------------------------------------------------
        preferred_categories = [cat for cat, _ in category_weights.most_common(5)]
        preferred_brands = [b for b, _ in brand_weights.most_common(5)]
        preferred_colors = [c for c, _ in color_weights.most_common(5)]

        price_min: Optional[Decimal] = None
        price_max: Optional[Decimal] = None

        if collected_prices:
            # Calculate 10% below min and 10% above max to create a realistic preference envelope
            min_val = min(collected_prices)
            max_val = max(collected_prices)
            price_min = round(Decimal(str(float(min_val) * 0.9)), 2)
            price_max = round(Decimal(str(float(max_val) * 1.1)), 2)

        return CustomerPreferences(
            customer_id=resolved_customer_id,
            preferred_categories=preferred_categories,
            preferred_brands=preferred_brands,
            preferred_colors=preferred_colors,
            preferred_price_min=price_min,
            preferred_price_max=price_max,
            preferred_product_ids=list(purchased_product_ids),
            total_events_analyzed=total_interactions,
        )
