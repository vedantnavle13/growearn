import uuid
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import String, Text, ForeignKey, DateTime, JSON, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.merchant import Merchant
    from app.models.customer import Customer


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    merchant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("merchants.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    customer_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    session_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )
    current_intent: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    last_search_results: Mapped[Optional[List[dict]]] = mapped_column(JSON, nullable=True)
    # Persists multi-step checkout state across requests.
    # Schema: {"mode": "SINGLE_PRODUCT"|"CART", "step": "awaiting_size"|"awaiting_address"|"awaiting_confirmation",
    #          "resolved_product_id": str|null, "resolved_variant_id": str|null,
    #          "quantity": int, "address_id": str|null, "summary": {...}}
    checkout_state: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    cart_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("carts.id", ondelete="SET NULL"),
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    merchant: Mapped["Merchant"] = relationship("Merchant", back_populates="agent_sessions")
    customer: Mapped[Optional["Customer"]] = relationship("Customer", back_populates="agent_sessions")
    cart: Mapped[Optional["Cart"]] = relationship("Cart", back_populates="agent_sessions")