"""
CustomerAddress model: Persistent address book for customers.

Each customer can have multiple saved addresses with an optional label.
At most one address per customer can be the default (is_default=True).
This is enforced at the application layer in AddressService.

IMPORTANT: When an order is placed, the address fields are SNAPSHOT-copied
onto the Order row. This ensures historical orders are immutable even if
the customer later edits or deletes their saved address.
"""

import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Boolean, ForeignKey, DateTime, func, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.customer import Customer


class CustomerAddress(Base):
    __tablename__ = "customer_addresses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Human-readable label for the address (e.g. "Home", "Office", "Parents")
    label: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # Name of the person receiving the delivery (defaults to customer name if null)
    recipient_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    address_line_1: Mapped[str] = mapped_column(String(500), nullable=False)
    address_line_2: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    state: Mapped[str] = mapped_column(String(100), nullable=False)
    postal_code: Mapped[str] = mapped_column(String(20), nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False, default="India")

    # At most ONE address per customer should have is_default=True.
    # AddressService.save_address / set_default clear others before setting this.
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationship
    customer: Mapped["Customer"] = relationship("Customer", back_populates="addresses")
