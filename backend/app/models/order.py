import uuid
from decimal import Decimal
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import String, Integer, Numeric, ForeignKey, DateTime, func, Enum as SQLEnum, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import OrderStatus, PaymentStatus

if TYPE_CHECKING:
    from app.models.merchant import Merchant
    from app.models.customer import Customer
    from app.models.cart import Cart
    from app.models.product import ProductVariant
    from app.models.customer_address import CustomerAddress


class Order(Base):
    __tablename__ = "orders"

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
    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    cart_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("carts.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    total_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    status: Mapped[OrderStatus] = mapped_column(
        SQLEnum(OrderStatus, native_enum=False),
        default=OrderStatus.PENDING,
        nullable=False,
        index=True
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

    # -----------------------------------------------------------------------
    # Shipping address snapshot
    # These fields are COPIED from CustomerAddress at order-creation time.
    # They are intentionally denormalized so that historical orders remain
    # correct even if the customer later edits or deletes their saved address.
    # -----------------------------------------------------------------------
    customer_address_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customer_addresses.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    shipping_recipient_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    shipping_address_line_1: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    shipping_address_line_2: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    shipping_city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    shipping_state: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    shipping_postal_code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    shipping_country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Relationships
    merchant: Mapped["Merchant"] = relationship("Merchant", back_populates="orders")
    customer: Mapped["Customer"] = relationship("Customer", back_populates="orders")
    cart: Mapped[Optional["Cart"]] = relationship("Cart", back_populates="orders")
    customer_address: Mapped[Optional["CustomerAddress"]] = relationship("CustomerAddress")
    items: Mapped[List["OrderItem"]] = relationship(
        "OrderItem",
        back_populates="order",
        cascade="all, delete-orphan"
    )
    payments: Mapped[List["Payment"]] = relationship(
        "Payment",
        back_populates="order",
        cascade="all, delete-orphan"
    )


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    order_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    variant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("product_variants.id", ondelete="RESTRICT"),
        nullable=False,
        index=True
    )
    quantity: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationships
    order: Mapped["Order"] = relationship("Order", back_populates="items")
    variant: Mapped["ProductVariant"] = relationship("ProductVariant", back_populates="order_items")


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    order_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    razorpay_order_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    razorpay_payment_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    razorpay_signature: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(10), default="INR", nullable=False)
    status: Mapped[PaymentStatus] = mapped_column(
        SQLEnum(PaymentStatus, native_enum=False),
        default=PaymentStatus.PENDING,
        nullable=False,
        index=True
    )
    method: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
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
    order: Mapped["Order"] = relationship("Order", back_populates="payments")
