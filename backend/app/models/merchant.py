import uuid
from datetime import datetime
from typing import List, TYPE_CHECKING
from sqlalchemy import String, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.customer import Customer
    from app.models.product import Product
    from app.models.order import Order
    from app.models.event import Event
    from app.models.agent_action import AgentAction
    from app.models.agent_session import AgentSession


class Merchant(Base):
    __tablename__ = "merchants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    store_name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
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
    customers: Mapped[List["Customer"]] = relationship(
        "Customer",
        back_populates="merchant",
        cascade="all, delete-orphan"
    )
    products: Mapped[List["Product"]] = relationship(
        "Product",
        back_populates="merchant",
        cascade="all, delete-orphan"
    )
    orders: Mapped[List["Order"]] = relationship(
        "Order",
        back_populates="merchant",
        cascade="all, delete-orphan"
    )
    events: Mapped[List["Event"]] = relationship(
        "Event",
        back_populates="merchant",
        cascade="all, delete-orphan"
    )
    agent_actions: Mapped[List["AgentAction"]] = relationship(
        "AgentAction",
        back_populates="merchant",
        cascade="all, delete-orphan"
    )
    agent_sessions: Mapped[List["AgentSession"]] = relationship(
        "AgentSession",
        back_populates="merchant",
        cascade="all, delete-orphan"
    )
