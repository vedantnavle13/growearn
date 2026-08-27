"""
AddressService: Manages customer address book with security isolation.

SECURITY GUARANTEES:
- All lookups use the trusted customer_id from authenticated session context.
- customer_id is NEVER accepted from the LLM or request body.
- A customer can only access their own addresses.
- set_default ensures at most one default address per customer.
"""

import logging
import uuid
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from app.models.customer_address import CustomerAddress

logger = logging.getLogger(__name__)


class AddressNotFoundError(Exception):
    """Raised when address resolution fails."""


class AddressService:
    """
    Manages the customer address book.

    All public methods accept a trusted customer_id (from authenticated
    session context) and enforce that the customer can only see/modify
    their own addresses.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def list_addresses(self, customer_id: uuid.UUID) -> List[CustomerAddress]:
        """Return all saved addresses for the customer, ordered by creation date."""
        return (
            self.db.query(CustomerAddress)
            .filter(CustomerAddress.customer_id == customer_id)
            .order_by(CustomerAddress.created_at.asc())
            .all()
        )

    def get_default_address(self, customer_id: uuid.UUID) -> Optional[CustomerAddress]:
        """Return the customer's default address, or None if none set."""
        return (
            self.db.query(CustomerAddress)
            .filter(
                CustomerAddress.customer_id == customer_id,
                CustomerAddress.is_default.is_(True),
            )
            .first()
        )

    def get_address_by_label(self, customer_id: uuid.UUID, label: str) -> Optional[CustomerAddress]:
        """Return address matching the label (case-insensitive), or None."""
        return (
            self.db.query(CustomerAddress)
            .filter(
                CustomerAddress.customer_id == customer_id,
                CustomerAddress.label.ilike(label.strip()),
            )
            .first()
        )

    def get_address_by_position(self, customer_id: uuid.UUID, position: int) -> Optional[CustomerAddress]:
        """
        Return the Nth saved address (1-based) for this customer.
        Ordered by creation date for stable ordering.
        """
        addresses = self.list_addresses(customer_id)
        idx = position - 1
        if 0 <= idx < len(addresses):
            return addresses[idx]
        return None

    def get_address_by_id(self, customer_id: uuid.UUID, address_id: uuid.UUID) -> Optional[CustomerAddress]:
        """Return a specific address, verifying it belongs to this customer."""
        return (
            self.db.query(CustomerAddress)
            .filter(
                CustomerAddress.id == address_id,
                CustomerAddress.customer_id == customer_id,
            )
            .first()
        )

    # ------------------------------------------------------------------
    # Address hint resolution
    # ------------------------------------------------------------------

    def resolve_address_hint(
        self, customer_id: uuid.UUID, hint: Optional[str]
    ) -> Optional[CustomerAddress]:
        """
        Resolve a natural-language address hint to a saved CustomerAddress.

        Mapping:
          None / "" / "default"  → default address (is_default=True)
          "home"                 → label="home" (case-insensitive)
          "office" / "work"      → label="office"
          "1" / "2" / integer    → Nth saved address (1-based)
          Any other string       → try label match

        Returns None if:
          - customer has no saved addresses
          - hint doesn't match any address

        The caller decides whether to ask the user for a new address.
        """
        if not hint or hint.strip().lower() in ("default", ""):
            return self.get_default_address(customer_id)

        normalized = hint.strip().lower()

        # Numeric position reference: "2", "3rd", etc.
        # Strip ordinal suffixes for robustness
        cleaned = normalized.rstrip("stndrh")  # removes "st","nd","rd","th"
        if cleaned.isdigit():
            position = int(cleaned)
            addr = self.get_address_by_position(customer_id, position)
            if addr:
                logger.debug(f"[AddressService] Resolved hint='{hint}' to address #{position}: {addr.id}")
                return addr

        # "work" is an alias for "office"
        if normalized == "work":
            normalized = "office"

        # Label match
        addr = self.get_address_by_label(customer_id, normalized)
        if addr:
            logger.debug(f"[AddressService] Resolved hint='{hint}' by label to: {addr.id}")
            return addr

        # Fallback: try default
        default = self.get_default_address(customer_id)
        logger.debug(
            f"[AddressService] hint='{hint}' not matched; fallback to default: {default.id if default else None}"
        )
        return default

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def save_address(
        self,
        customer_id: uuid.UUID,
        data: Dict[str, Any],
        set_as_default: bool = False,
    ) -> CustomerAddress:
        """
        Save a new address for the customer.

        If set_as_default=True (or customer has no addresses yet), clears
        any existing default before marking this one as default.
        """
        existing = self.list_addresses(customer_id)
        # First address ever → auto-set as default
        if not existing:
            set_as_default = True

        if set_as_default:
            self._clear_defaults(customer_id)

        address = CustomerAddress(
            customer_id=customer_id,
            label=data.get("label"),
            recipient_name=data.get("recipient_name"),
            address_line_1=data["address_line_1"],
            address_line_2=data.get("address_line_2"),
            city=data["city"],
            state=data["state"],
            postal_code=data["postal_code"],
            country=data.get("country", "India"),
            is_default=set_as_default,
        )
        self.db.add(address)
        self.db.flush()
        logger.info(f"[AddressService] Saved new address {address.id} for customer {customer_id}, default={set_as_default}")
        return address

    def set_default(self, customer_id: uuid.UUID, address_id: uuid.UUID) -> CustomerAddress:
        """
        Mark the specified address as default. Clears the previous default.
        Raises AddressNotFoundError if address not found or belongs to another customer.
        """
        address = self.get_address_by_id(customer_id, address_id)
        if not address:
            raise AddressNotFoundError(
                f"Address {address_id} not found for customer {customer_id}"
            )

        self._clear_defaults(customer_id)
        address.is_default = True
        self.db.flush()
        logger.info(f"[AddressService] Set default address to {address_id} for customer {customer_id}")
        return address

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _clear_defaults(self, customer_id: uuid.UUID) -> None:
        """Remove is_default from all addresses for this customer."""
        self.db.query(CustomerAddress).filter(
            CustomerAddress.customer_id == customer_id,
            CustomerAddress.is_default.is_(True),
        ).update({"is_default": False})
        self.db.flush()

    # ------------------------------------------------------------------
    # Snapshot helper
    # ------------------------------------------------------------------

    @staticmethod
    def build_snapshot(address: CustomerAddress, customer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Build a flat dict snapshot of the address for embedding into Order fields.
        Uses recipient_name if set, otherwise falls back to customer_name.
        """
        return {
            "recipient_name": address.recipient_name or customer_name or "",
            "address_line_1": address.address_line_1,
            "address_line_2": address.address_line_2,
            "city": address.city,
            "state": address.state,
            "postal_code": address.postal_code,
            "country": address.country,
        }
