"""
Agent API schemas for the AI Shopping Agent.
"""

import uuid
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AgentProductSummary(BaseModel):
    """Compact product summary returned to the LLM and frontend."""
    
    id: uuid.UUID
    title: str
    price: Decimal
    color: Optional[str] = None
    size: Optional[str] = None
    in_stock: bool
    position: int = Field(..., description="1-based position in search results for follow-up reference")
    variant_id: Optional[uuid.UUID] = Field(None, description="Primary variant ID for direct add-to-cart")


class AgentChatRequest(BaseModel):
    """Request for the agent chat endpoint."""
    
    message: str = Field(..., min_length=1, max_length=2000, description="User's message")
    session_id: str = Field(..., description="Session identifier")


class AgentChatResponse(BaseModel):
    """Response from the agent chat endpoint."""
    
    session_id: str
    message: str
    products: List[AgentProductSummary] = Field(default_factory=list)
    cart_updated: bool = False
    product_detail: Optional[Dict[str, Any]] = None
    cart_summary: Optional[Dict[str, Any]] = None
    # Checkout summary shown to user before confirmation.
    # Contains: product(s), variant, quantity, unit_price, subtotal, address, total.
    # Does NOT contain Razorpay fields (payment not yet integrated).
    checkout_summary: Optional[Dict[str, Any]] = None
    # Populated when the agent is mid-checkout flow (e.g. waiting for size or address)
    checkout_state: Optional[Dict[str, Any]] = None
    # True when the product has multiple variants and user hasn't specified one yet
    needs_variant_selection: bool = False
    # Available variants to select from (populated when needs_variant_selection=True)
    available_variants: List[Dict[str, Any]] = Field(default_factory=list)


class AgentToolSearchInput(BaseModel):
    """Input for search_products tool."""
    
    query: str = Field(..., min_length=1, max_length=500)


class AgentToolGetProductInput(BaseModel):
    """Input for get_product tool."""
    
    product_id: uuid.UUID


class AgentToolAddToCartInput(BaseModel):
    """Input for add_to_cart tool."""
    
    reference_position: Optional[int] = Field(None, ge=1, le=50, description="Position in the most recent search results (1-based)")
    product_id: Optional[uuid.UUID] = Field(None, description="Product UUID (alternative to reference_position)")
    variant_id: Optional[uuid.UUID] = Field(None, description="Product variant UUID (required if using product_id)")
    quantity: int = Field(default=1, ge=1, le=99)


class AgentToolRemoveFromCartInput(BaseModel):
    """Input for remove_from_cart tool."""

    item_position: Optional[int] = Field(None, ge=1, le=50, description="1-based position in customer's cart")
    product_name: Optional[str] = Field(None, description="Product name or keyword to remove")
    cart_item_id: Optional[str] = Field(None, description="Specific cart item UUID")
    remove_all: Optional[bool] = Field(False, description="Whether to clear all items from cart")


class AgentToolCheckoutSingleProductInput(BaseModel):
    """Input for checkout_single_product tool."""
    
    reference_position: Optional[int] = Field(None, ge=1, le=50, description="1-based position in last search results")
    product_id: Optional[str] = Field(None, description="Explicit product UUID (alternative to reference_position)")
    quantity: int = Field(default=1, ge=1, le=99)
    size: Optional[str] = Field(None, description="Size/variant selection (e.g. 'L', 'M', 'XL')")
    address_hint: Optional[str] = Field(None, description="Address hint: 'default', 'home', 'office', '2', etc.")


class AgentToolCheckoutCartInput(BaseModel):
    """Input for checkout_cart tool."""
    
    address_hint: Optional[str] = Field(None, description="Address hint: 'default', 'home', 'office', '2', etc.")


class AgentToolConfirmCheckoutInput(BaseModel):
    """Input for confirm_checkout tool - final confirmation step."""
    
    confirm: bool = Field(..., description="True = user confirmed, proceed to create order")