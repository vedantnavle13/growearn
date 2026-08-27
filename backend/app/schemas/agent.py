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