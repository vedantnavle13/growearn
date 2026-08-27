"""
Agent API router: HTTP endpoint for the AI Shopping Agent.

POST /api/agent/chat - Conversational shopping with tool orchestration.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.merchant_context import MerchantContext, get_merchant_context
from app.db.database import get_db
from app.models.customer import Customer
from app.schemas.agent import AgentChatRequest, AgentChatResponse
from app.services.agent_service import AgentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["Agent"])


# ---------------------------------------------------------------------------
# Development/Testing: Customer context resolution
# In production, this would come from authentication (JWT, session, etc.)
# For MVP, we support an optional X-Customer-Id header for testing.
# ---------------------------------------------------------------------------

def get_customer_context(
    x_customer_id: str | None = None,  # Header alias will be set in Depends
    merchant_context: MerchantContext = Depends(get_merchant_context),
    db: Session = Depends(get_db),
) -> Customer | None:
    """
    Resolve customer context for the request.
    
    Resolution order:
    1. HTTP Header 'X-Customer-Id' (for logged-in users)
    2. None (guest user - no cart access for MVP)
    
    Returns None for guest users. In production, replace with auth.
    """
    if not x_customer_id:
        return None

    try:
        customer_uuid = uuid.UUID(str(x_customer_id).strip())
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid customer ID format: '{x_customer_id}'. Must be a valid UUID.",
        )

    customer = db.query(Customer).filter(
        Customer.id == customer_uuid,
        Customer.merchant_id == merchant_context.merchant_id,
    ).first()

    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID '{customer_uuid}' not found in this merchant.",
        )

    return customer


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    summary="Chat with the AI Shopping Agent",
    description=(
        "Conversational shopping interface. The agent can search products, "
        "show product details, view cart, and add items to cart. "
        "Maintains structured session state for follow-up conversations."
    ),
)
async def chat(
    body: AgentChatRequest,
    merchant_context: MerchantContext = Depends(get_merchant_context),
    customer: Customer | None = Depends(get_customer_context),
    db: Session = Depends(get_db),
) -> AgentChatResponse:
    """
    Process a user message through the AI Shopping Agent.
    
    The agent uses Gemini's function calling to orchestrate tools:
    - search_products: Natural language product search
    - get_product: Detailed product information
    - get_cart: View current cart contents
    - add_to_cart: Add a product variant to cart
    
    Session state is maintained server-side for follow-up references
    (e.g., "the second one", "add it to cart").
    """
    try:
        service = AgentService(
            db=db,
            merchant_context=merchant_context,
            customer=customer,
        )

        response = service.chat(
            message=body.message,
            session_id=body.session_id,
        )

        return response

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Agent chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent processing failed. Please try again.",
        ) from e