"""
Agent API router: HTTP endpoint for the AI Shopping Agent.

POST /api/agent/chat - Conversational shopping with tool orchestration.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.config import settings
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
# Supports X-Customer-Id header, customer_id query param, and dev fallback.
# ---------------------------------------------------------------------------

def get_customer_context(
    x_customer_id: Optional[str] = Header(None, alias="X-Customer-Id", description="Customer UUID header"),
    customer_id: Optional[str] = Query(None, description="Customer UUID query parameter"),
    merchant_context: MerchantContext = Depends(get_merchant_context),
    db: Session = Depends(get_db),
) -> Customer | None:
    """
    Resolve customer context for the request.
    
    Resolution order:
    1. HTTP Header 'X-Customer-Id'
    2. Query parameter 'customer_id'
    3. Demo/Development Fallback: Primary active customer for merchant (if DEBUG is enabled)
    """
    target_id_str = x_customer_id or customer_id
    if target_id_str:
        try:
            customer_uuid = uuid.UUID(str(target_id_str).strip())
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid customer ID format: '{target_id_str}'. Must be a valid UUID.",
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

    # Development / Demo mode fallback: find primary customer for this merchant
    if getattr(settings, "DEBUG", False):
        fallback_customer = db.query(Customer).filter(
            Customer.merchant_id == merchant_context.merchant_id
        ).order_by(Customer.created_at.asc()).first()
        if fallback_customer:
            return fallback_customer

    return None


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
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower() or "rate limit" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="AI service quota / rate limit reached. Please retry in a few moments.",
            ) from e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent processing failed. Please try again.",
        ) from e