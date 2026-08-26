"""
Intent API router: HTTP endpoints for commerce intent extraction.

POST /api/intent/parse — parse intent only, no search.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.intent import CommerceIntent, IntentParseRequest
from app.services.intent_service import IntentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/intent", tags=["Intent"])


@router.post(
    "/parse",
    response_model=CommerceIntent,
    summary="Parse commerce intent from natural language",
    description="Extracts structured CommerceIntent from a user query without searching products.",
)
async def parse_intent(body: IntentParseRequest) -> CommerceIntent:
    """
    Parses a natural language query into a structured CommerceIntent.

    Does NOT search products. Returns only the extracted intent.
    """
    try:
        service = IntentService()
        intent = service.extract_intent(body.query)
        return intent
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Intent extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Intent extraction failed. Please try again.")
