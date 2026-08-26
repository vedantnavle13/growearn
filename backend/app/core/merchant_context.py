"""
Merchant Context abstraction and resolution for multi-tenant isolation.

Ensures that every request explicitly resolves a MerchantContext containing
the target merchant_id. No global mutable state is used.
"""

import uuid
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.merchant import Merchant


@dataclass(frozen=True)
class MerchantContext:
    """
    Immutable merchant context representing the current tenant boundary.
    """
    merchant_id: uuid.UUID
    merchant_name: Optional[str] = None


def get_merchant_context(
    x_merchant_id: Optional[str] = Header(None, alias="X-Merchant-Id", description="Merchant UUID header"),
    merchant_id: Optional[str] = Query(None, description="Merchant UUID query parameter"),
    db: Session = Depends(get_db),
) -> MerchantContext:
    """
    Resolves the active merchant context for the request.
    
    Resolution order:
    1. HTTP Header 'X-Merchant-Id'
    2. Query parameter 'merchant_id'
    3. Demo/Development Fallback: Active merchant from database (e.g. UrbanThreads)
    """
    target_id_str = x_merchant_id or merchant_id

    if target_id_str:
        try:
            target_uuid = uuid.UUID(str(target_id_str).strip())
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid merchant ID format: '{target_id_str}'. Must be a valid UUID.",
            )

        merchant = db.query(Merchant).filter(Merchant.id == target_uuid, Merchant.is_active.is_(True)).first()
        if not merchant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Merchant with ID '{target_uuid}' not found or inactive.",
            )
        return MerchantContext(merchant_id=merchant.id, merchant_name=merchant.name)

    # Fallback for Demo / Development Mode: find primary active merchant (e.g. UrbanThreads)
    merchant = db.query(Merchant).filter(Merchant.is_active.is_(True)).order_by(Merchant.created_at.asc()).first()
    if merchant:
        return MerchantContext(merchant_id=merchant.id, merchant_name=merchant.name)

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="No active merchant found. Please provide an 'X-Merchant-Id' header.",
    )
