from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def get_health():
    """Basic health check endpoint."""
    return {"status": "ok"}


@router.get("/db-health")
def get_db_health(db: Session = Depends(get_db)):
    """Database connectivity health check endpoint executing SELECT 1."""
    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "database": "connected"
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection failed"
        )
