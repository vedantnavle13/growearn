"""API routers and endpoints package."""

from app.api.agent import router as agent_router
from app.api.health import router as health_router
from app.api.intent import router as intent_router
from app.api.products import router as products_router

__all__ = [
    "agent_router",
    "health_router",
    "intent_router",
    "products_router",
]
