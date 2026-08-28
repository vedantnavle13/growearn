import sys
from pathlib import Path

from fastapi import FastAPI
from app.core.config import settings
from app.api.health import router as health_router
from app.api.products import router as products_router
from app.api.intent import router as intent_router
from app.api.agent import router as agent_router
from app.api.checkout import router as checkout_router
from app.api.payment_verification import router as payment_verification_router
from app.api.webhook import router as webhook_router

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Growearn - AI Commerce Agent API for SMB Merchants",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(products_router)
app.include_router(intent_router)
app.include_router(agent_router)
app.include_router(checkout_router)
app.include_router(payment_verification_router)
app.include_router(webhook_router)
