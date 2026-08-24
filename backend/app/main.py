import sys
from pathlib import Path



from fastapi import FastAPI
from app.core.config import settings
from app.api.health import router as health_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Growearn - AI Commerce Agent API for SMB Merchants",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register health check routes
app.include_router(health_router)


