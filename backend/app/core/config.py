from typing import Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Growearn API"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # PostgreSQL Connection Parameters
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "growearn_db"
    DATABASE_URL: Optional[str] = None

    # LLM Provider Configuration ('huggingface' or 'gemini')
    LLM_PROVIDER: str = "huggingface"

    # Hugging Face Configuration
    HF_TOKEN: Optional[str] = None
    HUGGINGFACEHUB_ACCESS_TOKEN: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    HF_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct"
    HF_API_BASE: Optional[str] = None

    @model_validator(mode="after")
    def sync_hf_token(self):
        if not self.HF_TOKEN:
            self.HF_TOKEN = self.HUGGINGFACEHUB_ACCESS_TOKEN or self.HUGGINGFACE_API_KEY
        return self

    # Gemini AI & Multimodal Embedding Configuration
    GEMINI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "gemini-embedding-2"
    EMBEDDING_DIMENSION: int = 1536
    GEMINI_INTENT_MODEL: str = "gemini-2.5-flash"

    # Search limits
    DEFAULT_SEARCH_LIMIT: int = 10
    MAX_SEARCH_LIMIT: int = 50

    # Razorpay Configuration
    RAZORPAY_KEY_ID: Optional[str] = None
    RAZORPAY_KEY_SECRET: Optional[str] = None
    RAZORPAY_WEBHOOK_SECRET: Optional[str] = None

    @property
    def sync_database_url(self) -> str:
        """Returns the active database URL or constructs one from components."""
        url = self.DATABASE_URL
        if not url:
            url = (
                f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        # Render often provides URLs starting with 'postgres://' which SQLAlchemy requires as 'postgresql://'
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
