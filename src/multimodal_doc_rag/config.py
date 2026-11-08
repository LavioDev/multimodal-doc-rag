"""Application configuration loaded from environment variables and `.env`."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings shared by the API, CLI, and processing pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Multimodal Document RAG"
    environment: str = "development"
    database_url: str = "postgresql+psycopg://rag:rag_dev_only@localhost:5433/multimodal_rag"
    data_dir: str = "data"
    frontend_origin: str = "http://localhost:3000"
    api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "qwen/qwen3-vl-8b-instruct"
    gotenberg_url: str = "http://localhost:3001"
    embedding_model: str = "intfloat/multilingual-e5-small"
    max_document_pages: int = 200
    worker_lease_seconds: int = 300


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings so all components use the same configuration."""
    return Settings()
