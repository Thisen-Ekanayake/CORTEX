"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Global application settings loaded from .env or environment."""

    # Database
    database_url: str = "postgresql://cortex:cortex@localhost:5432/cortex"

    # JWT
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # App
    app_name: str = "CORTEX API"
    debug: bool = True
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = {"env_file": "../.env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
