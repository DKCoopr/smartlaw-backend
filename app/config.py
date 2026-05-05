from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_env: str = "development"
    app_secret_key: str = "change-me"
    frontend_url: str = "http://localhost:5173"

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_key: str = ""

    # AI APIs
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Limits
    max_audio_mb: int = 25

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
