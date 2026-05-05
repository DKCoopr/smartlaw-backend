import os
from functools import lru_cache


class Settings:
    def __init__(self):
        self.app_env = os.environ.get("APP_ENV", "development")
        self.app_secret_key = os.environ.get("APP_SECRET_KEY", "change-me")
        self.frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5173")

        self.supabase_url = os.environ.get("SUPABASE_URL", "")
        self.supabase_anon_key = os.environ.get("SUPABASE_ANON_KEY", "")
        self.supabase_service_key = os.environ.get("SUPABASE_SERVICE_KEY", "")

        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.google_api_key = os.environ.get("GOOGLE_API_KEY", "")

        self.redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self.max_audio_mb = int(os.environ.get("MAX_AUDIO_MB", "25"))


@lru_cache()
def get_settings() -> Settings:
    return Settings()
