from supabase import create_client, Client
from app.config import get_settings
from functools import lru_cache


@lru_cache()
def get_supabase() -> Client:
    """
    Returns a Supabase client using the service role key.
    Service role bypasses Row Level Security — use only server-side.
    """
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_key)


def get_supabase_anon() -> Client:
    """
    Returns a Supabase client using the anon key.
    Respects Row Level Security — use for user-scoped operations.
    """
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_anon_key)
