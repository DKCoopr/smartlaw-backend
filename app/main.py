"""
SmartLaw Backend — FastAPI Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.routes import transcribe, analyze, cases, documents, billings, transactions, legal

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    import os
    print("🚀 SmartLaw API starting up...")
    print(f"   RAW APP_ENV from os.environ: {os.environ.get('APP_ENV', 'NOT_FOUND')}")
    print(f"   RAW ANTHROPIC from os.environ: {os.environ.get('ANTHROPIC_API_KEY', 'NOT_FOUND')[:10] if os.environ.get('ANTHROPIC_API_KEY') else 'NOT_FOUND'}")
    print(f"   Environment: {settings.app_env}")
    print(f"   Supabase: {'✅ configured' if settings.supabase_url else '❌ missing'}")
    print(f"   OpenAI:   {'✅ configured' if settings.openai_api_key else '❌ missing'}")
    print(f"   Claude:   {'✅ configured' if settings.anthropic_api_key else '❌ missing'}")
    print(f"   Gemini:   {'✅ configured' if settings.google_api_key else '❌ missing'}")
    yield
    print("SmartLaw API shutting down...")


app = FastAPI(
    title="SmartLaw API",
    description="Thai Legal AI — Voice to Police Report Pipeline",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc",     # ReDoc at /redoc
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Allow the React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:5173",    # Vite dev server
        "http://localhost:3000",    # Alternative dev port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(transcribe.router)
app.include_router(analyze.router)
app.include_router(cases.router)
app.include_router(documents.router)
app.include_router(billings.router)
app.include_router(transactions.router)
app.include_router(legal.router)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "SmartLaw API",
        "version": "1.0.0",
        "env": settings.app_env,
    }


@app.get("/")
async def root():
    return {
        "message": "SmartLaw API is running",
        "docs": "/docs",
        "health": "/health",
    }
