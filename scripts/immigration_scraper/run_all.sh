#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Thai Immigration Law — Full pipeline runner
# Run once to populate Supabase with all law chunks
#
# Prerequisites:
#   1. Run 01_setup_pgvector.sql in Supabase SQL Editor
#   2. Set env vars: OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY
#   3. pip install -r requirements_scraper.txt
# ══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env from backend root
if [ -f ../../.env ]; then
  export $(grep -v '^#' ../../.env | xargs)
  echo "✓ Loaded .env"
fi

# Check required env vars
if [ -z "$OPENAI_API_KEY" ]; then echo "✗ OPENAI_API_KEY not set"; exit 1; fi
if [ -z "$SUPABASE_URL" ]; then echo "✗ SUPABASE_URL not set"; exit 1; fi
if [ -z "$SUPABASE_SERVICE_KEY" ]; then echo "✗ SUPABASE_SERVICE_KEY not set"; exit 1; fi

echo ""
echo "═══════════════════════════════════════════════"
echo "  Thai Immigration Law RAG Pipeline"
echo "═══════════════════════════════════════════════"
echo ""

# Pick the right Python — prefer the macOS system python at /usr/bin/python3
# (currently 3.9) over user-installed Python 3.14 from python.org. We've seen
# 3.14 + this set of native deps (httpx/pypdf/pdfminer) get SIGKILL'd by the
# OS during PDF processing — likely a memory leak in a C-extension that's
# not yet 3.14-compatible. 3.9 ships with macOS so it's always available.
if [ -x /usr/bin/python3 ]; then
  PY="/usr/bin/python3"
elif [ -x /opt/homebrew/bin/python3.12 ]; then
  PY="/opt/homebrew/bin/python3.12"
elif [ -x /opt/homebrew/bin/python3.11 ]; then
  PY="/opt/homebrew/bin/python3.11"
else
  PY="$(command -v python3 || command -v python)"
fi
if [ -z "$PY" ]; then
  echo "✗ Python not found. Install python3 first."
  exit 1
fi
PY_VER="$("$PY" --version 2>&1)"
echo "  Using: $PY ($PY_VER)"

# Ensure deps are installed in this interpreter's site-packages.
# macOS system /usr/bin/python3 requires --user, so try that as a fallback.
"$PY" -c "import httpx, pypdf, openai, supabase" 2>/dev/null || {
  echo "  Installing scraper deps into $PY..."
  "$PY" -m pip install --user -q -r requirements_scraper.txt 2>&1 | tail -8 || {
    echo "  ✗ pip --user failed; trying regular install..."
    "$PY" -m pip install -q -r requirements_scraper.txt 2>&1 | tail -8
  }
  # Verify
  "$PY" -c "import httpx, pypdf, openai, supabase" 2>&1 | head -3
}
echo ""

# ── Step 1: Scrape all sources ───────────────────
echo "📡 Step 1/2 — Scraping all sources..."
"$PY" scraper.py --source all --output ./raw_chunks.jsonl

CHUNK_COUNT=$(wc -l < raw_chunks.jsonl)
echo ""
echo "  ✓ Scraped $CHUNK_COUNT chunks → raw_chunks.jsonl"
echo ""

# ── Step 2: Embed + upload to Supabase ──────────
echo "🧠 Step 2/2 — Embedding & uploading to Supabase..."
"$PY" embedder.py --input ./raw_chunks.jsonl --batch 50

echo ""
echo "✅ Pipeline complete!"
echo "   Law chunks are now searchable in Supabase law_chunks table."
echo "   The analysis endpoint will auto-retrieve relevant laws on immigration queries."
echo ""
