"""
Embedder — reads raw_chunks.jsonl → embeds with OpenAI → upserts to Supabase

Usage:
  python embedder.py --input ./raw_chunks.jsonl
  python embedder.py --input ./raw_chunks.jsonl --batch 50 --dry-run

Requires env vars (or .env in backend root):
  OPENAI_API_KEY
  SUPABASE_URL
  SUPABASE_SERVICE_KEY
"""

import asyncio
import json
import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Load .env from backend root if running locally
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

import openai
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EMBED_MODEL   = "text-embedding-3-small"   # 1536-dim, $0.02/1M tokens — very cheap
BATCH_SIZE    = 50                          # OpenAI allows up to 2048 inputs per call
RATE_LIMIT_S  = 0.1                         # seconds between batches (avoid 429)


def get_clients() -> tuple[openai.OpenAI, Client]:
    oai_key = os.environ.get("OPENAI_API_KEY", "")
    sb_url  = os.environ.get("SUPABASE_URL", "")
    sb_key  = os.environ.get("SUPABASE_SERVICE_KEY", "")

    if not oai_key:
        sys.exit("✗ OPENAI_API_KEY not set")
    if not sb_url or not sb_key:
        sys.exit("✗ SUPABASE_URL or SUPABASE_SERVICE_KEY not set")

    oai = openai.OpenAI(api_key=oai_key)
    sb  = create_client(sb_url, sb_key)
    return oai, sb


def load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    log.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def embed_batch(oai: openai.OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of 1536-dim vectors."""
    response = oai.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def upsert_batch(sb: Client, rows: list[dict]) -> int:
    """Upsert a batch of rows into law_chunks. Returns count inserted."""
    result = sb.table("law_chunks").upsert(rows, on_conflict="source_url,chunk_index").execute()
    return len(result.data) if result.data else 0


def run(input_path: str, batch_size: int, dry_run: bool, resume_from: int):
    oai, sb = get_clients()
    chunks = load_chunks(input_path)

    if resume_from > 0:
        log.info(f"Resuming from chunk {resume_from}")
        chunks = chunks[resume_from:]

    total       = len(chunks)
    embedded    = 0
    failed      = 0
    cost_tokens = 0

    log.info(f"Embedding {total} chunks with {EMBED_MODEL} (batch={batch_size})...")
    if dry_run:
        log.info("DRY RUN — no data will be written to Supabase")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["chunk_text"] for c in batch]

        try:
            vectors = embed_batch(oai, texts)
            cost_tokens += sum(len(t.split()) for t in texts)  # rough estimate
        except Exception as e:
            log.error(f"  ✗ Embed failed batch {i}–{i+len(batch)}: {e}")
            failed += len(batch)
            time.sleep(2)
            continue

        if not dry_run:
            rows = []
            for chunk, vector in zip(batch, vectors):
                row = {
                    "source_url":  chunk["source_url"],
                    "source_name": chunk["source_name"],
                    "law_title":   chunk.get("law_title", ""),
                    "section_ref": chunk.get("section_ref", ""),
                    "chunk_text":  chunk["chunk_text"],
                    "chunk_index": chunk["chunk_index"],
                    "language":    chunk.get("language", "th"),
                    "category":    chunk.get("category", "general"),
                    "embedding":   vector,
                    "metadata":    chunk.get("metadata", {}),
                }
                rows.append(row)

            try:
                count = upsert_batch(sb, rows)
                embedded += count
            except Exception as e:
                log.error(f"  ✗ Upsert failed batch {i}: {e}")
                failed += len(batch)
                continue
        else:
            embedded += len(batch)

        progress = (i + len(batch)) / total * 100
        log.info(f"  [{progress:5.1f}%] batch {i//batch_size + 1} — {len(batch)} chunks embedded")
        time.sleep(RATE_LIMIT_S)

    # Rough cost estimate: text-embedding-3-small = $0.020 per 1M tokens
    est_cost = cost_tokens / 1_000_000 * 0.020
    log.info(f"\n✅ Done — {embedded} embedded, {failed} failed")
    log.info(f"   ~{cost_tokens:,} tokens ≈ ${est_cost:.4f} USD")
    if dry_run:
        log.info("   (dry run — nothing written to Supabase)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed law chunks into Supabase pgvector")
    parser.add_argument("--input",       default="./raw_chunks.jsonl")
    parser.add_argument("--batch",       type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Skip first N chunks (resume interrupted run)")
    args = parser.parse_args()
    run(args.input, args.batch, args.dry_run, args.resume_from)
