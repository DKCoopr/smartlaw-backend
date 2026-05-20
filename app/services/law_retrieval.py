"""
Law Retrieval Service — RAG layer before Claude/GPT analysis

Flow:
  1. Embed the incoming query/transcript with OpenAI text-embedding-3-small
  2. Search Supabase pgvector for top-k relevant immigration law chunks
  3. Return formatted context string to inject into the Claude prompt
"""

import logging
from typing import Optional
import openai
from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

EMBED_MODEL     = "text-embedding-3-small"
TOP_K           = 6       # number of law chunks to retrieve per query
SIMILARITY_THRESHOLD = 0.68

# Keywords that signal an immigration-related query
IMMIGRATION_KEYWORDS = [
    "วีซ่า", "visa", "ตม", "immigration", "คนเข้าเมือง", "ต่ออายุ", "extension",
    "90 วัน", "90day", "90-day", "tm30", "tm47", "แจ้งที่พัก", "ใบอนุญาต",
    "work permit", "ใบอนุญาตทำงาน", "non-immigrant", "tourist visa", "retirement",
    "ลี้ภัย", "ผู้ลี้ภัย", "deportation", "blacklist", "ดำเนินคดี ตม",
    "overstay", "อยู่เกิน", "ค่าธรรมเนียม ตม", "residence permit",
]


def _is_immigration_related(text: str) -> bool:
    """Quick keyword check — avoid embedding cost when not relevant."""
    lower = text.lower()
    return any(kw.lower() in lower for kw in IMMIGRATION_KEYWORDS)


def _get_openai_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=settings.openai_api_key)


def _get_supabase():
    from supabase import create_client
    return create_client(settings.supabase_url, settings.supabase_service_key)


async def retrieve_relevant_laws(
    query: str,
    category: Optional[str] = None,
    force: bool = False,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Embed query → vector search in law_chunks → return top-k results.

    Args:
        query:    The user's question or case transcript
        category: Optional filter ('visa','extension','90day','tm30','csoc','fees')
        force:    Skip keyword check and always retrieve
        top_k:    Max chunks to return

    Returns:
        List of dicts with keys: law_title, section_ref, chunk_text,
                                  source_name, source_url, similarity
    """
    if not force and not _is_immigration_related(query):
        return []

    if not settings.openai_api_key or not settings.supabase_url:
        log.warning("law_retrieval: missing OpenAI or Supabase config — skipping RAG")
        return []

    try:
        # 1. Embed the query
        oai = _get_openai_client()
        response = oai.embeddings.create(model=EMBED_MODEL, input=[query])
        query_vector = response.data[0].embedding

        # 2. Search Supabase
        sb = _get_supabase()
        result = sb.rpc("search_law_chunks", {
            "query_embedding": query_vector,
            "match_threshold": SIMILARITY_THRESHOLD,
            "match_count": top_k,
            "filter_category": category,
        }).execute()

        chunks = result.data or []
        log.info(f"law_retrieval: found {len(chunks)} relevant law chunks for query")
        return chunks

    except Exception as e:
        log.warning(f"law_retrieval: RAG failed (non-fatal) — {e}")
        return []


def format_law_context(chunks: list[dict], lang: str = "th") -> str:
    """
    Format retrieved chunks into a context block to prepend to the Claude prompt.

    Returns empty string if no chunks (no-op for non-immigration cases).
    """
    if not chunks:
        return ""

    if lang == "en":
        header = "=== RELEVANT THAI IMMIGRATION LAW (retrieved) ===\n"
        footer = "\n=== END OF LAW REFERENCES ===\n"
        fmt = "[{i}] {title}{sec}\nSource: {src}\n{text}"
    else:
        header = "=== กฎหมาย ตม. ที่เกี่ยวข้อง (ดึงจากฐานข้อมูล) ===\n"
        footer = "\n=== สิ้นสุดข้อมูลกฎหมาย ===\n"
        fmt = "[{i}] {title}{sec}\nที่มา: {src}\n{text}"

    lines = [header]
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("law_title") or ""
        sec   = f" — {chunk['section_ref']}" if chunk.get("section_ref") else ""
        src   = chunk.get("source_name", "")
        text  = chunk.get("chunk_text", "").strip()
        lines.append(fmt.format(i=i, title=title, sec=sec, src=src, text=text))
        lines.append("")

    lines.append(footer)
    return "\n".join(lines)


async def get_law_context_for_prompt(
    query: str,
    category: Optional[str] = None,
    lang: str = "th",
) -> str:
    """
    Convenience wrapper: retrieve + format in one call.
    Returns empty string if not immigration-related or RAG unavailable.
    """
    chunks = await retrieve_relevant_laws(query, category=category)
    return format_law_context(chunks, lang=lang)
