"""
Thai Immigration Law Scraper
Scrapes 7 sources → cleans text → chunks → ready for embedding

Sources:
  1. ratchakitcha.soc.go.th  — Royal Gazette (official PDFs)
  2. immigration.go.th        — Immigration Bureau announcements
  3. krisdika.go.th           — Council of State (bilingual laws)
  4. dol.go.th                — Land Dept (foreign land ownership)
  5. samuiforsale.com         — EN translations / expat guides
  6. aseanlawyer.com          — EN immigration commentary
  7. thailaws.com             — EN translations

Run:  python scraper.py [--source all|ratchakitcha|immigration|krisdika|dol|samui|asean|thai]
      python scraper.py --output ./raw_chunks.jsonl
"""

# PEP 563: defer annotation evaluation so `dict | None`, `list[LawChunk]` etc.
# are treated as strings — works on Python 3.9 without throwing TypeError on
# the PEP-604 `|` union syntax (which became real at runtime only in 3.10).
from __future__ import annotations

import asyncio
import re
import json
import argparse
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import httpx
from bs4 import BeautifulSoup

# ── Optional PDF support ────────────────────────────────────────────────────
try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False
    logging.warning("pdfplumber not installed — PDF scraping disabled. pip install pdfplumber")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHUNK_SIZE   = 800   # characters per chunk (≈ 200 Thai tokens)
CHUNK_OVERLAP = 150  # overlap between chunks

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ThaiLawBot/1.0; +https://thai.law)",
    "Accept-Language": "th,en;q=0.9",
}


@dataclass
class LawChunk:
    source_url:  str
    source_name: str
    law_title:   str
    section_ref: str
    chunk_text:  str
    chunk_index: int
    language:    str   # 'th' | 'en'
    category:    str   # 'visa'|'extension'|'90day'|'tm30'|'csoc'|'general'
    metadata:    dict


# ─── Helpers ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove excess whitespace, zero-width chars, and HTML artifacts."""
    text = re.sub(r'​| |﻿', ' ', text)
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def chunk_text(text: str, source_url: str, source_name: str,
               law_title: str, language: str, category: str,
               metadata: dict | None = None) -> list[LawChunk]:
    """Split text into overlapping chunks and return LawChunk list.

    ⚠️ Bug fixed 2026-05-20: previous version had an infinite-loop bug when
    `chunk` (post sentence-break + strip) happened to be ≤ CHUNK_OVERLAP
    long: `start += len(chunk) - CHUNK_OVERLAP` → 0 or negative → loop never
    advanced → CPU pegged 100% → Mac thermal-shutdown / hard reboot.
    Now `progress = max(1, len(chunk) - CHUNK_OVERLAP)` so `start` always
    moves forward by ≥1 char. Hard iteration cap as belt-and-suspenders.
    """
    if not text or len(text) < 50:
        return []

    metadata = metadata or {}
    chunks = []
    start = 0
    idx = 0
    max_iters = len(text) + 100   # generous upper bound — won't be reached in normal use

    for _ in range(max_iters):
        if start >= len(text):
            break

        end = start + CHUNK_SIZE
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            for sep in ['\n\n', '।', '。', '\n', '. ', ' ']:
                pos = chunk.rfind(sep)
                if pos > CHUNK_SIZE * 0.6:
                    chunk = chunk[:pos + len(sep)]
                    break

        chunk_stripped = chunk.strip()
        if len(chunk_stripped) > 40:
            # Extract section reference from chunk if present
            section_ref = ""
            m = re.search(r'(มาตรา\s*\d+[\w/]*|ข้อ\s*\d+[\w/]*|Section\s+\d+[\w.]*)', chunk_stripped)
            if m:
                section_ref = m.group(0)

            chunks.append(LawChunk(
                source_url=source_url,
                source_name=source_name,
                law_title=law_title,
                section_ref=section_ref,
                chunk_text=chunk_stripped,
                chunk_index=idx,
                language=language,
                category=category,
                metadata=metadata,
            ))
            idx += 1

        # CRITICAL: always advance by ≥1 char to prevent infinite loop.
        progress = max(1, len(chunk) - CHUNK_OVERLAP)
        start += progress

    return chunks


def detect_category(text: str, url: str) -> str:
    """Auto-detect immigration category from content/URL."""
    combined = (text + url).lower()
    if any(k in combined for k in ['90 day', '90-day', '90วัน', '90 วัน', 'tm47', 'tm 47']):
        return '90day'
    if any(k in combined for k in ['tm30', 'tm 30', 'แจ้งที่พัก', 'notify']):
        return 'tm30'
    if any(k in combined for k in ['extension', 'ต่ออายุ', 'ต่อวีซ่า', 'ขยายเวลา']):
        return 'extension'
    if any(k in combined for k in ['visa', 'วีซ่า', 'ประเภทวีซ่า', 'non-immigrant']):
        return 'visa'
    if any(k in combined for k in ['คสช', 'คำสั่งที่', 'ncpo', 'order no']):
        return 'csoc'
    if any(k in combined for k in ['fee', 'ค่าธรรมเนียม', 'ค่าวีซ่า']):
        return 'fees'
    return 'general'


async def fetch_html(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        r = await client.get(url, headers=HEADERS, follow_redirects=True, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        log.warning(f"  ✗ fetch failed {url}: {e}")
        return None


async def fetch_pdf_bytes(client: httpx.AsyncClient, url: str) -> Optional[bytes]:
    try:
        r = await client.get(url, headers=HEADERS, follow_redirects=True, timeout=60)
        r.raise_for_status()
        return r.content
    except Exception as e:
        log.warning(f"  ✗ PDF fetch failed {url}: {e}")
        return None


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes. Uses pypdf — far lighter than pdfplumber
    (pdfplumber loads full layout/char/image cache per page; for batches of
    PDFs that adds up and got the scraper SIGKILL'd by macOS OOM killer).
    pypdf streams text-only and releases per-page state immediately."""
    import io
    import gc
    pages_text = []
    try:
        import pypdf
    except ImportError:
        # Fallback to pdfplumber if pypdf isn't installed
        if not PDF_OK:
            return ""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t: pages_text.append(t)
                    try: page.flush_cache()
                    except Exception: pass
            return "\n\n".join(pages_text)
        except Exception as e:
            log.warning(f"  ✗ pdfplumber fallback failed: {e}")
            return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        n = len(reader.pages)
        for i in range(n):
            try:
                t = reader.pages[i].extract_text() or ""
                if t.strip(): pages_text.append(t)
            except Exception as pe:
                log.warning(f"  ✗ page {i+1}/{n} failed: {pe}")
            if i % 50 == 49:
                gc.collect()
        return "\n\n".join(pages_text)
    except Exception as e:
        log.warning(f"  ✗ pypdf extract failed: {e}")
        return "\n\n".join(pages_text)


# ─── Source 1: ราชกิจจานุเบกษา ───────────────────────────────────────────────

RATCHAKITCHA_URLS = [
    # พ.ร.บ. คนเข้าเมือง 2522 — official PDF
    ("https://ratchakitcha.soc.go.th/documents/1781865.pdf",
     "พ.ร.บ. คนเข้าเมือง พ.ศ. 2522", "general"),
    # กฎกระทรวง ประเภท Visa
    ("https://ratchakitcha.soc.go.th/documents/17136563.pdf",
     "กฎกระทรวง ประเภทการตรวจลงตรา", "visa"),
    # แก้ไขเพิ่มเติม
    ("https://ratchakitcha.soc.go.th/documents/17040814.pdf",
     "พ.ร.บ. คนเข้าเมือง (ฉบับที่ 2)", "general"),
]

# Krisdika bilingual page for Immigration Act
KRISDIKA_IMMIGRATION_URLS = [
    ("https://www.krisdika.go.th/librarian/get?sysid=147486&ext=pdf",
     "พ.ร.บ. คนเข้าเมือง พ.ศ. 2522 (Krisdika)", "th", "general"),
    ("https://www.krisdika.go.th/librarian/get?sysid=147487&ext=pdf",
     "Immigration Act B.E. 2522 (EN)", "en", "general"),
]


async def scrape_ratchakitcha(client: httpx.AsyncClient) -> list[LawChunk]:
    log.info("📜 Scraping ราชกิจจานุเบกษา...")
    chunks = []
    import gc

    for url, title, category in RATCHAKITCHA_URLS:
        log.info(f"  → {title}")
        try:
            pdf_bytes = await fetch_pdf_bytes(client, url)
            if not pdf_bytes:
                log.warning(f"  ✗ No bytes from {url}")
                continue
            log.info(f"    fetched {len(pdf_bytes)} bytes")
            text = extract_pdf_text(pdf_bytes)
            log.info(f"    extracted {len(text)} chars")
            # Release the raw bytes immediately — pdf_bytes can be tens of MB
            del pdf_bytes
            gc.collect()
            if not text:
                log.warning(f"  ✗ No text extracted from {url}")
                continue
            text = clean_text(text)
            c = chunk_text(text, url, "ราชกิจจานุเบกษา", title, "th", category,
                           {"document_type": "royal_gazette"})
            chunks.extend(c)
            _checkpoint(c)   # flush this PDF's chunks to disk immediately
            log.info(f"  ✓ {len(c)} chunks from {title} (checkpointed)")
            del text
            gc.collect()
        except Exception as e:
            log.error(f"  ✗ {title} CRASHED: {type(e).__name__}: {e}")

    # Also scrape the search page for immigration-related documents
    search_url = "https://ratchakitcha.soc.go.th/search?keyword=%E0%B8%84%E0%B8%99%E0%B9%80%E0%B8%82%E0%B9%89%E0%B8%B2%E0%B9%80%E0%B8%A1%E0%B8%B7%E0%B8%AD%E0%B8%87&cat=1"
    html = await fetch_html(client, search_url)
    if html:
        soup = BeautifulSoup(html, "html.parser")
        pdf_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".pdf" in href.lower() and "ratchakitcha" in href.lower():
                pdf_links.append(href if href.startswith("http") else "https://ratchakitcha.soc.go.th" + href)
        log.info(f"  Found {len(pdf_links)} additional PDFs from search (will process 5 max)")
        for idx, link in enumerate(pdf_links[:5]):   # reduced from 10 → 5 to stay light on memory
            log.info(f"    [search {idx+1}/{min(5,len(pdf_links))}] {link[:80]}")
            try:
                pdf_bytes = await fetch_pdf_bytes(client, link)
                if not pdf_bytes:
                    continue
                text = clean_text(extract_pdf_text(pdf_bytes))
                del pdf_bytes
                gc.collect()
                if len(text) < 100:
                    continue
                cat = detect_category(text, link)
                c = chunk_text(text, link, "ราชกิจจานุเบกษา", "ราชกิจจา — ค้นหาคนเข้าเมือง",
                               "th", cat, {"document_type": "royal_gazette"})
                chunks.extend(c)
                _checkpoint(c)   # flush after each search-page PDF
                del text
                gc.collect()
            except Exception as e:
                log.warning(f"    ✗ search-pdf failed ({link[:60]}): {type(e).__name__}: {e}")

    return chunks


# ─── Source 2: immigration.go.th ──────────────────────────────────────────────

IMMIGRATION_PAGES = [
    ("https://www.immigration.go.th/content/visa",                 "ประเภทวีซ่า",        "visa"),
    ("https://www.immigration.go.th/content/extension",            "การต่ออายุ",          "extension"),
    ("https://www.immigration.go.th/content/90day",                "รายงาน 90 วัน",       "90day"),
    ("https://www.immigration.go.th/content/tm30",                 "TM30 แจ้งที่พัก",    "tm30"),
    ("https://www.immigration.go.th/content/fee",                  "ค่าธรรมเนียม",        "fees"),
    ("https://www.immigration.go.th/content/announcement",         "ประกาศ ตม.",          "general"),
]


async def scrape_immigration(client: httpx.AsyncClient) -> list[LawChunk]:
    log.info("🏛 Scraping immigration.go.th...")
    chunks = []

    for url, title, category in IMMIGRATION_PAGES:
        html = await fetch_html(client, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        # Remove nav/footer/script noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        # Main content
        main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|main|body", re.I))
        text = clean_text((main or soup).get_text(separator="\n"))
        if len(text) < 100:
            continue
        c = chunk_text(text, url, "immigration.go.th", title, "th", category,
                       {"document_type": "official_page"})
        chunks.extend(c)
        log.info(f"  ✓ {len(c)} chunks — {title}")

    # Also grab PDF announcements linked from the site
    index_html = await fetch_html(client, "https://www.immigration.go.th/content/announcement")
    if index_html:
        soup = BeautifulSoup(index_html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".pdf" in href.lower():
                pdf_url = href if href.startswith("http") else "https://www.immigration.go.th" + href
                pdf_bytes = await fetch_pdf_bytes(client, pdf_url)
                if not pdf_bytes:
                    continue
                text = clean_text(extract_pdf_text(pdf_bytes))
                if len(text) < 100:
                    continue
                cat = detect_category(text, pdf_url)
                c = chunk_text(text, pdf_url, "immigration.go.th",
                               a.get_text(strip=True) or "ประกาศ ตม.",
                               "th", cat, {"document_type": "announcement_pdf"})
                chunks.extend(c)

    return chunks


# ─── Source 3: krisdika.go.th ─────────────────────────────────────────────────

KRISDIKA_PAGES = [
    # Search results for immigration-related laws
    ("https://www.krisdika.go.th/th/law/search?keyword=%E0%B8%84%E0%B8%99%E0%B9%80%E0%B8%82%E0%B9%89%E0%B8%B2%E0%B9%80%E0%B8%A1%E0%B8%B7%E0%B8%AD%E0%B8%87",
     "กฤษฎีกา — ค้นหาคนเข้าเมือง", "th", "general"),
]


async def scrape_krisdika(client: httpx.AsyncClient) -> list[LawChunk]:
    log.info("📚 Scraping krisdika.go.th...")
    chunks = []

    # KRISDIKA_IMMIGRATION_URLS items are 4-tuples (url, title, lang, category)
    # — previously unpacked as 3 values which raised ValueError.
    for url, title, lang, category in KRISDIKA_IMMIGRATION_URLS:
        log.info(f"  → PDF: {url[:70]}")
        try:
            pdf_bytes = await fetch_pdf_bytes(client, url)
            if pdf_bytes:
                text = clean_text(extract_pdf_text(pdf_bytes))
                if len(text) > 100:
                    c = chunk_text(text, url, "krisdika.go.th", title,
                                   lang, category, {"document_type": "official_act", "bilingual": True})
                    chunks.extend(c)
                    _checkpoint(c)
                    log.info(f"  ✓ {len(c)} chunks [{lang}] — {title}")
        except Exception as e:
            log.warning(f"  ✗ krisdika PDF failed ({url[:60]}): {type(e).__name__}: {e}")

    # Scrape HTML pages
    for url, title, lang, category in KRISDIKA_PAGES:
        html = await fetch_html(client, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        # Find all law document links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/law/" in href and ("คนเข้าเมือง" in a.get_text() or "immigration" in href.lower()):
                law_url = href if href.startswith("http") else "https://www.krisdika.go.th" + href
                law_html = await fetch_html(client, law_url)
                if not law_html:
                    continue
                law_soup = BeautifulSoup(law_html, "html.parser")
                for tag in law_soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = clean_text(law_soup.get_text(separator="\n"))
                if len(text) > 100:
                    c = chunk_text(text, law_url, "krisdika.go.th",
                                   a.get_text(strip=True) or title, lang, category,
                                   {"document_type": "official_act"})
                    chunks.extend(c)

    return chunks


# ─── Source 4: dol.go.th ──────────────────────────────────────────────────────

DOL_PAGES = [
    ("https://www.dol.go.th/en/Pages/foreign-national.aspx",
     "Foreign National Land Ownership", "en", "general"),
    ("https://www.dol.go.th/th/Pages/land-foreign.aspx",
     "ชาวต่างชาติกับที่ดิน", "th", "general"),
    ("https://www.dol.go.th/en/Pages/condo-foreign.aspx",
     "Foreigners — Condominium Ownership", "en", "general"),
]


async def scrape_dol(client: httpx.AsyncClient) -> list[LawChunk]:
    log.info("🏠 Scraping dol.go.th...")
    chunks = []
    for url, title, lang, category in DOL_PAGES:
        html = await fetch_html(client, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = clean_text(soup.get_text(separator="\n"))
        if len(text) > 100:
            c = chunk_text(text, url, "dol.go.th", title, lang, category,
                           {"document_type": "government_page"})
            chunks.extend(c)
            log.info(f"  ✓ {len(c)} chunks — {title}")
    return chunks


# ─── Source 5: samuiforsale.com ───────────────────────────────────────────────

SAMUI_PAGES = [
    "https://www.samuiforsale.com/law-library/immigration-act.html",
    "https://www.samuiforsale.com/law-library/thai-visa-types.html",
    "https://www.samuiforsale.com/law-library/visa-extension.html",
    "https://www.samuiforsale.com/law-library/90-day-reporting.html",
    "https://www.samuiforsale.com/law-library/tm-30.html",
    "https://www.samuiforsale.com/law-library/thai-immigration-fees.html",
    "https://www.samuiforsale.com/law-library/non-immigrant-visa.html",
    "https://www.samuiforsale.com/law-library/retirement-visa.html",
    "https://www.samuiforsale.com/law-library/marriage-visa.html",
    "https://www.samuiforsale.com/law-library/work-permit.html",
]


async def scrape_samui(client: httpx.AsyncClient) -> list[LawChunk]:
    log.info("🌴 Scraping samuiforsale.com...")
    chunks = []
    for url in SAMUI_PAGES:
        html = await fetch_html(client, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", ".sidebar"]):
            tag.decompose()
        # Main article content
        main = (soup.find("div", class_=re.compile(r"entry|content|article|post", re.I))
                or soup.find("article") or soup)
        title_el = soup.find("h1") or soup.find("title")
        title = title_el.get_text(strip=True) if title_el else url.split("/")[-1].replace(".html","")
        text = clean_text(main.get_text(separator="\n"))
        if len(text) > 100:
            cat = detect_category(text, url)
            c = chunk_text(text, url, "samuiforsale.com", title, "en", cat,
                           {"document_type": "expat_guide"})
            chunks.extend(c)
            log.info(f"  ✓ {len(c)} chunks — {title[:60]}")
    return chunks


# ─── Source 6: aseanlawyer.com ────────────────────────────────────────────────

ASEAN_PAGES = [
    "https://aseanlawyer.com/thailand/immigration/",
    "https://aseanlawyer.com/thailand/immigration/visa-types/",
    "https://aseanlawyer.com/thailand/immigration/visa-extension/",
    "https://aseanlawyer.com/thailand/immigration/90-day-report/",
    "https://aseanlawyer.com/thailand/immigration/tm30/",
    "https://aseanlawyer.com/thailand/immigration/work-permit/",
]


async def scrape_asean(client: httpx.AsyncClient) -> list[LawChunk]:
    log.info("⚖️  Scraping aseanlawyer.com...")
    chunks = []
    for url in ASEAN_PAGES:
        html = await fetch_html(client, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        main = soup.find("article") or soup.find("main") or soup.find("div", class_="content")
        title_el = soup.find("h1")
        title = title_el.get_text(strip=True) if title_el else "ASEAN Lawyer — Thailand Immigration"
        text = clean_text((main or soup).get_text(separator="\n"))
        if len(text) > 100:
            cat = detect_category(text, url)
            c = chunk_text(text, url, "aseanlawyer.com", title, "en", cat,
                           {"document_type": "legal_commentary"})
            chunks.extend(c)
            log.info(f"  ✓ {len(c)} chunks — {title[:60]}")
    return chunks


# ─── Source 7: ThaiLaws.com ───────────────────────────────────────────────────

THAILAWS_PAGES = [
    ("https://www.thailaws.com/law/t_laws/tlaw0264.pdf",
     "Immigration Act B.E. 2522 (ThaiLaws EN)", "en", "general"),
    ("https://www.thailaws.com/aboutthailaws/immig.htm",
     "ThaiLaws — Immigration Overview", "en", "general"),
    ("https://www.thailaws.com/aboutthailaws/visatypes.htm",
     "ThaiLaws — Visa Types", "en", "visa"),
]


async def scrape_thailaws(client: httpx.AsyncClient) -> list[LawChunk]:
    log.info("📖 Scraping ThaiLaws.com...")
    chunks = []
    for url, title, lang, category in THAILAWS_PAGES:
        if url.endswith(".pdf"):
            pdf_bytes = await fetch_pdf_bytes(client, url)
            if pdf_bytes:
                text = clean_text(extract_pdf_text(pdf_bytes))
                if len(text) > 100:
                    c = chunk_text(text, url, "thailaws.com", title, lang, category,
                                   {"document_type": "en_translation"})
                    chunks.extend(c)
                    log.info(f"  ✓ {len(c)} chunks (PDF) — {title}")
        else:
            html = await fetch_html(client, url)
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = clean_text(soup.get_text(separator="\n"))
            if len(text) > 100:
                c = chunk_text(text, url, "thailaws.com", title, lang, category,
                               {"document_type": "en_translation"})
                chunks.extend(c)
                log.info(f"  ✓ {len(c)} chunks — {title}")
    return chunks


# ─── Main orchestrator ────────────────────────────────────────────────────────

SOURCE_MAP = {
    "ratchakitcha": scrape_ratchakitcha,
    "immigration":  scrape_immigration,
    "krisdika":     scrape_krisdika,
    "dol":          scrape_dol,
    "samui":        scrape_samui,
    "asean":        scrape_asean,
    "thai":         scrape_thailaws,
}


async def run_all(sources: list[str], output_path: str):
    """Resumable scrape — writes EVERY chunk immediately (after each PDF, not
    each source). This is critical because the user's Mac has been crashing
    mid-pipeline; writing per-PDF means previous PDFs survive a hard system
    crash. On rerun, source_url set already in the file is skipped.

    To keep the existing scrape_* functions simple (they return lists), we
    write after each function returns AND inside scrape_ratchakitcha we now
    also flush via a checkpoint helper passed through `_write_chunks_global`.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build set of (url, chunk_index) already written → exact dedupe on resume
    seen_keys: set[tuple] = set()
    if out.exists():
        try:
            with out.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        seen_keys.add((row.get("source_url", ""), row.get("chunk_index", -1)))
                    except json.JSONDecodeError:
                        continue
            log.info(f"📂 Resume mode — found {len(seen_keys)} chunks already on disk → will skip duplicates")
        except Exception as e:
            log.warning(f"Resume read failed (starting fresh): {e}")
            seen_keys.clear()

    # Write helper: appends new chunks immediately + fsync so system crash
    # within the next 1ms still preserves everything written so far.
    def flush_chunks(chunks):
        if not chunks:
            return 0
        written = 0
        with out.open("a", encoding="utf-8") as f:
            for c in chunks:
                key = (c.source_url, c.chunk_index)
                if key in seen_keys:
                    continue
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
                seen_keys.add(key)
                written += 1
            f.flush()
            import os as _os
            try: _os.fsync(f.fileno())
            except OSError: pass
        return written

    # Expose flush_chunks globally so scrape_* can checkpoint mid-source
    global _GLOBAL_FLUSH
    _GLOBAL_FLUSH = flush_chunks

    total_written = 0
    async with httpx.AsyncClient(timeout=60, verify=False) as client:
        for name in sources:
            fn = SOURCE_MAP.get(name)
            if not fn:
                log.warning(f"Unknown source: {name}")
                continue
            try:
                chunks = await fn(client)
                # flush_chunks dedupes via seen_keys, so we can pass all chunks
                written = flush_chunks(chunks)
                total_written += written
                log.info(f"  → {name}: {written} chunks written ({len(chunks) - written} dedupe/skipped)")
            except Exception as e:
                log.error(f"✗ {name} failed: {type(e).__name__}: {e}")

    log.info(f"\n✅ Done — {total_written} new chunks appended → {output_path}")
    final_count = 0
    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            final_count = sum(1 for _ in f)
    log.info(f"   Total chunks in file: {final_count}")
    return final_count


# Module-level checkpoint callback — set by run_all(), called by scrape_*
# to flush partial progress after each PDF so a system-level crash (not
# just Python crash) preserves what was already extracted.
_GLOBAL_FLUSH = None
def _checkpoint(chunks):
    if _GLOBAL_FLUSH and chunks:
        try: _GLOBAL_FLUSH(chunks)
        except Exception as e: log.warning(f"  ⚠ checkpoint write failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thai Immigration Law Scraper")
    parser.add_argument("--source", default="all",
                        help="Comma-separated sources or 'all'")
    parser.add_argument("--output", default="./raw_chunks.jsonl",
                        help="Output JSONL file path")
    args = parser.parse_args()

    sources = list(SOURCE_MAP.keys()) if args.source == "all" else args.source.split(",")
    asyncio.run(run_all(sources, args.output))
