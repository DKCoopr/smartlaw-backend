"""
Brave Search — Thai Legal Database
────────────────────────────────────────────────────────────────────────────────
ค้นหาฎีกา / กฎหมาย / ประมวลกฎหมายที่เกี่ยวข้องกับคดี
โดยการค้นหาจาก Brave Search API กรองเฉพาะแหล่งข้อมูลกฎหมายไทยที่น่าเชื่อถือ

แหล่งข้อมูลที่ค้นหา:
  - deka.supremecourt.or.th   → คำพิพากษาฎีกา
  - krisdika.go.th            → ประมวลกฎหมาย / พรบ.
  - law.go.th                 → กฎหมายและประกาศราชกิจจา
  - ilaw.or.th                → สรุปกฎหมายสำคัญ
────────────────────────────────────────────────────────────────────────────────
"""
import asyncio
import re
from typing import Optional
import httpx

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"

# แหล่งข้อมูลกฎหมายไทยที่น่าเชื่อถือ
LEGAL_SITES = [
    "deka.supremecourt.or.th",
    "krisdika.go.th",
    "law.go.th",
    "ilaw.or.th",
    "ocs.go.th",
]

SITE_LABELS = {
    "deka.supremecourt.or.th": "ฎีกาศาลสูงสุด",
    "krisdika.go.th":          "สำนักงานกฤษฎีกา",
    "law.go.th":               "ระบบกฎหมายไทย",
    "ilaw.or.th":              "iLaw",
    "ocs.go.th":               "สำนักงานคณะกรรมการกฤษฎีกา",
}

_MAX_RESULTS_PER_QUERY = 5
_TIMEOUT_SECONDS       = 8


def _site_filter() -> str:
    return " OR ".join(f"site:{s}" for s in LEGAL_SITES)


def _build_queries(
    case_type: str = "",
    plaintiff: str = "",
    defendant: str = "",
    claim_amount: float = 0,
    transcript: str = "",
    notes: str = "",
    doc_summaries: str = "",
) -> list[str]:
    """
    สร้าง 2-3 query ที่ตรงกับคดีมากที่สุด
    """
    queries = []
    combined_text = f"{transcript} {notes} {doc_summaries}".strip()
    site_q = _site_filter()

    # ── Query 1: ฎีกาตามประเภทคดี ──────────────────────────────────────────────
    if case_type:
        q = f"คำพิพากษาฎีกา {case_type}"
        # ถ้าทุนทรัพย์สูง เพิ่มขนาดคดีเข้า query
        if claim_amount and claim_amount >= 1_000_000:
            q += " ทุนทรัพย์"
        queries.append(f"{q} ({site_q})")

    # ── Query 2: มาตรากฎหมายที่ปรากฏในเอกสาร ──────────────────────────────────
    law_sections = re.findall(
        r"มาตรา\s*\d[\d/]*|ป\.พ\.พ\.|ป\.อ\.|พ\.ร\.บ\.[^\n]{0,40}",
        combined_text,
    )
    if law_sections:
        # เอาแค่ 3 มาตราแรก กัน query ยาวเกิน
        section_str = " ".join(dict.fromkeys(law_sections[:3]))
        queries.append(f"{section_str} ({site_q})")

    # ── Query 3: keyword จากเนื้อหาเอกสาร ─────────────────────────────────────
    # ดึง noun phrase สั้นๆ จาก transcript (เช่น "ฉ้อโกง" "ผิดสัญญา" "ละเมิด")
    legal_keywords = _extract_legal_keywords(combined_text)
    if legal_keywords:
        q = f"ฎีกา {' '.join(legal_keywords[:3])} กฎหมาย ({site_q})"
        queries.append(q)

    # Fallback ถ้าไม่มีข้อมูลเลย
    if not queries:
        queries.append(f"คำพิพากษาฎีกา กฎหมายแพ่งพาณิชย์ ({site_q})")

    return queries[:3]   # max 3 queries


def _extract_legal_keywords(text: str) -> list[str]:
    """ดึง keyword ทางกฎหมายจากข้อความ"""
    patterns = [
        r"ฉ้อโกง", r"ยักยอก", r"ลักทรัพย์", r"ฆ่า", r"ทำร้าย",
        r"ผิดสัญญา", r"ละเมิด", r"เลิกสัญญา", r"บอกเลิก",
        r"ค้ำประกัน", r"จำนอง", r"จำนำ", r"เช่า", r"ซื้อขาย",
        r"กู้ยืม", r"อาญา", r"แพ่ง", r"แรงงาน", r"ปกครอง",
        r"มรดก", r"หย่า", r"ครอบครัว", r"ภาษี", r"ล้มละลาย",
        r"บริษัท", r"หุ้น", r"ทรัพย์สิน", r"ที่ดิน", r"โฉนด",
    ]
    found = []
    for pat in patterns:
        if re.search(pat, text):
            found.append(pat.replace("\\", ""))
    return list(dict.fromkeys(found))  # dedupe, preserve order


def _source_label(url: str) -> str:
    for site, label in SITE_LABELS.items():
        if site in url:
            return label
    return "แหล่งข้อมูลกฎหมาย"


def _is_legal_source(url: str) -> bool:
    return any(site in url for site in LEGAL_SITES)


async def _search_one(client: httpx.AsyncClient, api_key: str, query: str) -> list[dict]:
    """เรียก Brave Search API 1 query → list of result dicts"""
    try:
        resp = await client.get(
            BRAVE_API_URL,
            params={
                "q":           query,
                "count":       _MAX_RESULTS_PER_QUERY,
                "search_lang": "th",
                "country":     "TH",
                "text_decorations": False,
                "safesearch":  "off",
            },
            headers={
                "X-Subscription-Token": api_key,
                "Accept":               "application/json",
                "Accept-Encoding":      "gzip",
            },
            timeout=_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("web", {}).get("results", [])
    except Exception as e:
        print(f"[brave_search] query failed: {e}")
        return []


async def search_thai_legal(
    api_key: str,
    case_type: str = "",
    plaintiff: str = "",
    defendant: str = "",
    claim_amount: float = 0,
    transcript: str = "",
    notes: str = "",
    doc_summaries: str = "",
) -> list[dict]:
    """
    ค้นหา 2-3 queries พร้อมกัน แล้วรวมผลลัพธ์
    Return list of {title, url, description, source_label}
    เรียงตาม relevance + กรองเฉพาะแหล่งกฎหมายไทย
    """
    if not api_key:
        return []

    queries = _build_queries(
        case_type=case_type,
        plaintiff=plaintiff,
        defendant=defendant,
        claim_amount=claim_amount,
        transcript=transcript,
        notes=notes,
        doc_summaries=doc_summaries,
    )

    async with httpx.AsyncClient() as client:
        results_per_query = await asyncio.gather(
            *[_search_one(client, api_key, q) for q in queries],
            return_exceptions=True,
        )

    # รวมผล + dedupe + กรองเฉพาะแหล่งกฎหมาย
    seen_urls: set[str] = set()
    merged: list[dict] = []
    for batch in results_per_query:
        if isinstance(batch, Exception):
            continue
        for r in batch:
            url = r.get("url", "")
            if url in seen_urls:
                continue
            if not _is_legal_source(url):
                continue
            seen_urls.add(url)
            merged.append({
                "title":        r.get("title", ""),
                "url":          url,
                "description":  r.get("description", ""),
                "source_label": _source_label(url),
            })

    # ฎีกา (deka) ขึ้นก่อน เพราะตรงกว่า
    merged.sort(key=lambda x: (0 if "deka.supremecourt" in x["url"] else 1))
    return merged[:10]   # max 10 ผล


def format_for_prompt(results: list[dict]) -> str:
    """
    แปลงผลการค้นหาเป็น text block สำหรับ inject เข้า Claude prompt
    """
    if not results:
        return ""

    lines = [
        "## 📚 ผลการค้นหาจากฐานข้อมูลกฎหมายไทย (Thai Legal Database)",
        "*(ระบบค้นพบฎีกาและกฎหมายต่อไปนี้ — ใช้เป็นข้อมูลอ้างอิงเพิ่มเติมในการวิเคราะห์)*",
        "",
    ]

    deka = [r for r in results if "deka.supremecourt" in r["url"]]
    laws = [r for r in results if "deka.supremecourt" not in r["url"]]

    if deka:
        lines.append("### คำพิพากษาฎีกาที่เกี่ยวข้อง:")
        for i, r in enumerate(deka, 1):
            lines.append(f"{i}. **{r['title']}**")
            if r["description"]:
                lines.append(f"   {r['description'][:200]}")
            lines.append(f"   🔗 {r['url']}")
            lines.append("")

    if laws:
        lines.append("### บทบัญญัติกฎหมายที่เกี่ยวข้อง:")
        for i, r in enumerate(laws, 1):
            lines.append(f"{i}. **{r['title']}** [{r['source_label']}]")
            if r["description"]:
                lines.append(f"   {r['description'][:200]}")
            lines.append(f"   🔗 {r['url']}")
            lines.append("")

    lines += [
        "---",
        "🚨 **คำสั่งสำคัญ:** อ้างอิงฎีกาและมาตรากฎหมายจากรายการด้านบนในการวิเคราะห์ด้วย",
        "— ระบุเลขฎีกา / ชื่อกฎหมาย / มาตราให้ครบถ้วนในหัวข้อ 5 (กฎหมายที่เกี่ยวข้อง) และ 14 (บทสรุป)",
        "",
    ]
    return "\n".join(lines)
