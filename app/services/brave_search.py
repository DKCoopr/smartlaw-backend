"""
Brave Search — Thai Legal Database
────────────────────────────────────────────────────────────────────────────────
ค้นหาฎีกา / กฎหมาย / ประมวลกฎหมายที่เกี่ยวข้องกับคดี
โดยการค้นหาจาก Brave Search API กรองเฉพาะแหล่งข้อมูลกฎหมายไทยที่น่าเชื่อถือ

แหล่งข้อมูลที่ค้นหา (8 แหล่ง ครอบคลุม ~95% ของคดีทั่วไป):
  ── ศาล ──────────────────────────────────────────────────────────────────────
  - deka.supremecourt.or.th   → คำพิพากษาฎีกาศาลสูงสุด
  - appeal.coj.go.th          → คำพิพากษาศาลอุทธรณ์
  - admincourt.go.th          → คำพิพากษาศาลปกครอง
  ── กฎหมาย / ประมวล ──────────────────────────────────────────────────────────
  - krisdika.go.th            → สำนักงานคณะกรรมการกฤษฎีกา / ประมวลกฎหมาย
  - law.go.th                 → ระบบกฎหมายไทย
  - ratchakitcha.soc.go.th    → ราชกิจจานุเบกษา (กฎหมายใหม่ / พรบ.)
  - revenue.go.th             → กรมสรรพากร (กฎหมายภาษี)
  ── แหล่งอ้างอิงเพิ่มเติม ────────────────────────────────────────────────────
  - ilaw.or.th                → iLaw สรุปกฎหมายสำคัญ
────────────────────────────────────────────────────────────────────────────────
"""
import asyncio
import re
from typing import Optional
import httpx

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"

# แหล่งข้อมูลกฎหมายไทยที่น่าเชื่อถือ
LEGAL_SITES = [
    # ── ศาล ──────────────────────────────────────────────────────────────────
    "deka.supremecourt.or.th",      # คำพิพากษาฎีกาศาลสูงสุด
    "appeal.coj.go.th",             # คำพิพากษาศาลอุทธรณ์
    "admincourt.go.th",             # คำพิพากษาศาลปกครอง
    "iptc.go.th",                   # ศาลทรัพย์สินทางปัญญาและการค้าระหว่างประเทศ ★ ใหม่
    "labourtribunal.go.th",         # ศาลแรงงาน ★ ใหม่
    # ── กฎหมาย / ประมวล ──────────────────────────────────────────────────────
    "krisdika.go.th",               # สำนักงานคณะกรรมการกฤษฎีกา
    "law.go.th",                    # ระบบกฎหมายไทย
    "ratchakitcha.soc.go.th",       # ราชกิจจานุเบกษา — กฎหมายใหม่
    "revenue.go.th",                # กรมสรรพากร — กฎหมายภาษี
    # ── แหล่งอ้างอิงเพิ่มเติม ────────────────────────────────────────────────
    "ilaw.or.th",                   # iLaw — สรุปกฎหมายสำคัญ
]

SITE_LABELS = {
    "deka.supremecourt.or.th":  "ฎีกาศาลสูงสุด",
    "appeal.coj.go.th":         "ศาลอุทธรณ์",
    "admincourt.go.th":         "ศาลปกครอง",
    "iptc.go.th":               "ศาลทรัพย์สินทางปัญญา",
    "labourtribunal.go.th":     "ศาลแรงงาน",
    "krisdika.go.th":           "สำนักงานกฤษฎีกา",
    "law.go.th":                "ระบบกฎหมายไทย",
    "ratchakitcha.soc.go.th":   "ราชกิจจานุเบกษา",
    "revenue.go.th":            "กรมสรรพากร",
    "ilaw.or.th":               "iLaw",
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

    # ── Query 1: ฎีกา/คำพิพากษาตามประเภทคดี ────────────────────────────────────
    if case_type:
        # ศาลเฉพาะทาง — ใช้ site filter ที่แคบกว่าเพื่อความแม่นยำ
        LABOUR_TYPES = {"แรงงาน", "เลิกจ้าง", "ค่าชดเชย"}
        IP_TYPES     = {"ทรัพย์สินทางปัญญา", "ลิขสิทธิ์", "สิทธิบัตร", "เครื่องหมายการค้า"}

        if any(t in case_type for t in LABOUR_TYPES) or any(
            re.search(t, combined_text) for t in [r"แรงงาน", r"เลิกจ้าง", r"ค่าชดเชย"]
        ):
            # คดีแรงงาน — เน้น labourtribunal + ฎีกาแรงงาน
            labour_site = "site:labourtribunal.go.th OR site:deka.supremecourt.or.th"
            queries.append(f"คำพิพากษา แรงงาน {case_type} ({labour_site})")
        elif any(t in case_type for t in IP_TYPES) or any(
            re.search(t, combined_text) for t in [r"ลิขสิทธิ์", r"สิทธิบัตร", r"เครื่องหมายการค้า"]
        ):
            # คดี IP — เน้น iptc + ฎีกา
            ip_site = "site:iptc.go.th OR site:deka.supremecourt.or.th"
            queries.append(f"คำพิพากษา ทรัพย์สินทางปัญญา {case_type} ({ip_site})")
        else:
            q = f"คำพิพากษาฎีกา {case_type}"
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
        # อาญา
        r"ฉ้อโกง", r"ยักยอก", r"ลักทรัพย์", r"ฆ่า", r"ทำร้าย",
        r"ปล้น", r"ชิงทรัพย์", r"ข่มขืน", r"ทุจริต", r"รับสินบน",
        # แพ่ง / สัญญา
        r"ผิดสัญญา", r"ละเมิด", r"เลิกสัญญา", r"บอกเลิก",
        r"ค้ำประกัน", r"จำนอง", r"จำนำ", r"เช่า", r"ซื้อขาย",
        r"กู้ยืม", r"ตั๋วเงิน", r"เช็ค", r"สัญญากู้",
        # ครอบครัว / มรดก
        r"มรดก", r"หย่า", r"ครอบครัว", r"รับบุตรบุญธรรม", r"อำนาจปกครอง",
        r"สินสมรส", r"สินส่วนตัว",
        # ธุรกิจ / หุ้น
        r"บริษัท", r"หุ้น", r"หุ้นส่วน", r"ล้มละลาย", r"ฟื้นฟูกิจการ",
        r"ผู้ชำระบัญชี", r"กรรมการ", r"ผู้ถือหุ้น",
        # ที่ดิน / ทรัพย์สิน
        r"ที่ดิน", r"โฉนด", r"น.ส.3", r"ครอบครองปรปักษ์", r"ภาระจำยอม",
        r"ทรัพย์สิน", r"เวนคืน", r"อสังหาริมทรัพย์",
        # ภาษี / ปกครอง
        r"ภาษี", r"สรรพากร", r"ภาษีมูลค่าเพิ่ม", r"ภาษีเงินได้",
        r"ปกครอง", r"ใบอนุญาต", r"สัมปทาน", r"คำสั่งทางปกครอง",
        # แรงงาน
        r"แรงงาน", r"เลิกจ้าง", r"ค่าชดเชย", r"ค่าจ้าง", r"นายจ้าง",
        r"ลูกจ้าง", r"สัญญาจ้าง", r"ค่าเสียหาย", r"สินจ้างแทนการบอกกล่าว",
        # ทรัพย์สินทางปัญญา
        r"ลิขสิทธิ์", r"สิทธิบัตร", r"เครื่องหมายการค้า", r"ความลับทางการค้า",
        r"ละเมิดลิขสิทธิ์", r"ปลอมแปลง", r"การค้าระหว่างประเทศ",
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

    # เรียงตามลำดับชั้นศาล: ฎีกา → อุทธรณ์ → ปกครอง → กฎหมาย → อื่นๆ
    def _sort_rank(r: dict) -> int:
        url = r["url"]
        if "deka.supremecourt" in url:    return 0
        if "appeal.coj" in url:           return 1
        if "admincourt" in url:           return 2
        if "iptc.go.th" in url:           return 3
        if "labourtribunal" in url:       return 4
        if "krisdika" in url:             return 5
        if "ratchakitcha" in url:         return 6
        if "law.go.th" in url:            return 7
        if "revenue.go.th" in url:        return 8
        return 9

    merged.sort(key=_sort_rank)
    return merged[:12]   # max 12 ผล (เพิ่มจาก 10 เพราะมีแหล่งเพิ่ม)


def format_for_prompt(results: list[dict]) -> str:
    """
    แปลงผลการค้นหาเป็น text block สำหรับ inject เข้า Claude prompt
    จัดกลุ่มตามประเภท: ฎีกา → อุทธรณ์/ปกครอง → กฎหมาย/ราชกิจจา
    """
    if not results:
        return ""

    lines = [
        "## 📚 ผลการค้นหาจากฐานข้อมูลกฎหมายไทย (Thai Legal Database)",
        f"*(ระบบค้นพบ {len(results)} รายการ — ใช้เป็นข้อมูลอ้างอิงในการวิเคราะห์)*",
        "",
    ]

    # จัดกลุ่ม
    COURT_URLS = ["deka.supremecourt", "appeal.coj", "admincourt", "iptc.go.th", "labourtribunal"]
    deka     = [r for r in results if "deka.supremecourt" in r["url"]]
    appeal   = [r for r in results if "appeal.coj" in r["url"]]
    admin    = [r for r in results if "admincourt" in r["url"]]
    iptc     = [r for r in results if "iptc.go.th" in r["url"]]
    labour   = [r for r in results if "labourtribunal" in r["url"]]
    laws     = [r for r in results if not any(x in r["url"] for x in COURT_URLS)]

    def _fmt_result(r: dict, i: int) -> list[str]:
        out = [f"{i}. **{r['title']}** [{r['source_label']}]"]
        if r["description"]:
            out.append(f"   {r['description'][:250]}")
        out.append(f"   🔗 {r['url']}")
        out.append("")
        return out

    if deka:
        lines.append("### ⚖️ คำพิพากษาฎีกา (ศาลสูงสุด):")
        for i, r in enumerate(deka, 1):
            lines.extend(_fmt_result(r, i))

    if appeal:
        lines.append("### ⚖️ คำพิพากษาศาลอุทธรณ์:")
        for i, r in enumerate(appeal, 1):
            lines.extend(_fmt_result(r, i))

    if admin:
        lines.append("### 🏛️ คำพิพากษาศาลปกครอง:")
        for i, r in enumerate(admin, 1):
            lines.extend(_fmt_result(r, i))

    if iptc:
        lines.append("### 💡 คำพิพากษาศาลทรัพย์สินทางปัญญาและการค้าระหว่างประเทศ:")
        for i, r in enumerate(iptc, 1):
            lines.extend(_fmt_result(r, i))

    if labour:
        lines.append("### 👷 คำพิพากษาศาลแรงงาน:")
        for i, r in enumerate(labour, 1):
            lines.extend(_fmt_result(r, i))

    if laws:
        lines.append("### 📜 บทบัญญัติกฎหมาย / ประกาศ:")
        for i, r in enumerate(laws, 1):
            lines.extend(_fmt_result(r, i))

    lines += [
        "---",
        "🚨 **คำสั่งสำคัญ:** ต้องอ้างอิงฎีกา/มาตรากฎหมายจากรายการด้านบนในการวิเคราะห์",
        "— ระบุเลขฎีกา / ชื่อพรบ. / มาตราให้ครบถ้วนในหัวข้อ 5 (กฎหมายที่เกี่ยวข้อง) และ 14 (บทสรุป)",
        "",
    ]
    return "\n".join(lines)
