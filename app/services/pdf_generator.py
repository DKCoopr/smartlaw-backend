"""
Thai.Law PDF Generator  (markdown → fpdf2, no HTML parsing)
============================================================
Receives raw markdown text + case metadata → produces PDF bytes.
Fonts embedded from /app/fonts/ (downloaded during Docker build).
Uses Sarabun (Thai+Latin) so dates, numbers, and Thai text all render correctly.
"""

import re
import os
from datetime import datetime

FONTS_DIR = os.environ.get("PDF_FONTS_DIR", "/app/fonts")


# Colours (R, G, B)
BLUE       = (29,  78, 216)
GREEN      = (15, 118, 110)
DARK       = (15,  26,  74)
GRAY       = (100,116, 139)
WHITE      = (255,255, 255)
LIGHT_BLUE = (239,244, 255)
LIGHT_GRAY = (241,245, 249)

MARGIN_L  = 16
MARGIN_R  = 16
MARGIN_T  = 20
MARGIN_B  = 18
A4_W      = 210
A4_H      = 297
CONTENT_W = A4_W - MARGIN_L - MARGIN_R   # 178 mm


def _fmt_thb(amount, lang: str = "th") -> str:
    try:
        n = float(amount or 0)
    except Exception:
        return "—"
    # NotoSansCJK doesn't include ฿ glyph; use "THB" text instead for Chinese.
    if lang == "zh":
        return "{:,.2f} 泰铢".format(n)
    if lang == "en":
        return "THB {:,.2f}".format(n)
    return "฿{:,.2f}".format(n)


def _strip_md(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",     r"\1", text)
    return text.strip()


# ── Font setup ────────────────────────────────────────────────────────────────

def _build_fpdf(lang: str):
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(MARGIN_L, MARGIN_T, MARGIN_R)
    pdf.set_auto_page_break(auto=True, margin=MARGIN_B)

    sarabun_r = os.path.join(FONTS_DIR, "Sarabun-Regular.ttf")
    sarabun_b = os.path.join(FONTS_DIR, "Sarabun-Bold.ttf")
    sc_r      = os.path.join(FONTS_DIR, "NotoSansSC-Regular.otf")

    def _size(p):
        try: return os.path.getsize(p)
        except OSError: return 0

    # CJK languages — Japanese & Korean fall back to the NotoSansSC font we
    # already ship. SC's glyph forms differ slightly from JP/KR variants but
    # the underlying Han characters + Hiragana/Katakana/Hangul coverage is
    # adequate for legal output (vs blank squares with Sarabun). For perfect
    # native typography, ship NotoSansJP / NotoSansKR separately later.
    if lang in ("zh", "ja", "ko"):
        sz = _size(sc_r)
        if sz < 1_000_000:
            raise RuntimeError(
                f"CJK font missing or corrupted at {sc_r} (size={sz}B). "
                "Rebuild Docker image — font-download step should have downloaded ~12MB OTF."
            )
        print(f"[pdf_generator] Using NotoSansSC for CJK lang={lang} ({sz} bytes)", flush=True)
        pdf.add_font("Main", "",  sc_r, uni=True)
        pdf.add_font("Main", "B", sc_r, uni=True)
    else:
        # Sarabun covers Latin (+ Latin Extended for French diacritics) and
        # Cyrillic well enough for Russian. Arabic glyphs are NOT in Sarabun —
        # output will show empty boxes for Arabic-script text. To render Arabic
        # PDFs properly, ship Noto Naskh Arabic and switch on lang=="ar".
        sz = _size(sarabun_r)
        if sz < 50_000:
            raise RuntimeError(
                f"Sarabun font missing or corrupted at {sarabun_r} (size={sz}B). "
                "Rebuild Docker image."
            )
        bold = sarabun_b if _size(sarabun_b) > 50_000 else sarabun_r
        print(f"[pdf_generator] Using Sarabun for lang={lang} ({sz} bytes)", flush=True)
        pdf.add_font("Main", "",  sarabun_r, uni=True)
        pdf.add_font("Main", "B", bold,      uni=True)

    pdf.set_font("Main", size=11)
    return pdf


# ── Cover page ────────────────────────────────────────────────────────────────

ACCENT      = (201, 151,  58)       # Thai.Law brand gold (#C9973A) — primary accent
TEAL        = (  0, 196, 154)       # secondary accent (success/active indicators)
CARD_BG     = (248, 250, 253)
CARD_BORDER = (225, 232, 240)
LOGO_DARK   = (15,  26,  74)        # deep navy for "Thai" half of wordmark
LOGO_SUB    = (140, 156, 180)       # gray for subtitle

# Gold ramp for logo bars — matches landing-page palette (#E8B84B → #C9973A).
# Outer bars use the lightest tones, inner bars darken toward the center.
LOGO_GOLD_RAMP = [
    (244, 224, 175),   # very light gold (opacity ~0.25 over white)
    (236, 200, 130),   # light gold      (opacity ~0.45)
    (222, 175,  95),   # medium gold     (opacity ~0.65)
    (201, 151,  58),   # deep gold       (#C9973A)
]
LOGO_CENTER = (  0, 196, 154)       # teal center bar — matches landing's gold-primary + teal-secondary scheme


# Cover-page disclaimer — language matches the analysis content.
# Crafted as formal but accessible legal prose.
DISCLAIMER_TEXT = {
    "th": (
        "เอกสารฉบับนี้จัดทำขึ้นโดย Thai.Law เพื่อใช้เป็นข้อมูลประกอบการวิเคราะห์ "
        "และช่วยในการตัดสินใจเบื้องต้นเท่านั้น — มิได้มีสถานะเป็นเอกสารทางราชการ "
        "และไม่อาจนำไปใช้อ้างอิงเป็นพยานหลักฐานในชั้นพิจารณาคดี ทั้งนี้ "
        "ผู้จัดทำมิได้มีส่วนได้เสีย หรือล่วงรู้ข้อเท็จจริงเฉพาะของคดีนี้แต่อย่างใด"
    ),
    "en": (
        "This document has been prepared by Thai.Law solely to assist with "
        "case analysis and preliminary decision-making. It is not an official "
        "record and may not be relied upon as evidence in any judicial or "
        "administrative proceeding. The author has no interest in, nor any "
        "prior knowledge of, the specific facts of this matter."
    ),
    "zh": (
        "本文件由 Thai.Law 编制，仅用于辅助案件分析及初步决策参考之用，"
        "不具备公文性质，亦不得作为任何司法或行政程序中的证据加以援引。"
        "编制方与本案件并无任何利害关系，亦未事先知悉本案的具体事实。"
    ),
    "ja": (
        "本文書は Thai.Law により、案件分析および予備的な意思決定の補助"
        "資料としてのみ作成されたものです。公文書としての性質を有さず、"
        "司法手続きまたは行政手続きにおける証拠として援用することはでき"
        "ません。作成者は本件について利害関係を有さず、固有の事実関係を"
        "事前に知るものでもありません。"
    ),
    "ko": (
        "본 문서는 Thai.Law가 사건 분석 및 예비 의사결정을 보조하기 위한 "
        "목적으로만 작성하였습니다. 공문서로서의 성격을 갖지 않으며, "
        "어떠한 사법 또는 행정 절차에서도 증거로 인용될 수 없습니다. "
        "작성자는 본 사건에 대한 이해관계가 없으며 구체적 사실 관계를 "
        "사전에 알지 못합니다."
    ),
    "ru": (
        "Настоящий документ подготовлен Thai.Law исключительно в качестве "
        "вспомогательного материала для анализа дела и предварительного "
        "принятия решений. Документ не имеет статуса официального и не может "
        "использоваться в качестве доказательства в каком-либо судебном "
        "или административном производстве. Составитель не имеет заинтересованности "
        "в данном деле и не осведомлён о конкретных обстоятельствах дела."
    ),
    "fr": (
        "Le présent document a été préparé par Thai.Law uniquement dans le but "
        "d'aider à l'analyse de l'affaire et à la prise de décision préliminaire. "
        "Il n'a pas valeur de document officiel et ne peut être invoqué comme "
        "preuve dans une quelconque procédure judiciaire ou administrative. "
        "L'auteur n'a aucun intérêt dans cette affaire ni connaissance préalable "
        "des faits spécifiques."
    ),
    "ar": (
        "تم إعداد هذه الوثيقة من قبل Thai.Law للمساعدة في تحليل القضية واتخاذ "
        "القرارات الأولية فحسب. لا تتمتع هذه الوثيقة بصفة المحرر الرسمي ولا "
        "يجوز الاستناد إليها كدليل في أي إجراء قضائي أو إداري. ليس للمعد أي "
        "مصلحة في هذه القضية ولا علم سابق بوقائعها المحددة."
    ),
}


def _is_filled(v) -> bool:
    """True if v represents real content (not None/empty/dash/zero)."""
    if v is None:
        return False
    s = str(v).strip()
    if not s:
        return False
    if s in {"—", "-", "–", "None", "null", "undefined"}:
        return False
    try:
        if float(s.replace(",", "").replace("฿", "").replace("THB", "").strip()) == 0:
            return False
    except (ValueError, AttributeError):
        pass
    return True


def _draw_logo(pdf, top_y: float = 36) -> float:
    """Draw the Thai.Law soundwave logo centered at page top — 9 vertical bars
    in a blue gradient with a single green accent bar in the middle, matching
    favicon.svg. Returns the Y position immediately below the logo block."""
    cx = A4_W / 2

    # 9-bar symmetric soundwave (heights scaled from favicon's pixel pattern):
    #   favicon px: [8, 16, 24, 18, 28, 18, 24, 16,  8]  (max 28px in 40px viewport)
    bar_heights = [3.0, 6.0, 9.0, 6.8, 11.0, 6.8, 9.0, 6.0, 3.0]
    # Color index per bar — mirror outward, center is green accent.
    # 0 = lightest blue → 3 = deepest blue; CENTER = green accent
    bar_colors  = [LOGO_GOLD_RAMP[0], LOGO_GOLD_RAMP[1], LOGO_GOLD_RAMP[2], LOGO_GOLD_RAMP[3],
                   LOGO_CENTER,
                   LOGO_GOLD_RAMP[3], LOGO_GOLD_RAMP[2], LOGO_GOLD_RAMP[1], LOGO_GOLD_RAMP[0]]

    bar_w   = 1.6
    bar_gap = 1.1
    total_w = len(bar_heights) * bar_w + (len(bar_heights) - 1) * bar_gap
    icon_x  = cx - total_w / 2
    max_h   = max(bar_heights)

    for i, (h, color) in enumerate(zip(bar_heights, bar_colors)):
        x = icon_x + i * (bar_w + bar_gap)
        y = top_y + (max_h - h) / 2
        pdf.set_fill_color(*color)
        # Try rounded bars; fall back to plain rect on older fpdf2.
        try:
            pdf.rect(x, y, bar_w, h, style="F",
                     round_corners=True, corner_radius=bar_w / 2)
        except TypeError:
            pdf.rect(x, y, bar_w, h, "F")

    # ── "Thai.Law" wordmark ──
    text_y = top_y + max_h + 5
    pdf.set_xy(0, text_y)
    pdf.set_font("Main", "B", 22)
    pdf.set_text_color(*LOGO_DARK)
    pdf.cell(A4_W, 9, "Thai.Law", align="C", ln=1)

    # ── "AI Legal Assistant" subtitle (letter-spaced) ──
    pdf.set_font("Main", "B", 8)
    pdf.set_text_color(*LOGO_SUB)
    pdf.cell(A4_W, 4, "A I   L E G A L   A S S I S T A N T", align="C", ln=1)

    return pdf.get_y()


def _add_cover(pdf, case_title: str, meta: dict, perspective: str, lang: str) -> None:
    """Minimal, formal cover page:
        Logo (top-center) → Title (upper-middle) → Case No. (if any) → Disclaimer (footer).
    Uses absolute Y positioning for a balanced, predictable layout."""
    pdf.add_page()

    # ─── Logo (top-center) ──────────────────────────────────────────────
    _draw_logo(pdf, top_y=38)

    # ─── Title block (upper-middle) ─────────────────────────────────────
    pdf.set_y(108)
    pdf.set_font("Main", "B", 28)
    pdf.set_text_color(*BLUE)
    title = (case_title or meta.get("title") or "Thai.Law Case Analysis").strip()
    pdf.multi_cell(CONTENT_W, 13, title, align="C")
    pdf.ln(2)

    # Short centered accent divider
    cx  = A4_W / 2
    div_y = pdf.get_y()
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.7)
    pdf.line(cx - 22, div_y, cx + 22, div_y)
    pdf.ln(11)

    # ─── Case number (if available) ─────────────────────────────────────
    case_number = meta.get("case_number")
    if _is_filled(case_number):
        label = {
            "th": "เลขที่คดี",
            "en": "CASE NO.",
            "zh": "案件编号",
            "ja": "事件番号",
            "ko": "사건번호",
            "ru": "№ ДЕЛА",
            "fr": "N° D'AFFAIRE",
            "ar": "رقم القضية",
        }.get(lang, "CASE NO.")

        pdf.set_font("Main", "B", 8)
        pdf.set_text_color(*GRAY)
        pdf.cell(CONTENT_W, 4, label.upper() if lang == "en" else label, ln=1, align="C")
        pdf.ln(1)

        pdf.set_font("Main", "B", 14)
        pdf.set_text_color(*DARK)
        pdf.cell(CONTENT_W, 7, str(case_number), ln=1, align="C")

    # ─── Footer disclaimer (pinned to bottom) ───────────────────────────
    disc = DISCLAIMER_TEXT.get(lang, DISCLAIMER_TEXT["en"])

    # Reserve a generous bottom area for the disclaimer (it wraps to 3-4 lines).
    pdf.set_y(A4_H - MARGIN_B - 28)

    # Thin hairline above the disclaimer, indented to feel formal
    pdf.set_draw_color(*CARD_BORDER)
    pdf.set_line_width(0.2)
    pdf.line(MARGIN_L + 22, pdf.get_y(), A4_W - MARGIN_R - 22, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Main", "", 8)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(CONTENT_W, 4.2, disc, align="C")


# ── Markdown table renderer ───────────────────────────────────────────────────


def _wrap_cell_text(pdf, text: str, max_w: float) -> list:
    """Character-level word wrap — works correctly for CJK (no spaces),
    Thai, and Latin text alike. Returns a list of lines that each fit in max_w."""
    text = (text or "").replace("\r", "")
    if not text:
        return [""]
    out_lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            out_lines.append("")
            continue
        line = ""
        for ch in paragraph:
            test = line + ch
            try:
                if pdf.get_string_width(test) <= max_w - 1.0:
                    line = test
                else:
                    if line:
                        out_lines.append(line)
                    line = ch
            except Exception:
                if line:
                    out_lines.append(line)
                line = ch
        if line:
            out_lines.append(line)
    return out_lines or [""]


def _compute_col_widths(pdf, head, body, col_count) -> list:
    """Distribute content width proportionally to natural text length per column.
    Each column gets at least MIN mm, at most CONTENT_W * 0.45. Remainder is
    allocated by ratio of max(content-width-needed) per column."""
    MIN_W = 16.0
    MAX_W = CONTENT_W * 0.45

    all_rows = ([head] if head else []) + body
    if not all_rows:
        return [CONTENT_W / col_count] * col_count

    # Measure widest "natural" (unwrapped) width per column
    needs = [0.0] * col_count
    for row in all_rows:
        padded = (row + [""] * col_count)[:col_count]
        for i, c in enumerate(padded):
            txt = _strip_md(c)
            # Cap measurement at MAX_W so a single huge cell doesn't dominate
            try:
                w = min(pdf.get_string_width(txt) + 4, MAX_W)
            except Exception:
                w = MIN_W
            if w > needs[i]:
                needs[i] = w

    # Floor at MIN_W
    needs = [max(MIN_W, n) for n in needs]
    total = sum(needs)

    # Scale to fit CONTENT_W
    if total <= CONTENT_W:
        # Distribute leftover proportionally
        scale = CONTENT_W / total
        widths = [n * scale for n in needs]
    else:
        # Need to shrink — clamp each column above the floor proportionally
        excess = total - CONTENT_W
        # Sort indices by how much room they have above MIN_W
        slack = [needs[i] - MIN_W for i in range(col_count)]
        total_slack = sum(slack)
        if total_slack <= 0:
            widths = [CONTENT_W / col_count] * col_count
        else:
            widths = [needs[i] - excess * (slack[i] / total_slack) for i in range(col_count)]

    # Final safety clamp
    widths = [max(MIN_W, w) for w in widths]
    s = sum(widths)
    if s > 0:
        widths = [w * (CONTENT_W / s) for w in widths]
    return widths


def _render_md_table(pdf, table_lines: list) -> None:
    def parse_row(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]

    rows = [parse_row(l) for l in table_lines if l.strip()]
    if not rows:
        return

    sep_idx = None
    for idx, row in enumerate(rows):
        if all(re.match(r"^:?-+:?$", c) for c in row if c):
            sep_idx = idx
            break

    head = rows[0] if sep_idx == 1 else None
    body = rows[sep_idx + 1:] if sep_idx is not None else rows

    col_count = len(head if head else (body[0] if body else []))
    if col_count == 0:
        return

    # Compute proportional column widths from actual content
    pdf.set_font("Main", "", 9)
    col_widths = _compute_col_widths(pdf, head, body, col_count)

    LINE_H  = 4.5    # line height inside a cell
    PAD_Y   = 1.2    # vertical padding inside cell
    PAD_X   = 1.5    # horizontal padding inside cell

    def render_row(row, is_header: bool, row_idx: int = 0):
        cells = [_strip_md(c) for c in (row + [""] * col_count)[:col_count]]

        # Set font for measuring
        pdf.set_font("Main", "B" if is_header else "", 9)

        # Wrap each cell + find max line count
        wrapped = []
        max_lines = 1
        for i, txt in enumerate(cells):
            lines = _wrap_cell_text(pdf, txt, col_widths[i] - 2 * PAD_X)
            wrapped.append(lines)
            if len(lines) > max_lines:
                max_lines = len(lines)

        row_h = max(LINE_H * max_lines + 2 * PAD_Y, 6.0)

        # Page break if needed
        if pdf.get_y() + row_h > A4_H - MARGIN_B:
            pdf.add_page()

        y0 = pdf.get_y()
        x0 = MARGIN_L

        # Background + border (draw rectangles per cell)
        if is_header:
            pdf.set_fill_color(*GREEN)
            text_color = WHITE
        else:
            if row_idx % 2 == 1:
                pdf.set_fill_color(*LIGHT_GRAY)
                fill = True
            else:
                pdf.set_fill_color(*WHITE)
                fill = True
            text_color = DARK

        pdf.set_draw_color(200, 210, 225)
        pdf.set_line_width(0.2)

        x = x0
        for i in range(col_count):
            pdf.rect(x, y0, col_widths[i], row_h, style="FD")
            x += col_widths[i]

        # Text on top
        pdf.set_text_color(*text_color)
        x = x0
        for i, lines in enumerate(wrapped):
            cell_w = col_widths[i]
            # Vertical centering for short content
            extra_space = row_h - (LINE_H * len(lines)) - 2 * PAD_Y
            cy = y0 + PAD_Y + max(0, extra_space / 2)
            for line in lines:
                pdf.set_xy(x + PAD_X, cy)
                pdf.cell(cell_w - 2 * PAD_X, LINE_H, line, border=0)
                cy += LINE_H
            x += cell_w

        pdf.set_xy(MARGIN_L, y0 + row_h)

    pdf.ln(2)

    if head:
        render_row(head, is_header=True)

    for r_idx, row in enumerate(body):
        render_row(row, is_header=False, row_idx=r_idx)

    pdf.set_fill_color(*WHITE)
    pdf.set_text_color(*DARK)
    pdf.set_draw_color(*GRAY)
    pdf.ln(3)


# ── Markdown content renderer ─────────────────────────────────────────────────

def _render_markdown(pdf, markdown: str) -> None:
    """Parse markdown line-by-line and emit fpdf2 output directly."""

    HSIZES  = {"#": 17, "##": 14, "###": 12, "####": 11}
    HCOLORS = {"#": BLUE, "##": GREEN, "###": DARK, "####": DARK}

    # fpdf2 raises FPDFUnicodeEncodingException when a glyph isn't in the
    # current font's character set (most commonly: box-drawing chars `│` `─`,
    # decorative arrows, or emoji on fonts without the right CMAP table).
    # That exception used to bubble all the way out and cut the PDF off after
    # the first unsupported line. Strip the offenders to a safe equivalent so
    # the render keeps going. Keep the set deliberately narrow — we still want
    # supported Unicode (Thai, Chinese, common arrows) to pass through.
    _UNSUPPORTED_REPLACE = {
        "│": " ",  "─": " ",  "┃": " ",  "━": " ",
        "└": " ",  "┘": " ",  "┌": " ",  "┐": " ",
        "├": " ",  "┤": " ",  "┬": " ",  "┴": " ",  "┼": " ",
    }
    def _sanitize(s: str) -> str:
        # Quick path: if none of the offenders appear, return unchanged
        if not any(c in s for c in _UNSUPPORTED_REPLACE):
            return s
        return "".join(_UNSUPPORTED_REPLACE.get(c, c) for c in s)

    lines = [_sanitize(l) for l in str(markdown or "").split("\n")]
    i = 0
    while i < len(lines):
        raw  = lines[i]
        line = raw.rstrip()

        # Blank line
        if not line.strip():
            pdf.ln(2)
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^---+$", line.strip()):
            pdf.set_draw_color(*GRAY)
            pdf.set_line_width(0.3)
            pdf.ln(2)
            pdf.line(MARGIN_L, pdf.get_y(), A4_W - MARGIN_R, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue

        # (Note: previous "orphan Flowchart heading skip" removed — we now
        # actually render the Mermaid block as an embedded PNG, so the
        # accompanying section heading "## 3.2 Flowchart Mermaid" is no
        # longer orphan and should be rendered normally.)

        # Fenced code block — ```mermaid is rendered to PNG via mermaid.ink
        # SVG diagrams in fpdf2 and showing the raw code would just be noise).
        # Other languages render as a tinted monospace block so JSON/code
        # snippets stay readable.
        #
        # Regex is intentionally permissive:
        #   - leading whitespace                    (\s*)
        #   - 3+ backticks OR 3+ tildes             ( ```+ | ~~~+ )
        #   - optional language identifier          ([a-zA-Z_][\w-]*)?
        #   - any trailing whitespace incl \r       (\s*$)
        # Some markdown emitters use 4+ backticks or ~~~ as the fence char,
        # and Anthropic streams occasionally include a trailing \r before \n —
        # the previous tighter regex missed those and dumped the raw source.
        fence_m = re.match(r"^\s*(?:`{3,}|~{3,})\s*([a-zA-Z_][\w-]*)?\s*\r?$", line)
        if fence_m:
            lang = (fence_m.group(1) or "").lower()
            j = i + 1
            # Match a closing fence of the SAME family (backticks vs tildes)
            close_re = re.compile(r"^\s*(?:`{3,}|~{3,})\s*\r?$")
            while j < len(lines) and not close_re.match(lines[j]):
                j += 1
            code_lines = lines[i + 1:j]
            if lang in ("mermaid", "mmd"):
                # Mermaid flowcharts are intentionally skipped in PDF output.
                # We tried server-side rendering (mermaid.ink → PNG) and
                # browser pre-rendering through several iterations — both kept
                # producing layout / sizing issues. The user prefers PDFs
                # without flowcharts to PDFs with broken flowcharts. The
                # on-screen UI still renders Mermaid as interactive SVG.
                pass
            else:
                # Generic code block — render as monospace tinted box
                pdf.ln(1)
                pdf.set_fill_color(245, 248, 252)
                pdf.set_draw_color(*GRAY)
                try:
                    pdf.set_font("Mono", size=9)
                except Exception:
                    pdf.set_font("Main", size=9)
                for cl in code_lines:
                    pdf.set_text_color(50, 70, 100)
                    pdf.multi_cell(CONTENT_W, 5, cl, fill=True)
                pdf.set_font("Main", size=11)
                pdf.set_text_color(*DARK)
                pdf.ln(2)
            i = j + 1  # skip past closing fence
            continue

        # Heading
        m = re.match(r"^(#{1,4})\s+(.+)$", line)
        if m:
            hashes = m.group(1)
            text   = _strip_md(m.group(2))
            size   = HSIZES.get(hashes, 11)
            color  = HCOLORS.get(hashes, DARK)
            pdf.ln(5 if hashes == "#" else 3)
            pdf.set_font("Main", "B", size)
            pdf.set_text_color(*color)
            pdf.multi_cell(CONTENT_W, 7 if len(hashes) <= 2 else 6, text)
            if hashes == "#":
                pdf.set_draw_color(*GREEN)
                pdf.set_line_width(0.4)
                pdf.line(MARGIN_L, pdf.get_y(), A4_W - MARGIN_R, pdf.get_y())
            pdf.ln(2)
            pdf.set_font("Main", size=11)
            pdf.set_text_color(*DARK)
            i += 1
            continue

        # Table — collect consecutive pipe lines
        if re.match(r"^\s*\|.+\|", line):
            tbl = []
            while i < len(lines) and re.match(r"^\s*\|.+\|", lines[i].rstrip()):
                tbl.append(lines[i].strip())
                i += 1
            _render_md_table(pdf, tbl)
            continue

        # Bullet
        if re.match(r"^[•\-]\s+", line):
            text = _strip_md(re.sub(r"^[•\-]\s+", "", line))
            pdf.set_font("Main", size=11)
            pdf.set_text_color(*DARK)
            pdf.set_x(MARGIN_L + 4)
            pdf.multi_cell(CONTENT_W - 4, 6, "•  " + text)
            pdf.ln(1)
            i += 1
            continue

        # Numbered list
        m = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m:
            text = _strip_md(m.group(2))
            pdf.set_font("Main", size=11)
            pdf.set_text_color(*DARK)
            pdf.set_x(MARGIN_L + 4)
            pdf.multi_cell(CONTENT_W - 4, 6, f"{m.group(1)}. {text}")
            pdf.ln(1)
            i += 1
            continue

        # Plain paragraph — wrapped in try/except so an unsupported glyph
        # we didn't catch in _sanitize() still doesn't truncate the rest of
        # the document. On failure, fall back to a best-effort ASCII version
        # of the line (drops the unsupported chars) instead of bailing.
        text = _strip_md(line)
        if text:
            pdf.set_font("Main", size=11)
            pdf.set_text_color(*DARK)
            try:
                pdf.multi_cell(CONTENT_W, 6, text)
            except Exception as e:
                # Strip every non-BMP/non-Thai/non-ASCII codepoint and retry.
                safe = "".join(c for c in text if (
                    ord(c) < 0x80 or                          # ASCII
                    0x0E00 <= ord(c) <= 0x0E7F or             # Thai block
                    0x4E00 <= ord(c) <= 0x9FFF or             # CJK
                    0x2010 <= ord(c) <= 0x203F                # General punctuation
                ))
                try:
                    if safe.strip():
                        pdf.multi_cell(CONTENT_W, 6, safe)
                except Exception:
                    # Last resort: log and continue. Better an empty line
                    # than a half-truncated PDF.
                    print(f"[pdf_generator] skipped unrenderable line: {e}")
            pdf.ln(1)
        i += 1


# ── Public API ────────────────────────────────────────────────────────────────

def generate_pdf(
    markdown: str,
    lang: str = "th",
    case_meta: dict | None = None,
    perspective: str = "",
) -> bytes:
    meta = case_meta or {}
    case_title = meta.get("title") or "Thai.Law Analysis"

    pdf = _build_fpdf(lang)
    _add_cover(pdf, case_title, meta, perspective, lang)

    pdf.add_page()
    _render_markdown(pdf, markdown)

    # Footer
    pdf.ln(6)
    footer = {
        "th": "⚠ Thai.Law — เอกสารนี้สร้างขึ้นอัตโนมัติ ไม่ใช่คำปรึกษาทางกฎหมาย",
        "en": "⚠ Thai.Law — Auto-generated. Not legal advice.",
        "zh": "⚠ Thai.Law — 自动生成，不构成法律建议。",
        "ja": "⚠ Thai.Law — 自動生成。法的助言ではありません。",
        "ko": "⚠ Thai.Law — 자동 생성. 법률 자문 아님.",
        "ru": "⚠ Thai.Law — Сгенерировано автоматически. Не является юридической консультацией.",
        "fr": "⚠ Thai.Law — Généré automatiquement. Ne constitue pas un conseil juridique.",
        "ar": "⚠ Thai.Law — تم إنشاؤه تلقائيًا. ليس استشارة قانونية.",
    }.get(lang, "")
    pdf.set_font("Main", size=8)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(CONTENT_W, 5, footer, align="C")

    return bytes(pdf.output())
