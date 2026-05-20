FROM python:3.11-slim

# Install system dependencies:
#   ffmpeg   → pydub audio processing
#   antiword → extract text from legacy .doc files (Word binary format)
#   catdoc   → fallback when antiword can't parse a particular .doc
RUN apt-get update && apt-get install -y \
    ffmpeg \
    antiword \
    catdoc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download fonts for PDF generation (embedded directly in PDFs via fpdf2)
# Sarabun: Thai + Latin (official Thai government font, covers all ASCII + Thai glyphs)
# NotoSansCJKsc: Simplified Chinese — full CJK font, ~12MB, includes all common SC glyphs
# Using jsdelivr CDN (mirrors GitHub) for reliability + size check to fail build if download is broken.
RUN mkdir -p /app/fonts && \
    curl -fsSL "https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/sarabun/Sarabun-Regular.ttf" \
         -o /app/fonts/Sarabun-Regular.ttf && \
    curl -fsSL "https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/sarabun/Sarabun-Bold.ttf" \
         -o /app/fonts/Sarabun-Bold.ttf && \
    curl -fsSL "https://cdn.jsdelivr.net/gh/notofonts/noto-cjk@main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf" \
         -o /app/fonts/NotoSansSC-Regular.otf && \
    sz_th="$(stat -c%s /app/fonts/Sarabun-Regular.ttf)" && \
    sz_zh="$(stat -c%s /app/fonts/NotoSansSC-Regular.otf)" && \
    echo "Font sizes — Thai: ${sz_th} bytes, Chinese: ${sz_zh} bytes" && \
    [ "$sz_th" -gt 50000  ] || (echo "ERROR: Thai font download corrupted"    && exit 1) && \
    [ "$sz_zh" -gt 1000000 ] || (echo "ERROR: Chinese font download corrupted" && exit 1)

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Start server
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
