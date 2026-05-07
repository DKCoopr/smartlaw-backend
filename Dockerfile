FROM python:3.11-slim

# Install system dependencies:
#   ffmpeg   → pydub audio processing
#   antiword → extract text from legacy .doc files (Word binary format)
#   catdoc   → fallback when antiword can't parse a particular .doc
RUN apt-get update && apt-get install -y \
    ffmpeg \
    antiword \
    catdoc \
    && rm -rf /var/lib/apt/lists/*

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
