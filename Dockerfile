# ---------------------------------------------------------
# .dockerignore — create this file in the project root with:
#
#   data/
#   .git/
#   __pycache__/
#   *.pyc
#   .env
#   *.egg-info/
#   .venv/
#   oos/
#   README.md
#   README_2.md
#
# ---------------------------------------------------------

FROM python:3.11-slim

WORKDIR /app

# System deps for pyarrow / numpy
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (no blpapi — no Bloomberg Terminal in Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY download_ohlcv.py plotly_dash.py run_pipeline.py ./
COPY src/ src/
COPY assets/ assets/

# Create data directory
RUN mkdir -p data

# Run the yfinance-only pipeline at build time (steps 1-2, 4-5; skip 3)
# This bakes the data into the image so startup is instant.
RUN python run_pipeline.py

EXPOSE 8050

CMD ["python", "plotly_dash.py"]
