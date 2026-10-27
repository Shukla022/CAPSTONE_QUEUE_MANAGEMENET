# Multi-Objective AI-Driven Cloud-Based Queue Management System (QMS)
# Production Container Image for GCP Cloud Run, AWS App Runner, Azure Container Apps, or Render

FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

WORKDIR /app

# Install minimal OS dependencies for healthchecks and builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first for efficient layer caching
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[ml,app,genai]"

# Copy application source code, models, and preprocessed data
COPY . .

# Run pre-flight verification tests during image build
RUN python tests/test_smoke.py && python tests/test_novelty.py

# Expose default Streamlit port
EXPOSE 8501

# Healthcheck to verify app is serving
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Launch the Streamlit Multi-Page Web Portal
ENTRYPOINT ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false"]
