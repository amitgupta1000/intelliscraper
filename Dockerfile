# Dockerfile for GCP Cloud Run
FROM mcr.microsoft.com/playwright/python:v1.50.0-noble

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Ensure a non-root user exists
RUN id appuser || useradd -m -s /bin/bash appuser || true

# Copy application code (preserve ownership)
# Copy backend directory; do not attempt to copy module specifiers like "backend.src.api"
COPY --chown=appuser:appuser backend/ ./backend/

# Install Python dependencies from backend requirements
RUN pip install --no-cache-dir -r backend/requirements-scraper.txt

# Ensure Playwright browsers are present (base image typically includes them,
# but run install to be safe in case of variant images)
RUN python -m playwright install --with-deps || true

# Ensure playwright browser path is writable by the app user (Playwright image uses /ms-playwright)
RUN mkdir -p /ms-playwright && chown -R appuser:appuser /ms-playwright || true

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
	CMD curl -f http://localhost:8080/api/health || exit 1

# Switch to non-root user
USER appuser

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "backend.src.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--log-level", "info"]