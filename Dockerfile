# This Dockerfile builds the final application image using a custom base image
# where all heavy dependencies are pre-installed.
# Update the FROM line to point to your base image in your container registry.
# Use the updated base image with Playwright system deps installed
FROM asia-south2-docker.pkg.dev/gen-lang-client-0665888431/deepsearch-repo/deepsearch-base:1.1
# FROM deepsearch-base:1.0
WORKDIR /app
# Ensure required runtime system packages are present (non-interactive)
# This installs dbus and keyboard libraries needed by Chromium/Playwright.
USER root
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gosu dbus libxkbcommon0 xkb-data \
 && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# The entrypoint will run as root to set up dbus, then use gosu to drop to the 'appuser'
# before executing the main application.

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--timeout", "120", "--log-level", "info"]
