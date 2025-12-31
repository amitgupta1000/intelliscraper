"""Minimal configuration for scraper-only deployments.

This file provides a small set of settings that the scraper backend needs:
- AIOHTTP timeout and concurrency
- Playwright fallback toggle
- Minimal logging level

Keep this file under version control and load it in your scraper entrypoint
when you want to run only the scraping functionality.
"""
import os

SCRAPER_ENABLE_PLAYWRIGHT = bool(os.getenv('SCRAPER_ENABLE_PLAYWRIGHT', 'True') == 'True')
AIOHTTP_REQUEST_TIMEOUT = int(os.getenv('AIOHTTP_REQUEST_TIMEOUT', '30'))
MAX_CONCURRENT_SCRAPES = int(os.getenv('MAX_CONCURRENT_SCRAPES', '4'))
DEFAULT_USER_AGENT = os.getenv('DEFAULT_USER_AGENT', 'intelliscraper/1.0 (+https://example.com)')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Storage toggles (disable by default for local scraper-only runs)
FIRESTORE_ENABLED = bool(os.getenv('FIRESTORE_ENABLED', 'False') == 'True')
GCS_ENABLED = bool(os.getenv('GCS_ENABLED', 'False') == 'True')
GCS_BUCKET = os.getenv('GCS_BUCKET', '')

SCRAPE_RESULT_TTL = int(os.getenv('SCRAPE_RESULT_TTL', '864000'))  # seconds (default: 10 days)
