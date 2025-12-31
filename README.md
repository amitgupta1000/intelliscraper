Backend (minimal scraper)
=========================

This `backend/` folder contains a minimal scraper-only backend for the
project. It exposes a FastAPI endpoint at `/scrape` which accepts JSON
{ "url": "https://..." } and returns the scraped content as Markdown.

Key files:
- `src/api.py` — FastAPI app with `/scrape` endpoint
- `src/scraper.py` — Scraper implementation (aiohttp primary, Playwright fallback)
- `src/utils.py` — helpers: HTML->Markdown, cleaning, GCS upload
- `config_minimal.py` — minimal runtime knobs (set `GCS_BUCKET` to upload results)

GCS Uploads
-----------
Set the environment variable `GCS_BUCKET` or edit `src/config_minimal.py`.
If `google-cloud-storage` is installed and credentials are available, the
service will attempt a signed URL. Otherwise it will return a public
`https://storage.googleapis.com/<bucket>/<object>` URL.

Running locally (quick test):

1. Install dev dependencies (see `requirements-scraper.txt`).
2. From repository root run:

```powershell
python backend\run_scrape_direct.py
```

Deployment
----------
Use `uvicorn backend.src.api:app --host 0.0.0.0 --port 8080` to run the
service in production (behind a reverse proxy). Ensure credentials for
GCS are provided when enabling uploads.
