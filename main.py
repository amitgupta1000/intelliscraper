"""Top-level WSGI/ASGI entrypoint for GCP Cloud Run.

This module exposes the FastAPI `app` object so platforms that expect
`main:app` (or `app:app`) can import it.

It also provides a convenient local runner when executed directly.
"""
from backend.src.api import app  # expose the FastAPI app at module level
import asyncio
import sys
import logging
import os
from starlette.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bootstrap")

# Configure CORS: allow origins from env `FRONTEND_ORIGINS` (comma-separated)
# Fallback to common local dev origins used by Vite
_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:3002",
    "http://127.0.0.1:3002",
    "*"
]
raw = os.environ.get("FRONTEND_ORIGINS")
if raw:
    origins = [o.strip() for o in raw.split(",") if o.strip()]
else:
    origins = _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            policy = asyncio.get_event_loop_policy()
            if type(policy).__name__ != "WindowsProactorEventLoopPolicy":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                log.info("Set WindowsProactorEventLoopPolicy for this process")
        except Exception as e:
            log.warning(f"Failed to set WindowsProactorEventLoopPolicy: {e}")

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info")
