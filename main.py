import asyncio
import sys
from contextlib import asynccontextmanager
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List

# --- Third-party imports ---
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from backend.src.logging_setup import logger, start_logging, stop_logging
from typing import Optional, Dict, Any
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.src.scraper import Scraper

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    - Initializes logging.
    - Starts and stops the shared scraper instance.
    """
    # --- Startup Logic ---
    start_logging()
    logger.debug("Logging listener started.")
    SCRAPER_ENABLE_PLAYWRIGHT = 'True'


    if SCRAPER_ENABLE_PLAYWRIGHT:
        logger.debug("Application startup: Initializing and starting shared scraper instance...")
        try:
            scraper = Scraper()
            await scraper.start()
            logger.info("Shared scraper instance started successfully (browser is warm).")
        except Exception as e:
            logger.exception(f"Failed to start shared scraper at startup: {e}")

    yield

    # --- Shutdown Logic ---
    if SCRAPER_ENABLE_PLAYWRIGHT and hasattr(scraper, 'stop'):
        await scraper.stop()
        logger.info("Shared scraper instance closed successfully.")
    stop_logging()
    logger.debug("Logging listener stopped. API Shutting Down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:5173",
        "http://localhost:8000",
        "https://intelliscraper-44c6f.web.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SESSIONS = 50

# --- /scrape endpoint (original backend/src/api.py) ---
class ScrapeRequest(BaseModel):
    url: str

@app.post('/scrape')
async def scrape(req: ScrapeRequest):
    try:
        scraper = Scraper()
        # prefer aiohttp path; Scraper.scrape may be async
        if asyncio.iscoroutinefunction(scraper.scrape):
            scraped = await scraper.scrape(req.url)
        else:
            loop = asyncio.get_event_loop()
            scraped = await loop.run_in_executor(None, scraper.scrape, req.url)

        # Prefer `markdown` when available, fall back to `text`.
        markdown = None
        text = None
        gcs_path = None

        if hasattr(scraped, 'markdown'):
            markdown = getattr(scraped, 'markdown')
        elif isinstance(scraped, dict) and 'markdown' in scraped:
            markdown = scraped.get('markdown')

        if hasattr(scraped, 'text'):
            text = getattr(scraped, 'text')
        elif isinstance(scraped, dict) and 'text' in scraped:
            text = scraped.get('text')

        if hasattr(scraped, 'gcs_path'):
            gcs_path = getattr(scraped, 'gcs_path')
        elif isinstance(scraped, dict) and 'gcs_path' in scraped:
            gcs_path = scraped.get('gcs_path')

        # Choose primary content to return
        primary = markdown if markdown else text
        if not primary:
            raise HTTPException(status_code=500, detail='Scraper returned no content')

        # If we have a GCS path, try to produce a usable download URL:
        download_url = None
        if gcs_path:
            try:
                # parse gs://bucket/name
                bucket = None
                blob_name = None
                if isinstance(gcs_path, str) and gcs_path.startswith('gs://'):
                    _, rest = gcs_path.split('://', 1)
                    parts = rest.split('/', 1)
                    bucket = parts[0]
                    blob_name = parts[1] if len(parts) > 1 else ''
                elif isinstance(gcs_path, str) and gcs_path.startswith('https://storage.googleapis.com/'):
                    # https://storage.googleapis.com/bucket/path
                    rest = gcs_path[len('https://storage.googleapis.com/'):]
                    parts = rest.split('/', 1)
                    bucket = parts[0]
                    blob_name = parts[1] if len(parts) > 1 else ''

                if bucket and blob_name:
                    try:
                        # Try to create a signed URL if google-cloud-storage is available
                        from google.cloud import storage as gcs_lib
                        client = gcs_lib.Client()
                        bucket_obj = client.bucket(bucket)
                        blob = bucket_obj.blob(blob_name)
                        try:
                            # v4 signed URL if supported
                            download_url = blob.generate_signed_url(version='v4', expiration=3600)
                        except Exception:
                            # Fallback to a public URL format
                            from urllib.parse import quote
                            download_url = f"https://storage.googleapis.com/{bucket}/{quote(blob_name)}"
                    except Exception:
                        # google lib not available or signing failed; give public URL
                        from urllib.parse import quote
                        download_url = f"https://storage.googleapis.com/{bucket}/{quote(blob_name)}" if bucket and blob_name else None
            except Exception:
                download_url = None

        return {
            'url': getattr(scraped, 'url', req.url) if not isinstance(scraped, dict) else scraped.get('url', req.url),
            'title': getattr(scraped, 'title', None) if not isinstance(scraped, dict) else scraped.get('title'),
            'markdown': primary,
            'gcs_path': gcs_path,
            'download_url': download_url,
            'text': text if text and text != primary else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/', include_in_schema=False)
async def root():
    return {'message': 'intelliscraper backend. Use POST /scrape to submit a URL.'}


@app.get('/health', include_in_schema=False)
async def health():
    return {'status': 'ok'}


# --- Minimal research session manager (in-memory) ---
SESSIONS: Dict[str, Dict[str, Any]] = {}


class ResearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    prompt_type: Optional[str] = None


def _make_session(initial_query: str) -> str:
    sid = uuid4().hex
    SESSIONS[sid] = {
        'status': 'initializing',
        'progress': 0,
        'current_step': 'queued',
        'error': None,
        'conclusion_message': None,
        'processing_urls': [],
        'used_cache': False,
        'query': initial_query,
        'result': None,
    }
    return sid


async def _run_research_background(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        return
    try:
        sess['status'] = 'running'
        query = sess.get('query', '')
        # If query looks like a URL, use the Scraper to fetch it and use scraped markdown/text
        if isinstance(query, str) and query.startswith(('http://', 'https://')):
            sess['current_step'] = 'scraping'
            try:
                scraper = Scraper()
                if asyncio.iscoroutinefunction(scraper.scrape):
                    scraped = await scraper.scrape(query)
                else:
                    loop = asyncio.get_event_loop()
                    scraped = await loop.run_in_executor(None, scraper.scrape, query)

                # prefer markdown then text
                content = None
                sources = []
                if hasattr(scraped, 'markdown') and getattr(scraped, 'markdown'):
                    content = getattr(scraped, 'markdown')
                elif isinstance(scraped, dict) and scraped.get('markdown'):
                    content = scraped.get('markdown')
                elif hasattr(scraped, 'text') and getattr(scraped, 'text'):
                    content = getattr(scraped, 'text')
                elif isinstance(scraped, dict) and scraped.get('text'):
                    content = scraped.get('text')
                else:
                    content = str(scraped)

                # collect source URL if available
                if hasattr(scraped, 'url') and getattr(scraped, 'url'):
                    sources.append(getattr(scraped, 'url'))
                elif isinstance(scraped, dict) and scraped.get('url'):
                    sources.append(scraped.get('url'))

                sess['result'] = {
                    'analysis_content': content,
                    'appendix_content': None,
                    'analysis_filename': None,
                    'appendix_filename': None,
                    'sources': sources,
                    'qa_pairs': [],
                }
                sess['progress'] = 100
                sess['status'] = 'completed'
                sess['current_step'] = 'done'
                return
            except Exception as e:
                sess['status'] = 'failed'
                sess['error'] = str(e)
                return
        # Otherwise, treat query as a freeform prompt: for now echo it as analysis
        sess['current_step'] = 'analyzing'
        sess['progress'] = 50
        await asyncio.sleep(0.5)
        analysis = f"Echo analysis for query: {query}\n\n(No scraping performed; to analyze URLs provide a URL)"
        sess['result'] = {
            'analysis_content': analysis,
            'appendix_content': None,
            'analysis_filename': None,
            'appendix_filename': None,
            'sources': [],
            'qa_pairs': [],
        }
        sess['progress'] = 100
        sess['status'] = 'completed'
        sess['current_step'] = 'done'
    except Exception as e:
        sess['status'] = 'failed'
        sess['error'] = str(e)


@app.post('/research')
async def start_research(req: ResearchRequest):
    try:
        sid = req.session_id or _make_session(req.query)
        # If new session, start background processing
        if sid not in SESSIONS:
            sid = _make_session(req.query)

        # If session is initializing, kick off background task
        if SESSIONS[sid]['status'] in ('initializing', 'pending'):
            asyncio.create_task(_run_research_background(sid))

        return {'session_id': sid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/research/{session_id}/status')
async def research_status(session_id: str):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail='session not found')
    return {
        'status': s['status'],
        'progress': s.get('progress', 0),
        'current_step': s.get('current_step', ''),
        'error': s.get('error'),
        'conclusion_message': s.get('conclusion_message'),
        'processing_urls': s.get('processing_urls', []),
        'used_cache': s.get('used_cache', False),
    }


@app.get('/research/{session_id}/result')
async def research_result(session_id: str):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail='session not found')
    if s.get('status') != 'completed' or not s.get('result'):
        raise HTTPException(status_code=409, detail='result not available')
    return s['result']


@app.post('/research/{session_id}/conclude')
async def conclude_research(session_id: str):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail='session not found')
    s['status'] = 'concluded'
    s['conclusion_message'] = 'Concluded by user'
    return {'success': True, 'message': 'concluded'}


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            policy = asyncio.get_event_loop_policy()
            if type(policy).__name__ != "WindowsProactorEventLoopPolicy":
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                logger.info("Set WindowsProactorEventLoopPolicy for this process")
        except Exception as e:
            logger.warning(f"Failed to set WindowsProactorEventLoopPolicy: {e}")

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info")
