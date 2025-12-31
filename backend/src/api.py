"""Deprecated API module.

All endpoints were consolidated into the top-level `main.py`. Importing
this module will raise to avoid accidental usage.
"""
raise ImportError("backend.src.api has been removed; use main:app instead")


# --- Minimal research session manager (in-memory) ---
from typing import Optional, Dict, Any
from uuid import uuid4

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
                    'scrape_content': content,
                    'sources': sources,

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
            'scrape_content': analysis,
            'sources': [],
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
