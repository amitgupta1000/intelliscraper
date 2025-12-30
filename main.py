import os
import uuid
import sys, asyncio
from datetime import datetime
from typing import Any, Dict, Optional, List
from contextlib import asynccontextmanager

# --- Third-party imports ---
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from backend.src.logging_setup import logger, start_logging, stop_logging

# --- Application-specific imports for lifecycle management ---
from backend.src import nodes
from backend.src.config import SCRAPER_ENABLE_PLAYWRIGHT

# --- Firestore and LangChain Setup ---
try:
    from google.cloud import firestore
    db = firestore.Client()
except ImportError:
    db = None
    logger.warning("google-cloud-firestore not installed. Firestore features will be disabled.")

try:
    from langchain_core.runnables import RunnableConfig
    config = RunnableConfig(recursion_limit=100)
except Exception:
    config = None

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")


#==============
from backend.src.nodes import SHARED_SCRAPER

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

    if SCRAPER_ENABLE_PLAYWRIGHT:
        logger.debug("Application startup: Initializing and starting shared scraper instance...")
        try:
            await SHARED_SCRAPER.start()
            logger.info("Shared scraper instance started successfully (browser is warm).")
        except Exception as e:
            logger.exception(f"Failed to start shared scraper at startup: {e}")

    yield

    # --- Shutdown Logic ---
    if SCRAPER_ENABLE_PLAYWRIGHT and hasattr(SHARED_SCRAPER, 'stop'):
        await SHARED_SCRAPER.stop()
        logger.info("Shared scraper instance closed successfully.")
    stop_logging()
    logger.debug("Logging listener stopped. API Shutting Down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://localhost:8000",
        "https://deepsearch-56755551-95627.web.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SESSIONS = 50

# --- Pydantic Models ---
class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research query or question")
    prompt_type: Optional[str] = Field(default="general", description="Type of research prompt to use.")
    session_id: Optional[str] = Field(default=None, description="ID of an existing session for follow-up questions.")

# The rest of the Pydantic models remain the same
class ResearchSession(BaseModel):
    session_id: str
    query: str
    status: str
    created_at: datetime
    updated_at: datetime
    progress: int = 0
    current_step: str = ""
    analysis_content: Optional[str] = None
    conclusion_message: Optional[str] = None
    analysis_filename: Optional[str] = None
    error_message: Optional[str] = None

class ResearchStatus(BaseModel):
    session_id: str
    status: str
    progress: int
    current_step: str
    conclusion_message: Optional[str] = None
    estimated_completion: Optional[datetime] = None

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None

#==============
from backend.src.graph import app as workflow_app

async def run_workflow(initial_query: str, session_id: str, prompt_type: str, qa_pairs: Optional[List[Dict]] = None) -> Dict[str, Any]:
    if workflow_app is None:
        raise RuntimeError("Workflow not compiled. LangGraph not available.")

    from backend.src.config import (
        MAX_SEARCH_QUERIES, MAX_SEARCH_RESULTS, MAX_AI_ITERATIONS
    )

    initial_state = {
        "session_id": session_id,
        "new_query": initial_query,
        "search_queries": [],
        "data": [],
        "qa_pairs": qa_pairs or [], # Load previous qa_pairs if available
        "relevant_contexts": {},
        "relevant_chunks": [],
        "proceed": True,
        "visited_urls": [],
        "failed_urls": [],
        "iteration_count": 0,
        "analysis_content": None,
        "appendix_content": None,
        "error": None,
        "evaluation_response": None,
        "suggested_follow_up_queries": [],
        "approval_iteration_count": 0,
        "search_iteration_count": 0,
        "report_type": None,
        "max_search_queries": MAX_SEARCH_QUERIES,
        "max_search_results": MAX_SEARCH_RESULTS,
        "max_ai_iterations": MAX_AI_ITERATIONS,
    }

    try:
        # Direct one-shot execution
        final_state = await workflow_app.ainvoke(initial_state)
        return final_state
    except Exception as e:
        logger.exception(f"Workflow execution failed: {e}")
        raise

async def run_research_pipeline(session_id: str, request: ResearchRequest, prompt_type: str, qa_pairs: Optional[List[Dict]] = None):
    import time
    start_time = time.time()
    result = None
    pipeline_error = None
    try:
        if db:
            db.collection("research_sessions").document(session_id).update({
                "status": "running",
                "current_step": "Starting research pipeline...",
                "progress": 5,
                "updated_at": datetime.now(),
            })

        result = await run_workflow(request.query, session_id, prompt_type, qa_pairs)

        session_update = {}
        if result:
            session_update = {
                "analysis_content": result.get("analysis_content"),
                "appendix_content": result.get("appendix_content"),
                "status": "completed",
                "progress": 100,
                "current_step": "Research completed",
                "updated_at": datetime.now(),
                "qa_pairs": result.get("qa_pairs"), # Persist qa_pairs for next turn
                "prompt_type": result.get("prompt_type"), # Persist deduced prompt type
                "analysis_filename": result.get("analysis_filename"),
                "appendix_filename": result.get("appendix_filename"),
                "used_cache": False,
            }
            logger.info(f"Research session {session_id} completed successfully")
        else:
            session_update = {
                "status": "failed",
                "error_message": "No result returned from workflow.",
                "current_step": "Workflow returned no result.",
                "updated_at": datetime.now(),
            }
            logger.error(f"Workflow returned no result for session {session_id}")
        
        if db:
            # Explicitly check the 'used_cache' flag from the workflow state
            if result and result.get('used_cache'):
                session_update['used_cache'] = True

            # Update the live session document
            doc_ref = db.collection("research_sessions").document(session_id)
            doc_ref.update(session_update)

            # If the session completed successfully, optionally archive and remove the live session doc
            try:
                # Only archive+delete when a session is explicitly concluded.
                # Keep sessions in `research_sessions` after completion to allow follow-ups.
                if session_update.get("status") == "concluded":
                    from backend.src.config import ARCHIVE_ON_COMPLETE, ARCHIVE_COLLECTION
                    if ARCHIVE_ON_COMPLETE:
                        # Read the latest document state to archive full metadata
                        live_doc = doc_ref.get()
                        if live_doc.exists:
                            archived_data = live_doc.to_dict() or {}
                            archived_data["archived_at"] = datetime.now()
                            archived_data["archived_from"] = "research_sessions"
                            # Write to configured archive collection using same id
                            db.collection(ARCHIVE_COLLECTION).document(session_id).set(archived_data)
                            # Delete the original live document
                            doc_ref.delete()
                            logger.debug(f"Archived and deleted research session {session_id} to {ARCHIVE_COLLECTION}")
            except Exception as e:
                logger.warning(f"Failed to archive/delete session {session_id}: {e}")

    except Exception as e:
        pipeline_error = e
        logger.error(f"Research pipeline failed for session {session_id}: {e}")
        if db:
            db.collection("research_sessions").document(session_id).update({
                "status": "failed", "error_message": str(e), "current_step": f"Error: {str(e)}", "updated_at": datetime.now()
            })
    finally:
        # Always emit a workflow summary log, even if the pipeline errored or was interrupted.
        try:
            end_time = time.time()
            total_time_taken = end_time - start_time
            # We now always use the classic retriever
            retrieval_method = (result.get("retrieval_method") if result else None) or "classic"
            search_mode = "ultra" if (result and result.get("max_search_queries", 0) > 10) else "fast"
            error_message = (result.get("error") if result else None) or (str(pipeline_error) if pipeline_error else None)

            summary_lines = [
                "\n\n" + "="*25 + " WORKFLOW SUMMARY " + "="*25,
                f"Session ID:         {session_id}",
                f"Search Mode:        {search_mode.upper()}",
                f"Retrieval Method:   {retrieval_method}",
                f"Total Time Taken:   {total_time_taken:.2f} seconds",
            ]
            if error_message:
                summary_lines.append(f"Workflow Errors:    Yes (see logs for details)")
                summary_lines.append(f"Error Detail:       {error_message}")
            summary_lines.append("="*70 + "\n")
            logger.info('\n'.join(summary_lines))
        except Exception:
            logger.exception("Failed to emit workflow summary log.")

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "CRYSTAL DEEPSEARCH API is running", "status": "active", "timestamp": datetime.now().isoformat()}


@app.get('/api/health')
async def health():
    """Startup health endpoint reporting scraper/Playwright readiness."""
    try:
        scraper = nodes.SHARED_SCRAPER
        scraper_present = scraper is not None
        browser_ready = False
        if scraper_present:
            # Check common internal attributes used for readiness
            browser_ready = bool(getattr(scraper, '_browser', None) or getattr(scraper, '_playwright', None))
        return {
            'ok': True,
            'scraper_present': scraper_present,
            'playwright_initialized': browser_ready,
        }
    except Exception as e:
        return {'ok': False, 'error': str(e)}

@app.post("/api/research")
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    if not db:
        raise HTTPException(status_code=503, detail="Firestore client not available. Cannot start research.")
    # Use provided `session_id` if present (follow-up) else create a new session.
    prompt_type = request.prompt_type
    logger.debug("Received request to start or resume a research session.")
    # Protect against too many concurrent sessions
    active_sessions = db.collection("research_sessions").where(filter=firestore.FieldFilter("status", "in", ["running", "pending"])).stream()
    if len(list(active_sessions)) >= MAX_SESSIONS:
        raise HTTPException(status_code=429, detail="Maximum number of active research sessions reached")
    now = datetime.now()
    # If client provided a session_id, attempt to resume that session
    if request.session_id:
        session_id = request.session_id
        doc_ref = db.collection("research_sessions").document(session_id)
        doc = doc_ref.get()
        if doc.exists:
            existing = doc.to_dict() or {}
            qa_pairs: List[Dict] = existing.get("qa_pairs", []) or []
            # Update minimal session metadata to reflect the resumed session
            update_data = {
                "query": request.query,
                "status": "pending",
                "prompt_type": prompt_type,
                "updated_at": now,
                "progress": 0,
                "current_step": "Queued",
            }
            doc_ref.update(update_data)
        else:
            # Provided session_id does not exist; create a new session document with that id
            session_id = request.session_id
            qa_pairs: List[Dict] = []
            session_data = {
                "query": request.query,
                "status": "pending",
                "prompt_type": prompt_type,
                "created_at": now,
                "updated_at": now,
                "progress": 0,
                "current_step": "Queued",
                
                "qa_pairs": [],
                "analysis_content": None,
                "appendix_content": None,
                "analysis_filename": None,
                "appendix_filename": None,
                "error_message": None,
            }
            db.collection("research_sessions").document(session_id).set(session_data)
    else:
        # New session
        session_id = str(uuid.uuid4())
        qa_pairs: List[Dict] = []
        session_data = {
            "query": request.query,
            "status": "pending",
            "prompt_type": prompt_type,
            "created_at": now,
            "updated_at": now,
            "progress": 0,
            "current_step": "Queued",
            
            "qa_pairs": [],
            "analysis_content": None,
            "appendix_content": None,
            "analysis_filename": None,
            "appendix_filename": None,
            "error_message": None,
        }
        db.collection("research_sessions").document(session_id).set(session_data)
    background_tasks.add_task(run_research_pipeline, session_id, request, prompt_type, qa_pairs)
    logger.debug(f"Background task started for session_id='{session_id}'")
    return {"session_id": session_id, "status": "started"}

@app.get("/api/research/{session_id}/status")
async def get_research_status(session_id: str):
    if not db:
        raise HTTPException(status_code=503, detail="Firestore client not available")

    doc_ref = db.collection("research_sessions").document(session_id)
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Research session not found")

    session_data = doc.to_dict()
    # Support processing_urls for frontend visibility; fall back to visited_urls if present
    processing_urls = session_data.get("processing_urls") or session_data.get("visited_urls") or []

    return {
        "session_id": session_id,
        "status": session_data.get("status"),
        "progress": session_data.get("progress"),
        "current_step": session_data.get("current_step"),
        "updated_at": session_data.get("updated_at"),
        "conclusion_message": session_data.get("conclusion_message"),
        "used_cache": session_data.get("used_cache", False),
        "processing_urls": processing_urls,
    }

@app.get("/api/research/{session_id}/result")
async def get_research_result(session_id: str):
    if not db:
        raise HTTPException(status_code=503, detail="Firestore client not available")

    doc_ref = db.collection("research_sessions").document(session_id)
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Research session not found")
    
    session_data = doc.to_dict()
    if session_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Research not completed. Status: {session_data.get('status')}")
    
    return {
        "session_id": session_id,
        "query": session_data.get("query"),
        "analysis_content": session_data.get("analysis_content"),
        "appendix_content": session_data.get("appendix_content"),
        "created_at": session_data.get("created_at"),
        "completed_at": session_data.get("updated_at"),
        "analysis_filename": session_data.get("analysis_filename"),
        "appendix_filename": session_data.get("appendix_filename"),
        "qa_pairs": session_data.get("qa_pairs", []),
    }

@app.get("/api/research/sessions")
async def list_research_sessions(limit: int = 10, offset: int = 0):
    """List research sessions with pagination."""
    if not db:
        raise HTTPException(status_code=503, detail="Firestore client not available")
    query = db.collection("research_sessions").order_by("created_at", direction=firestore.Query.DESCENDING).offset(offset).limit(limit)
    sessions = [doc.to_dict() for doc in query.stream()]
    return {"success": True, "data": {"sessions": sessions, "total": len(sessions), "limit": limit, "offset": offset}}

@app.post("/api/research/{session_id}/conclude")
async def conclude_research_session(session_id: str):
    """Explicitly conclude a research session to prevent further follow-ups."""
    if not db:
        raise HTTPException(status_code=503, detail="Firestore client not available")
    
    doc_ref = db.collection("research_sessions").document(session_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Research session not found")
    
    conclusion_msg = "Research session concluded successfully. Results are saved for your reference, but no further follow-up questions can be processed for this session."
    doc_ref.update({
        "status": "concluded", 
        "updated_at": datetime.now(),
        "conclusion_message": conclusion_msg
    })
    return {"success": True, "message": conclusion_msg}

@app.delete("/api/research/{session_id}")
async def delete_research_session(session_id: str):
    """Delete a research session and its associated report files."""
    if not db:
        logger.warning("Firestore client not available. Cannot delete report files from Firestore.")
        raise HTTPException(status_code=503, detail="Firestore client not available")
    session_ref = db.collection("research_sessions").document(session_id)
    session_doc = session_ref.get()
    if not session_doc.exists:
        raise HTTPException(status_code=404, detail="Research session not found")
    session = session_doc.to_dict()
    for key in ["analysis_filename", "appendix_filename"]:
        filename = session.get(key)
        if filename and db:
            try:
                db.collection("report_files").document(filename).delete()
                logger.debug(f"Deleted report file from Firestore: {filename}")
            except Exception as e:
                logger.warning(f"Failed to delete file {filename} from Firestore: {e}")
    session_ref.delete()
    return {"success": True, "message": "Research session deleted successfully"}

# --- File Download Endpoint ---
@app.get("/api/download/{filename}")
async def download_file_from_firestore(filename: str):
    """Download a report file directly from Firestore."""
    if not db:
        raise HTTPException(status_code=503, detail="Firestore client not available")
    try:
        doc_ref = db.collection("report_files").document(filename)
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="File not found in Firestore")
        file_content = doc.to_dict().get("content")
        return Response(content=file_content, media_type="text/plain", headers={
            'Content-Disposition': f'attachment; filename="{filename}"'
        })
    except Exception as e:
        logger.error(f"Error retrieving file {filename} from Firestore: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving file from Firestore: {str(e)}")


from backend.src.config import CONFIG_SOURCES
@app.get("/api/config")
async def get_config():
    """Get config info and sources."""
    return {
        "prompt_types": ["general", "legal", "macro", "deepsearch", "person_search", "investment"],
        "limits": {
            "min_words": 500,
            "max_words": 2000,
            "max_query_length": 500,
        },
        "sources": CONFIG_SOURCES,
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))

    reload_flag = os.getenv("ENVIRONMENT") != "production"
    # Exclude noisy paths (logs, build artifacts, virtual envs) from the reload watcher to avoid frequent restarts
    reload_excludes = [
        "__pycache__",
        "venv",
        "node_modules",
        "*.pyc",
        # Exclude logging outputs and related files so log writes don't trigger reloads
        "logs",
        "logs/*",
        "logs/**",
        "*.log",
        "*.txt",
    ]
    # Add a small debounce to avoid thrashing on rapid file changes
    reload_delay = 0.5
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload_flag,
        reload_excludes=reload_excludes,
        reload_delay=reload_delay,
        loop="asyncio",
    )
