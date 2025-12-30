"""Bootstrap script to run the FastAPI app with proper Windows Proactor policy.

Usage: python run.py

This sets asyncio.WindowsProactorEventLoopPolicy() on Windows before importing
any modules that may create subprocesses (Playwright, uvicorn, etc.), then
starts uvicorn programmatically.
"""
import sys
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bootstrap")

if sys.platform == "win32":
    try:
        policy = asyncio.get_event_loop_policy()
        if type(policy).__name__ != "WindowsProactorEventLoopPolicy":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            log.info("Set WindowsProactorEventLoopPolicy for this process")
    except Exception as e:
        log.warning(f"Failed to set WindowsProactorEventLoopPolicy: {e}")

# Start uvicorn programmatically so the policy is already applied before imports
if __name__ == "__main__":
    import uvicorn

    # Use the same invocation as before; adjust host/port as needed
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
