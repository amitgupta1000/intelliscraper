"""Optional auto-applied startup hooks.

If this repository root is on `PYTHONPATH`, Python will import `sitecustomize`
automatically at startup. This file sets the Windows Proactor policy early so
that subprocess-creating libraries (Playwright, others) work correctly on
Windows.

Note: This has global effect for the process and should be used only when you
intend it for local development.
"""
import sys
import asyncio
import logging

logging.basicConfig(level=logging.WARNING)
_log = logging.getLogger("sitecustomize")

if sys.platform == "win32":
    try:
        policy = asyncio.get_event_loop_policy()
        if type(policy).__name__ != "WindowsProactorEventLoopPolicy":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            _log.info("sitecustomize: Set WindowsProactorEventLoopPolicy")
    except Exception as e:
        _log.warning(f"sitecustomize: Failed to set Proactor policy: {e}")
