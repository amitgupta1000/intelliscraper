import logging
import logging.handlers
import os
import queue
from typing import Optional

LOG_DIR = os.path.join(os.path.dirname(__file__), '../../logs')
LOG_FILE = os.path.join(LOG_DIR, 'app.log')

os.makedirs(LOG_DIR, exist_ok=True)

# Create a multiprocessing/thread-safe queue for log records
_log_queue: "queue.Queue[logging.LogRecord]" = queue.Queue(-1)

# Handlers that will run in the background (consumer)
_file_handler = logging.FileHandler(LOG_FILE)
_stream_handler = logging.StreamHandler()
for h in (_file_handler, _stream_handler):
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))

# QueueListener will consume records from the queue and dispatch to handlers
_queue_listener: Optional[logging.handlers.QueueListener] = logging.handlers.QueueListener(
    _log_queue, _file_handler, _stream_handler
)

# Root logger uses a QueueHandler that enqueues records quickly (non-blocking)
_queue_handler = logging.handlers.QueueHandler(_log_queue)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# remove default handlers if any
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)
root_logger.addHandler(_queue_handler)

# Named logger for the application
logger = logging.getLogger('deepsearch')


def start_logging():
    """Start the background QueueListener. Call on application startup."""
    global _queue_listener
    try:
        if _queue_listener and not _queue_listener._thread:
            _queue_listener.start()
        elif _queue_listener:
            # already started
            pass
    except Exception:
        # Best-effort; avoid crashing startup if logging can't start
        try:
            _queue_listener.start()
        except Exception:
            pass


def stop_logging():
    """Stop the background QueueListener. Call on application shutdown."""
    global _queue_listener
    try:
        if _queue_listener:
            _queue_listener.stop()
    except Exception:
        pass


# Start listener eagerly so imports (like main.py) get immediate logging
try:
    _queue_listener.start()
except Exception:
    # If start fails during import, it's non-fatal; user can call start_logging()
    pass
