# config.py - Unified Configuration System
# All configuration values loaded from environment variables (.env file)

import os
from typing import List, Any, Callable
import sys
from .logging_setup import logger

# Dictionary to track config values and their sources
CONFIG_SOURCES = {}

def get_config_value(key: str, default: Any, cast_func: Callable = str) -> Any:
    value = os.getenv(key)
    if value is not None:
        try:
            casted_value = cast_func(value)
        except Exception:
            casted_value = default
        CONFIG_SOURCES[key] = {"value": casted_value, "source": "env"}
        return casted_value
    else:
        CONFIG_SOURCES[key] = {"value": default, "source": "default"}
        return default

# Load environment variables (for local dev only)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_env_bool(key: str, default: bool = False) -> bool:
    def cast_bool(val):
        return val.lower() in ('true', '1', 'yes', 'on')
    return get_config_value(key, default, cast_bool)

def get_env_int(key: str, default: int) -> int:
    def cast_int(val):
        try:
            return int(val)
        except ValueError:
            logger.warning(f"Invalid integer value for {key}, using default: {default}")
            return default
    return get_config_value(key, default, cast_int)

def get_env_float(key: str, default: float) -> float:
    def cast_float(val):
        try:
            return float(val)
        except ValueError:
            logger.warning(f"Invalid float value for {key}, using default: {default}")
            return default
    return get_config_value(key, default, cast_float)

def get_env_list(key: str, default: List[str] = None, separator: str = ',') -> List[str]:
    if default is None:
        default = []
    def cast_list(val):
        return [item.strip() for item in val.split(separator) if item.strip()]
    return get_config_value(key, default, cast_list)

# =============================================================================
# API KEYS  
# =============================================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

GOOGLE_MODEL = get_config_value("GOOGLE_MODEL", "gemini-2.0-flash")

# Embedding Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")  # Updated to latest model

# Enhanced embedding configuration
EMBEDDING_TASK_TYPE = os.getenv("EMBEDDING_TASK_TYPE", "RETRIEVAL_DOCUMENT")  # Default task type
EMBEDDING_DIMENSIONALITY = get_env_int("EMBEDDING_DIMENSIONALITY", 768)  # Efficient default
EMBEDDING_NORMALIZE = get_env_bool("EMBEDDING_NORMALIZE", True)  # Auto-normalize for non-3072 dims
EMBEDDING_BATCH_SIZE = get_env_int("EMBEDDING_BATCH_SIZE", 100)  # Batch processing
USE_ENHANCED_EMBEDDINGS = get_env_bool("USE_ENHANCED_EMBEDDINGS", True)  # Use new implementation

# LLM Settings
LLM_TEMPERATURE = get_env_float("LLM_TEMPERATURE", 0.1)  # Low temperature for factual research
MAX_TOKENS = get_env_int("MAX_TOKENS", 30000)  # High token limit for comprehensive reports
DEFAULT_LLM_TIMEOUT = get_env_int("DEFAULT_LLM_TIMEOUT", 120)  # Extended timeout for complex queries
MAX_RETRIES = get_env_int("MAX_RETRIES", 5) # Increased retries for robustness
BASE_DELAY = get_env_int("BASE_DELAY", 1) # seconds
MAX_CONCURRENT_CALLS = get_env_int("MAX_CONCURRENT_CALLS", 20) # Limit the number of concurrent calls
MAX_CALLS_PER_SECOND = get_env_int("MAX_CALLS_PER_SECOND", 60)
MAX_SESSIONS = get_env_int("MAX_SESSIONS", 50) # Maximum concurrent research sessions

# =============================================================================
# SEARCH AND PROCESSING CONFIGURATION
# =============================================================================
SEARCH_CACHE_COLLECTION = os.getenv("SEARCH_CACHE_COLLECTION", "search_cache")
SCRAPE_RESULT_COLLECTION = os.getenv("SCRAPE_RESULT_COLLECTION", "scrape_results")
SCRAPED_CONTENT_COLLECTION = os.getenv("SCRAPED_CONTENT_COLLECTION", "scraped_content")
# -- Fast Search (Default) Mode Settings --
MAX_SEARCH_QUERIES = get_env_int("MAX_SEARCH_QUERIES", 6)  # Default for "fast" search
MAX_SEARCH_RESULTS = get_env_int("MAX_SEARCH_RESULTS", 8)  # Balanced between quality and performance
MAX_AI_ITERATIONS = get_env_int("MAX_AI_ITERATIONS", 1)

# -- Ultra Search Mode Settings --
ULTRA_MAX_SEARCH_QUERIES = get_env_int("ULTRA_MAX_SEARCH_QUERIES", 8)
ULTRA_MAX_SEARCH_RESULTS = get_env_int("ULTRA_MAX_SEARCH_RESULTS", 8)
ULTRA_MAX_AI_ITERATIONS = get_env_int("ULTRA_MAX_AI_ITERATIONS", 3)

# Other Search limits
MAX_CONCURRENT_SCRAPES = get_env_int("MAX_CONCURRENT_SCRAPES", 25)  # Reasonable concurrency for stability
MAX_SEARCH_RETRIES = get_env_int("MAX_SEARCH_RETRIES", 2)  # Limited retries to prevent hanging

# Content processing
CHUNK_SIZE = get_env_int("CHUNK_SIZE", 1000)  # Optimized for embedding model context
CHUNK_OVERLAP = get_env_int("CHUNK_OVERLAP", 200)  # Minimal overlap for efficiency
MAX_CONTENT_LENGTH = get_env_int("MAX_CONTENT_LENGTH", 25000)  # Reasonable limit per source
URL_TIMEOUT = get_env_int("URL_TIMEOUT", 30)  # Quick timeout to prevent hanging


# =============================================================================
# HYBRID RETRIEVAL CONFIGURATION
# =============================================================================
RETRIEVAL_METHOD = os.getenv("RETRIEVAL_METHOD", "hybrid")
HYBRID_VECTOR_WEIGHT = get_env_float("HYBRID_VECTOR_WEIGHT", 0.6)
HYBRID_BM25_WEIGHT = get_env_float("HYBRID_BM25_WEIGHT", 0.4)
HYBRID_FUSION_METHOD = os.getenv("HYBRID_FUSION_METHOD", "rrf")
HYBRID_RRF_K = get_env_int("HYBRID_RRF_K", 60)
VECTOR_SCORE_THRESHOLD = get_env_float("VECTOR_SCORE_THRESHOLD", 0.1)
VECTOR_FETCH_K_MULTIPLIER = get_env_int("VECTOR_FETCH_K_MULTIPLIER", 2)
RETRIEVAL_TOP_K = get_env_int("RETRIEVAL_TOP_K", 20)
MIN_CHUNK_LENGTH = get_env_int("MIN_CHUNK_LENGTH", 50)
MIN_WORD_COUNT = get_env_int("MIN_WORD_COUNT", 10)
USE_MULTI_QUERY_RETRIEVAL = get_env_bool("USE_MULTI_QUERY_RETRIEVAL", True)

MAX_RETRIEVAL_QUERIES = get_env_int("MAX_RETRIEVAL_QUERIES", 10)
QUERY_CHUNK_DISTRIBUTION = get_env_bool("QUERY_CHUNK_DISTRIBUTION", True)
USE_HYBRID_RETRIEVAL = get_env_bool("USE_HYBRID_RETRIEVAL", True)
USE_RERANKING = get_env_bool("USE_RERANKING", True)
RERANKER_CANDIDATES_MULTIPLIER = get_env_int("RERANKER_CANDIDATES_MULTIPLIER", 3)
USE_CROSS_ENCODER_RERANKING = get_env_bool("USE_CROSS_ENCODER_RERANKING", True)

# Only the fast cross-encoder model is supported
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_TOP_K = get_env_int("CROSS_ENCODER_TOP_K", 50)
RERANK_TOP_K = get_env_int("RERANK_TOP_K", 20)
CROSS_ENCODER_BATCH_SIZE = get_env_int("CROSS_ENCODER_BATCH_SIZE", 32)
MAX_RESULTS = MAX_SEARCH_RESULTS
MAX_RETRIES = MAX_SEARCH_RETRIES

# Log cross-encoder reranking config at startup
logger.debug(f"USE_CROSS_ENCODER_RERANKING={USE_CROSS_ENCODER_RERANKING}, CROSS_ENCODER_MODEL={CROSS_ENCODER_MODEL}, CROSS_ENCODER_TOP_K={CROSS_ENCODER_TOP_K}, RERANK_TOP_K={RERANK_TOP_K}, CROSS_ENCODER_BATCH_SIZE={CROSS_ENCODER_BATCH_SIZE}")


# =============================================================================
# REPORT CONFIGURATION
# =============================================================================

REPORT_FORMAT = os.getenv("REPORT_FORMAT", "md")
DEFAULT_REPORT_TYPE = os.getenv("DEFAULT_REPORT_TYPE", "detailed")

# Legacy support
REPORT_FILENAME_TEXT = os.getenv("REPORT_FILENAME_TEXT", "Crystal_DeepSearch.txt")

# =============================================================================
# WEB SCRAPING CONFIGURATION
# =============================================================================

USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
DEFAULT_USER_AGENT = USER_AGENT  # Legacy support
REQUEST_TIMEOUT = get_env_int("REQUEST_TIMEOUT", 45)
REQUEST_DELAY = get_env_int("REQUEST_DELAY", 1)
DEFAULT_REFERER = "https://www.google.com/"

# Blocked domains and extensions
BLOCKED_DOMAINS = get_env_list("BLOCKED_DOMAINS", [
    "instagram.com", "youtube.com/watch", "youtu.be", "nsearchives.nseindia.com", 
    "bseindia.com", 
])
SKIP_EXTENSIONS = get_env_list("SKIP_EXTENSIONS", [
    ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mp3", ".zip", 
    ".exe", ".dmg", ".rar", ".7z"
])


# =============================================================================
# CACHING AND PERFORMANCE
# =============================================================================
# Caching configuration
# In GCP deployments Firestore (and GCS for large blobs) are the canonical
# stores. Local filesystem cache is kept only as an optional fallback for
# developer/testing environments. All TTLs are in seconds.

# Global toggle to enable/disable caching behavior
CACHE_ENABLED = get_env_bool("CACHE_ENABLED", True)

# Default time-to-live for generic caches (seconds). Set to 10 days (864000).
CACHE_TTL = get_env_int("CACHE_TTL", 864000)

# Specific TTLs for search and scrape results (10 days = 864000 seconds)
SEARCH_CACHE_TTL = get_env_int("SEARCH_CACHE_TTL", 864000)
SCRAPE_RESULT_TTL = get_env_int("SCRAPE_RESULT_TTL", 864000)
# GCS bucket for storing full scraped content (optional)
SCRAPED_CONTENT_GCS_BUCKET = os.getenv("SCRAPED_CONTENT_GCS_BUCKET", "deepsearch_document_store")
USE_PERSISTENCE = get_env_bool("USE_PERSISTENCE", True)
# Whether the scraper should use an in-memory cache. Disable to force GCS-backed persistence.
SCRAPER_USE_IN_MEMORY_CACHE = get_env_bool("SCRAPER_USE_IN_MEMORY_CACHE", True)
SCRAPER_ENABLE_PLAYWRIGHT = get_env_bool("SCRAPER_ENABLE_PLAYWRIGHT", True)    
SEARCH_USE_LOCAL_CACHE = get_env_bool("SEARCH_USE_LOCAL_CACHE", False)

# UA rotation and proxy options for scraping
ENABLE_UA_ROTATION = get_env_bool("ENABLE_UA_ROTATION", True)
UA_ROTATION_LIST = get_env_list("UA_ROTATION_LIST", [
    USER_AGENT,
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
])

# Global proxy used by aiohttp requests (e.g. http://user:pass@host:port)
SCRAPER_PROXY = os.getenv("SCRAPER_PROXY", "")
# Playwright-specific proxy server (can be same as SCRAPER_PROXY)
PLAYWRIGHT_PROXY = os.getenv("PLAYWRIGHT_PROXY", "")
# Enable a small Playwright stealth init script to unset navigator.webdriver
PLAYWRIGHT_STEALTH = get_env_bool("PLAYWRIGHT_STEALTH", True)

# Domains (comma-separated) for which Playwright should be used instead of aiohttp.
# Example: PLAYWRIGHT_FORCE_DOMAINS=twitter.com,reddit.com,finance.yahoo.com
PLAYWRIGHT_FORCE_DOMAINS = get_env_list("PLAYWRIGHT_FORCE_DOMAINS", [])

# Redis configuration for high-speed, shared caching (e.g., Memorystore)
REDIS_HOST = os.getenv("REDIS_HOST", None)
REDIS_PORT = get_env_int("REDIS_PORT", 6379)


# =============================================================================
# ENHANCED DEDUPLICATION SETTINGS
# =============================================================================

# Enable LLM-powered intelligent deduplication
DEDUPLICATION_CACHE_ENABLED = get_env_bool("DEDUPLICATION_CACHE_ENABLED", True)
DEDUPLICATION_CACHE_TTL = get_env_int("DEDUPLICATION_CACHE_TTL", 864000)  # 2 hours cache

# Deduplication thresholds
SIMILARITY_THRESHOLD = get_env_float("SIMILARITY_THRESHOLD", 0.9)  # When to consider content similar
MIN_SENTENCE_LENGTH = get_env_int("MIN_SENTENCE_LENGTH", 3)  # Minimum words per sentence

API_REQUESTS_PER_MINUTE = get_env_int("API_REQUESTS_PER_MINUTE", 60)  # API-friendly limits
SCRAPING_REQUESTS_PER_MINUTE = get_env_int("SCRAPING_REQUESTS_PER_MINUTE", 30)  # Respectful scraping

## All automation settings removed

# =============================================================================
# LOGGING AND DEBUGGING
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
DEBUG_MODE = get_env_bool("DEBUG_MODE", False)
VERBOSE = get_env_bool("VERBOSE", False)

# =============================================================================
# SECURITY SETTINGS
# =============================================================================

SSL_VERIFY = get_env_bool("SSL_VERIFY", True)

# =============================================================================
# LEGACY SUPPORT AND COLORS
# =============================================================================

# Color constants for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
ENDC = '\033[0m'

# Legacy persistence setting


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate critical configuration values."""
    errors = []
    warnings = []
    
    # Check API keys
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is required")
    if not SERPER_API_KEY:
        errors.append("SERPER_API_KEY not set - search functionality may be limited")
    
    # Check numeric limits
    if MAX_SEARCH_QUERIES <= 0:
        errors.append("MAX_SEARCH_QUERIES must be positive")
    if CACHE_TTL <= 0:
        warnings.append("CACHE_TTL should be positive for effective caching")
    
    # Log results
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
    if warnings:
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
    
    return len(errors) == 0

# Validate configuration on import
config_valid = validate_config()

logger.debug("config.py loaded successfully with unified environment-based configuration")

# Export commonly used values for backward compatibility
__all__ = [
    # API Keys (Google-only)
    'GOOGLE_API_KEY', 'SERPER_API_KEY',
    
    # LLM Configuration  
    'GOOGLE_MODEL', 'EMBEDDING_MODEL',
    'LLM_TEMPERATURE', 'MAX_TOKENS',

    # Enhanced Embedding Configuration
    'EMBEDDING_PROVIDER', 'EMBEDDING_TASK_TYPE', 'EMBEDDING_DIMENSIONALITY',
    'EMBEDDING_NORMALIZE', 'EMBEDDING_BATCH_SIZE', 'USE_ENHANCED_EMBEDDINGS',
    
    # Search and Processing
    'MAX_SEARCH_QUERIES', 'MAX_SEARCH_RESULTS', 'MAX_AI_ITERATIONS',
    'ULTRA_MAX_SEARCH_QUERIES', 'ULTRA_MAX_SEARCH_RESULTS', 'ULTRA_MAX_AI_ITERATIONS',
    'MAX_CONCURRENT_SCRAPES', 'CHUNK_SIZE', 'CHUNK_OVERLAP',
    
    # Hybrid Retrieval
    'RETRIEVAL_METHOD', 'HYBRID_VECTOR_WEIGHT', 'HYBRID_BM25_WEIGHT',
    'HYBRID_FUSION_METHOD', 'HYBRID_RRF_K', 'VECTOR_SCORE_THRESHOLD',
    'RETRIEVAL_TOP_K', 'USE_HYBRID_RETRIEVAL', 'USE_RERANKING',
    'USE_MULTI_QUERY_RETRIEVAL', 'MAX_RETRIEVAL_QUERIES', 'QUERY_CHUNK_DISTRIBUTION',
    'USE_CROSS_ENCODER_RERANKING', 'CROSS_ENCODER_MODEL', 'CROSS_ENCODER_TOP_K', 'RERANK_TOP_K', 'CROSS_ENCODER_BATCH_SIZE',
    
    # Reports
    'REPORT_FORMAT', 'REPORT_FILENAME_TEXT',
    
    # Web Scraping
    'USER_AGENT', 'BLOCKED_DOMAINS', 'SKIP_EXTENSIONS', 'REQUEST_TIMEOUT',
    'ENABLE_UA_ROTATION', 'UA_ROTATION_LIST', 'SCRAPER_PROXY', 'PLAYWRIGHT_PROXY', 'PLAYWRIGHT_STEALTH', 'PLAYWRIGHT_FORCE_DOMAINS',
    
    # Caching
    'CACHE_ENABLED', 'CACHE_TTL',
    
    # Debug
    'DEBUG_MODE',
    
    # Colors
    'RED', 'GREEN', 'BLUE', 'YELLOW', 'ENDC',
    
    # Legacy support
    'MAX_RESULTS', 'DEFAULT_USER_AGENT'
]
