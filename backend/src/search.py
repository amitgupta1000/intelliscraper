"""Async search with Firestore + in-memory caching.

Lightweight `UnifiedSearcher` that uses Serper and preserves caching behavior:
- Tries Firestore cache first (if available).
- Falls back to an in-memory TTL cache per process.
- Uses `aiohttp` when available; otherwise offloads `requests` to a thread.
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import redis

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except Exception:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

import requests

# Import unified configuration
from .config import (
    SERPER_API_KEY,
    MAX_SEARCH_RESULTS,
    CACHE_ENABLED,
    SEARCH_CACHE_TTL,
    SEARCH_CACHE_COLLECTION,
    BLOCKED_DOMAINS,
    REDIS_HOST,
    REDIS_PORT,
)

logger = logging.getLogger(__name__)

# Firestore client (optional)
try:
    from google.cloud import firestore as _firestore_mod
    try:
        FIRESTORE_CLIENT = _firestore_mod.Client()
        logger.info("Firestore client initialized successfully.")
    except Exception:
        FIRESTORE_CLIENT = None
        logger.warning(f"Firestore client initialization failed: {e}. Firestore caching will be disabled.")
except ImportError:
    FIRESTORE_CLIENT = None
    logger.info("google-cloud-firestore not installed. Firestore caching will be disabled.")

# Redis client for high-speed caching (optional)
try:
    if REDIS_HOST:
        REDIS_CLIENT = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        REDIS_CLIENT.ping() # Check connection
        logger.info(f"Redis client connected to {REDIS_HOST}:{REDIS_PORT}")
    else:
        REDIS_CLIENT = None
except Exception as e:
    REDIS_CLIENT = None
    logger.warning(f"Redis client initialization failed: {e}. Redis caching will be disabled.")

SERPER_ENDPOINT = os.getenv("SERPER_ENDPOINT", "https://google.serper.dev/search")


@dataclass
class SearchResult:
    """Dataclass to store individual search results."""
    url: str
    title: str
    snippet: str
    source: str = "Serper"

    def to_dict(self):
        """Convert SearchResult object to a dictionary format for serialization."""
        return {"url": self.url, "title": self.title, "snippet": self.snippet, "source": self.source}


class UnifiedSearcher:
    """Searcher with Firestore primary cache and in-memory fallback."""

    def __init__(self, max_results: int = MAX_SEARCH_RESULTS, cache_enabled: bool = CACHE_ENABLED, cache_ttl: int = SEARCH_CACHE_TTL):
        self.max_results = max(1, min(int(max_results or 8), 12))
        self.cache_enabled = bool(cache_enabled)
        self.cache_ttl = int(cache_ttl or 864000)
        # simple in-memory cache: key -> (timestamp, results_list)
        self._local_cache: Dict[str, Tuple[float, List[SearchResult]]] = {}
        logger.info(f"UnifiedSearcher initialized (max_results={self.max_results}, cache_enabled={self.cache_enabled}, ttl={self.cache_ttl}s)")

    def _cache_key(self, query: str, engine: str = "serper") -> str:
        """Generate a stable MD5 hash for a given query and engine."""
        return hashlib.md5(f"{engine}:{query}".encode("utf-8")).hexdigest()

    async def _get_firestore_cache(self, key: str) -> Optional[List[SearchResult]]:
        """Asynchronously retrieve search results from Firestore cache."""
        if not FIRESTORE_CLIENT:
            return None
        try:
            def _sync_get(doc_id: str):
                doc_ref = FIRESTORE_CLIENT.collection(SEARCH_CACHE_COLLECTION).document(doc_id)
                doc = doc_ref.get()
                return doc.to_dict() if doc.exists else None

            # Try primary (hashed) key first
            doc = await asyncio.to_thread(_sync_get, key)
            if not doc:
                # Fallback: attempt legacy doc id format 'engine::query' for backward compatibility
                try:
                    legacy_id = f"serper::{key}"  # safe fallback placeholder
                    # If original code used raw query as id, try that too via a direct lookup by query field
                    doc = await asyncio.to_thread(_sync_get, legacy_id)
                except Exception:
                    doc = None

            if not doc:
                # As a last resort, attempt to find a document that matches the 'query' field
                try:
                    def _sync_find():
                        col = FIRESTORE_CLIENT.collection(SEARCH_CACHE_COLLECTION)
                        q = col.where('query', '==', key) if isinstance(key, str) else None
                        if q is None:
                            return None
                        docs = q.limit(1).stream()
                        for d in docs:
                            return d.to_dict()
                        return None
                    doc = await asyncio.to_thread(_sync_find)
                except Exception:
                    doc = None

            if not doc:
                return None

            expires = doc.get("expires_at")
            # Ensure comparison is timezone-aware
            now = datetime.now(timezone.utc)
            if expires and isinstance(expires, datetime):
                # If expires is naive, assume UTC
                expires_aware = expires.replace(tzinfo=timezone.utc) if expires.tzinfo is None else expires
                if expires_aware < now:
                    logger.info(f"Firestore cache expired for key: {key}")
                    return None

            results = [SearchResult(**item) for item in doc.get("results", [])]
            return results
        except Exception as e:
            logger.debug(f"Firestore cache read failed for key {key}: {e}")
            return None

    async def _save_firestore_cache(self, key: str, query: str, results: List[SearchResult]):
        """Asynchronously save search results to Firestore cache."""
        if not FIRESTORE_CLIENT:
            return
        try:
            doc = {
                "engine": "serper",
                "query": query,
                "results": [r.to_dict() for r in results],
                "created_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc) + timedelta(seconds=self.cache_ttl),
            }

            def _sync_set():
                FIRESTORE_CLIENT.collection(SEARCH_CACHE_COLLECTION).document(key).set(doc)

            await asyncio.to_thread(_sync_set)
            logger.info(f"CACHE SAVE: search (firestore) for key: {key}")
        except Exception as e:
            logger.warning(f"Failed to save search cache to Firestore for key {key}: {e}")

    def _get_local_cache(self, key: str) -> Optional[List[SearchResult]]:
        """Retrieve search results from the in-memory cache."""
        rec = self._local_cache.get(key)
        if not rec:
            return None
        ts, results = rec
        if time.time() - ts > self.cache_ttl:
            self._local_cache.pop(key, None)
            return None
        return results

    def _set_local_cache(self, key: str, results: List[SearchResult]):
        """Save search results to the in-memory cache."""
        self._local_cache[key] = (time.time(), results)

    async def _search_serper(self, query: str) -> List[SearchResult]:
        """Perform an asynchronous search using the Serper API."""
        if not SERPER_API_KEY:
            logger.warning("SERPER_API_KEY not configured; skipping Serper search.")
            return []

        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        payload = {"q": query, "num": min(self.max_results, 10), "sort": "date"}

        try:
            if AIOHTTP_AVAILABLE:
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(SERPER_ENDPOINT, json=payload, headers=headers) as resp:
                        resp.raise_for_status()
                        body = await resp.json()
            else:
                # Fallback to threaded requests if aiohttp is not available
                resp = await asyncio.to_thread(requests.post, SERPER_ENDPOINT, json=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                body = resp.json()

            organic = body.get("organic", []) if isinstance(body, dict) else []
            out = []
            for item in organic:
                url = item.get("link") or item.get("url")
                if not url or urlparse(url).scheme not in ("http", "https"):
                    continue
                
                domain = urlparse(url).netloc.lower()
                if any(blocked_domain in domain for blocked_domain in BLOCKED_DOMAINS):
                    logger.debug(f"Skipping blocked domain URL: {url}")
                    continue

                out.append(SearchResult(url=url, title=item.get("title", ""), snippet=item.get("snippet", "")))
                if len(out) >= self.max_results:
                    break
            return out
        except Exception as e:
            logger.error(f"Serper search failed for query={query!r}: {e}")
            return []

    async def search(self, query: str, force_refresh: bool = False) -> Tuple[List[SearchResult], bool]:
        """
        Orchestrates the search process, including caching and live search.

        Args:
            query: The search query string.
            force_refresh: If True, bypasses the cache and performs a live search.

        Returns:
            A tuple containing:
            - A list of SearchResult objects.
            - A boolean indicating if the result was served from cache.
        """
        year = datetime.now(timezone.utc).year
        enhanced_query = query if (str(year) in query or str(year - 1) in query or "recent" in query.lower()) else f"{query} recent latest {year}"
        
        key = self._cache_key(enhanced_query)

        if self.cache_enabled and not force_refresh:
            # 1. Check Firestore cache
            fs_results = await self._get_firestore_cache(key)
            if fs_results is not None:
                logger.info(f"CACHE HIT: search (firestore) for '{query}'")
                self._set_local_cache(key, fs_results) # Populate local cache
                return fs_results, True

            # 2. Check in-memory cache
            local_results = self._get_local_cache(key)
            if local_results is not None:
                logger.info(f"CACHE HIT: search (local) for '{query}'")
                return local_results, True

        # 3. Perform live search
        logger.info(f"CACHE MISS: Performing live search for '{query}'")
        results = await self._search_serper(enhanced_query)

        # 4. Save to caches if results were found
        if self.cache_enabled and results:
            self._set_local_cache(key, results)
            await self._save_firestore_cache(key, enhanced_query, results)

        return results, False
