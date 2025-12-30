"""Storage helpers for scraped content.

Design (clear responsibilities):
- Firestore: metadata, lookup, TTL and light-weight fields (url, title, snippet,
  gcs paths, content_hash, canonical_url, fetched_at, expires_at, id).
- GCS: large blob storage for full `text` and optional `html` content. Firestore
  stores the `gcs_text_path` / `gcs_html_path` pointing to the GCS object.

Typical workflow:
1. When scraping, compute a `content_hash` and upload large blobs to GCS.
2. Save a metadata record in Firestore via `save_scraped_record` (or
   `store_scraped_content` which uploads + saves).
3. To read: query Firestore for a matching metadata record (via
   `get_by_url`). If found and `gcs_text_path` (or `gcs_html_path`) exist,
   download the text/html from GCS using `download_text_from_gcs`.

This module provides a convenience wrapper `fetch_scraped_content` which
performs the canonical read path (Firestore lookup then optional GCS fetch)
to reduce duplication and confusion in the codebase.

Usage example:
    from google.cloud import firestore
    from backend.src.storage import fetch_scraped_content

    db = firestore.Client()
    rec = fetch_scraped_content(db, 'https://example.com')
    if rec:
        # rec includes 'text' loaded from GCS when available
        print(rec['text'])
"""
from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from .logging_setup import logger
try:
    from .config import SCRAPE_RESULT_TTL
except Exception:
    SCRAPE_RESULT_TTL = 864000  # 10 days fallback

try:
    from google.cloud import storage as gcs_lib
    from google.cloud import firestore
except Exception:  # pragma: no cover - will raise if libs missing at runtime
    gcs_lib = None
    firestore = None


def _compute_sha256(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# Simple in-memory metrics for Firestore/GCS operations. These are process-local
# counters intended for lightweight observation during runtime or tests. For a
# production deployment you'd wire these into Prometheus / Cloud Monitoring.
_metrics_lock = threading.Lock()
_metrics: Dict[str, int] = {}


def _inc_metric(name: str, amount: int = 1):
    with _metrics_lock:
        _metrics[name] = _metrics.get(name, 0) + int(amount)


def get_metrics() -> Dict[str, int]:
    """Return a snapshot of collected storage metrics."""
    with _metrics_lock:
        return dict(_metrics)


def reset_metrics():
    """Reset all in-memory metrics to zero. Useful for tests."""
    with _metrics_lock:
        _metrics.clear()


def _normalize_url_for_match(u: str) -> str:
    """Return a normalized URL string suitable for equality matching.

    Normalization steps:
    - lower-case scheme and host
    - remove default ports (80, 443)
    - strip fragment
    - remove common tracking params from query
    - strip trailing slash
    """
    try:
        from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
        p = urlparse(u)
        scheme = p.scheme.lower() if p.scheme else 'http'
        netloc = p.netloc.lower()
        # remove default ports
        if netloc.endswith(':80') and scheme == 'http':
            netloc = netloc[:-3]
        if netloc.endswith(':443') and scheme == 'https':
            netloc = netloc[:-4]
        # strip fragment
        fragment = ''
        # remove tracking params
        pairs = parse_qsl(p.query, keep_blank_values=True)
        TRACKING_PREFIXES = ('utm_', 'fbclid', 'gclid', 'mc_cid', 'mc_eid')
        # Filter and sort the query parameters for a stable order
        filtered = sorted([(k, v) for (k, v) in pairs if not any(k.startswith(pref) for pref in TRACKING_PREFIXES)])
        query = urlencode(filtered)
        path = p.path or ''
        if path.endswith('/') and path != '/':
            path = path.rstrip('/')
        norm = urlunparse((scheme, netloc, path, '', query, fragment))
        return norm
    except Exception:
        return u.lower()


def upload_text_to_gcs(bucket_name: str, destination: str, content: str, content_type: str = 'text/plain') -> str:
    """Upload text content to GCS and return the gs:// path.

    Raises if google.cloud.storage is not available or upload fails.
    """
    if gcs_lib is None:
        raise RuntimeError('google.cloud.storage is not installed')
    client = gcs_lib.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination)
    blob.upload_from_string(content, content_type=content_type)
    # metric: upload performed
    try:
        _inc_metric('gcs_uploads')
        if content_type == 'text/html':
            _inc_metric('gcs_html_uploads')
        else:
            _inc_metric('gcs_text_uploads')
    except Exception:
        pass
    return f'gs://{bucket_name}/{destination}'


def download_text_from_gcs(gs_path: str) -> str:
    """Download text from a gs://bucket/path and return its content as string.

    Raises RuntimeError if GCS client is not available or download fails.
    """
    if gcs_lib is None:
        raise RuntimeError('google.cloud.storage is not installed')
    if not gs_path.startswith('gs://'):
        raise ValueError('gs_path must start with gs://')
    # parse gs://bucket/path
    _, rest = gs_path.split('://', 1)
    parts = rest.split('/', 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ''
    client = gcs_lib.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    try:
        text = blob.download_as_text()
        try:
            _inc_metric('gcs_downloads')
            _inc_metric('gcs_text_downloads')
        except Exception:
            pass
        return text
    except Exception as e:
        raise RuntimeError(f'Failed to download {gs_path}: {e}')


def save_scraped_record(firestore_client, record: Dict[str, Any], collection: str = 'scraped_contents') -> str:
    """Save metadata record to Firestore and return document id."""
    if firestore is None and firestore_client is None:
        raise RuntimeError('google.cloud.firestore is not installed or client not provided')
    doc_ref = firestore_client.collection(collection).document()
    now = datetime.utcnow()
    record.setdefault('created_at', now)
    # set expires_at according to configured TTL (seconds)
    try:
        expires_at = now + timedelta(seconds=int(SCRAPE_RESULT_TTL))
    except Exception:
        expires_at = now + timedelta(days=10)
    record.setdefault('expires_at', expires_at)
    doc_ref.set(record)
    return doc_ref.id


def get_by_url(firestore_client, url: str, collection: str = 'scraped_contents') -> Optional[Dict[str, Any]]:
    """Return first matching document for a given URL, or None.

    If an exact match is not found, this function will try a set of
    normalized variants (strip fragment, remove common tracking params,
    strip trailing slash, www/non-www, http/https) and return the first
    valid document found. When a variant is matched, the returned dict
    will include a `matched_url_variant` key with the variant string.
    """
    def _is_doc_valid(doc: Dict[str, Any]) -> bool:
        now = datetime.utcnow()
        expires = doc.get('expires_at')
        try:
            return (expires is None) or (expires >= now)
        except Exception:
            return True

    from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

    def _strip_tracking_params(pairs):
        TRACKING_PREFIXES = ('utm_', 'fbclid', 'gclid', 'mc_cid', 'mc_eid')
        return [(k, v) for (k, v) in pairs if not any(k.startswith(pref) for pref in TRACKING_PREFIXES)]

    def _variants(u: str):
        # yield original first
        yield u
        p = urlparse(u)
        # strip fragment
        if p.fragment:
            yield urlunparse(p._replace(fragment=""))
        # strip query entirely
        if p.query:
            yield urlunparse(p._replace(query=""))
        # strip common tracking params
        if p.query:
            pairs = parse_qsl(p.query, keep_blank_values=True)
            filtered = _strip_tracking_params(pairs)
            if len(filtered) != len(pairs):
                yield urlunparse(p._replace(query=urlencode(filtered)))
        # trailing slash variations
        path = p.path or ''
        if path.endswith('/'):
            yield urlunparse(p._replace(path=path.rstrip('/')))
        else:
            yield urlunparse(p._replace(path=path + '/'))
        # www/non-www
        netloc = p.netloc
        if netloc.startswith('www.'):
            yield urlunparse(p._replace(netloc=netloc[4:]))
        else:
            yield urlunparse(p._replace(netloc='www.' + netloc))
        # scheme switches
        if p.scheme == 'http':
            yield urlunparse(p._replace(scheme='https'))
        elif p.scheme == 'https':
            yield urlunparse(p._replace(scheme='http'))

    tried = set()
    for candidate in _variants(url):
        if candidate in tried:
            continue
        tried.add(candidate)
        try:
            # First try exact URL match
            q = firestore_client.collection(collection).where(filter=firestore.FieldFilter('url', '==', candidate)).limit(1)
            docs = q.stream() if hasattr(q, 'stream') else q
            for d in docs:
                doc = d.to_dict()
                if _is_doc_valid(doc):
                    try:
                        _inc_metric('firestore_hits')
                    except Exception:
                        pass
                    doc['matched_url_variant'] = candidate
                    return doc

            # Next try canonical_url field match
            q2 = firestore_client.collection(collection).where(filter=firestore.FieldFilter('canonical_url', '==', candidate)).limit(1)
            docs2 = q2.stream() if hasattr(q2, 'stream') else q2
            for d in docs2:
                doc = d.to_dict()
                if _is_doc_valid(doc):
                    try:
                        _inc_metric('firestore_hits')
                    except Exception:
                        pass
                    doc['matched_url_variant'] = candidate
                    return doc
        except Exception:
            # if a query fails for some reason, continue to next variant
            continue

    # If nothing matched, try resolving redirects (HEAD) and try variants of final URL
    try:
        import requests
        resp = requests.head(url, allow_redirects=True, timeout=5)
        final = resp.url
        if final and final not in tried:
            for candidate in _variants(final):
                if candidate in tried:
                    continue
                tried.add(candidate)
                try:
                    q = firestore_client.collection(collection).where(filter=firestore.FieldFilter('url', '==', candidate)).limit(1)
                    docs = q.stream() if hasattr(q, 'stream') else q
                    for d in docs:
                        doc = d.to_dict()
                        if _is_doc_valid(doc):
                            doc['matched_url_variant'] = candidate
                            doc['resolved_from'] = final
                            return doc
                    q2 = firestore_client.collection(collection).where(filter=firestore.FieldFilter('canonical_url', '==', candidate)).limit(1)
                    docs2 = q2.stream() if hasattr(q2, 'stream') else q2
                    for d in docs2:
                        doc = d.to_dict()
                        if _is_doc_valid(doc):
                            doc['matched_url_variant'] = candidate
                            doc['resolved_from'] = final
                            return doc
                except Exception:
                    continue
    except Exception:
        # network/requests not available or failed; return None
        pass
    # As a last resort, try normalized canonical fields equality
    try:
        norm = _normalize_url_for_match(url)
        qn = firestore_client.collection(collection).where(filter=firestore.FieldFilter('canonical_url_normalized', '==', norm)).limit(1)
        docsn = qn.stream() if hasattr(qn, 'stream') else qn
        for d in docsn:
            doc = d.to_dict()
            if _is_doc_valid(doc):
                doc['matched_url_variant'] = doc.get('canonical_url') or doc.get('url')
                doc['matched_by'] = 'canonical_url_normalized'
                return doc
    except Exception:
        pass

    return None


def exists_by_hash(firestore_client, content_hash: str, collection: str = 'scraped_contents') -> bool:
    q = firestore_client.collection(collection).where(filter=firestore.FieldFilter('content_hash', '==', content_hash)).limit(1)
    docs = q.stream() if hasattr(q, 'stream') else q
    now = datetime.utcnow()
    for d in docs:
        doc = d.to_dict()
        expires = doc.get('expires_at')
        try:
            if expires is None or expires >= now:
                try:
                    _inc_metric('exists_by_hash_hits')
                except Exception:
                    pass
                return True
        except Exception:
            return True
    return False


def get_record_by_hash(firestore_client, content_hash: str, collection: str = 'scraped_contents') -> Optional[Dict[str, Any]]:
    """Return the first Firestore document matching a content_hash, or None.

    The returned dict will include the document id as `id`.
    """
    try:
        q = firestore_client.collection(collection).where(filter=firestore.FieldFilter('content_hash', '==', content_hash)).limit(1)
        docs = q.stream() if hasattr(q, 'stream') else q
        now = datetime.utcnow()
        for d in docs:
            doc = d.to_dict()
            expires = doc.get('expires_at')
            try:
                if expires is None or expires >= now:
                    doc['id'] = getattr(d, 'id', None)
                    return doc
            except Exception:
                doc['id'] = getattr(d, 'id', None)
                return doc
    except Exception:
        return None
    return None


def store_scraped_content(firestore_client, gcs_bucket: str, scraped: Dict[str, Any], session_id: Optional[str] = None, collection: str = 'scraped_contents') -> Dict[str, Any]:
    """Upload scraped content to GCS and save metadata in Firestore.

    Args:
        firestore_client: an instance of google.cloud.firestore.Client
        gcs_bucket: target GCS bucket name
        scraped: dict with keys: 'url', 'title', 'text', optional 'html'
        session_id: optional session id to associate the record with
    Returns: saved Firestore record (including generated doc id as 'id')
    """
    text = scraped.get('text', '') or ''
    html = scraped.get('html')
    url = scraped.get('url')
    title = scraped.get('title') or url

    # Compute content hash for duplicate detection
    content_for_hash = (html if html is not None else text)[:100000]
    content_hash = _compute_sha256(content_for_hash)

    # Quick duplicate check using content hash; if a record exists return its id
    if firestore_client:
        existing = get_record_by_hash(firestore_client, content_hash, collection=collection)
        if existing:
            try:
                _inc_metric('duplicate_skipped')
            except Exception:
                pass
            try:
                logger.info(f"store_scraped_content: duplicate detected for url={url} content_hash={content_hash} existing_id={existing.get('id')}")
            except Exception:
                pass
            return {'status': 'exists', 'content_hash': content_hash, 'existing_id': existing.get('id'), 'existing_doc': existing}

    # Upload text and optional html to GCS
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    base_name = f"{content_hash[:12]}_{timestamp}"
    text_path = f"scraped/{base_name}.txt"
    gcs_text_path = upload_text_to_gcs(gcs_bucket, text_path, text, content_type='text/plain')

    gcs_html_path = None
    if html:
        html_path = f"scraped/{base_name}.html"
        gcs_html_path = upload_text_to_gcs(gcs_bucket, html_path, html, content_type='text/html')
    try:
        logger.info(f"Uploaded scraped content to GCS: text={gcs_text_path} html={gcs_html_path}")
    except Exception:
        pass
    snippet = text[:1000]

    record = {
        'url': url,
        'title': title,
        'gcs_text_path': gcs_text_path,
        'gcs_html_path': gcs_html_path,
        'content_hash': content_hash,
        'snippet': snippet,
        'session_id': session_id,
        'fetched_at': datetime.utcnow(),
    }

    # Attempt to determine a canonical URL for more robust lookups.
    try:
        import requests
        from bs4 import BeautifulSoup
        # follow redirects to get final URL
        try:
            head = requests.head(url, allow_redirects=True, timeout=5)
            final_url = head.url or url
        except Exception:
            final_url = url

        canonical_url = final_url
        # Try fetching page HTML to find <link rel="canonical">
        try:
            r = requests.get(final_url, timeout=5)
            if r.status_code == 200 and r.text:
                soup = BeautifulSoup(r.text, 'html.parser')
                link = soup.find('link', rel=lambda x: x and 'canonical' in x.lower())
                if link and link.get('href'):
                    canonical_url = link.get('href')
        except Exception:
            pass

        record['canonical_url'] = canonical_url
    except Exception:
        # If requests/bs4 not available or network fails, skip canonical_url
        pass

    if firestore_client:
        doc_id = save_scraped_record(firestore_client, record, collection=collection)
        record['id'] = doc_id
        try:
            _inc_metric('firestore_writes')
        except Exception:
            pass
        try:
            logger.info(f"Saved scraped metadata to Firestore: id={doc_id} url={url} content_hash={content_hash}")
        except Exception:
            pass
    else:
        record['id'] = None

    return record


def fetch_scraped_content(firestore_client, url: str, collection: str = 'scraped_contents') -> Optional[Dict[str, Any]]:
    """Convenience helper: lookup metadata in Firestore and download GCS text/html if present.

    Returns a dict with the Firestore document fields. If `gcs_text_path` or
    `gcs_html_path` are present, their contents will be downloaded and added as
    `text` and/or `html` keys in the returned dict.
    """
    if firestore_client is None:
        return None
    try:
        doc = get_by_url(firestore_client, url, collection=collection)
        if not doc:
            return None
        # If there's a GCS path, download the text/html and attach
        gcs_text = doc.get('gcs_text_path')
        if gcs_text:
            try:
                doc['text'] = download_text_from_gcs(gcs_text)
            except Exception:
                # Don't fail hard; preserve metadata and proceed
                doc['text'] = None
        # Optionally load html if present
        gcs_html = doc.get('gcs_html_path')
        if gcs_html:
            try:
                doc['html'] = download_text_from_gcs(gcs_html)
            except Exception:
                doc['html'] = None
        return doc
    except Exception:
        return None
