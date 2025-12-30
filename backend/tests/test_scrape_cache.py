from datetime import datetime, timedelta
from backend.src import storage
import importlib


def test_scrape_cache_expiry(monkeypatch):
    # Import DummyFirestoreClient from the existing test helper module
    ts = importlib.import_module('backend.tests.test_storage')
    DummyFirestoreClient = ts.DummyFirestoreClient

    dummy_db = DummyFirestoreClient()

    scraped = {
        'url': 'https://example.com/page-cache',
        'title': 'Cache Test',
        'text': 'Cached content',
        'html': '<html><body>Cache</body></html>',
        'content_hash': 'hash-123'
    }

    # Save record
    doc_id = storage.save_scraped_record(dummy_db, dict(scraped))
    assert doc_id is not None

    # Should be retrievable by URL
    rec = storage.get_by_url(dummy_db, scraped['url'])
    assert rec is not None
    assert rec.get('id') is not None

    # exists_by_hash should return True
    assert storage.exists_by_hash(dummy_db, 'hash-123') is True

    # Simulate expiry by setting expires_at in the past
    assert len(dummy_db._data) == 1
    dummy_db._data[0]['expires_at'] = datetime.utcnow() - timedelta(days=11)

    # Now get_by_url should return None and exists_by_hash False
    rec2 = storage.get_by_url(dummy_db, scraped['url'])
    assert rec2 is None
    assert storage.exists_by_hash(dummy_db, 'hash-123') is False
