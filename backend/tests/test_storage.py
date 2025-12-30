import types
from backend.src import storage


class DummyBlob:
    def __init__(self):
        self.content = None

    def upload_from_string(self, content, content_type=None):
        self.content = content


class DummyBucket:
    def __init__(self):
        self._blobs = {}

    def blob(self, name):
        b = DummyBlob()
        self._blobs[name] = b
        return b


class DummyGCSClient:
    def __init__(self):
        self._bucket = DummyBucket()

    def bucket(self, name):
        return self._bucket


class DummyFirestoreClient:
    def __init__(self):
        self._data = []

    def collection(self, name):
        return self

    def document(self, doc_id=None):
        outer = self

        class Doc:
            def __init__(self):
                self._id = str(len(outer._data) + 1)

            def set(self, record):
                rec = dict(record)
                rec['id'] = self._id
                outer._data.append(rec)

            @property
            def id(self):
                return self._id

        return Doc()

    def where(self, field, op, value):
        outer = self

        class Q:
            def __init__(self):
                pass

            def limit(self, n):
                def stream():
                    for d in outer._data:
                        if d.get(field) == value:
                            class D:
                                def __init__(self, dd):
                                    self._d = dd

                                def to_dict(self):
                                    return self._d

                            yield D(d)

                return stream()

        return Q()


def test_store_and_duplicate(monkeypatch):
    # Monkeypatch storage.gcs_lib to return our dummy client
    dummy_gcs_mod = types.SimpleNamespace()
    dummy_gcs_mod.Client = lambda: DummyGCSClient()
    monkeypatch.setattr(storage, 'gcs_lib', dummy_gcs_mod)

    # Use dummy firestore client instance
    dummy_db = DummyFirestoreClient()

    scraped = {
        'url': 'https://example.com/page1',
        'title': 'Example Page',
        'text': 'This is some example content for testing.',
        'html': '<html><body><p>Test</p></body></html>'
    }

    # First store should create a record
    record = storage.store_scraped_content(dummy_db, 'test-bucket', scraped, session_id='sess-1')
    assert record.get('id') is not None
    assert record.get('gcs_text_path', '').startswith('gs://test-bucket/scraped/')
    assert 'content_hash' in record

    # Second store with identical content should return exists
    record2 = storage.store_scraped_content(dummy_db, 'test-bucket', scraped, session_id='sess-1')
    assert record2.get('status') == 'exists'
