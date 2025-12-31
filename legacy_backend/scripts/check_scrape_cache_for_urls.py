import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.src import storage

def get_db_client():
    try:
        from google.cloud import firestore
        return firestore.Client()
    except Exception:
        # Try to use DummyFirestoreClient from tests if available
        try:
            ts = __import__('backend.tests.test_storage', fromlist=['DummyFirestoreClient'])
            return ts.DummyFirestoreClient()
        except Exception:
            return None

def check_urls(urls):
    db = get_db_client()
    results = []
    for u in urls:
        doc = None
        if db:
            try:
                doc = storage.get_by_url(db, u)
            except Exception as e:
                doc = {'error': str(e)}
        results.append({'url': u, 'doc': doc})
    return results

def main():
    if len(sys.argv) > 1:
        urls = sys.argv[1:]
    else:
        urls = [
            'https://example.com/run_scrape_twice',
            'https://en.wikipedia.org/wiki/Post-quantum_cryptography',
            'https://www.sciencedirect.com/science/article/pii/S0167404824001846'
        ]

    res = check_urls(urls)
    print(json.dumps(res, indent=2, default=str))

if __name__ == '__main__':
    main()
