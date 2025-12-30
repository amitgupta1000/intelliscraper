"""Report Firestore cache state and storage metrics.

Usage:
  PYTHONPATH=. python backend/scripts/report_cache_state.py
"""
from google.cloud import firestore
from backend.src import storage
from datetime import datetime


def main():
    try:
        db = firestore.Client()
    except Exception as e:
        print("Failed to create Firestore client:", e)
        return

    def sample_collection(name, limit=5):
        coll = db.collection(name)
        docs = list(coll.limit(limit).stream())
        print(f"Collection '{name}' sample (count <= {limit}): {len(docs)}")
        for d in docs:
            try:
                data = d.to_dict()
                print(f" - id={d.id} url={data.get('url') or data.get('query')} expires={data.get('expires_at')}")
            except Exception:
                print(f" - id={d.id} (could not read)")

    print("Storage metrics:", storage.get_metrics())
    print("\nFirestore samples:")
    sample_collection('search_cache', limit=10)
    sample_collection('scraped_contents', limit=10)


if __name__ == '__main__':
    main()
