"""Cleanup script for Firestore search and scrape caches.

Usage:
  python backend/scripts/cleanup_search_cache.py

This script deletes expired documents from the Firestore collections:
- search_cache
- scraped_contents

Alternatively, prefer configuring Firestore TTL policies via the console or gcloud (see README_CACHE.md).
"""
from datetime import datetime
import os
import sys

try:
    from google.cloud import firestore
except Exception:
    print("google-cloud-firestore not installed or credentials not available.")
    sys.exit(1)

def cleanup_collection(client, collection_name):
    coll = client.collection(collection_name)
    now = datetime.utcnow()
    print(f"Cleaning collection: {collection_name} at {now.isoformat()}")
    docs = coll.stream()
    deleted = 0
    for d in docs:
        data = d.to_dict() or {}
        expires = data.get('expires_at')
        try:
            if expires and expires < now:
                coll.document(d.id).delete()
                deleted += 1
        except Exception:
            # If expires is not a datetime (string), attempt parse
            try:
                from dateutil import parser
                dt = parser.parse(expires)
                if dt < now:
                    coll.document(d.id).delete()
                    deleted += 1
            except Exception:
                pass
    print(f"Deleted {deleted} expired documents from {collection_name}.")

def main():
    client = firestore.Client()
    cleanup_collection(client, 'search_cache')
    cleanup_collection(client, 'scraped_contents')

if __name__ == '__main__':
    main()
