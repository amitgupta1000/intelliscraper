Firestore TTL and cleanup
=========================

Recommended approaches to expire cache documents in Firestore:

1) Firestore TTL policy (recommended)

- In Google Cloud Console -> Firestore -> Indexes & TTL -> Add TTL
- Choose the field `expires_at` and enable TTL.
- Firestore will automatically delete documents after the timestamp.

2) Manual cleanup script (fallback)

You can run the included cleanup script which deletes expired documents:

```bash
python backend/scripts/cleanup_search_cache.py
```

3) Notes

- The codebase writes `expires_at` (UTC datetimes) to search cache and scraped content records.
- TTL in config: `SEARCH_CACHE_TTL` (seconds, default 86400 = 1 day) and `SCRAPE_RESULT_TTL` (seconds, default 864000 = 10 days).
