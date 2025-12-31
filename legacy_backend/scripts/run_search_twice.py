"""Run a simulated search twice to demonstrate cache behavior.

This script monkeypatches UnifiedSearcher._search_serper to return a deterministic
result so we can test local + Firestore cache code paths without external API keys.
"""
import time
import logging
from backend.src import search as search_mod
from backend.src.search import UnifiedSearcher

logging.basicConfig(level=logging.DEBUG)

async def fake_search(self, query: str):
    # Return a simple list of SearchResult-like dicts
    from backend.src.search import SearchResult
    return [SearchResult(url='https://example.com/test', title='Example', snippet='snippet', source='fake')]

def main():
    s = UnifiedSearcher(max_results=3, cache_enabled=True, cache_ttl=3600)
    # monkeypatch the method with an async function that has proper name
    import types
    async def serper_stub(self, q):
        return await fake_search(self, q)
    serper_stub.__name__ = '_search_serper'
    s._search_serper = types.MethodType(serper_stub, s)

    q = 'test caching'
    print('First search (should miss and write cache)')
    res1 = s.search_sync(q)
    print('Results:', [r.url for r in res1])
    print('LAST_CACHE_HIT after first:', getattr(search_mod, 'LAST_CACHE_HIT', None))

    time.sleep(1)

    print('\nSecond search (should hit cache)')
    res2 = s.search_sync(q)
    print('Results:', [r.url for r in res2])
    print('LAST_CACHE_HIT after second:', getattr(search_mod, 'LAST_CACHE_HIT', None))

if __name__ == '__main__':
    main()
