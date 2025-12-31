import asyncio
import tempfile
import shutil
import os
from backend.src.search import UnifiedSearcher


def test_file_cache_save_and_load(tmp_path, monkeypatch):
    # Use a temporary cache dir
    temp_dir = tmp_path / "cache"
    monkeypatch.setenv('HOME', str(tmp_path))
    # Override global CACHE_DIR by patching module variable
    import importlib
    search_mod = importlib.import_module('backend.src.search')
    monkeypatch.setattr(search_mod, 'CACHE_DIR', str(temp_dir))
    # Ensure directory exists
    os.makedirs(str(temp_dir), exist_ok=True)

    searcher = UnifiedSearcher(max_results=3, cache_enabled=True, cache_ttl=60)

    async def _run():
        results = [search_mod.SearchResult(url='https://example.com', title='Ex', snippet='S', source='test')]
        ok = await searcher._save_to_cache('hello world', 'serper', results)
        assert ok
        cached = await searcher._check_cache('hello world', 'serper')
        assert cached is not None
        assert len(cached) == 1

    asyncio.run(_run())
