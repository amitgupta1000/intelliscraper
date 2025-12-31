This directory records backend modules that have been archived to simplify the
project for a scraper-only deployment.

Why archived:
- These modules are large, dependency-heavy, and are unrelated to the core
  URL scraping workflow (LLMs, vector stores, hybrid retrieval, reporting).

How to restore:
- The original contents are available in your Git history. To restore a file,
  checkout the commit that contained it or copy the file from history into
  `backend/src/`.

Archived files (moved/stubbed in-place):
- backend/src/storage.py
- backend/src/search.py
- backend/src/prompt.py
- backend/src/nodes.py
- backend/src/llm_utils.py
- backend/src/hybrid_retriever.py
- backend/src/graph.py
- backend/src/gemini_fss_sample.py
- backend/src/fss_retriever.py
- backend/src/enhanced_embeddings.py
- backend/src/download_model.py
- backend/src/deprecated_cross_encoder_hybrid_retriever/*

Note: The original files were replaced with small stubs that explicitly raise
an error at import time to avoid accidental usage. This makes a scraper-only
deployment easier to reason about. Restore any archived module when you need
its functionality again.
