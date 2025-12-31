# types.py
# Minimal local type stubs to help static typing and linting within the workspace.
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""
    source: str = ""

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]
