"""Web search providers and the parallel search executor.

The pipeline builds its own queries and fires them all concurrently as
plain HTTP calls — the LLM never drives searching. Providers are
interchangeable behind the SearchProvider protocol; TavilyProvider is
the default (TAVILY_API_KEY in .env).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Protocol, Sequence

# Simultaneous HTTP search requests.
SEARCH_WORKERS = 16


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int = 3) -> str:
        """Return formatted result snippets for one query."""
        ...


class TavilyProvider:
    """Tavily (tavily.com) — search API built for LLM agents."""

    def __init__(self, api_key: Optional[str] = None):
        from tavily import TavilyClient

        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise ValueError(
                "TAVILY_API_KEY not set — add it to 2026/.env (see .env.example)"
            )
        self._client = TavilyClient(api_key=key)

    def search(self, query: str, max_results: int = 3) -> str:
        response = self._client.search(
            query=query, max_results=max_results, search_depth="basic"
        )
        parts = []
        for result in response.get("results", []):
            title = result.get("title", "")
            url = result.get("url", "")
            content = (result.get("content") or "").strip()
            parts.append(f"{title} ({url})\n{content}")
        return "\n".join(parts) or "(no results)"


def search_many(
    provider: SearchProvider,
    queries: Sequence[tuple[str, int]],
    max_workers: int = SEARCH_WORKERS,
) -> list[str]:
    """Run (query, max_results) pairs concurrently, preserving order.

    A failed query becomes an inline note rather than an exception — one
    flaky search must not kill a 50-query fan-out.
    """

    def one(pair: tuple[str, int]) -> str:
        query, max_results = pair
        try:
            return provider.search(query, max_results)
        except Exception as exc:  # noqa: BLE001 — degrade, don't die
            return f"(search failed: {exc})"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(one, queries))
