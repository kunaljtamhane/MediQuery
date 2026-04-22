"""
Web Search Agent — Jaya (Weeks 7-8)
Handles general context retrieval for queries that go beyond the indexed
medical literature corpus. Uses DuckDuckGo (no API key required) with
structured result normalization.
"""
import logging
import os

log = logging.getLogger(__name__)

MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", 5))


def _ddg_search(query: str, max_results: int) -> list[dict]:
    """DuckDuckGo text search via duckduckgo_search library."""
    from duckduckgo_search import DDGS
    results = []
    with DDGS() as ddg:
        for r in ddg.text(query, max_results=max_results):
            results.append({
                "source": "web",
                "doc_id": f"web:{r.get('href', '')}",
                "title": r.get("title", ""),
                "summary": r.get("body", ""),
                "url": r.get("href", ""),
                "published": "Unknown",
                "authors": [],
            })
    return results


def search_web(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """
    Perform a web search and return structured results.
    Falls back to an empty list with a warning if the library is unavailable.
    """
    try:
        return _ddg_search(query, max_results)
    except ImportError:
        log.warning(
            "duckduckgo_search not installed — web search unavailable. "
            "Install with: pip install duckduckgo-search"
        )
        return []
    except Exception as e:
        log.error(f"Web search error: {e}")
        return []


def web_search_agent_node(state: dict) -> dict:
    """
    LangGraph node: performs a web search for the user's query,
    normalizes results, and returns them as sources.
    """
    query = state["query"]
    results = search_web(query)

    if not results:
        return {
            **state,
            "final_answer": "No web results found for this query.",
            "sources": [],
            "web_results": [],
        }

    lines = [f"Found {len(results)} web results.\n"]
    for r in results:
        snippet = r["summary"][:300] + "..." if len(r["summary"]) > 300 else r["summary"]
        lines.append(f"**{r['title']}**\n{snippet}\n[{r['url']}]\n")

    sources = [
        {
            "doc_id": r["doc_id"],
            "title": r["title"],
            "score": 1.0,
            "source": "web",
            "url": r["url"],
        }
        for r in results
    ]

    return {
        **state,
        "final_answer": "\n".join(lines),
        "sources": sources,
        "web_results": results,
    }
