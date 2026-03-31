"""
Person C — Research Agent (Weeks 7-8)
Queries arXiv API live for papers not in the local corpus.
Owned jointly by Person C (integration) and Person D (data patterns).
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import logging
import os

log = logging.getLogger(__name__)
ARXIV_API = "http://export.arxiv.org/api/query"
NS = "{http://www.w3.org/2005/Atom}"


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Query the arXiv API and return structured paper metadata."""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            xml_data = resp.read()
    except Exception as e:
        log.error(f"arXiv API error: {e}")
        return []

    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall(f"{NS}entry"):
        papers.append({
            "arxiv_id": entry.find(f"{NS}id").text.split("/abs/")[-1],
            "title": entry.find(f"{NS}title").text.strip(),
            "summary": entry.find(f"{NS}summary").text.strip(),
            "authors": [a.find(f"{NS}name").text for a in entry.findall(f"{NS}author")],
            "published": entry.find(f"{NS}published").text[:10],
            "url": entry.find(f"{NS}id").text.strip(),
        })
    return papers


def research_agent_node(state: dict) -> dict:
    """LangGraph node: fetches live arXiv results and summarises them."""
    query = state["query"]
    papers = search_arxiv(query, max_results=5)

    if not papers:
        return {**state, "final_answer": "No papers found on arXiv for this query.", "sources": []}

    summary_lines = [f"Found {len(papers)} recent arXiv papers:\n"]
    for p in papers:
        summary_lines.append(f"**{p['title']}** ({p['published']})\n{p['summary'][:200]}...\n[{p['url']}]\n")

    sources = [{"doc_id": p["arxiv_id"], "title": p["title"], "score": 1.0} for p in papers]

    return {
        **state,
        "final_answer": "\n".join(summary_lines),
        "sources": sources,
        "research_results": papers,
    }
