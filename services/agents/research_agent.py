"""
Medical Papers Agent — Kunal (Weeks 7-8)
Queries PubMed, arXiv, and medRxiv APIs concurrently for live literature retrieval.
Results are reranked by the cross-encoder reranker before returning to the Supervisor.

Replaces the old "Research Agent" which only covered arXiv + PubMed.
"""
import asyncio
import json
import logging
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import httpx

log = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
PUBMED_SEARCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
MEDRXIV_API = "https://api.biorxiv.org/details/medrxiv"
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8003")
NS = "{http://www.w3.org/2005/Atom}"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean(value: str | None) -> str:
    return " ".join((value or "").split())


def _normalize_month(value: str) -> str:
    m = _clean(value).lower()
    if not m:
        return "01"
    if m.isdigit():
        return m.zfill(2)
    return {
        "jan": "01", "january": "01", "feb": "02", "february": "02",
        "mar": "03", "march": "03", "apr": "04", "april": "04",
        "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
        "aug": "08", "august": "08", "sep": "09", "sept": "09",
        "september": "09", "oct": "10", "october": "10",
        "nov": "11", "november": "11", "dec": "12", "december": "12",
    }.get(m, "01")


def _pubmed_date(article: ET.Element) -> str:
    for path in (".//Article/ArticleDate", ".//JournalIssue/PubDate"):
        node = article.find(path)
        if node is not None:
            year = _clean(node.findtext("Year"))
            if year:
                month = _normalize_month(node.findtext("Month") or "")
                day = (_clean(node.findtext("Day")) or "01").zfill(2)
                return f"{year}-{month}-{day}"
    return "Unknown"


def _pubmed_abstract(article: ET.Element) -> str:
    sections = []
    for ab in article.findall(".//Article/Abstract/AbstractText"):
        label = _clean(ab.attrib.get("Label"))
        text = _clean("".join(ab.itertext()))
        if text:
            sections.append(f"{label}: {text}" if label else text)
    return "\n".join(sections) or "No abstract available."


def _pubmed_authors(article: ET.Element) -> list[str]:
    authors = []
    for a in article.findall(".//Article/AuthorList/Author"):
        cn = _clean(a.findtext("CollectiveName"))
        if cn:
            authors.append(cn)
            continue
        fn, ln = _clean(a.findtext("ForeName")), _clean(a.findtext("LastName"))
        if fn and ln:
            authors.append(f"{fn} {ln}")
        elif ln:
            authors.append(ln)
    return authors


# ── Source-specific search functions ──────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    try:
        with urllib.request.urlopen(f"{ARXIV_API}?{params}", timeout=10) as r:
            root = ET.fromstring(r.read())
    except Exception as e:
        log.error(f"arXiv API error: {e}")
        return []

    papers = []
    for entry in root.findall(f"{NS}entry"):
        arxiv_id = entry.find(f"{NS}id").text.split("/abs/")[-1]
        papers.append({
            "source": "arxiv",
            "doc_id": f"arxiv:{arxiv_id}",
            "title": _clean(entry.find(f"{NS}title").text),
            "summary": _clean(entry.find(f"{NS}summary").text),
            "authors": [_clean(a.find(f"{NS}name").text) for a in entry.findall(f"{NS}author")],
            "published": entry.find(f"{NS}published").text[:10],
            "url": entry.find(f"{NS}id").text.strip(),
        })
    return papers


def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    email = os.getenv("NCBI_EMAIL", "")
    api_key = os.getenv("NCBI_API_KEY", "")

    search_params: dict = {"db": "pubmed", "term": query, "retmode": "json",
                           "retmax": max_results, "sort": "pub date",
                           "tool": "mediquery-agent"}
    if email:
        search_params["email"] = email
    if api_key:
        search_params["api_key"] = api_key

    try:
        with urllib.request.urlopen(
            f"{PUBMED_SEARCH_API}?{urllib.parse.urlencode(search_params)}", timeout=10
        ) as r:
            ids = json.loads(r.read()).get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        log.error(f"PubMed search error: {e}")
        return []

    if not ids:
        return []

    fetch_params: dict = {"db": "pubmed", "id": ",".join(ids),
                          "retmode": "xml", "rettype": "abstract",
                          "tool": "mediquery-agent"}
    if email:
        fetch_params["email"] = email
    if api_key:
        fetch_params["api_key"] = api_key

    try:
        with urllib.request.urlopen(
            f"{PUBMED_FETCH_API}?{urllib.parse.urlencode(fetch_params)}", timeout=15
        ) as r:
            root = ET.fromstring(r.read())
    except Exception as e:
        log.error(f"PubMed fetch error: {e}")
        return []

    papers = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _clean(article.findtext(".//MedlineCitation/PMID"))
        title_node = article.find(".//Article/ArticleTitle")
        title = _clean("".join(title_node.itertext())) if title_node is not None else ""
        if not pmid or not title:
            continue
        papers.append({
            "source": "pubmed",
            "doc_id": f"pubmed:{pmid}",
            "title": title,
            "summary": _pubmed_abstract(article),
            "authors": _pubmed_authors(article),
            "published": _pubmed_date(article),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "journal": _clean(article.findtext(".//Article/Journal/Title")),
        })
    return papers


def search_medrxiv(query: str, max_results: int = 5) -> list[dict]:
    """
    Search medRxiv using the bioRxiv/medRxiv API.
    Fetches recent papers (last 180 days) and filters by keyword match in title/abstract.
    """
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=180)).isoformat()
    url = f"{MEDRXIV_API}/{start_date}/{end_date}/0/json"

    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        log.error(f"medRxiv API error: {e}")
        return []

    keywords = query.lower().split()
    papers = []
    for item in data.get("collection", []):
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        text = (title + " " + abstract).lower()
        if not any(kw in text for kw in keywords):
            continue
        doi = item.get("doi", "")
        papers.append({
            "source": "medrxiv",
            "doc_id": f"medrxiv:{doi}",
            "title": title,
            "summary": abstract[:600] if abstract else "No abstract.",
            "authors": [a.get("author_name", "") for a in item.get("authors", [])],
            "published": item.get("date", "Unknown"),
            "url": f"https://www.medrxiv.org/content/{doi}",
        })
        if len(papers) >= max_results:
            break
    return papers


def _rerank_papers(query: str, papers: list[dict]) -> list[dict]:
    """Score papers with cross-encoder reranker; falls back to original order."""
    if not papers:
        return papers
    try:
        resp = httpx.post(
            f"{RERANKER_URL}/rerank",
            json={"query": query, "candidates": [p["summary"] for p in papers]},
            timeout=5.0,
        )
        resp.raise_for_status()
        scores = resp.json()["scores"]
        for p, s in zip(papers, scores):
            p["rerank_score"] = s
        return sorted(papers, key=lambda x: x.get("rerank_score", 0), reverse=True)
    except Exception as e:
        log.warning(f"Reranker unavailable for medical papers ({e})")
        return papers


def _format_paper(paper: dict) -> str:
    authors = ", ".join(paper.get("authors", [])[:3])
    if len(paper.get("authors", [])) > 3:
        authors += " et al."
    journal = paper.get("journal", "")
    venue = f" | {journal}" if journal else ""
    author_line = f"\nAuthors: {authors}" if authors else ""
    return (
        f"**{paper['title']}** ({paper['published']}, {paper['source'].title()}{venue})"
        f"{author_line}\n{paper['summary'][:250]}...\n[{paper['url']}]\n"
    )


# ── LangGraph node ─────────────────────────────────────────────────────────────

def medical_papers_agent_node(state: dict) -> dict:
    """
    LangGraph node: queries PubMed, arXiv, and medRxiv concurrently,
    reranks all results, and returns structured sources.
    """
    query = state["query"]

    async def _gather():
        loop = asyncio.get_event_loop()
        arxiv_f = loop.run_in_executor(None, search_arxiv, query, 5)
        pubmed_f = loop.run_in_executor(None, search_pubmed, query, 5)
        medrxiv_f = loop.run_in_executor(None, search_medrxiv, query, 5)
        return await asyncio.gather(arxiv_f, pubmed_f, medrxiv_f)

    try:
        arxiv_papers, pubmed_papers, medrxiv_papers = asyncio.run(_gather())
    except RuntimeError:
        # Fallback if event loop already running (e.g. Jupyter)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as ex:
            a = ex.submit(search_arxiv, query, 5)
            p = ex.submit(search_pubmed, query, 5)
            m = ex.submit(search_medrxiv, query, 5)
            arxiv_papers, pubmed_papers, medrxiv_papers = a.result(), p.result(), m.result()

    all_papers = _rerank_papers(query, arxiv_papers + pubmed_papers + medrxiv_papers)

    if not all_papers:
        return {**state, "final_answer": "No papers found across PubMed, arXiv, or medRxiv.", "sources": []}

    lines = [f"Found {len(all_papers)} papers across PubMed, arXiv, and medRxiv.\n"]
    for source_name, source_papers in [
        ("arXiv", arxiv_papers), ("PubMed", pubmed_papers), ("medRxiv", medrxiv_papers)
    ]:
        if source_papers:
            lines.append(f"### {source_name}\n")
            lines.extend(_format_paper(p) for p in source_papers)

    sources = [
        {"doc_id": p["doc_id"], "title": p["title"],
         "score": p.get("rerank_score", 1.0), "source": p["source"], "url": p["url"]}
        for p in all_papers
    ]

    return {**state, "final_answer": "\n".join(lines), "sources": sources,
            "research_results": all_papers}


# Backward-compatibility alias
def research_agent_node(state: dict) -> dict:
    return medical_papers_agent_node(state)
